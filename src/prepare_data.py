"""CRISP-DM stage 3 - build patient-independent heartbeat datasets from MIT-BIH.

Processing order for every record:
    raw MLII signal (mV)
    -> zero-phase Butterworth band-pass on the whole record (0.5-40 Hz)
    -> segmentation: fixed window [R - pre, R + post) around each annotated beat
    -> label mapping (MIT-BIH symbol -> N/S/V/F, other symbols excluded)
    -> per-window z-score normalization
    -> clipping to +-clip and scaling to [-1, 1]
    -> model input of shape (1, pre + post)

Outputs:
    data/processed/{train,val,test}.npz      X, y, record, sample, symbol, rr_pre, rr_post, rr_local
    results/data_stats/record_stats.csv      per-record class counts, exclusions, amplitude/quality stats
    results/data_stats/split_summary.csv     per-split N/S/V/F counts (report table 3.8)
    results/data_stats/annotation_mapping.csv  every annotation symbol seen -> class or exclusion reason
    results/data_stats/prep_manifest.json    parameters, software versions, seed, output checksums

Usage:
    python src/prepare_data.py [--config configs/data.yaml]
"""
import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import wfdb
from scipy import signal as sps

from config import class_names, load_config, resolve, split_records, symbol_to_class


# ---------------------------------------------------------------------------
# Reusable building blocks (also used by noise_test.py and the EDA notebook)
# ---------------------------------------------------------------------------

def load_record(mitdb_dir, record: int, lead: str):
    """Return (signal of the requested lead in mV, fs, annotation samples, annotation symbols)."""
    path = str(resolve(mitdb_dir) / str(record))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    if lead not in rec.sig_name:
        raise KeyError(f"Record {record}: lead {lead} not available (has {rec.sig_name})")
    x = rec.p_signal[:, rec.sig_name.index(lead)].astype(np.float64)
    return x, rec.fs, np.asarray(ann.sample), np.asarray(ann.symbol)


def preprocess_signal(x: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Filtering applied to a whole (raw or noise-corrupted) record before segmentation."""
    bp = cfg["signal"]["bandpass"]
    if bp["enabled"]:
        sos = sps.butter(bp["order"], [bp["low_hz"], bp["high_hz"]], btype="bandpass", fs=fs, output="sos")
        x = sps.sosfiltfilt(sos, x)
    return x


def normalize_windows(W: np.ndarray, cfg: dict) -> np.ndarray:
    norm = cfg["normalization"]
    if norm["method"] == "zscore":
        W = (W - W.mean(axis=1, keepdims=True)) / np.maximum(W.std(axis=1, keepdims=True), 1e-6)
    elif norm["method"] != "none":
        raise ValueError(f"Unknown normalization method {norm['method']}")
    if norm.get("clip"):
        W = np.clip(W, -norm["clip"], norm["clip"]) / norm["clip"]
    return W.astype(np.float32)


def rr_features(qrs_samples: np.ndarray, beat_samples: np.ndarray, fs: float, n_local: int = 10):
    """RR intervals (s) of each beat: previous, next, and mean of the last n_local previous intervals."""
    q = np.sort(qrs_samples)
    rr = np.diff(q) / fs
    idx = np.searchsorted(q, beat_samples)
    pre = np.where(idx > 0, rr[np.clip(idx - 1, 0, len(rr) - 1)], np.nan)
    post = np.where(idx < len(q) - 1, rr[np.clip(idx, 0, len(rr) - 1)], np.nan)
    csum = np.concatenate([[0.0], np.cumsum(rr)])
    lo = np.clip(idx - n_local, 0, None)
    n = idx - lo
    local = np.where(n > 0, (csum[idx] - csum[lo]) / np.maximum(n, 1), np.nan)
    return pre.astype(np.float32), post.astype(np.float32), local.astype(np.float32)


def extract_beats(x: np.ndarray, fs: float, samples: np.ndarray, symbols: np.ndarray, cfg: dict):
    """Segment and label all mappable beats of one filtered record.

    Returns (arrays dict, Counter of excluded symbols, number of beats dropped at record edges).
    """
    mapping = symbol_to_class(cfg)
    pre, post = cfg["window"]["pre_samples"], cfg["window"]["post_samples"]

    is_beat = np.array([s in mapping for s in symbols])
    excluded = Counter(symbols[~is_beat].tolist())
    in_bounds = (samples - pre >= 0) & (samples + post <= len(x))
    keep = is_beat & in_bounds
    edge_dropped = int((is_beat & ~in_bounds).sum())

    s_keep = samples[keep]
    W = np.stack([x[s - pre:s + post] for s in s_keep]) if len(s_keep) else np.empty((0, pre + post))
    qrs = samples[np.isin(symbols, cfg["qrs_symbols"])]
    rr_pre, rr_post, rr_local = rr_features(qrs, s_keep, fs)
    arrays = {
        "X": normalize_windows(W, cfg)[:, None, :],
        "y": np.array([mapping[s] for s in symbols[keep]], dtype=np.int64),
        "sample": s_keep.astype(np.int64),
        "symbol": symbols[keep].astype("U1"),
        "rr_pre": rr_pre,
        "rr_post": rr_post,
        "rr_local": rr_local,
    }
    return arrays, excluded, edge_dropped


def process_record(record: int, cfg: dict, noise: np.ndarray | None = None):
    """Load -> (optionally add noise to the raw signal) -> filter -> extract beats."""
    x, fs, samples, symbols = load_record(cfg["paths"]["mitdb_dir"], record, cfg["signal"]["lead"])
    assert fs == cfg["signal"]["fs"], f"Record {record}: fs={fs}, expected {cfg['signal']['fs']}"
    if noise is not None:
        x = x + noise[: len(x)]
    arrays, excluded, edge_dropped = extract_beats(preprocess_signal(x, fs, cfg), fs, samples, symbols, cfg)
    arrays["record"] = np.full(len(arrays["y"]), record, dtype=np.int16)
    return arrays, excluded, edge_dropped, x


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _annotation_descriptions() -> dict[str, str]:
    try:
        table = wfdb.io.annotation.ann_label_table
        return dict(zip(table["symbol"], table["description"]))
    except AttributeError:
        return {}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])

    classes = class_names(cfg)
    mapping = symbol_to_class(cfg)
    splits = split_records(cfg)
    out_dir = resolve(cfg["paths"]["processed_dir"])
    stats_dir = resolve(cfg["paths"]["stats_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    record_rows, symbol_counts, summary_rows, checksums = [], {}, [], {}
    for split, records in splits.items():
        parts = []
        for rec in records:
            arrays, excluded, edge_dropped, raw = process_record(rec, cfg)
            parts.append(arrays)
            counts = np.bincount(arrays["y"], minlength=len(classes))
            for sym, n in Counter(arrays["symbol"].tolist()).items():
                symbol_counts.setdefault(sym, Counter())[split] += n
            for sym, n in excluded.items():
                symbol_counts.setdefault(sym, Counter())[split] += n
            record_rows.append({
                "record": rec, "split": split,
                **{c: int(n) for c, n in zip(classes, counts)},
                "total": int(counts.sum()),
                "excluded_beats": sum(n for s, n in excluded.items() if s in cfg["qrs_symbols"]),
                "non_beat_annotations": sum(n for s, n in excluded.items() if s not in cfg["qrs_symbols"]),
                "edge_dropped": edge_dropped,
                "n_samples": len(raw),
                "nan_samples": int(np.isnan(raw).sum()),
                "raw_min_mV": float(np.nanmin(raw)), "raw_max_mV": float(np.nanmax(raw)),
                "raw_p1_mV": float(np.nanpercentile(raw, 1)), "raw_p99_mV": float(np.nanpercentile(raw, 99)),
                "raw_std_mV": float(np.nanstd(raw)),
                "clipped_fraction": float((np.abs(arrays["X"]) >= 1.0).mean()) if len(arrays["y"]) else 0.0,
                "median_rr_s": float(np.nanmedian(arrays["rr_pre"])) if len(arrays["y"]) else np.nan,
            })
            print(f"  [{split:5s}] record {rec}: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))

        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        path = out_dir / f"{split}.npz"
        np.savez_compressed(path, **data, classes=np.array(classes))
        checksums[path.name] = _sha256(path)
        counts = np.bincount(data["y"], minlength=len(classes))
        summary_rows.append({
            "split": split, "n_records": len(records), "records": " ".join(map(str, records)),
            **{c: int(n) for c, n in zip(classes, counts)}, "total": int(counts.sum()),
        })

    record_df = pd.DataFrame(record_rows)
    summary_df = pd.DataFrame(summary_rows)
    total = {"split": "all", "n_records": int(summary_df["n_records"].sum()), "records": ""}
    total.update({c: int(summary_df[c].sum()) for c in classes + ["total"]})
    summary_df = pd.concat([summary_df, pd.DataFrame([total])], ignore_index=True)

    desc = _annotation_descriptions()
    mapping_rows = []
    for sym in sorted(symbol_counts):
        if sym in mapping:
            target, reason = classes[mapping[sym]], ""
        elif sym in cfg["qrs_symbols"]:
            target, reason = "excluded", "beat type outside N/S/V/F (paced, fusion of paced/normal, unclassifiable)"
        else:
            target, reason = "excluded", "non-beat annotation (rhythm/quality/artifact marker)"
        c = symbol_counts[sym]
        mapping_rows.append({
            "symbol": sym, "description": desc.get(sym, ""), "final_class": target, "reason": reason,
            **{s: c.get(s, 0) for s in splits}, "total": sum(c.values()),
        })
    mapping_df = pd.DataFrame(mapping_rows).sort_values(["final_class", "total"], ascending=[True, False])

    record_df.to_csv(stats_dir / "record_stats.csv", index=False)
    summary_df.to_csv(stats_dir / "split_summary.csv", index=False)
    mapping_df.to_csv(stats_dir / "annotation_mapping.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": cfg,
        "input_shape": [1, cfg["window"]["pre_samples"] + cfg["window"]["post_samples"]],
        "classes": classes,
        "splits": splits,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "wfdb": wfdb.__version__, "platform": platform.platform(),
        },
        "outputs_sha256": checksums,
    }
    with open(stats_dir / "prep_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + summary_df.drop(columns="records").to_string(index=False))
    print(f"\nWrote datasets to {out_dir} and statistics to {stats_dir}")


if __name__ == "__main__":
    main()
