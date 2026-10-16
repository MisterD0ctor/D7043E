"""Shared data helpers for the exploratory model comparisons on DS1 (explore_baselines.py, explore_edge_models.py).

Exploration only: nothing here is the final training or evaluation pipeline, and DS2 is never read.
Every function takes its records from `splits.ds1` of configs/data.yaml.

Noise comes from the parts of each NSTDB noise record defined in `noise.parts` (configs/data.yaml):
"train" for augmentation, "cv" for noisy held-out records during model selection. The "eval" part is never
used here; it is reserved for the final robustness test on DS2.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from config import development_records
from features import FEATURES, _corr, _windows, beat_features
from noise import load_noise, noise_power, qrs_power, scaled_noise
from prepare_data import extract_beats, load_record, preprocess_signal

# Measures that a device cannot compute at decision time (see src/features.py)
NEXT_BEAT = ["rr_post", "rr_post_ratio", "rr_pre_post", "rr_diff"]
WHOLE_RECORD = ["template_corr", "template_rmse", "qrs_ptp_rel", "qrs_width_rel", "energy_rel"]
# Causal replacements of the whole-record measures: reference = the previous RUN_BEATS beats of the record
RUN_BEATS = 32
RUNNING = [f"{name}_run" for name in WHOLE_RECORD]


def class_metrics(y_true, y_pred, classes) -> dict:
    """Same metrics as evaluate.compute_metrics, kept here so the baseline workers do not have to import torch."""
    labels = list(range(len(classes)))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {"accuracy": accuracy_score(y_true, y_pred),
               "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)}
    for i, c in enumerate(classes):
        metrics.update({f"precision_{c}": p[i], f"recall_{c}": r[i], f"f1_{c}": f[i], f"support_{c}": int(support[i])})
    return metrics


def feature_sets() -> dict[str, list[str]]:
    causal = [f for f in FEATURES if f not in NEXT_BEAT + WHOLE_RECORD]
    rhythm = [f for f, (family, _) in FEATURES.items() if family == "rhythm" and f not in NEXT_BEAT]
    return {
        "all measures (not causal)": list(FEATURES),
        "causal": causal,
        "causal + running template": causal + RUNNING,
        "causal + running template + next beat": causal + RUNNING + NEXT_BEAT,
        "no rhythm": [f for f in causal + RUNNING if f not in rhythm],
        "rhythm only": rhythm,
    }


def ds1_records(cfg: dict) -> list[int]:
    """DS1 records available for development: DS1 without the records of `splits.exclude`."""
    return development_records(cfg)


def running_relative_features(x: np.ndarray, fs: float, feats: pd.DataFrame, cfg: dict, k: int = RUN_BEATS) -> pd.DataFrame:
    """The WHOLE_RECORD measures recomputed against the median of the previous k beats (NaN for the first beat)."""
    s = feats["sample"].to_numpy()
    pre, post = cfg["window"]["pre_samples"], cfg["window"]["post_samples"]
    ms = lambda t: int(round(t * fs / 1000))  # noqa: E731
    base = np.median(_windows(x, s, -pre, post), axis=1, keepdims=True)
    seg = _windows(x, s, -ms(100), ms(150)) - base
    template = np.full_like(seg, np.nan)
    for i in range(1, min(k, len(s))):
        template[i] = np.median(seg[:i], axis=0)
    if len(s) > k:
        template[k:] = np.median(sliding_window_view(seg, k, axis=0)[:-1], axis=-1)
    reference = lambda name: feats[name].shift(1).rolling(k, min_periods=1).median()  # noqa: E731
    return pd.DataFrame({
        "template_corr_run": _corr(seg, template),
        "template_rmse_run": np.sqrt(((seg - template) ** 2).mean(axis=1)),
        "qrs_ptp_rel_run": feats["qrs_ptp"] / reference("qrs_ptp"),
        "qrs_width_rel_run": feats["qrs_width"] / reference("qrs_width"),
        "energy_rel_run": feats["energy"] / reference("energy"),
    }, index=feats.index)


@lru_cache(maxsize=None)
def _noise(cfg_noise_key: tuple, noise_type: str, part: str):
    assert part in ("train", "cv"), "exploration must not use the noise reserved for the final test"
    cfg = _CFG[cfg_noise_key]
    return load_noise(cfg, noise_type, part=part), noise_power(cfg, noise_type)


_CFG: dict[tuple, dict] = {}


def exploration_noise(cfg: dict, record: int, x: np.ndarray, fs: float, samples, symbols, noise_type: str, snr_db: float,
                      part: str = "cv", copy: int = 0):
    """NSTDB noise for one record at the nst SNR from one part of the noise record; deterministic per argument set."""
    key = (cfg["paths"]["nstdb_dir"], cfg["noise"]["channel"], tuple(cfg["noise"]["parts"].items()))
    _CFG[key] = cfg
    noise, n_power = _noise(key, noise_type, part)
    rng = np.random.default_rng([cfg["seed"], record, list(cfg["noise"]["types"]).index(noise_type),
                                 int(round(snr_db * 10)) + 1000, ["train", "cv"].index(part), copy])
    return scaled_noise(noise, len(x), qrs_power(x, fs, samples, symbols), n_power, snr_db, rng)


def build_record(record: int, cfg: dict, noise: tuple | None = None, with_features: bool = True) -> dict:
    """Beat windows and (optionally) all measures of one DS1 record.

    `noise` = None, (noise_type, snr_db) for a noisy held-out record (noise part "cv"), or
    (noise_type, snr_db, part, copy) to choose the part and the random copy explicitly.
    """
    assert record in cfg["splits"]["ds1"], f"record {record} is not in DS1 - exploration must not read DS2"
    x, fs, samples, symbols = load_record(cfg["paths"]["mitdb_dir"], record, cfg["signal"]["lead"])
    if noise is not None:
        x = x + exploration_noise(cfg, record, x, fs, samples, symbols, *noise)
    xf = preprocess_signal(x, fs, cfg)
    arrays, _, _ = extract_beats(xf, fs, samples, symbols, cfg)
    out = {"X": arrays["X"], "y": arrays["y"], "record": np.full(len(arrays["y"]), record)}
    if with_features:
        feats = beat_features(xf, fs, samples, symbols, cfg)
        assert (feats["sample"].to_numpy() == arrays["sample"]).all()
        out["features"] = pd.concat([feats[list(FEATURES)], running_relative_features(xf, fs, feats, cfg)], axis=1)
    return out


def noise_conditions(cfg: dict) -> list[tuple | None]:
    return [None] + [(t, s) for t in cfg["noise"]["types"] for s in cfg["noise"]["snr_db"]]


def condition_name(noise: tuple | None) -> str:
    return "clean" if noise is None else f"{noise[0]} {noise[1]:g} dB"
