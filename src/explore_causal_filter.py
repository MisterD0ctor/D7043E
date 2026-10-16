"""Exploratory check on DS1: does a causal band-pass filter change classification compared with the zero-phase one?

The prepared data use a zero-phase filter (forward and backward over the whole record), which needs future samples.
A device has to filter causally. This script prepares the development records both ways and compares gradient
boosting on the causal measures with leave-one-record-out cross-validation. It also reports how far the largest
deflection of the QRS complex moves relative to the annotation when the filter is causal.

DS2 is never read. Output: results/exploration/causal_filter.csv

Usage:
    python src/explore_causal_filter.py [--jobs 4]
"""
import argparse
import copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

from config import class_names, load_config, resolve
from explore_data import build_record, class_metrics, ds1_records, feature_sets
from prepare_data import load_record, preprocess_signal


def fit_predict(X_train, y_train, X_test, seed):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
    return model.fit(X_train, y_train).predict(X_test)


def peak_offset_ms(record: int, cfg: dict) -> float:
    """Median distance from the annotation to the largest deflection within +-50 ms of it (ms)."""
    x, fs, samples, symbols = load_record(cfg["paths"]["mitdb_dir"], record, cfg["signal"]["lead"])
    xf, half = preprocess_signal(x, fs, cfg), int(round(0.05 * fs))
    beats = samples[np.isin(symbols, cfg["labels"]["N"]) & (samples > half) & (samples < len(x) - half)]
    return float(np.median(np.abs(xf[beats[:, None] + np.arange(-half, half + 1)[None, :]]).argmax(axis=1) - half) / fs * 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    base = load_config(args.config)
    classes, records = class_names(base), ds1_records(base)
    cols = feature_sets()["causal + running template"]

    rows = []
    for zero_phase in (True, False):
        cfg = copy.deepcopy(base)
        cfg["signal"]["bandpass"]["zero_phase"] = zero_phase
        built = Parallel(n_jobs=args.jobs)(delayed(build_record)(r, cfg) for r in records)
        X = pd.concat([b["features"][cols] for b in built], ignore_index=True).to_numpy()
        y, rec = np.concatenate([b["y"] for b in built]), np.concatenate([b["record"] for b in built])
        folds = Parallel(n_jobs=args.jobs)(delayed(fit_predict)(X[rec != r], y[rec != r], X[rec == r], base["seed"])
                                           for r in records)
        pred = np.zeros_like(y)
        for r, p in zip(records, folds):
            pred[rec == r] = p
        offset = np.median([peak_offset_ms(r, cfg) for r in records])
        rows.append({"filter": "zero-phase" if zero_phase else "causal", "qrs_peak_offset_ms": offset,
                     **class_metrics(y, pred, classes)})
    table = pd.DataFrame(rows)
    out_dir = resolve("results/exploration")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "causal_filter.csv", index=False)
    show = ["filter", "qrs_peak_offset_ms", "accuracy", "macro_f1"] + [f"f1_{c}" for c in classes]
    print(f"Gradient boosting on {len(cols)} causal measures, leave-one-record-out over {len(records)} development records:")
    print(table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
