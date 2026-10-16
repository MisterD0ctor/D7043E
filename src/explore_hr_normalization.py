"""Exploratory check on DS1: does normalizing the previous RR interval for heart rate help, and with which reference?

Compares the absolute previous RR interval with its ratio to several estimates of the patient's current rhythm:
    mean of the 10 previous intervals with the current one included   (rr_pre_ratio of src/features.py)
    mean of the 10 previous intervals without the current one
    median of the 32 or 128 previous intervals without the current one  (rr_pre_ratio_32, rr_pre_ratio_128)
    median of the whole record                                        (not causal, upper bound)

Three results are printed and the first is saved to results/exploration/hr_normalization.csv:
    1. separability of S and V from N by each single measure (AUC), pooled over DS1 and within records
    2. rank correlation of QRS width with heart rate among N beats (does shape depend on heart rate?)
    3. gradient boosting, leave-one-record-out, without any ratio, with rr_pre_ratio only, and with all three

DS2 is never read.

Usage:
    python src/explore_hr_normalization.py [--jobs 4]
"""
import argparse

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from config import class_names, load_config, resolve, symbol_to_class
from explore_data import build_record, class_metrics, ds1_records, feature_sets

REFERENCES = {  # name -> (number of previous intervals, current interval included, statistic)
    "ratio: mean of 10, current included (rr_pre_ratio)": (10, True, np.mean),
    "ratio: mean of 10, current excluded": (10, False, np.mean),
    "ratio: median of previous 32": (32, False, np.median),
    "ratio: median of previous 128": (128, False, np.median),
}
WHOLE_RECORD = "ratio: median of whole record (not causal)"
RATIOS = ["rr_pre_ratio", "rr_pre_ratio_32", "rr_pre_ratio_128"]  # the ratios among the measures of src/features.py


def rr_table(record: int, cfg: dict) -> pd.DataFrame:
    """Previous RR interval of every mappable beat and its ratio to each reference."""
    mapping, fs = symbol_to_class(cfg), cfg["signal"]["fs"]
    pre, post = cfg["window"]["pre_samples"], cfg["window"]["post_samples"]
    path = str(resolve(cfg["paths"]["mitdb_dir"]) / str(record))
    ann, n_samples = wfdb.rdann(path, "atr"), wfdb.rdheader(path).sig_len
    samples, symbols = np.asarray(ann.sample), np.asarray(ann.symbol)
    keep = np.array([s in mapping for s in symbols]) & (samples - pre >= 0) & (samples + post <= n_samples)
    qrs = np.sort(samples[np.isin(symbols, cfg["qrs_symbols"])])
    rr = np.diff(qrs) / fs
    idx = np.searchsorted(qrs, samples[keep])  # rr[idx - 1] is the interval that ends at the beat
    rr_pre = np.where(idx > 0, rr[np.clip(idx - 1, 0, len(rr) - 1)], np.nan)
    out = pd.DataFrame({"record": record, "y": [mapping[s] for s in symbols[keep]], "previous RR, absolute (s)": rr_pre})
    for name, (k, included, statistic) in REFERENCES.items():
        ends = idx if included else idx - 1
        out[name] = rr_pre / np.array([statistic(rr[max(e - k, 0):e]) if e >= 1 else np.nan for e in ends])
    out[WHOLE_RECORD] = rr_pre / np.median(rr)
    return out


def auc(v, pos, neg) -> float:
    ok = ~np.isnan(v) & (pos | neg)
    ranks = stats.rankdata(v[ok])
    n_pos, n_neg = pos[ok].sum(), neg[ok].sum()
    return (ranks[pos[ok]].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_predict(X_train, y_train, X_test, seed):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
    return model.fit(X_train, y_train).predict(X_test)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes, records = class_names(cfg), ds1_records(cfg)

    rrd = pd.concat([rr_table(r, cfg) for r in records], ignore_index=True)
    y, rec = rrd.y.to_numpy(), rrd.record.to_numpy()

    rows = []
    for name in rrd.columns[2:]:
        v, row = rrd[name].to_numpy(), {"measure": name}
        for k, c in ((1, "S"), (2, "V")):
            a = auc(v, y == k, y == 0)
            per_record = np.array([auc(v[rec == r], y[rec == r] == k, y[rec == r] == 0) for r in records
                                   if (y[rec == r] == k).sum() >= 10])
            row[f"{c} pooled"], row[f"{c} within"] = max(a, 1 - a), np.median(np.maximum(per_record, 1 - per_record))
        rows.append(row)
    table = pd.DataFrame(rows)
    out_dir = resolve("results/exploration")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "hr_normalization.csv", index=False)
    print("1. Separability from N by the single measure (AUC):")
    print(table.round(3).to_string(index=False))

    absolute, long_ratio = rrd["previous RR, absolute (s)"], rrd["ratio: median of previous 128"]
    current = rrd["ratio: mean of 10, current included (rr_pre_ratio)"]
    fast = max(records, key=lambda r: -np.nanmedian(absolute[(rec == r) & (y == 0)]))
    m = (rec == fast) & (y == 0)
    print(f"\n   Record {fast} (fastest rhythm): {np.mean(absolute[m] < 0.6):.0%} of its N beats have a previous RR below 0.6 s "
          f"(median of all S beats: {np.nanmedian(absolute[y == 1]):.2f} s); {np.mean(long_ratio[m] < 0.85):.0%} have a ratio below 0.85")
    runs = max(records, key=lambda r: ((rec == r) & (y == 1)).sum())
    m = (rec == runs) & (y == 1)
    print(f"   Record {runs} (most S beats, many in runs): previous RR below 0.6 s for {np.mean(absolute[m] < 0.6):.0%} of its S beats; "
          f"ratio below 0.85 for {np.mean(current[m] < 0.85):.0%} with the 10-beat mean and {np.mean(long_ratio[m] < 0.85):.0%} with the 128-beat median")

    built = [build_record(r, cfg) for r in records]
    X = pd.concat([b["features"] for b in built], ignore_index=True)
    assert (np.concatenate([b["y"] for b in built]) == y).all()
    normal = X[y == 0].assign(record=rec[y == 0])
    within = [stats.spearmanr(g.qrs_width, g.rr_local, nan_policy="omit")[0] for _, g in normal.groupby("record")]
    medians = normal.groupby("record")[["qrs_width", "rr_local"]].median()
    print(f"\n2. N beats: rank correlation of QRS width with the local RR interval: {np.nanmedian(within):+.2f} inside a record "
          f"(median over records), {stats.spearmanr(medians.qrs_width, medians.rr_local)[0]:+.2f} between records")

    for own, measure in (("ratio: median of previous 32", "rr_pre_ratio_32"), ("ratio: median of previous 128", "rr_pre_ratio_128")):
        assert np.allclose(rrd[own], X[measure], equal_nan=True), f"{measure} differs from the definition in this script"
    reference = feature_sets()["causal + running template"]
    print("\n3. Gradient boosting, leave-one-record-out:")
    for name, cols in (("without any RR ratio", [c for c in reference if c not in RATIOS]),
                       ("with rr_pre_ratio only", [c for c in reference if c not in RATIOS[1:]]),
                       ("with all three RR ratios", reference)):
        data = X[cols].to_numpy()
        folds = Parallel(n_jobs=args.jobs)(delayed(fit_predict)(data[rec != r], y[rec != r], data[rec == r], cfg["seed"])
                                           for r in records)
        pred = np.zeros_like(y)
        for r, p in zip(records, folds):
            pred[rec == r] = p
        m = class_metrics(y, pred, classes)
        print(f"   {name:34s} accuracy {m['accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}, S precision {m['precision_S']:.2f} "
              f"recall {m['recall_S']:.2f} F1 {m['f1_S']:.2f}, V F1 {m['f1_V']:.2f}")


if __name__ == "__main__":
    main()
