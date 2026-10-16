"""Paired patient bootstrap: network configurations against the gradient-boosting baseline - Task 2, exploration.

Both are scored on the same pooled leave-one-record-out predictions of the 21 development records. Each bootstrap
sample draws 21 records with replacement and recomputes the macro F1 of both models on those records, averaged
over the seeds of each model. The baseline is gradient boosting on the 75 causal measures without class weights,
seeds 0-2 (results/exploration/ideas/refine_predictions.npz).

Usage:
    python src/explore_compare.py --plan round1 [round2 ...]
"""
import argparse
from collections import defaultdict

import numpy as np

from config import resolve

BASELINE = [f"gradient boosting | none | reference | seed {s}" for s in range(3)]


def record_confusions(y, pred, rec, records):
    """4x4 confusion matrix per record, shape (records, 4, 4)."""
    out = np.zeros((len(records), 4, 4), np.int64)
    for i, r in enumerate(records):
        m = rec == r
        np.add.at(out[i], (y[m], pred[m]), 1)
    return out


def macro_f1(cm):
    """Macro F1 over the four classes from confusion matrices (..., 4, 4); absent classes count as 0."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-1) + cm.sum(-2)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=5000)
    args = ap.parse_args()
    base = np.load(resolve("results/exploration/ideas/refine_predictions.npz"))
    y, rec = base["y"], base["record"]
    records = np.unique(rec)
    base_cm = np.stack([record_confusions(y, base[k], rec, records) for k in BASELINE])  # (seeds, records, 4, 4)
    rng = np.random.default_rng(0)
    draws = np.stack([np.bincount(rng.integers(0, len(records), len(records)), minlength=len(records))
                      for _ in range(args.samples)])  # (samples, records): how often each record is drawn
    boot = lambda cm: macro_f1(np.einsum("br,srij->bsij", draws, cm)).mean(1)  # noqa: E731
    base_point, base_boot = macro_f1(base_cm.sum(1)).mean(), boot(base_cm)
    print(f"baseline (gradient boosting, seeds 0-2): macro F1 {base_point:.3f}, "
          f"95% interval {np.percentile(base_boot, 2.5):.3f}-{np.percentile(base_boot, 97.5):.3f}\n")
    print(f"{'configuration':72s} {'F1':>6s} {'diff':>7s} {'95% interval':>16s} {'P(better)':>9s}")
    for plan in args.plan:
        z = np.load(resolve(f"results/exploration/networks/{plan}_predictions.npz"))
        seeds = defaultdict(list)
        for k in z.files:
            if not k.endswith("| record"):
                tag = k.rsplit(" | seed", 1)[0]
                if len(z[k]) != len(rec):  # e.g. a longer pre-window drops the first beats of some records
                    seeds.setdefault(tag, None)
                    continue
                assert (z[k + " | record"] == rec).all(), "beat order differs from the baseline"
                seeds[tag].append(record_confusions(y, z[k], rec, records))
        for tag, cms in seeds.items():
            if cms is None:
                print(f"{tag[:72]:72s} (other beats than the baseline, not compared)")
                continue
            cm = np.stack(cms)
            point, diff = macro_f1(cm.sum(1)).mean(), boot(cm) - base_boot
            lo, hi = np.percentile(diff, [2.5, 97.5])
            print(f"{tag[:72]:72s} {point:6.3f} {point - base_point:+7.3f} {lo:+7.3f} to {hi:+.3f} "
                  f"{(diff > 0).mean():9.2f}")


if __name__ == "__main__":
    main()
