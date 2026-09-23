"""Choose the record-level validation split inside DS1 (reproducible selection criterion).

Criterion: over all DS1 record subsets of size 4-6, minimise
    sum_{c in N,S,V} |share_val(c) - target| + 2 * |min(share_val(F), f_cap) - f_cap|
where share_val(c) is the fraction of DS1 beats of class c that fall in the validation records.
Records in `--keep-in-train` are never validation candidates:
    208 - holds ~90 % of all DS1 fusion (F) beats
    207 - holds all DS1 ventricular escape (E) beats and all ventricular flutter episodes

Class counts per record are read from results/data_stats/record_stats.csv (they do not depend on the split).

Usage:
    python src/select_val_records.py
"""
import argparse
import itertools

import pandas as pd

from config import load_config, resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--target", type=float, default=0.2)
    ap.add_argument("--f-cap", type=float, default=0.1)
    ap.add_argument("--keep-in-train", type=int, nargs="*", default=[207, 208])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)

    classes = list(cfg["labels"])
    stats = pd.read_csv(resolve(cfg["paths"]["stats_dir"]) / "record_stats.csv").set_index("record")
    ds1 = stats.loc[cfg["splits"]["ds1"], classes]
    total = ds1.sum()
    candidates = [r for r in ds1.index if r not in args.keep_in_train]

    rows = []
    for k in args.sizes:
        for combo in itertools.combinations(candidates, k):
            share = ds1.loc[list(combo)].sum() / total
            score = sum(abs(share[c] - args.target) for c in classes if c != "F")
            score += 2 * abs(min(share["F"], args.f_cap) - args.f_cap)
            rows.append({"records": " ".join(map(str, sorted(combo))), "n_records": k, "score": round(score, 4),
                         **ds1.loc[list(combo)].sum().to_dict(), **{f"share_{c}": round(share[c], 3) for c in classes}})

    ranked = pd.DataFrame(rows).sort_values(["score", "n_records"]).head(args.top)
    print(f"DS1 totals: {total.to_dict()} | kept in train: {args.keep_in_train}")
    print(ranked.to_string(index=False))
    print(f"\nBest validation records: [{ranked.iloc[0]['records'].replace(' ', ', ')}]")


if __name__ == "__main__":
    main()
