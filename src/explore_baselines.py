"""Exploratory comparison of feature-based baselines on DS1 (CRISP-DM stage 4 - exploration, not final results).

Protocols (DS2 is never read):
    loro    leave-one-record-out over the 22 DS1 records. The out-of-fold predictions of all records are pooled
            before the metrics are computed, so every DS1 beat is predicted by a model that never saw its record.
    split   the project's fixed split: fit on the 18 training records, predict the 4 validation records.

Inputs are the signal measures of src/features.py; the feature sets are defined in src/explore_data.py.
The reference baseline is also evaluated on the held-out record with NSTDB noise added (bw, em, ma at 12/6/0 dB).

Outputs in results/exploration/:
    baseline_metrics.csv        one row per model x feature set x protocol x noise condition
    baseline_per_record.csv     results per held-out record for the reference baseline
    baseline_predictions.npz    pooled out-of-fold predictions of every run (clean condition)

Usage:
    python src/explore_baselines.py [--jobs 6]
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from config import class_names, load_config, resolve, split_records
from explore_data import build_record, class_metrics, condition_name, ds1_records, feature_sets, noise_conditions

REFERENCE_SET = "causal + running template"
REFERENCE_MODEL = "gradient boosting"


def model_factories(seed: int) -> dict:
    def scaled(classifier):  # heavy-tailed measures (RR gaps, ratios) -> rank-based mapping to a normal distribution
        return make_pipeline(SimpleImputer(strategy="median"),
                             QuantileTransformer(n_quantiles=256, output_distribution="normal", subsample=50_000,
                                                 random_state=seed), classifier)
    return {
        "logistic regression": lambda: scaled(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "logistic regression, unweighted": lambda: scaled(LogisticRegression(max_iter=1000)),
        "LDA": lambda: scaled(LinearDiscriminantAnalysis()),
        "random forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=1)),
        "gradient boosting": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed),
    }


def build_measures(record: int, cfg: dict, noise) -> dict:
    """Measures only: the beat windows are not needed here and would cost about 1 GB over all conditions."""
    built = build_record(record, cfg, noise)
    return {"features": built["features"], "y": built["y"], "record": built["record"]}


def fit_predict(make_model, X_train, y_train, tests: dict) -> dict:
    model = make_model().fit(X_train, y_train)
    return {name: model.predict(X) for name, X in tests.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes, records = class_names(cfg), ds1_records(cfg)
    val_records = split_records(cfg)["val"]
    out_dir = resolve("results/exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    conditions = noise_conditions(cfg)
    built = Parallel(n_jobs=args.jobs)(delayed(build_measures)(r, cfg, c) for c in conditions for r in records)
    tables, y, rec = {}, None, None
    for i, c in enumerate(conditions):
        parts = built[i * len(records):(i + 1) * len(records)]
        tables[condition_name(c)] = pd.concat([p["features"] for p in parts], ignore_index=True)
        y_c, rec_c = np.concatenate([p["y"] for p in parts]), np.concatenate([p["record"] for p in parts])
        assert y is None or ((y == y_c).all() and (rec == rec_c).all())
        y, rec = y_c, rec_c
    del built
    print(f"built {len(y):,} DS1 beats x {tables['clean'].shape[1]} measures for {len(conditions)} conditions "
          f"in {time.time() - t0:.0f} s", flush=True)

    sets, factories = feature_sets(), model_factories(cfg["seed"])
    runs = [(REFERENCE_MODEL, s) for s in sets]
    runs += [("logistic regression", s) for s in (REFERENCE_SET, "causal", "rhythm only", "no rhythm")]
    runs += [("logistic regression, unweighted", REFERENCE_SET), ("LDA", REFERENCE_SET),
             ("random forest", REFERENCE_SET), ("random forest", "causal")]

    always_n = np.zeros_like(y)
    rows = [{"model": "always N", "features": "-", "n_features": 0, "protocol": "loro", "condition": "clean",
             **class_metrics(y, always_n, classes)},
            {"model": "always N", "features": "-", "n_features": 0, "protocol": "split", "condition": "clean",
             **class_metrics(y[np.isin(rec, val_records)], always_n[np.isin(rec, val_records)], classes)}]
    predictions = {"y": y, "record": rec}
    for model_name, set_name in runs:
        t0 = time.time()
        cols, make_model = sets[set_name], factories[model_name]
        with_noise = (set_name == REFERENCE_SET) and model_name in (REFERENCE_MODEL, "logistic regression")
        conds = list(tables) if with_noise else ["clean"]
        X = {c: tables[c][cols].to_numpy() for c in conds}

        folds = Parallel(n_jobs=args.jobs)(
            delayed(fit_predict)(make_model, X["clean"][rec != r], y[rec != r], {c: X[c][rec == r] for c in conds})
            for r in records)
        pooled = {c: np.zeros_like(y) for c in conds}
        for r, fold in zip(records, folds):
            for c in conds:
                pooled[c][rec == r] = fold[c]
        base = {"model": model_name, "features": set_name, "n_features": len(cols)}
        for c in conds:
            rows.append({**base, "protocol": "loro", "condition": c, **class_metrics(y, pooled[c], classes)})
        predictions[f"{model_name} | {set_name}"] = pooled["clean"]

        val = np.isin(rec, val_records)
        split_pred = fit_predict(make_model, X["clean"][~val], y[~val], {"clean": X["clean"][val]})["clean"]
        rows.append({**base, "protocol": "split", "condition": "clean", **class_metrics(y[val], split_pred, classes)})

        m = next(row for row in rows if row["model"] == model_name and row["features"] == set_name
                 and row["protocol"] == "loro" and row["condition"] == "clean")
        print(f"{model_name:32s} | {set_name:38s} | loro macro-F1 {m['macro_f1']:.3f} "
              f"(F1 N {m['f1_N']:.2f} S {m['f1_S']:.2f} V {m['f1_V']:.2f} F {m['f1_F']:.2f}) | "
              f"split macro-F1 {rows[-1]['macro_f1']:.3f} | {time.time() - t0:.0f} s", flush=True)

        if model_name == REFERENCE_MODEL and set_name == REFERENCE_SET:
            per_record = []
            for r in records:
                m_r, p_r, t_r = rec == r, pooled["clean"][rec == r], y[rec == r]
                row = {"record": r, "split": "val" if r in val_records else "train", "beats": int(m_r.sum()),
                       "accuracy": float((p_r == t_r).mean())}
                for k, c in enumerate(classes):
                    row[f"support_{c}"] = int((t_r == k).sum())
                    row[f"recall_{c}"] = float((p_r[t_r == k] == k).mean()) if (t_r == k).any() else np.nan
                    row[f"predicted_{c}"] = int((p_r == k).sum())
                per_record.append(row)
            pd.DataFrame(per_record).round(4).to_csv(out_dir / "baseline_per_record.csv", index=False)

        pd.DataFrame(rows).round(4).to_csv(out_dir / "baseline_metrics.csv", index=False)
    np.savez_compressed(out_dir / "baseline_predictions.npz", **predictions)
    print(f"Saved {out_dir / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
