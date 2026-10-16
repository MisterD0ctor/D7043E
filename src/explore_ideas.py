"""Exploratory baseline experiments on the 21 development records (DS1 without record 201) - Task 2, exploration.

Every experiment is scored with leave-one-record-out cross-validation (`config.cv_folds`): each development record is
predicted by a model fitted on the other 20, and the predictions of all records are pooled and scored once. DS2 is
never read. Noisy held-out records use the "cv" part of the noise records, noise augmentation the "train" part.

Groups (--group, several allowed, run in the order given):
    models      classifier families on the reference measures, without and with balancing
    imbalance   the four balancing strategies of report section 3.6, for gradient boosting and logistic regression
    features    which information helps: rhythm, morphology, deviations from the patient's own beats, raw waveform
    priors      adjustment of the decision thresholds, tuned on the other 20 records only
    noise       robustness on noisy held-out records, and recorded-noise augmentation (report section 3.7)
    refine      second round: the best settings of the first combined, seeds, regularization, two stages,
                ensembles, the previous beat as context, and noise augmentation for the best feature set

Outputs in results/exploration/ideas/: <group>.csv (pooled metrics per experiment), <group>_per_record.csv
(accuracy and recall per class of every held-out record) and <group>_predictions.npz.

Usage:
    python src/explore_ideas.py --group models imbalance features priors noise refine [--jobs 4]
"""
import argparse
import inspect
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC

from config import class_names, cv_folds, load_config, resolve
from explore_data import NEXT_BEAT, RUNNING, build_record, class_metrics, condition_name, feature_sets, noise_conditions
from features import FEATURES

REFERENCE = feature_sets()["causal + running template"]
RHYTHM = [f for f in REFERENCE if FEATURES.get(f, ("",))[0] == "rhythm"]
MORPHOLOGY = [f for f in REFERENCE if f not in RHYTHM]
DEVIATION_SOURCE = [f for f in MORPHOLOGY if f not in RUNNING and f != "prev_beat_corr"]
DEVIATION_BEATS = 32  # reference for the deviation measures: median of the previous 32 beats of the record
STRATEGIES = {"none": None, "sqrt weights": 0.5, "inverse weights": 1.0, "balanced sampling": "sampling"}
DEVIATIONS = [f"{f}_dev" for f in DEVIATION_SOURCE]
# Everything measured relative to the patient: rhythm ratios, comparisons with recent beats, deviations
PATIENT_RELATIVE = RHYTHM + RUNNING + ["prev_beat_corr"] + DEVIATIONS
LAG_SOURCE = RHYTHM + RUNNING + ["prev_beat_corr"]  # the previous beat's values as context
LAGGED = [f"{f}_prev" for f in LAG_SOURCE]


# --------------------------------------------------------------------------- data

def deviations(table: pd.DataFrame, rec: np.ndarray, whole_record: bool = False) -> pd.DataFrame:
    """Each morphology measure minus its typical value in the same record: the median of the previous
    DEVIATION_BEATS beats (causal), or the median of the whole record (not causal, an upper bound)."""
    grouped = table[DEVIATION_SOURCE].groupby(rec)
    if whole_record:
        reference = grouped.transform("median")
    else:
        reference = grouped.transform(lambda s: s.shift(1).rolling(DEVIATION_BEATS, min_periods=1).median())
    suffix = "_devrec" if whole_record else "_dev"
    return (table[DEVIATION_SOURCE] - reference).add_suffix(suffix)


def lagged(table: pd.DataFrame, rec: np.ndarray) -> pd.DataFrame:
    """The previous beat's rhythm and comparison measures, as context for the current beat (causal)."""
    return table[LAG_SOURCE].groupby(rec).shift(1).add_suffix("_prev")


def raw_waveform(X: np.ndarray) -> pd.DataFrame:
    """The prepared 512-sample window averaged over blocks of 8 samples: 64 values per beat."""
    pooled = X[:, 0, :].reshape(len(X), -1, 8).mean(axis=2)
    return pd.DataFrame(pooled, columns=[f"raw_{i:02d}" for i in range(pooled.shape[1])])


def build_tables(cfg: dict, records: list[int], conditions: list, jobs: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Measure tables of all development records for every condition (+ deviation and raw-waveform columns)."""
    built = Parallel(n_jobs=jobs)(delayed(build_record)(r, cfg, c) for c in conditions for r in records)
    tables, y, rec = {}, None, None
    for i, c in enumerate(conditions):
        parts = built[i * len(records):(i + 1) * len(records)]
        table = pd.concat([p["features"] for p in parts], ignore_index=True)
        y_c, rec_c = np.concatenate([p["y"] for p in parts]), np.concatenate([p["record"] for p in parts])
        assert y is None or ((y == y_c).all() and (rec == rec_c).all())
        y, rec = y_c, rec_c
        table = pd.concat([table, deviations(table, rec), deviations(table, rec, whole_record=True),
                           raw_waveform(np.concatenate([p["X"] for p in parts]))], axis=1)
        table = pd.concat([table, lagged(table, rec)], axis=1)
        tables[condition_name(c)] = table
    return tables, y, rec


# --------------------------------------------------------------------------- models

def quantile(seed):  # heavy-tailed measures -> rank-based mapping to a normal distribution
    return QuantileTransformer(n_quantiles=256, output_distribution="normal", subsample=50_000, random_state=seed)


def scaled(classifier, seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("q", quantile(seed)), ("clf", classifier)])


class TwoStage(BaseEstimator, ClassifierMixin):
    """Gradient boosting in two stages: N against all ectopic beats, then S, V or F among the ectopic ones."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.arange(4)
        self.first_ = HistGradientBoostingClassifier(max_iter=300, random_state=self.seed).fit(X, (y != 0).astype(int))
        ectopic = y != 0
        self.second_ = HistGradientBoostingClassifier(max_iter=300, random_state=self.seed).fit(X[ectopic], y[ectopic])
        return self

    def predict_proba(self, X):
        p_ectopic = self.first_.predict_proba(X)[:, 1]
        out = np.zeros((len(X), 4))
        out[:, 0] = 1 - p_ectopic
        for j, c in enumerate(self.second_.classes_):
            out[:, c] = p_ectopic * self.second_.predict_proba(X)[:, j]
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


MODELS = {
    "logistic regression": lambda s: scaled(LogisticRegression(max_iter=2000), s),
    "LDA": lambda s: scaled(LinearDiscriminantAnalysis(), s),
    "Gaussian naive Bayes": lambda s: scaled(GaussianNB(), s),
    "linear SVM": lambda s: scaled(LinearSVC(C=0.1), s),
    "RBF SVM (Nystroem, 500 components)": lambda s: Pipeline([
        ("imp", SimpleImputer(strategy="median")), ("q", quantile(s)),
        ("rbf", Nystroem(n_components=500, random_state=s)), ("clf", LinearSVC(C=0.1))]),
    "k nearest neighbours (k=25)": lambda s: scaled(KNeighborsClassifier(n_neighbors=25, weights="distance"), s),
    "random forest": lambda s: Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, random_state=s, n_jobs=1))]),
    "extra trees": lambda s: Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", ExtraTreesClassifier(
        n_estimators=150, min_samples_leaf=2, random_state=s, n_jobs=1))]),
    "gradient boosting": lambda s: Pipeline([("clf", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, random_state=s))]),
    "gradient boosting, regularized": lambda s: Pipeline([("clf", HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=False, random_state=s))]),
    "gradient boosting, slower learning": lambda s: Pipeline([("clf", HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, random_state=s))]),
    "gradient boosting, no early stopping": lambda s: Pipeline([("clf", HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=False, random_state=s))]),
    "gradient boosting, two stages": lambda s: Pipeline([("clf", TwoStage(seed=s))]),
    "gradient boosting, 20 best measures per fold": lambda s: Pipeline([
        ("imp", SimpleImputer(strategy="median")), ("sel", SelectKBest(f_classif, k=20)),
        ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=s))]),
    "small MLP (64-32)": lambda s: scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=256,
                                                        max_iter=60, random_state=s), s),
}


def balanced_indices(y: np.ndarray, rng) -> np.ndarray:
    """Draw len(y) beats with replacement, every class with the same probability (the balanced sampler of dataset.py)."""
    p = (1.0 / np.bincount(y, minlength=4).clip(min=1))[y]
    return rng.choice(len(y), size=len(y), replace=True, p=p / p.sum())


def fit_fold(model_name, strategy, X_all, y_all, train_idx, tests_all: dict, test_idx, seed):
    """Fit one fold and predict every test table; returns {test name: (labels, probabilities or None)}.
    The full arrays are passed once (joblib memory-maps them) and sliced here, to keep memory low."""
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    tests = {name: X[test_idx] for name, X in tests_all.items()}
    model, setting = MODELS[model_name](seed), STRATEGIES[strategy]
    fit_kwargs = {}
    if setting == "sampling" or (setting is not None and "sample_weight" not in inspect.signature(
            model.steps[-1][1].fit).parameters):
        idx = balanced_indices(y_train, np.random.default_rng(seed))
        X_train, y_train = X_train[idx], y_train[idx]
    elif setting is not None:
        counts = np.bincount(y_train, minlength=4).clip(min=1)
        weights = (counts.sum() / counts) ** setting
        fit_kwargs["clf__sample_weight"] = (weights / weights.mean())[y_train]
    model.fit(X_train, y_train, **fit_kwargs)
    out = {}
    for name, X in tests.items():
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        out[name] = (model.predict(X), proba)
    return out


# --------------------------------------------------------------------------- experiments

def run_experiment(spec: dict, tables: dict, y, rec, folds, classes, jobs, seed):
    """Leave-one-record-out for one experiment; returns metric rows, per-record rows and pooled predictions."""
    seed = spec.get("seed", seed)
    cols, test_names = spec["features"], spec.get("test", ["clean"])
    train_tables = spec.get("train", ["clean"])
    X_test = {name: tables[name][cols].to_numpy(np.float32) for name in test_names}
    X_train_all = np.concatenate([tables[name][cols].to_numpy(np.float32) for name in train_tables])
    y_train_all, rec_train_all = np.tile(y, len(train_tables)), np.tile(rec, len(train_tables))
    results = Parallel(n_jobs=jobs)(
        delayed(fit_fold)(spec["model"], spec["strategy"], X_train_all, y_train_all,
                          np.flatnonzero(np.isin(rec_train_all, train)), X_test,
                          np.flatnonzero(np.isin(rec, held_out)), seed)
        for train, held_out in folds)
    pooled = {name: np.zeros_like(y) for name in test_names}
    proba = {name: np.zeros((len(y), len(classes)), np.float32) for name in test_names}
    for (_, held_out), fold in zip(folds, results):
        m = np.isin(rec, held_out)
        for name in test_names:
            pooled[name][m] = fold[name][0]
            if fold[name][1] is not None:
                proba[name][m] = fold[name][1]
    base = {"group": spec["group"], "experiment": spec["name"], "model": spec["model"], "strategy": spec["strategy"],
            "seed": seed, "n_features": len(cols), "training data": " + ".join(train_tables)}
    rows = [{**base, "condition": name, **class_metrics(y, pooled[name], classes)} for name in test_names]
    per_record = []
    for r in np.unique(rec):
        m = rec == r
        row = {"experiment": spec["name"], "record": int(r), "accuracy": float((pooled["clean"][m] == y[m]).mean())}
        for k, c in enumerate(classes):
            row[f"recall_{c}"] = float((pooled["clean"][m & (y == k)] == k).mean()) if (m & (y == k)).any() else np.nan
        per_record.append(row)
    return rows, per_record, pooled, proba


def tune_priors(proba: np.ndarray, y: np.ndarray, rec: np.ndarray, classes, grid=np.arange(-3, 3.01, 0.25)):
    """Add a log-offset per class to the predicted log-probabilities. For every held-out record the offsets are
    chosen to maximize the macro F1 on the other 20 records only (coordinate search), then applied to that record."""
    logp = np.log(np.clip(proba, 1e-9, 1))

    def macro_f1(pred, truth):
        return class_metrics(truth, pred, classes)["macro_f1"]

    pred, chosen = np.zeros_like(y), {}
    for r in np.unique(rec):
        train = rec != r
        offsets = np.zeros(len(classes))
        for _ in range(3):
            for k in range(1, len(classes)):
                scores = []
                for value in grid:
                    trial = offsets.copy()
                    trial[k] = value
                    scores.append(macro_f1((logp[train] + trial).argmax(axis=1), y[train]))
                offsets[k] = grid[int(np.argmax(scores))]
        chosen[int(r)] = offsets.round(2).tolist()
        pred[rec == r] = (logp[rec == r] + offsets).argmax(axis=1)
    return pred, chosen


def specs_for(group: str) -> list[dict]:
    ref = REFERENCE
    if group == "models":
        out = []
        for model in MODELS:
            if model.startswith("gradient boosting, 20"):
                continue
            strategies = ["inverse weights"] if model in ("random forest", "extra trees") else ["none", "inverse weights"]
            for strategy in strategies:
                if model in ("k nearest neighbours (k=25)", "LDA", "small MLP (64-32)") and strategy == "inverse weights":
                    strategy = "balanced sampling"
                out.append({"group": group, "name": f"{model} | {strategy}", "model": model, "strategy": strategy,
                            "features": ref})
        return out
    if group == "imbalance":
        return [{"group": group, "name": f"{model} | {strategy}", "model": model, "strategy": strategy, "features": ref}
                for model in ("gradient boosting", "logistic regression") for strategy in STRATEGIES]
    if group == "features":
        dev = [f"{f}_dev" for f in DEVIATION_SOURCE]
        devrec = [f"{f}_devrec" for f in DEVIATION_SOURCE]
        raw = [f"raw_{i:02d}" for i in range(64)]
        sets = {
            "rhythm only": RHYTHM,
            "morphology only": MORPHOLOGY,
            "reference without running template": [f for f in ref if f not in RUNNING],
            "reference": ref,
            "reference + deviations (previous 32 beats)": ref + dev,
            "rhythm + running template + deviations only": RHYTHM + RUNNING + ["prev_beat_corr"] + dev,
            "reference + deviations from the whole record (not causal)": ref + devrec,
            "reference + next beat (decision one beat later)": ref + NEXT_BEAT,
            "raw waveform (64 values) + rhythm": raw + RHYTHM,
            "raw waveform + reference": raw + ref,
        }
        out = [{"group": group, "name": f"gradient boosting | {name}", "model": "gradient boosting",
                "strategy": "inverse weights", "features": cols} for name, cols in sets.items()]
        out.append({"group": group, "name": "gradient boosting, 20 best measures per fold | reference",
                    "model": "gradient boosting, 20 best measures per fold", "strategy": "inverse weights", "features": ref})
        return out
    if group == "priors":
        return [{"group": group, "name": f"gradient boosting | {strategy}", "model": "gradient boosting",
                 "strategy": strategy, "features": ref} for strategy in ("none", "inverse weights")]
    if group == "noise":
        dev = ref + [f"{f}_dev" for f in DEVIATION_SOURCE]
        noisy = ["clean"] + [f"{t} {s:g} dB" for t in ("bw", "em", "ma") for s in (12, 6, 0)]
        augmented = ["clean", "augment bw", "augment em", "augment ma"]
        return [{"group": group, "name": f"gradient boosting | {name}", "model": "gradient boosting",
                 "strategy": "inverse weights", "features": cols, "train": train, "test": noisy}
                for name, cols, train in (("reference", ref, ["clean"]),
                                          ("reference, noise augmentation", ref, augmented),
                                          ("reference + deviations", dev, ["clean"]),
                                          ("reference + deviations, noise augmentation", dev, augmented))]
    if group == "refine":
        p, ref_ = PATIENT_RELATIVE, ref
        gb = "gradient boosting"
        out = [{"group": group, "name": f"{gb} | none | reference | seed {s}", "model": gb, "strategy": "none",
                "features": ref_, "seed": s} for s in (0, 1, 2)]
        out += [{"group": group, "name": f"{gb} | none | patient-relative | seed {s}", "model": gb, "strategy": "none",
                 "features": p, "seed": s} for s in (0, 1, 2)]
        out += [{"group": group, "name": name, "model": model, "strategy": strategy, "features": cols}
                for name, model, strategy, cols in (
                    (f"{gb} | none | patient-relative + previous beat", gb, "none", p + LAGGED),
                    (f"{gb} | sqrt weights | patient-relative", gb, "sqrt weights", p),
                    (f"{gb}, regularized | none | patient-relative", f"{gb}, regularized", "none", p),
                    (f"{gb}, slower learning | none | patient-relative", f"{gb}, slower learning", "none", p),
                    (f"{gb}, no early stopping | none | patient-relative", f"{gb}, no early stopping", "none", p),
                    (f"{gb}, two stages | none | patient-relative", f"{gb}, two stages", "none", p),
                    ("logistic regression | inverse weights | patient-relative", "logistic regression", "inverse weights", p),
                    ("small MLP (64-32) | none | patient-relative", "small MLP (64-32)", "none", p),
                    ("small MLP (64-32) | balanced sampling | patient-relative", "small MLP (64-32)", "balanced sampling", p))]
        noisy = ["clean"] + [f"{t} {s:g} dB" for t in ("bw", "em", "ma") for s in (12, 6, 0)]
        augmented = ["clean", "augment bw", "augment em", "augment ma"]
        out += [{"group": group, "name": f"{gb} | none | {name}", "model": gb, "strategy": "none", "features": cols,
                 "train": train, "test": noisy}
                for name, cols, train in (("reference, noise augmentation", ref_, augmented),
                                          ("patient-relative, clean training", p, ["clean"]),
                                          ("patient-relative, noise augmentation", p, augmented))]
        return out
    raise ValueError(group)


ENSEMBLES = {  # averaged probabilities of experiments of the same group
    "refine": [
        ("ensemble | gradient boosting, 3 seeds | patient-relative",
         [f"gradient boosting | none | patient-relative | seed {s}" for s in (0, 1, 2)]),
        ("ensemble | gradient boosting + logistic regression | patient-relative",
         ["gradient boosting | none | patient-relative | seed 0", "logistic regression | inverse weights | patient-relative"]),
        ("ensemble | gradient boosting + MLP | patient-relative",
         ["gradient boosting | none | patient-relative | seed 0", "small MLP (64-32) | balanced sampling | patient-relative"]),
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--group", nargs="+", required=True,
                    choices=["models", "imbalance", "features", "priors", "noise", "refine"])
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes, folds, seed = class_names(cfg), cv_folds(cfg), cfg["seed"]
    records = sorted(r for _, held_out in folds for r in held_out)
    out_dir = resolve("results/exploration/ideas")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    conditions = [None]
    with_noise = bool({"noise", "refine"} & set(args.group))
    if with_noise:
        conditions += [c for c in noise_conditions(cfg) if c is not None]
    tables, y, rec = build_tables(cfg, records, conditions, args.jobs)
    if with_noise:  # one noisy copy of every training record per noise type, SNR drawn from 0-18 dB
        rng = np.random.default_rng(seed)
        snr = {(r, t): float(rng.uniform(0, 18)) for r in records for t in cfg["noise"]["types"]}
        for t in cfg["noise"]["types"]:
            built = Parallel(n_jobs=args.jobs)(delayed(build_record)(r, cfg, (t, snr[(r, t)], "train", 0)) for r in records)
            table = pd.concat([b["features"] for b in built], ignore_index=True)
            table = pd.concat([table, deviations(table, rec)], axis=1)
            tables[f"augment {t}"] = pd.concat([table, lagged(table, rec)], axis=1)
    print(f"{len(y):,} beats of {len(records)} development records, {len(tables)} tables, {time.time() - t0:.0f} s", flush=True)

    for group in args.group:
        rows, per_record, predictions, probabilities = [], [], {"y": y, "record": rec}, {}
        for spec in specs_for(group):
            t1 = time.time()
            r_rows, r_per_record, pooled, proba = run_experiment(spec, tables, y, rec, folds, classes, args.jobs, seed)
            rows += r_rows
            per_record += r_per_record
            predictions[spec["name"]] = pooled["clean"]
            probabilities[spec["name"]] = proba["clean"]
            m = r_rows[0]
            print(f"[{group}] {spec['name'][:78]:78s} macro F1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f} | "
                  f"F1 S {m['f1_S']:.2f} V {m['f1_V']:.2f} F {m['f1_F']:.2f} | {time.time() - t1:.0f} s", flush=True)
            for row in r_rows[1:]:
                print(f"    {row['condition']:>9s}: macro F1 {row['macro_f1']:.3f}, acc {row['accuracy']:.3f}", flush=True)
            if group == "priors":
                pred, chosen = tune_priors(proba["clean"], y, rec, classes)
                tuned = {**r_rows[0], "experiment": spec["name"] + " | priors tuned on the other records",
                         **class_metrics(y, pred, classes)}
                rows.append(tuned)
                predictions[tuned["experiment"]] = pred
                median_offsets = np.median(np.array(list(chosen.values())), axis=0).round(2).tolist()
                print(f"    priors tuned: macro F1 {tuned['macro_f1']:.3f} acc {tuned['accuracy']:.3f} | F1 S "
                      f"{tuned['f1_S']:.2f} V {tuned['f1_V']:.2f} F {tuned['f1_F']:.2f} | median offsets {median_offsets}",
                      flush=True)
            pd.DataFrame(rows).round(4).to_csv(out_dir / f"{group}.csv", index=False)
            pd.DataFrame(per_record).round(4).to_csv(out_dir / f"{group}_per_record.csv", index=False)
        for name, members in ENSEMBLES.get(group, []):
            pred = np.mean([probabilities[m] for m in members], axis=0).argmax(axis=1)
            m = class_metrics(y, pred, classes)
            rows.append({"group": group, "experiment": name, "model": "ensemble", "strategy": "-",
                         "n_features": "-", "training data": "clean", "condition": "clean", **m})
            predictions[name] = pred
            print(f"[{group}] {name[:78]:78s} macro F1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f} | "
                  f"F1 S {m['f1_S']:.2f} V {m['f1_V']:.2f} F {m['f1_F']:.2f}", flush=True)
        pd.DataFrame(rows).round(4).to_csv(out_dir / f"{group}.csv", index=False)
        np.savez_compressed(out_dir / f"{group}_predictions.npz", **predictions)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
