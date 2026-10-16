"""Exploratory comparison of edge-model candidates (small neural networks) - Task 2, exploration.

Leave-one-record-out over the 21 development records (`config.cv_folds`), pooled predictions scored once, several
seeds per configuration. DS2 is never read. Noisy held-out records use the "cv" part of the noise records,
recorded-noise augmentation the "train" part (one noisy copy per record and noise type, SNR drawn from 0-18 dB).

Every network uses only operations the MAX78002 accelerator supports: 1D convolutions (kernel <= 9, padding <= 2,
at most 64 input channels when padding > 0), batch normalization (folded into the convolutions), ReLU, max pooling
and linear layers.

Networks (key "net"):
    cnn   ECGNet of src/model.py on the beat window. Optional extra input channels:
              template   the running template, the median of the previous 32 windows of the same record (causal)
              measures   a set of measures, each as a constant channel (e.g. "context": rhythm measures and the
                         comparisons with the patient's recent beats)
    mlp   a network on measures only, written as 1x1 convolutions with batch normalization (FusedConv1dBNReLU in
          ai8x) and a linear output layer

Each configuration of a plan is a dict that overrides DEFAULTS. Outputs in results/exploration/networks/:
<plan>.csv (pooled metrics per configuration, seed and condition), <plan>_per_record.csv,
<plan>_curves.csv (training loss and pooled held-out macro F1 after every epoch) and <plan>_predictions.npz (pooled
clean predictions, for bootstrap comparisons). The held-out records are only watched, never used to stop training.

With --wandb every configuration and seed is also logged as one Weights & Biases run (grouped by plan, job type =
configuration). The API key is taken from WANDB_API_KEY, else from the git-ignored key.txt in the repository root.

Usage:
    python src/explore_networks.py --plan round1 [--seeds 0 1 2] [--wandb]
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from config import class_names, cv_folds, load_config, resolve
from explore_data import build_record, class_metrics, condition_name, noise_conditions
from explore_ideas import LAG_SOURCE, PATIENT_RELATIVE, REFERENCE, RHYTHM, deviations
from model import ECGNet, count_parameters

CONTEXT = LAG_SOURCE  # rhythm measures, running-template comparisons and the correlation with the previous beat
MEASURE_SETS = {"reference": REFERENCE, "patient-relative": PATIENT_RELATIVE, "context": CONTEXT, "rhythm": RHYTHM,
                "none": []}
BALANCE = {"none": None, "sqrt": 0.5, "inverse": 1.0, "sampling": "sampling"}
TRAINING = {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "batch": 256, "schedule": "cosine, no restarts"}
BASELINE_MACRO_F1 = 0.556  # gradient boosting on the 75 causal measures, mean over seeds 0-2 (decision log 2026-10-14)
WANDB_PROJECT = "D7043E Edge-Ready ECG Classification"

DEFAULTS = {"net": "cnn", "template": False, "measures": "none", "balance": "none", "augment": "window",
            "noisy_eval": False, "pre": 320, "channels": (16, 32, 32, 64, 64), "epochs": None,
            "snr": (0, 18), "noise_share": None}
# snr: range (dB) of the SNR drawn for each noisy training copy; noise_share: fraction of every epoch drawn from the
# noisy copies (None: drawn uniformly from clean and noisy beats together, i.e. 3/4 noisy with three noise types)
WIDE = (32, 64, 64, 64, 64)

PLANS = {
    "round1": [
        {"balance": "none"},
        {"balance": "sqrt"},
        {"balance": "sampling"},
        {"net": "mlp", "measures": "reference", "augment": "none"},
        {"net": "mlp", "measures": "patient-relative", "augment": "none"},
        {"net": "mlp", "measures": "patient-relative", "balance": "sqrt", "augment": "none"},
        {"measures": "context"},
        {"measures": "context", "balance": "sqrt"},
    ],
    "round2": [  # round 1: comparisons with the patient's recent beats help; give the CNN the template itself
        {"template": True},
        {"template": True, "measures": "context"},
        {"template": True, "measures": "context", "balance": "sqrt"},
    ],
    "round3": [  # round 2: the template channel is the first network above the baseline; context measures hurt it
        {"template": True, "epochs": 40},  # its curve was still rising after 20 epochs
        {"template": True, "channels": WIDE},
        {"template": True, "measures": "rhythm"},
        {"template": True, "balance": "sqrt"},
        {"template": True, "noisy_eval": True},
        {"template": True, "augment": "window+noise", "noisy_eval": True},
        {"template": True, "pre": 448},
    ],
    "round4": [  # round 3: noise augmentation makes the template net robust but costs about 0.025 on clean records
        {"template": True, "augment": "window+noise", "noisy_eval": True, "noise_share": 0.5},
        {"template": True, "augment": "window+noise", "noisy_eval": True, "snr": (6, 24)},
        {"template": True, "augment": "window+noise", "noisy_eval": True, "snr": (6, 24), "noise_share": 0.5},
    ],
}


def describe(c: dict) -> str:
    """Short name of a configuration: the network and every setting that differs from DEFAULTS."""
    extra = [f"{k}: {v}" for k, v in c.items() if k not in ("net", "noisy_eval") and v != DEFAULTS[k]]
    return " | ".join([c["net"] + ("+template" if c["template"] else "")] +
                      [e for e in extra if not e.startswith("template")])


# --------------------------------------------------------------------------- networks

class MeasureNet(nn.Module):
    """Fully connected network on the measures, as 1x1 convolutions over an input of length 1."""

    def __init__(self, n_in, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, c = [], n_in
        for h in hidden:
            layers += [nn.Conv1d(c, h, 1), nn.BatchNorm1d(h), nn.ReLU()]
            c = h
        self.body, self.dropout, self.fc = nn.Sequential(*layers), nn.Dropout(dropout), nn.Linear(c, 4)

    def forward(self, window, measures):
        return self.fc(self.dropout(self.body(measures[:, :, None]).flatten(1)))


class WindowNet(nn.Module):
    """ECGNet on the beat window (and template); measures, if any, enter as extra constant input channels."""

    def __init__(self, n_signals, n_measures, input_len, channels):
        super().__init__()
        assert n_signals + n_measures <= 64, "padding > 0 needs at most 64 input channels on the MAX78002"
        assert all(c <= 64 for c in channels[:-1]), "padding > 0 needs at most 64 input channels on the MAX78002"
        self.n_measures = n_measures
        self.net = ECGNet(num_classes=4, in_channels=n_signals + n_measures, input_len=input_len, channels=channels)

    def forward(self, window, measures):
        if self.n_measures:
            window = torch.cat([window, measures[:, :, None].expand(-1, -1, window.shape[-1])], dim=1)
        return self.net(window)


def build_network(c: dict, n_measures: int, input_len: int) -> nn.Module:
    if c["net"] == "mlp":
        return MeasureNet(n_measures)
    return WindowNet(1 + c["template"], n_measures, input_len, c["channels"])


# --------------------------------------------------------------------------- data

def running_template(X: np.ndarray, rec: np.ndarray, k: int = 32) -> np.ndarray:
    """Median of the previous k windows of the same record (fewer at the start, the beat itself for the first)."""
    T = np.empty_like(X)
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        W, out = X[idx, 0].astype(np.float32), np.empty((len(idx), X.shape[-1]), np.float32)
        for i in range(min(k, len(W))):
            out[i] = np.median(W[:max(i, 1)], axis=0)
        if len(W) > k:
            view = np.lib.stride_tricks.sliding_window_view(W[:-1], k, axis=0)  # view[j] = W[j:j+k] -> beat j+k
            for j in range(0, len(view), 256):
                out[k + j:k + j + 256] = np.median(view[j:j + 256], axis=-1)
        T[idx, 0] = out
    return T


def build_condition(cfg, records, noises, jobs, measure_cols, template):
    """Windows (float16), measures (with deviations), labels and records of the given records for one condition.
    `noises` is one noise setting for all records or a list with one per record."""
    noises = noises if isinstance(noises, list) else [noises] * len(records)
    built = Parallel(n_jobs=jobs)(delayed(build_record)(r, cfg, n) for r, n in zip(records, noises))
    rec = np.concatenate([b["record"] for b in built])
    table = pd.concat([b["features"] for b in built], ignore_index=True)
    table = pd.concat([table, deviations(table, rec)], axis=1)
    X = np.concatenate([b["X"] for b in built]).astype(np.float16)
    if template:
        X = np.concatenate([X, running_template(X, rec)], axis=1)
    return {"X": X, "M": table[measure_cols].to_numpy(np.float32),
            "y": np.concatenate([b["y"] for b in built]), "record": rec}


def augment_name(noise_type, snr) -> str:
    return f"augment {noise_type} ({snr[0]}-{snr[1]})"  # must not end in "dB" like the held-out conditions


def load_data(cfg, records, pre, plan, jobs, measure_cols) -> dict:
    """All conditions needed by the configurations of `plan` that use pre-window `pre`."""
    cfg = copy.deepcopy(cfg)
    cfg["window"]["pre_samples"] = pre
    plan = [c for c in plan if c["pre"] == pre]
    template = any(c["template"] for c in plan)
    data = {"clean": build_condition(cfg, records, None, jobs, measure_cols, template)}
    if any(c["noisy_eval"] for c in plan):
        for c in noise_conditions(cfg)[1:]:
            data[condition_name(c)] = build_condition(cfg, records, c, jobs, measure_cols, template)
    for lo, hi in sorted({c["snr"] for c in plan if c["augment"] == "window+noise"}):
        rng = np.random.default_rng(cfg["seed"])
        snr = {(r, t): float(rng.uniform(lo, hi)) for r in records for t in cfg["noise"]["types"]}
        for t in cfg["noise"]["types"]:
            data[augment_name(t, (lo, hi))] = build_condition(
                cfg, records, [(t, snr[(r, t)], "train", 0) for r in records], jobs, measure_cols, template)
    return data


def augment_window(x: torch.Tensor, g: torch.Generator, max_shift=8, scale=(0.9, 1.1), noise_std=0.02):
    """Amplitude scaling, circular shift and white noise (report section 3.7, sinusoidal wander dropped).
    The scale applies to every channel (same recording gain); shift and noise only to the beat window (channel 0),
    so a template channel stays put and the network sees beats that are not perfectly aligned with it."""
    b, _, n = x.shape
    u = lambda lo, hi: torch.rand(b, 1, 1, device=x.device, generator=g) * (hi - lo) + lo  # noqa: E731
    x = x * u(*scale)
    shift = torch.randint(-max_shift, max_shift + 1, (b, 1), device=x.device, generator=g)
    w = x[:, :1].gather(2, ((torch.arange(n, device=x.device)[None, :] - shift) % n)[:, None, :])
    w = w + torch.randn(w.shape, device=x.device, generator=g) * u(0, noise_std)
    return torch.cat([w, x[:, 1:]], dim=1).clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- training

def predict(model, t, device):
    model.eval()
    with torch.no_grad():
        X, M = torch.from_numpy(t["X"]).to(device).float(), torch.from_numpy(t["M"]).to(device)
        return torch.cat([model(X[i:i + 4096], M[i:i + 4096]).argmax(1) for i in range(0, len(X), 4096)]).cpu().numpy()


def train_and_predict(c, train, tests, seed, epochs, device, n_measures, input_len):
    """Train one fold on `train` (dict of arrays, possibly with noisy copies) and predict every test dict.
    Also returns the mean training loss and the clean held-out predictions after every epoch (watched only)."""
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    model = build_network(c, n_measures, input_len).to(device)
    Xt, Mt = torch.from_numpy(train["X"]).to(device).float(), torch.from_numpy(train["M"]).to(device)
    yt = torch.from_numpy(train["y"]).to(device)
    n_epoch = train["n_epoch"]  # beats per epoch (the clean training set size, also with noisy copies)
    setting, weight = BALANCE[c["balance"]], None
    counts = torch.bincount(yt, minlength=4).clamp(min=1).float()
    if setting not in (None, "sampling"):
        weight = (counts.sum() / counts) ** setting
        weight = weight / weight.mean()
    p_draw = (1.0 / counts)[yt] if setting == "sampling" else torch.ones(len(yt), device=device)
    if c["noise_share"] is not None:  # the clean beats come first in `train`, the noisy copies after them
        assert setting != "sampling" and len(yt) > n_epoch
        p_draw[:n_epoch], p_draw[n_epoch:] = (1 - c["noise_share"]) / n_epoch, c["noise_share"] / (len(yt) - n_epoch)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
    batch = TRAINING["batch"]
    steps = n_epoch // batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps)
    losses, curve = [], []
    for _ in range(epochs):
        model.train()
        order = torch.multinomial(p_draw, n_epoch, replacement=setting == "sampling", generator=g)
        loss_sum = torch.zeros((), device=device)
        for i in range(steps):
            idx = order[i * batch:(i + 1) * batch]
            xb = augment_window(Xt[idx], g) if c["augment"] != "none" else Xt[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, Mt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach()
        losses.append(loss_sum.item() / steps)
        curve.append(predict(model, tests["clean"], device))
    out = {name: curve[-1] if name == "clean" else predict(model, t, device) for name, t in tests.items()}
    return out, count_parameters(model), losses, curve


# --------------------------------------------------------------------------- Weights & Biases

def short_tag(tag: str) -> str:
    """W&B allows at most 64 characters for the job type."""
    for long, short in (("augment: window+noise", "noise aug"), ("noise_share", "share"), ("measures: ", ""),
                        ("balance: ", ""), ("channels: ", "")):
        tag = tag.replace(long, short) if len(tag) > 64 else tag
    return tag[:64]


def wandb_start(args, c, tag, seed, epochs, n_beats):
    import wandb  # only needed with --wandb
    if not os.environ.get("WANDB_API_KEY") and resolve("key.txt").exists():
        os.environ["WANDB_API_KEY"] = resolve("key.txt").read_text().strip()
    for attempt in (1, 2):  # a W&B hiccup must not stop the experiment: retry once, then go on without logging
        try:
            run = wandb.init(project=args.wandb_project, group=args.plan, job_type=short_tag(tag),
                             name=f"{tag} | seed {seed}", dir=resolve("."),
                             config={**c, "configuration": tag, "plan": args.plan, "seed": seed, "epochs": epochs,
                                     **TRAINING, "beats": n_beats, "baseline_macro_f1": BASELINE_MACRO_F1})
            break
        except Exception as e:  # noqa: BLE001
            print(f"W&B init failed (attempt {attempt}): {e}", flush=True)
            time.sleep(10)
    else:
        return None
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    return run


def wandb_finish(run, classes, y, rec, records, pooled, losses, curves, n_params):
    import wandb
    run.config.update({"parameters": n_params})
    for e, loss in enumerate(losses):
        m = class_metrics(y, curves[e], classes)
        run.log({"epoch": e + 1, "train_loss": loss, **{k: m[k] for k in ["macro_f1", "accuracy"] +
                                                        [f"f1_{c}" for c in classes]}})
    for name, pred in pooled.items():
        m = class_metrics(y, pred, classes)
        run.summary.update({name if name == "clean" else f"noise/{name}": m})
    run.summary["macro_f1_minus_baseline"] = class_metrics(y, pooled["clean"], classes)["macro_f1"] - BASELINE_MACRO_F1
    run.log({"confusion_matrix": wandb.plot.confusion_matrix(y_true=y, preds=pooled["clean"], class_names=classes)})
    table = wandb.Table(columns=["record", "beats", "accuracy"] + [f"recall_{c}" for c in classes])
    for r in records:
        m = rec == r
        recalls = [float((pooled["clean"][m & (y == i)] == i).mean()) if (m & (y == i)).any() else None
                   for i in range(len(classes))]
        table.add_data(int(r), int(m.sum()), float((pooled["clean"][m] == y[m]).mean()), *recalls)
    run.log({"per_record": table})
    run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--plan", required=True, choices=list(PLANS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs-cnn", type=int, default=20)
    ap.add_argument("--epochs-mlp", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--wandb", action="store_true", help="log every configuration and seed to Weights & Biases")
    ap.add_argument("--wandb-project", default=WANDB_PROJECT)
    ap.add_argument("--configs", type=int, nargs="+",
                    help="run only these configurations of the plan (0-based); their results replace the same "
                         "configurations in the existing output files, the others are kept")
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes, folds = class_names(cfg), cv_folds(cfg)
    records = sorted(r for _, held_out in folds for r in held_out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    plan = [{**DEFAULTS, **c} for c in PLANS[args.plan]]
    out_dir = resolve("results/exploration/networks")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_measures = sorted({m for c in plan for m in MEASURE_SETS[c["measures"]]}, key=lambda m: (m not in REFERENCE, m))

    t0 = time.time()
    rows, per_record, curve_rows, predictions, data, data_pre = [], [], [], {}, None, None
    if args.configs:
        plan = [plan[i] for i in args.configs]
        tags = {describe(c) for c in plan}
        old = [pd.read_csv(out_dir / f"{args.plan}{s}.csv") for s in ("", "_per_record", "_curves")]
        rows, per_record, curve_rows = [d[~d.configuration.isin(tags)].to_dict("records") for d in old]
        old = np.load(out_dir / f"{args.plan}_predictions.npz")
        predictions = {k: old[k] for k in old.files if k.rsplit(" | seed", 1)[0] not in tags}
    for c in sorted(plan, key=lambda c: c["pre"]):  # build the data once per pre-window length
        if c["pre"] != data_pre:
            data, data_pre = None, c["pre"]
            data = load_data(cfg, records, data_pre, plan, args.jobs, all_measures)
            y, rec = data["clean"]["y"], data["clean"]["record"]
            print(f"pre-window {data_pre}: {len(y):,} beats, {len(all_measures)} measures, {len(data)} conditions, "
                  f"built in {time.time() - t0:.0f} s", flush=True)
        input_len = data_pre + cfg["window"]["post_samples"]
        cols = [all_measures.index(m) for m in MEASURE_SETS[c["measures"]]]
        tag = describe(c)
        test_names = ["clean"] + ([n for n in data if n.endswith("dB")] if c["noisy_eval"] else [])
        train_names = ["clean"] + ([augment_name(t, c["snr"]) for t in cfg["noise"]["types"]]
                                   if c["augment"] == "window+noise" else [])
        epochs = c["epochs"] or (args.epochs_mlp if c["net"] == "mlp" else args.epochs_cnn)
        for seed in args.seeds:
            t1 = time.time()
            run = wandb_start(args, c, tag, seed, epochs, len(y)) if args.wandb else None
            pooled = {n: np.full_like(y, -1) for n in test_names}
            curves, losses = np.full((epochs, len(y)), -1), np.zeros(epochs)
            for k, (train_records, held_out) in enumerate(folds):
                tr, te = np.isin(rec, train_records), np.isin(rec, held_out)
                if cols:  # scale the measures with statistics of the training records only
                    scaler = make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(
                        n_quantiles=256, output_distribution="normal", subsample=50_000, random_state=seed))
                    scaler.fit(data["clean"]["M"][tr][:, cols])
                    scale = lambda m: np.clip(scaler.transform(m[:, cols]) / 3.0, -1, 1).astype(np.float32)  # noqa: E731
                else:
                    scale = lambda m: np.zeros((len(m), 0), np.float32)  # noqa: E731
                ch = slice(0, 1 + c["template"])  # the data carry the template channel if any configuration uses it
                train = {"X": np.concatenate([data[n]["X"][tr, ch] for n in train_names]),
                         "y": np.concatenate([data[n]["y"][tr] for n in train_names])}
                train["M"] = np.concatenate([scale(data[n]["M"][tr]) for n in train_names])
                train["n_epoch"] = int(tr.sum())
                tests = {n: {"X": data[n]["X"][te, ch], "M": scale(data[n]["M"][te])} for n in test_names}
                preds, n_params, fold_losses, curve = train_and_predict(c, train, tests, seed * 1000 + k, epochs,
                                                                        device, len(cols), input_len)
                for n in test_names:
                    pooled[n][te] = preds[n]
                curves[:, te] = np.stack(curve)
                losses += np.array(fold_losses) / len(folds)
            for e in range(epochs):
                curve_rows.append({"configuration": tag, "seed": seed, "epoch": e + 1, "train_loss": losses[e],
                                   "macro_f1": class_metrics(y, curves[e], classes)["macro_f1"]})
            for n in test_names:
                rows.append({"configuration": tag, **{k: str(v) for k, v in c.items()}, "seed": seed,
                             "parameters": n_params, "condition": n, **class_metrics(y, pooled[n], classes)})
            for r in records:
                m = rec == r
                per_record.append({"configuration": tag, "seed": seed, "record": r,
                                   "accuracy": float((pooled["clean"][m] == y[m]).mean())})
            predictions[f"{tag} | seed {seed}"] = pooled["clean"].astype(np.int8)
            predictions[f"{tag} | seed {seed} | record"] = rec.astype(np.int16)
            m = next(row for row in rows[::-1] if row["condition"] == "clean")
            print(f"{tag[:70]:70s} seed {seed} | macro F1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f} | F1 S "
                  f"{m['f1_S']:.2f} V {m['f1_V']:.2f} F {m['f1_F']:.2f} | {n_params:,} parameters | {time.time() - t1:.0f} s",
                  flush=True)
            for row in rows[-len(test_names) + 1:] if c["noisy_eval"] else []:
                print(f"    {row['condition']:>9s}: macro F1 {row['macro_f1']:.3f}", flush=True)
            if run is not None:
                try:
                    wandb_finish(run, classes, y, rec, records, pooled, losses, curves, n_params)
                except Exception as e:  # noqa: BLE001
                    print(f"W&B logging failed: {e}", flush=True)
            pd.DataFrame(rows).round(4).to_csv(out_dir / f"{args.plan}.csv", index=False)
            pd.DataFrame(per_record).round(4).to_csv(out_dir / f"{args.plan}_per_record.csv", index=False)
            pd.DataFrame(curve_rows).round(4).to_csv(out_dir / f"{args.plan}_curves.csv", index=False)
            np.savez_compressed(out_dir / f"{args.plan}_predictions.npz", **predictions)

    summary = pd.DataFrame(rows).query("condition == 'clean'").groupby("configuration", sort=False).macro_f1.agg(
        ["mean", "min", "max"])
    print(f"\nmacro F1 over seeds (gradient-boosting baseline: mean {BASELINE_MACRO_F1}, range 0.553-0.561):")
    print(summary.round(3).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
