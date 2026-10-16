"""Exploratory comparison of candidate edge models (small neural networks) on DS1 - exploration, not final results.

Protocols as in explore_baselines.py (DS2 is never read):
    loro    leave-one-record-out over the 22 DS1 records, out-of-fold predictions pooled before scoring
    split   fit on the 18 training records, predict the 4 validation records (also logs the per-epoch curve)

Every model is trained for a fixed number of epochs with cosine learning-rate decay and no early stopping,
so no held-out data influences the model that is scored. Optimizer, batch size, class weighting and
augmentation follow configs/train.yaml unless the variant changes them.

Variants (--variant):
    cnn            ECGNet of src/model.py on the 512-sample beat window
    cnn_noaug      no augmentation
    cnn_noweight   no class weights
    cnn_invweight  full inverse-frequency class weights (configs/train.yaml uses the square root)
    cnn_small      half the channels per layer
    cnn_rr         three extra constant input channels: previous RR, previous RR / local mean, local mean
    cnn_clip8      beat windows re-prepared with clip level 8 instead of 5
    cnn_win256     beat windows re-prepared with 128 + 128 samples instead of 320 + 192
    cnn_causal     beat windows re-prepared with a causal band-pass filter instead of the zero-phase one
    mlp            fully connected network on the causal signal measures (src/features.py, src/explore_data.py)

With --noise each fold's model is also scored on its held-out record with NSTDB noise (bw, em, ma at 12/6/0 dB).

Outputs in results/exploration/: edge_<variant>_s<seed>_<protocol>.csv (metrics), ..._per_record.csv,
..._predictions.npz and, for the split protocol, ..._curve.csv.

Usage:
    python src/explore_edge_models.py --variant cnn --protocol loro --noise
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from config import class_names, load_config, resolve, split_records
from explore_data import build_record, class_metrics, condition_name, ds1_records, feature_sets, noise_conditions
from model import ECGNet, count_parameters

RR_CHANNELS = ["rr_pre", "rr_pre_ratio", "rr_local"]  # each clipped to 0-2 and shifted to [-1, 1]
MLP_FEATURES = "causal + running template"
VARIANTS = ["cnn", "cnn_noaug", "cnn_noweight", "cnn_invweight", "cnn_small", "cnn_rr", "cnn_clip8", "cnn_win256",
            "cnn_causal", "mlp"]


def prepare_config(cfg: dict, variant: str) -> dict:
    cfg = copy.deepcopy(cfg)
    if variant == "cnn_clip8":
        cfg["normalization"]["clip"] = 8.0
    if variant == "cnn_win256":
        cfg["window"].update(pre_samples=128, post_samples=128)
    if variant == "cnn_causal":
        cfg["signal"]["bandpass"]["zero_phase"] = False
    return cfg


def record_inputs(record: int, cfg: dict, variant: str, noise=None) -> dict:
    """Raw model inputs of one record: beat windows (plus RR channels) or the table of measures."""
    needs_features = variant in ("cnn_rr", "mlp")
    built = build_record(record, cfg, noise, with_features=needs_features)
    out = {"y": built["y"], "record": built["record"]}
    if variant == "mlp":
        out["x"] = built["features"][feature_sets()[MLP_FEATURES]].to_numpy(dtype=np.float32)
    else:
        x = built["X"]
        if variant == "cnn_rr":
            rr = built["features"][RR_CHANNELS].to_numpy(dtype=np.float32)
            rr = np.clip(np.nan_to_num(rr, nan=1.0), 0.0, 2.0) - 1.0
            x = np.concatenate([x, np.repeat(rr[:, :, None], x.shape[2], axis=2)], axis=1)
        out["x"] = x.astype(np.float32)
    return out


def build_model(variant: str, in_shape: tuple) -> nn.Module:
    if variant == "mlp":
        return nn.Sequential(nn.Linear(in_shape[0], 64), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
    channels = (8, 16, 16, 32, 32) if variant == "cnn_small" else (16, 32, 32, 64, 64)
    return ECGNet(num_classes=4, in_channels=in_shape[0], input_len=in_shape[1], channels=channels)


def augment_batch(x: torch.Tensor, a: dict, g: torch.Generator) -> torch.Tensor:
    """GPU version of ECGBeatDataset._augment, applied to the ECG channel only."""
    sig, rest = x[:, :1], x[:, 1:]
    b, _, n = sig.shape
    u = lambda lo, hi: torch.rand(b, 1, 1, device=x.device, generator=g) * (hi - lo) + lo  # noqa: E731
    sig = sig * u(*a["amplitude_scale"])
    shift = torch.randint(-a["max_shift"], a["max_shift"] + 1, (b, 1), device=x.device, generator=g)
    idx = (torch.arange(n, device=x.device)[None, :] - shift) % n
    sig = sig.gather(2, idx[:, None, :])
    bw = a["baseline_wander"]
    if bw.get("enabled", True):
        t = torch.arange(n, device=x.device, dtype=x.dtype)[None, None, :] / bw["fs"]
        sig = sig + u(0, bw["amplitude"]) * torch.sin(2 * torch.pi * u(0.05, bw["max_freq_hz"]) * t + u(0, 2 * torch.pi))
    sig = sig + torch.randn(sig.shape, device=x.device, generator=g) * u(0, a["gaussian_noise_std"])
    return torch.cat([sig.clamp(-1.0, 1.0), rest], dim=1)


def class_weights(y: torch.Tensor, power: float | None) -> torch.Tensor | None:
    if power is None:
        return None
    counts = torch.bincount(y, minlength=4).clamp(min=1).float()
    w = (counts.sum() / counts) ** power
    return w / w.mean()


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.cat([model(x[i:i + 4096]).argmax(1) for i in range(0, len(x), 4096)]).cpu().numpy()


def train(model, x, y, tcfg, weight_power, augmentation, epochs, seed, on_epoch=None):
    g = torch.Generator(device=x.device).manual_seed(seed)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y, weight_power))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    batch, steps = tcfg["batch_size"], len(y) // tcfg["batch_size"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(y), device=x.device, generator=g)
        for i in range(steps):  # the last incomplete batch is dropped
            idx = perm[i * batch:(i + 1) * batch]
            xb = augment_batch(x[idx], augmentation, g) if augmentation else x[idx]
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), y[idx]).backward()
            optimizer.step()
            scheduler.step()
        if on_epoch:
            on_epoch(epoch, model)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--protocol", default="loro", choices=["loro", "split"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", action="store_true", help="also score every fold on its held-out record with noise")
    ap.add_argument("--jobs", type=int, default=1, help="processes for preparing the records")
    ap.add_argument("--out", default="results/exploration", help="output directory")
    args = ap.parse_args()

    train_cfg = load_config("configs/train.yaml")
    base_cfg = load_config(train_cfg["data_config"])
    cfg = prepare_config(base_cfg, args.variant)
    classes, records = class_names(base_cfg), ds1_records(base_cfg)
    val_records = split_records(base_cfg)["val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    tcfg = train_cfg["training"]
    weight_power = {"cnn_noweight": None, "cnn_invweight": 1.0}.get(args.variant, train_cfg["imbalance"]["weight_power"])
    augmentation = None if args.variant in ("cnn_noaug", "mlp") else train_cfg["augmentation"]

    parts = Parallel(n_jobs=args.jobs)(delayed(record_inputs)(r, cfg, args.variant) for r in records)
    x_all = np.concatenate([p["x"] for p in parts])
    y_all = np.concatenate([p["y"] for p in parts])
    rec_all = np.concatenate([p["record"] for p in parts])
    conditions = [c for c in noise_conditions(base_cfg) if c is not None] if args.noise else []
    folds = [[r] for r in records] if args.protocol == "loro" else [val_records]

    tag = f"edge_{args.variant}_s{args.seed}_{args.protocol}"
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled = {"clean": np.full_like(y_all, -1), **{condition_name(c): np.full_like(y_all, -1) for c in conditions}}
    curve, n_params, t0 = [], None, time.time()
    for k, held_out in enumerate(folds):
        test = np.isin(rec_all, held_out)
        x_train, x_test = x_all[~test], x_all[test]
        if args.variant == "mlp":  # scale with training-fold statistics, then to roughly [-1, 1] for 8-bit inputs
            scaler = make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(
                n_quantiles=256, output_distribution="normal", subsample=50_000, random_state=args.seed)).fit(x_train)
            to_input = lambda a: np.clip(scaler.transform(a) / 3.0, -1.0, 1.0).astype(np.float32)  # noqa: E731
        else:
            to_input = lambda a: a  # noqa: E731
        xt = torch.from_numpy(to_input(x_train)).to(device)
        yt = torch.from_numpy(y_all[~test]).to(device)
        xv = torch.from_numpy(to_input(x_test)).to(device)

        torch.manual_seed(args.seed * 1000 + k)
        model = build_model(args.variant, tuple(xt.shape[1:])).to(device)
        n_params = count_parameters(model)

        def on_epoch(epoch, m, y_true=y_all[test], xv=xv):
            metrics = class_metrics(y_true, predict(m, xv), classes)
            curve.append({"epoch": epoch, **{key: metrics[key] for key in ("accuracy", "macro_f1", "f1_N", "f1_S", "f1_V", "f1_F")}})

        train(model, xt, yt, tcfg, weight_power, augmentation, args.epochs, args.seed * 1000 + k,
              on_epoch if args.protocol == "split" else None)
        pooled["clean"][test] = predict(model, xv)
        for c in conditions:
            noisy = np.concatenate([record_inputs(r, cfg, args.variant, c)["x"] for r in held_out])
            pooled[condition_name(c)][test] = predict(model, torch.from_numpy(to_input(noisy)).to(device))
        print(f"[{tag}] fold {k + 1}/{len(folds)} (records {held_out}) done, {time.time() - t0:.0f} s elapsed", flush=True)

    scored = pooled["clean"] >= 0
    rows = [{"model": args.variant, "seed": args.seed, "protocol": args.protocol, "epochs": args.epochs,
             "parameters": n_params, "condition": name, **class_metrics(y_all[scored], pred[scored], classes)}
            for name, pred in pooled.items()]
    pd.DataFrame(rows).round(4).to_csv(out_dir / f"{tag}.csv", index=False)
    np.savez_compressed(out_dir / f"{tag}_predictions.npz", y=y_all, record=rec_all, **{k_: v for k_, v in pooled.items()})
    if curve:
        pd.DataFrame(curve).round(4).to_csv(out_dir / f"{tag}_curve.csv", index=False)

    per_record = []
    for r in np.unique(rec_all[scored]):
        m, row = rec_all == r, {"record": int(r), "split": "val" if r in val_records else "train"}
        row.update(beats=int(m.sum()), accuracy=float((pooled["clean"][m] == y_all[m]).mean()))
        for i, c in enumerate(classes):
            cls = m & (y_all == i)
            row[f"support_{c}"] = int(cls.sum())
            row[f"recall_{c}"] = float((pooled["clean"][cls] == i).mean()) if cls.any() else np.nan
            row[f"predicted_{c}"] = int((pooled["clean"][m] == i).sum())
        per_record.append(row)
    pd.DataFrame(per_record).round(4).to_csv(out_dir / f"{tag}_per_record.csv", index=False)

    m = rows[0]
    print(f"[{tag}] {n_params:,} parameters | clean: accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f} "
          f"(F1 N {m['f1_N']:.2f} S {m['f1_S']:.2f} V {m['f1_V']:.2f} F {m['f1_F']:.2f}) | {time.time() - t0:.0f} s")
    for row in rows[1:]:
        print(f"[{tag}] {row['condition']:>9s}: accuracy {row['accuracy']:.3f}, macro-F1 {row['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
