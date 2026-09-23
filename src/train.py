"""CRISP-DM stage 4 - train the FP32 edge model on DS1 (train split), select on the validation split.

Usage:
    python src/train.py [--config configs/train.yaml] [--epochs N]
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from config import load_config, resolve
from dataset import ECGBeatDataset
from evaluate import compute_metrics, predict
from model import build_model, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--epochs", type=int, help="override training.epochs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    if args.epochs:
        tcfg["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = load_config(cfg["data_config"])
    train_ds = ECGBeatDataset("train", augment=cfg.get("augmentation"), data_cfg=data_cfg, seed=cfg["seed"])
    val_ds = ECGBeatDataset("val", data_cfg=data_cfg)
    print(f"train: {dict(zip(train_ds.classes, train_ds.class_counts()))}")
    print(f"val:   {dict(zip(val_ds.classes, val_ds.class_counts()))}")

    strategy = cfg["imbalance"]["strategy"]
    sampler = train_ds.balanced_sampler() if strategy == "balanced_sampler" else None
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=sampler is None, sampler=sampler,
                          num_workers=tcfg["num_workers"], generator=torch.Generator().manual_seed(cfg["seed"]))
    weights = train_ds.class_weights(cfg["imbalance"]["weight_power"]).to(device) if strategy == "weighted_loss" else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    print(f"model parameters: {count_parameters(model):,} | device: {device}")

    run_dir = resolve("results/runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best, best_epoch, history = -1.0, 0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, total_loss = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        val_metrics = compute_metrics(val_ds.y.numpy(), predict(model, val_ds.X, device), val_ds.classes)
        score = val_metrics[tcfg["selection_metric"]]
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_ds), **val_metrics})
        print(f"epoch {epoch:3d} | loss {total_loss / len(train_ds):.4f} | val acc {val_metrics['accuracy']:.4f} "
              f"| val macro-F1 {val_metrics['macro_f1']:.4f} | {time.time() - t0:.1f}s")

        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch, "val_metrics": val_metrics},
                       run_dir / "best.pt")
        elif epoch - best_epoch >= tcfg["early_stopping_patience"]:
            print(f"early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"best val {tcfg['selection_metric']} = {best:.4f} at epoch {best_epoch}; checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
