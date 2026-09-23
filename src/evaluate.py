"""CRISP-DM stage 5 - evaluation metrics and reporting.

Usage:
    python src/evaluate.py --checkpoint results/runs/<run>/best.pt --split val
    python src/evaluate.py --checkpoint results/runs/<run>/best.pt --split test --final   # DS2, final evaluation only
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from config import load_config, resolve
from dataset import ECGBeatDataset
from model import build_model, count_parameters


def compute_metrics(y_true, y_pred, classes) -> dict:
    labels = list(range(len(classes)))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    for i, c in enumerate(classes):
        metrics.update({f"precision_{c}": p[i], f"recall_{c}": r[i], f"f1_{c}": f[i], f"support_{c}": int(support[i])})
    return metrics


@torch.no_grad()
def predict(model, X: torch.Tensor, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    preds = [model(X[i:i + batch_size].to(device)).argmax(1).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(preds).numpy()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def plot_confusion_matrix(cm: np.ndarray, classes, path: Path, title: str):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def append_metrics_row(row: dict, path=resolve("results/metrics.csv")):
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="required to evaluate on DS2 (test)")
    ap.add_argument("--model-label", default="fp32")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("DS2 is reserved for the final evaluation - pass --final once model selection is frozen.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    data_cfg = load_config(ckpt["config"]["data_config"])
    ds = ECGBeatDataset(args.split, data_cfg=data_cfg)
    y_pred = predict(model, ds.X, device)
    y_true = ds.y.numpy()

    metrics = compute_metrics(y_true, y_pred, ds.classes)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(ds.classes)))
    run = ckpt["config"]["run_name"]
    out_dir = resolve("results")
    out_dir.mkdir(exist_ok=True)
    stem = f"confusion_matrix_{run}_{args.model_label}_{args.split}"
    pd.DataFrame(cm, index=ds.classes, columns=ds.classes).to_csv(out_dir / f"{stem}.csv")
    plot_confusion_matrix(cm, ds.classes, out_dir / f"{stem}.png", f"{run} ({args.model_label}) - {args.split}")
    append_metrics_row({"run": run, "model": args.model_label, "split": args.split,
                        "parameters": count_parameters(model), **metrics})

    for k, v in metrics.items():
        print(f"{k:>14s}: {v:.4f}" if isinstance(v, float) else f"{k:>14s}: {v}")
    print(pd.DataFrame(cm, index=ds.classes, columns=ds.classes))


if __name__ == "__main__":
    main()
