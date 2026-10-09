"""CRISP-DM stage 5 - noise-robustness evaluation with the MIT-BIH Noise Stress Test Database.

Noise is added to the RAW held-out ECG signal before the unchanged preprocessing pipeline
(filter -> segment -> normalize), mimicking a noisy recording rather than a noisy model input.

    SNR (dB) = 10 log10(S / N), defined as in the WFDB `nst` tool that generated the NSTDB records
    S = (peak-to-peak amplitude of the first 300 normal QRS complexes of the clean raw record)^2 / 8
    N = (RMS amplitude of the scaled noise in one-second windows, first 300 s of the noise record)^2

Unlike the default nst protocol (two-minute noisy and clean segments alternating), the noise covers
the whole record.

Leakage guard: DS1 validation records get the `cv` part of each noise record and DS2 the `eval` part
(`noise.parts` in configs/data.yaml); the `train` part is reserved for training-time augmentation.

Usage:
    python src/noise_test.py --checkpoint results/runs/<run>/best.pt --split val
    python src/noise_test.py --checkpoint results/runs/<run>/best.pt --split test --final
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import class_names, load_config, resolve, split_records
from evaluate import compute_metrics, load_checkpoint, predict
from noise import load_noise, noise_power, qrs_power, scaled_noise
from prepare_data import load_record, process_record

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (bw, em, ma)


def plot_noise_results(df: pd.DataFrame, path, metric="macro_f1"):
    clean = df.loc[df["noise_type"] == "clean", metric].iloc[0]
    snrs = sorted(df.loc[df["noise_type"] != "clean", "snr_db"].unique(), reverse=True)
    x_labels = ["clean"] + [f"{s:g} dB" for s in snrs]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.axhline(clean, color="#898781", lw=1, ls="--", zorder=1, label="clean reference")
    for color, (ntype, g) in zip(SERIES_COLORS, df[df["noise_type"] != "clean"].groupby("noise_type")):
        ys = [clean] + [g.loc[g["snr_db"] == s, metric].iloc[0] for s in snrs]
        ax.plot(range(len(ys)), ys, color=color, lw=2, marker="o", ms=6, label=ntype, zorder=3)
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required to evaluate on DS2 (test)")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("DS2 is reserved for the final evaluation - pass --final once model selection is frozen.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = load_config(ckpt["config"]["data_config"])
    classes = class_names(cfg)
    records = split_records(cfg)[args.split]
    rng = np.random.default_rng(cfg["seed"])

    conditions = [("clean", None)] + [(t, s) for t in cfg["noise"]["types"] for s in cfg["noise"]["snr_db"]]
    part = "eval" if args.split == "test" else "cv"
    noises = {t: (load_noise(cfg, t, part), noise_power(cfg, t)) for t in cfg["noise"]["types"]}
    ecg_power = {}
    for rec in records:
        x, fs, samples, symbols = load_record(cfg["paths"]["mitdb_dir"], rec, cfg["signal"]["lead"])
        ecg_power[rec] = (qrs_power(x, fs, samples, symbols), len(x))

    rows = []
    for ntype, snr in conditions:
        y_true, y_pred = [], []
        for rec in records:
            power, length = ecg_power[rec]
            noise = None if snr is None else scaled_noise(noises[ntype][0], length, power, noises[ntype][1], snr, rng)
            arrays, *_ = process_record(rec, cfg, noise=noise)
            y_true.append(arrays["y"])
            y_pred.append(predict(model, torch.from_numpy(arrays["X"]), device))
        m = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred), classes)
        rows.append({"noise_type": ntype, "snr_db": snr, "split": args.split, **m})
        print(f"{ntype:>5s} {'' if snr is None else f'{snr:>3g} dB':>6s} | acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    run = ckpt["config"]["run_name"]
    out = resolve("results")
    df.to_csv(out / f"noise_results_{run}_{args.split}.csv", index=False)
    plot_noise_results(df, out / f"noise_results_{run}_{args.split}.png")
    print(f"Saved results/noise_results_{run}_{args.split}.[csv|png]")


if __name__ == "__main__":
    main()
