"""Collect the exploratory results of explore_baselines.py and explore_edge_models.py into tables and figures.

Reads results/exploration/baseline_metrics.csv and edge_*_{loro,split}.csv, writes
    results/exploration/summary_clean.csv     one row per model and protocol (mean and spread over seeds)
    results/exploration/summary_noise.csv     macro F1 and per-class F1 under NSTDB noise (loro)
    results/exploration/summary_per_record.csv   accuracy per held-out record for the main models
    results/exploration/explore_f1_per_class.png, explore_noise.png, explore_per_record.png

Usage:
    python src/explore_summary.py
"""
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from config import resolve

CLASSES = ["N", "S", "V", "F"]
BLUE, GRAY, INK2 = "#2a78d6", "#898781", "#52514e"
NOISE_COLORS = {"bw": "#2a78d6", "em": "#eb6834", "ma": "#1baf7a"}
NOISE_NAMES = {"bw": "baseline wander", "em": "electrode motion", "ma": "muscle artifact"}
EDGE_LABELS = {
    "cnn": "CNN on beat window", "cnn_rr": "CNN + RR channels", "mlp": "MLP on measures",
    "cnn_noaug": "CNN, no augmentation", "cnn_noweight": "CNN, no class weights",
    "cnn_invweight": "CNN, inverse-frequency weights", "cnn_small": "CNN, half the channels",
    "cnn_clip8": "CNN, clip level 8", "cnn_win256": "CNN, 256-sample window", "cnn_causal": "CNN, causal filter",
}
METRICS = ["accuracy", "macro_f1"] + [f"{m}_{c}" for c in CLASSES for m in ("precision", "recall", "f1")]


def load() -> pd.DataFrame:
    out = resolve("results/exploration")
    base = pd.read_csv(out / "baseline_metrics.csv")
    base["label"] = np.where(base.features == "-", base.model, base.model + " | " + base.features)
    base["kind"], base["seed"], base["parameters"] = "baseline", 0, np.nan
    files = [p for p in sorted(out.glob("edge_*_s*_*.csv")) if not p.stem.endswith(("_curve", "_per_record"))]
    edge = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    edge["label"], edge["kind"] = edge.model.map(EDGE_LABELS), "edge model"
    return pd.concat([base, edge], ignore_index=True)


def main():
    out = resolve("results/exploration")
    df = load()
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.grid": True,
                         "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})

    # ---- clean results: mean and spread over seeds
    clean = df[df.condition == "clean"]
    g = clean.groupby(["kind", "label", "protocol"], sort=False)
    summary = g[METRICS].mean()
    summary.insert(0, "seeds", g.seed.nunique())
    summary.insert(1, "parameters", g.parameters.first())
    summary.insert(2, "macro_f1_min", g.macro_f1.min())
    summary.insert(3, "macro_f1_max", g.macro_f1.max())
    summary = summary.reset_index()
    summary.round(4).to_csv(out / "summary_clean.csv", index=False)

    loro = summary[summary.protocol == "loro"].set_index("label")
    split = summary[summary.protocol == "split"].set_index("label")
    order = list(loro.sort_values(["kind", "macro_f1"], ascending=[True, True]).index)
    ys = np.arange(len(order))
    fig, axes = plt.subplots(1, 5, figsize=(13, 0.27 * len(order) + 1.3), sharey=True)
    titles = ["macro F1"] + [f"F1 of {c}" for c in CLASSES]
    for ax, col, title in zip(axes, ["macro_f1"] + [f"f1_{c}" for c in CLASSES], titles):
        if col == "macro_f1":  # range over seeds where several were run
            ax.hlines(ys, loro.loc[order, "macro_f1_min"], loro.loc[order, "macro_f1_max"], color=BLUE, lw=2)
        ax.scatter(split.reindex(order)[col], ys, s=30, color=GRAY, edgecolor="white", linewidth=1.2, zorder=3,
                   label="fixed split: 18 training records -> 4 validation records")
        ax.scatter(loro.loc[order, col], ys, s=38, color=BLUE, edgecolor="white", linewidth=1.2, zorder=4,
                   label="leave-one-record-out over the 22 DS1 records")
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.axhline((loro.loc[order, "kind"] == "baseline").sum() - 0.5, color="#c3c2b7", lw=1)
    axes[0].set_yticks(ys, order, fontsize=8)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -1.6 / len(order)), ncols=2, fontsize=8)
    fig.savefig(out / "explore_f1_per_class.png", dpi=200, bbox_inches="tight")

    # ---- noise robustness (loro)
    scored = df[df.condition != "clean"][["label", "seed"]].drop_duplicates()  # only the runs that were scored with noise
    noise = df[df.protocol == "loro"].merge(scored, on=["label", "seed"])
    noise[["noise_type", "snr_db"]] = noise.condition.str.extract(r"(\w+) (-?\d+) dB")
    values = ["accuracy", "macro_f1"] + [f"f1_{c}" for c in CLASSES]
    table = noise.groupby(["label", "condition", "noise_type", "snr_db"], dropna=False, sort=False)[values].mean().reset_index()
    table.round(4).to_csv(out / "summary_noise.csv", index=False)
    labels = list(dict.fromkeys(table.label))
    fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.3), sharey=True, squeeze=False)
    for ax, label in zip(axes[0], labels):
        d = table[table.label == label]
        clean_value = d.loc[d.condition == "clean", "macro_f1"].iloc[0]
        for t, color in NOISE_COLORS.items():
            dn = d[d.noise_type == t].assign(snr=lambda x: x.snr_db.astype(float)).sort_values("snr", ascending=False)
            ax.plot(range(len(dn) + 1), [clean_value] + list(dn.macro_f1), color=color, lw=2, marker="o", ms=6,
                    markeredgecolor="white", label=NOISE_NAMES[t])
        ax.set_xticks(range(4), ["clean", "12 dB", "6 dB", "0 dB"])
        ax.set_ylim(0, 1)
        ax.set_title(textwrap.fill(label.replace(" | ", " on: "), 30), fontsize=9)
    axes[0, 0].set_ylabel("macro F1 (leave-one-record-out, DS1)")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.savefig(out / "explore_noise.png", dpi=200, bbox_inches="tight")

    # ---- accuracy per held-out record (loro): which patients fail, and how stable that is
    columns = {"gradient boosting\non measures": "baseline_per_record", "MLP on\nmeasures": "edge_mlp_s0_loro_per_record",
               "CNN + RR\nchannels": "edge_cnn_rr_s0_loro_per_record", "CNN\nseed 0": "edge_cnn_s0_loro_per_record",
               "CNN\nseed 1": "edge_cnn_s1_loro_per_record", "CNN\nseed 2": "edge_cnn_s2_loro_per_record"}
    first = pd.read_csv(out / "baseline_per_record.csv").set_index("record")
    per_record = pd.DataFrame({name: pd.read_csv(out / f"{stem}.csv").set_index("record").accuracy
                               for name, stem in columns.items()})
    per_record.rename(columns=lambda c: c.replace("\n", " ")).round(4).to_csv(out / "summary_per_record.csv")
    seq = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.4, 0.27 * len(per_record) + 1.2))
    ax.imshow(per_record.to_numpy(), aspect="auto", cmap=seq, vmin=0, vmax=1)
    for i in range(per_record.shape[0]):
        for j in range(per_record.shape[1]):
            v = per_record.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v >= 0.45 else "#0b0b0b")
    counts = first[[f"support_{c}" for c in CLASSES]]
    ax.set_yticks(range(len(per_record)), [f"{r}  ({', '.join(f'{n} {c}' for c, n in zip(CLASSES, row) if n)})"
                                           for r, row in zip(counts.index, counts.to_numpy())], fontsize=7)
    ax.set_xticks(range(per_record.shape[1]), per_record.columns, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, per_record.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, len(per_record)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("accuracy on each DS1 record when that record is held out of training", fontsize=8)
    fig.savefig(out / "explore_per_record.png", dpi=200, bbox_inches="tight")

    show = ["seeds", "parameters", "accuracy", "macro_f1"] + [f"{m}_{c}" for c in CLASSES[1:] for m in ("precision", "recall")]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        for protocol in ("loro", "split"):
            print(f"\n== clean, {protocol}")
            print(summary[summary.protocol == protocol].set_index("label")[show].round(3).to_string())
        print("\n== noise (loro), macro F1")
        print(table.pivot_table(index="label", columns="condition", values="macro_f1", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
