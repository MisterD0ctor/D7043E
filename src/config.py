"""Shared configuration helpers. All paths in configs/*.yaml are relative to the repo root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path="configs/data.yaml") -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_names(cfg: dict) -> list[str]:
    return list(cfg["labels"].keys())


def symbol_to_class(cfg: dict) -> dict[str, int]:
    """Map an MIT-BIH beat symbol to its class index (order = order of cfg['labels'])."""
    return {sym: idx for idx, syms in enumerate(cfg["labels"].values()) for sym in syms}


def split_records(cfg: dict) -> dict[str, list[int]]:
    """Return train/val/test record lists and assert that no record is shared between splits."""
    s = cfg["splits"]
    ds1, ds2, val = set(s["ds1"]), set(s["ds2"]), set(s["val"])
    assert not ds1 & ds2, f"DS1/DS2 overlap: {sorted(ds1 & ds2)}"
    assert val <= ds1, f"Validation records must come from DS1: {sorted(val - ds1)}"
    return {
        "train": sorted(ds1 - val),
        "val": sorted(val),
        "test": sorted(ds2),
    }
