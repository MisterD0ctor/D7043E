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


def development_records(cfg: dict) -> list[int]:
    """DS1 records available for training, validation and model selection (DS1 without `splits.exclude`)."""
    return sorted(set(cfg["splits"]["ds1"]) - set(cfg["splits"].get("exclude", [])))


def cv_folds(cfg: dict) -> list[tuple[list[int], list[int]]]:
    """(training records, held-out records) of every cross-validation fold over the development records."""
    scheme = cfg["cross_validation"]["scheme"]
    if scheme != "leave_one_record_out":
        raise ValueError(f"Unknown cross-validation scheme {scheme}")
    dev = development_records(cfg)
    return [([r for r in dev if r != held_out], [held_out]) for held_out in dev]


def split_records(cfg: dict) -> dict[str, list[int]]:
    """Return the record lists per split and assert that no record is shared between splits.

    "excluded" holds the DS1 records of `splits.exclude`: they are prepared and described, but never
    used for training, validation or model selection.
    """
    s = cfg["splits"]
    ds1, ds2, val, excluded = set(s["ds1"]), set(s["ds2"]), set(s["val"]), set(s.get("exclude", []))
    assert not ds1 & ds2, f"DS1/DS2 overlap: {sorted(ds1 & ds2)}"
    assert val <= ds1, f"Validation records must come from DS1: {sorted(val - ds1)}"
    assert excluded <= ds1, f"Excluded records must come from DS1: {sorted(excluded - ds1)}"
    assert not val & excluded, f"Excluded records cannot be validation records: {sorted(val & excluded)}"
    return {
        "train": sorted(ds1 - val - excluded),
        "val": sorted(val),
        "excluded": sorted(excluded),
        "test": sorted(ds2),
    }
