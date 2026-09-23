"""MIT-BIH Noise Stress Test Database helpers (shared by noise_test.py, augmentation and EDA)."""
import numpy as np
import wfdb

from config import resolve


def load_noise(cfg: dict, noise_type: str, part: str = "eval") -> np.ndarray:
    """Zero-mean noise signal ('bw', 'em' or 'ma').

    part='train' returns the first `noise.train_fraction` of the record (augmentation / EDA),
    part='eval' the remainder (robustness evaluation only) - the two never overlap.
    """
    nc = cfg["noise"]
    rec = wfdb.rdrecord(str(resolve(cfg["paths"]["nstdb_dir"]) / noise_type))
    assert rec.fs == cfg["signal"]["fs"]
    n = rec.p_signal[:, nc["channel"]].astype(np.float64)
    cut = int(len(n) * nc["train_fraction"])
    n = n[cut:] if part == "eval" else n[:cut]
    return n - n.mean()


def scaled_noise(noise: np.ndarray, length: int, ecg_power: float, snr_db: float, rng) -> np.ndarray:
    """Tile the noise from a random offset to `length` samples, scaled so 10*log10(ecg_power / P_noise) = snr_db."""
    start = int(rng.integers(len(noise)))
    seg = np.resize(np.roll(noise, -start), length)
    return seg * np.sqrt(ecg_power / (seg.var() * 10 ** (snr_db / 10)))
