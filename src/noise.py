"""MIT-BIH Noise Stress Test Database helpers (shared by noise_test.py, augmentation and EDA).

SNR is defined as in the WFDB `nst` tool that generated the NSTDB records (amplitudes measured as by `sigamp`):

    SNR (dB) = 10 log10(S / (a^2 N))        a = gain applied to the noise record
    S = (peak-to-peak QRS amplitude of the clean ECG)^2 / 8
    N = (RMS amplitude of the noise in one-second windows)^2

Both amplitudes are trimmed means of up to 300 measurements (largest and smallest 5 % discarded), taken on
the unfiltered signals. Measuring the noise around the mean of each window keeps drift below 1 Hz out of N.
"""
import numpy as np
import wfdb

from config import resolve

# Beat symbols that WFDB's map1() maps to NORMAL (every QRS that is neither ventricular nor fusion);
# `sigamp -a` measures the QRS amplitude on these beats only.
NST_NORMAL_SYMBOLS = ["N", "L", "R", "B", "A", "a", "J", "S", "e", "j", "n", "/", "f", "Q"]
NST_MEASUREMENTS = 300         # measurements per amplitude estimate
NST_QRS_HALF_WINDOW_S = 0.05   # QRS range is taken within +-50 ms of the annotation
NST_RMS_WINDOW_S = 1.0         # noise RMS is taken per window, around the mean of that window


def _read_noise(cfg: dict, noise_type: str) -> np.ndarray:
    rec = wfdb.rdrecord(str(resolve(cfg["paths"]["nstdb_dir"]) / noise_type))
    assert rec.fs == cfg["signal"]["fs"]
    return rec.p_signal[:, cfg["noise"]["channel"]].astype(np.float64)


def _trimmed_mean(values) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    k = len(v) // 20
    return float(v[k:len(v) - k].mean())


def load_noise(cfg: dict, noise_type: str, part: str = "eval") -> np.ndarray:
    """Zero-mean noise signal ('bw', 'em' or 'ma') from one of the consecutive, non-overlapping parts
    of the record listed in `noise.parts`, in that order:

        train  training-time augmentation and the EDA
        cv     noisy held-out records in cross-validation / validation (model selection)
        eval   robustness evaluation on DS2 (final test only)
    """
    n = _read_noise(cfg, noise_type)
    parts = cfg["noise"]["parts"]
    if part not in parts:
        raise ValueError(f"Unknown noise part {part}; expected one of {list(parts)}")
    bounds = np.cumsum([0.0] + list(parts.values()))
    assert np.isclose(bounds[-1], 1.0), f"noise.parts must add up to 1, got {bounds[-1]}"
    i = list(parts).index(part)
    n = n[int(len(n) * bounds[i]):int(len(n) * bounds[i + 1])]
    return n - n.mean()


def noise_power(cfg: dict, noise_type: str) -> float:
    """nst noise power N, calibrated like nst on the first 300 s of the noise record (used for both parts)."""
    n = _read_noise(cfg, noise_type)
    w = int(round(NST_RMS_WINDOW_S * cfg["signal"]["fs"]))
    windows = n[: min(len(n) // w, NST_MEASUREMENTS) * w].reshape(-1, w)
    return _trimmed_mean(windows.std(axis=1)) ** 2


def qrs_power(x: np.ndarray, fs: float, samples: np.ndarray, symbols: np.ndarray) -> float:
    """nst signal power S of a raw ECG signal, from its first 300 normal QRS annotations."""
    h = int(round(NST_QRS_HALF_WINDOW_S * fs))
    beats = samples[np.isin(symbols, NST_NORMAL_SYMBOLS)]
    beats = beats[(beats >= h) & (beats + h < len(x))][:NST_MEASUREMENTS]
    return _trimmed_mean([np.ptp(x[s - h:s + h + 1]) for s in beats]) ** 2 / 8


def scaled_noise(noise: np.ndarray, length: int, s_power: float, n_power: float, snr_db: float, rng) -> np.ndarray:
    """Tile the noise from a random offset to `length` samples and apply the nst gain for `snr_db`."""
    start = int(rng.integers(len(noise)))
    seg = np.resize(np.roll(noise, -start), length)
    return seg * np.sqrt(s_power / (n_power * 10 ** (snr_db / 10)))
