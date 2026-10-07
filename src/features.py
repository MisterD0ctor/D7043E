"""Signal measures per heartbeat: explored in the EDA (CRISP-DM stage 2) and candidate inputs for a classifier.

All measures are computed from the band-pass-filtered MLII signal (mV) of one record and its annotations,
on three segments around the annotated R-peak:

    beat segment  B   256 samples, -278 ... +433 ms: one beat from P wave to T wave
    QRS segment   Q   +-60 ms (narrower or wider where the description says so)
    beat window   W   the 512 samples prepared by prepare_data.py (window.pre_samples / post_samples)

FEATURES maps every measure to (family, description). To add a measure, add an entry to FEATURES and
assign the column of the same name in beat_features().

Availability on a device: template_corr, template_rmse, qrs_ptp_rel, qrs_width_rel and energy_rel use all
beats of the record, and rr_post, rr_post_ratio, rr_pre_post and rr_diff need the next beat. A classifier
that runs in real time needs causal versions of these (a running template, a delayed decision).

Usage (see notebooks/EDA.ipynb):
    df = beat_features(xf, fs, samples, symbols, cfg)   # one row per mappable beat of the record
"""
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

from config import symbol_to_class
from prepare_data import rr_features

B_PRE, B_POST = 100, 156  # beat segment; 256 samples = 2^8 for the FFT and the Haar wavelet levels

FEATURES = {
    # --- amplitude statistics of the beat segment
    "mean": ("amplitude", "mean (mV)"),
    "median": ("amplitude", "median (mV)"),
    "std": ("amplitude", "standard deviation (mV)"),
    "rms": ("amplitude", "root mean square (mV)"),
    "min": ("amplitude", "minimum (mV)"),
    "max": ("amplitude", "maximum (mV)"),
    "ptp": ("amplitude", "peak-to-peak range (mV)"),
    "iqr": ("amplitude", "interquartile range (mV)"),
    "mav": ("amplitude", "mean absolute value (mV)"),
    "skewness": ("amplitude", "skewness of the sample distribution"),
    "kurtosis": ("amplitude", "excess kurtosis of the sample distribution"),
    "crest_factor": ("amplitude", "largest absolute value / RMS"),
    "energy": ("amplitude", "sum of squares / fs (mV^2 s)"),
    "abs_area": ("amplitude", "area between signal and baseline (mV s)"),
    # --- shape and complexity of the beat segment
    "zero_crossings": ("shape", "crossings of the segment median"),
    "line_length": ("shape", "sum of absolute sample-to-sample differences (mV)"),
    "max_slope": ("shape", "largest absolute slope (mV/s)"),
    "slope_sign_changes": ("shape", "number of local extrema (sign changes of the slope)"),
    "hjorth_mobility": ("shape", "Hjorth mobility: std of the derivative / std of the signal"),
    "hjorth_complexity": ("shape", "Hjorth complexity: mobility of the derivative / mobility of the signal"),
    "teager_energy": ("shape", "mean Teager-Kaiser energy x[n]^2 - x[n-1] x[n+1]"),
    "n_peaks": ("shape", "peaks of |signal - baseline| with prominence >= 25 % of the largest"),
    "perm_entropy": ("shape", "permutation entropy, order 3, normalized to 0-1"),
    "amp_entropy": ("shape", "Shannon entropy of a 16-bin amplitude histogram, normalized to 0-1"),
    # --- QRS complex and neighbouring waves
    "r_amp": ("qrs", "amplitude at the annotated R-peak (mV)"),
    "qrs_ptp": ("qrs", "peak-to-peak range within +-50 ms (mV)"),
    "qrs_polarity": ("qrs", "sign of the largest deflection within +-50 ms"),
    "qrs_width": ("qrs", "time within +-120 ms above half of the largest deflection (ms)"),
    "qrs_area": ("qrs", "area between signal and baseline within +-60 ms (mV s)"),
    "qrs_energy_share": ("qrs", "energy within +-60 ms / energy of the beat segment"),
    "max_upslope": ("qrs", "steepest rise within +-60 ms (mV/s)"),
    "max_downslope": ("qrs", "steepest fall within +-60 ms (mV/s)"),
    "q_depth": ("qrs", "minimum in the 60 ms before the R-peak (mV)"),
    "s_depth": ("qrs", "minimum in the 80 ms after the R-peak (mV)"),
    "qrs_asymmetry": ("qrs", "(area after - area before the R-peak) / QRS area"),
    "t_amp": ("qrs", "largest deflection 150-400 ms after the R-peak, signed (mV)"),
    "t_qrs_ratio": ("qrs", "T deflection / QRS peak-to-peak range"),
    "p_amp": ("qrs", "largest deflection 250-80 ms before the R-peak, signed (mV)"),
    "p_energy_share": ("qrs", "energy 250-80 ms before the R-peak / energy of the beat segment"),
    "st_level": ("qrs", "mean level 80-120 ms after the R-peak (mV)"),
    # --- rhythm, from the annotated beat positions
    "rr_pre": ("rhythm", "interval to the previous beat (s)"),
    "rr_post": ("rhythm", "interval to the next beat (s)"),
    "rr_local": ("rhythm", "mean of the 10 previous intervals (s)"),
    "rr_pre_ratio": ("rhythm", "previous interval / local mean (prematurity)"),
    "rr_pre_ratio_32": ("rhythm", "previous interval / median of the 32 intervals before it"),
    "rr_pre_ratio_128": ("rhythm", "previous interval / median of the 128 intervals before it"),
    "rr_post_ratio": ("rhythm", "next interval / local mean (pause after the beat)"),
    "rr_pre_post": ("rhythm", "previous interval / next interval"),
    "rr_diff": ("rhythm", "next interval - previous interval (s)"),
    "heart_rate": ("rhythm", "60 / previous interval (bpm)"),
    "rr_local_std": ("rhythm", "standard deviation of the 10 previous intervals (s)"),
    "rr_pre_delta": ("rhythm", "previous interval - the interval before it (s)"),
    "beats_in_window": ("rhythm", "annotated beats inside the 512-sample beat window"),
    # --- spectrum of the beat segment (Hann window, 0.5-60 Hz)
    "bp_low": ("frequency", "share of spectral power in 0.5-5 Hz"),
    "bp_mid": ("frequency", "share of spectral power in 5-15 Hz"),
    "bp_high": ("frequency", "share of spectral power in 15-40 Hz"),
    "spec_centroid": ("frequency", "power-weighted mean frequency (Hz)"),
    "spec_spread": ("frequency", "power-weighted standard deviation of frequency (Hz)"),
    "median_freq": ("frequency", "frequency below which half of the power lies (Hz)"),
    "peak_freq": ("frequency", "frequency of the largest spectral peak (Hz)"),
    "edge_95": ("frequency", "frequency below which 95 % of the power lies (Hz)"),
    "spec_entropy": ("frequency", "Shannon entropy of the power spectrum, normalized to 0-1"),
    "spec_flatness": ("frequency", "geometric mean / arithmetic mean of the power spectrum"),
    # --- Haar wavelet decomposition of the beat segment (energy share per level)
    "dwt_d2": ("wavelet", "detail level 2, about 45-90 Hz"),
    "dwt_d3": ("wavelet", "detail level 3, about 22-45 Hz"),
    "dwt_d4": ("wavelet", "detail level 4, about 11-22 Hz"),
    "dwt_d5": ("wavelet", "detail level 5, about 6-11 Hz"),
    "dwt_a5": ("wavelet", "approximation level 5, below about 6 Hz"),
    # --- relative to the other beats of the same record (no labels used)
    "template_corr": ("patient-relative", "correlation with the record's median beat, -100 ... +150 ms"),
    "template_rmse": ("patient-relative", "RMS difference to the record's median beat (mV)"),
    "prev_beat_corr": ("patient-relative", "correlation with the previous beat, -100 ... +150 ms"),
    "qrs_ptp_rel": ("patient-relative", "QRS peak-to-peak range / median of the record"),
    "qrs_width_rel": ("patient-relative", "QRS width / median of the record"),
    "energy_rel": ("patient-relative", "energy / median of the record"),
    # --- the 512-sample beat window after normalization
    "win_std": ("prepared window", "standard deviation of the window, the normalization divisor (mV)"),
    "win_ptp": ("prepared window", "peak-to-peak range of the window (mV)"),
    "win_max_z": ("prepared window", "largest absolute z-score in the window"),
    "win_clipped_share": ("prepared window", "share of samples beyond the clip level"),
    "win_kurtosis": ("prepared window", "excess kurtosis of the window"),
}
FAMILIES = list(dict.fromkeys(family for family, _ in FEATURES.values()))


def _windows(x: np.ndarray, centers: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Rows x[c + start : c + stop] for every centre c."""
    return x[centers[:, None] + np.arange(start, stop)[None, :]]


def _sign_changes(a: np.ndarray) -> np.ndarray:
    s = np.sign(a)
    return (s[:, 1:] * s[:, :-1] < 0).sum(axis=1)


def _signed_extreme(a: np.ndarray) -> np.ndarray:
    return np.take_along_axis(a, np.abs(a).argmax(axis=1)[:, None], axis=1)[:, 0]


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row (a distribution), normalized by the number of bins."""
    logp = np.log(np.where(p > 0, p, 1.0))
    return -(p * logp).sum(axis=1) / np.log(p.shape[1])


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of a with the matching row of b (b may be a single row)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denom = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=-1))
    return (a * b).sum(axis=1) / np.maximum(denom, 1e-12)


def _haar_energy_shares(b: np.ndarray, levels: int = 5) -> np.ndarray:
    """Energy share of the Haar detail levels 1..levels and of the final approximation (last column)."""
    a = b - b.mean(axis=1, keepdims=True)
    energies = []
    for _ in range(levels):
        d = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
        a = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2)
        energies.append((d ** 2).sum(axis=1))
    e = np.stack(energies + [(a ** 2).sum(axis=1)], axis=1)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _rr_context(qrs_samples: np.ndarray, beat_samples: np.ndarray, fs: float, n_local: int = 10):
    """Standard deviation of the n_local previous RR intervals, and change of the previous interval (s)."""
    q = np.sort(qrs_samples)
    rr = np.diff(q) / fs
    idx = np.searchsorted(q, beat_samples)  # rr[idx - 1] is the interval that ends at the beat
    c1 = np.concatenate([[0.0], np.cumsum(rr)])
    c2 = np.concatenate([[0.0], np.cumsum(rr ** 2)])
    lo = np.clip(idx - n_local, 0, None)
    n = idx - lo
    mean = (c1[idx] - c1[lo]) / np.maximum(n, 1)
    var = (c2[idx] - c2[lo]) / np.maximum(n, 1) - mean ** 2
    local_std = np.where(n > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)
    last = np.clip(idx - 1, 0, len(rr) - 1)
    delta = np.where(idx > 1, rr[last] - rr[np.clip(idx - 2, 0, len(rr) - 1)], np.nan)
    return local_std, delta


def _rr_reference(qrs_samples: np.ndarray, beat_samples: np.ndarray, fs: float, k: int) -> np.ndarray:
    """Median of the k RR intervals before the one that ends at each beat (s): the rhythm the beat is compared with."""
    q = np.sort(qrs_samples)
    rr = np.diff(q) / fs
    ends = np.searchsorted(q, beat_samples) - 1  # rr[ends] is the interval that ends at the beat
    return np.array([np.median(rr[max(e - k, 0):e]) if e >= 1 else np.nan for e in ends])


def beat_features(x: np.ndarray, fs: float, samples: np.ndarray, symbols: np.ndarray, cfg: dict) -> pd.DataFrame:
    """All FEATURES for every mappable beat of one filtered record (same beats and order as prepare_data)."""
    mapping = symbol_to_class(cfg)
    pre, post = cfg["window"]["pre_samples"], cfg["window"]["post_samples"]
    keep = np.array([sym in mapping for sym in symbols]) & (samples - pre >= 0) & (samples + post <= len(x))
    s = samples[keep]
    ms = lambda t: int(round(t * fs / 1000))  # noqa: E731

    W = _windows(x, s, -pre, post)
    base = np.median(W, axis=1, keepdims=True)
    B = _windows(x, s, -B_PRE, B_POST)
    Bb = B - base
    dB = np.diff(B, axis=1)
    f = {}

    # amplitude statistics
    f["mean"], f["median"], f["std"] = B.mean(axis=1), np.median(B, axis=1), B.std(axis=1)
    f["rms"] = np.sqrt((B ** 2).mean(axis=1))
    f["min"], f["max"], f["ptp"] = B.min(axis=1), B.max(axis=1), np.ptp(B, axis=1)
    f["iqr"] = np.subtract(*np.percentile(B, [75, 25], axis=1))
    f["mav"] = np.abs(B).mean(axis=1)
    f["skewness"], f["kurtosis"] = stats.skew(B, axis=1), stats.kurtosis(B, axis=1)
    f["crest_factor"] = np.abs(B).max(axis=1) / np.maximum(f["rms"], 1e-9)
    f["energy"] = (B ** 2).sum(axis=1) / fs
    f["abs_area"] = np.abs(Bb).sum(axis=1) / fs

    # shape and complexity
    f["zero_crossings"] = _sign_changes(B - f["median"][:, None])
    f["line_length"] = np.abs(dB).sum(axis=1)
    f["max_slope"] = np.abs(dB).max(axis=1) * fs
    f["slope_sign_changes"] = _sign_changes(dB)
    mobility = dB.std(axis=1) / np.maximum(B.std(axis=1), 1e-9)
    f["hjorth_mobility"] = mobility
    f["hjorth_complexity"] = np.diff(dB, axis=1).std(axis=1) / np.maximum(dB.std(axis=1), 1e-9) / np.maximum(mobility, 1e-9)
    f["teager_energy"] = (B[:, 1:-1] ** 2 - B[:, :-2] * B[:, 2:]).mean(axis=1)
    f["n_peaks"] = np.array([len(sps.find_peaks(row, prominence=0.25 * row.max())[0]) for row in np.abs(Bb)])
    a, b, c = B[:, :-2], B[:, 1:-1], B[:, 2:]
    codes = (a < b).astype(int) + 2 * (b < c) + 4 * (a < c)  # ordinal pattern of each sample triple
    patterns = np.stack([(codes == k).mean(axis=1) for k in (0, 1, 2, 5, 6, 7)], axis=1)  # the 6 possible orders
    f["perm_entropy"] = _entropy(patterns / np.maximum(patterns.sum(axis=1, keepdims=True), 1e-12))
    bins = np.minimum(((B - f["min"][:, None]) / np.maximum(f["ptp"][:, None], 1e-9) * 16).astype(int), 15)
    f["amp_entropy"] = _entropy(np.stack([(bins == k).mean(axis=1) for k in range(16)], axis=1))

    # QRS complex and neighbouring waves
    q50 = _windows(x, s, -ms(50), ms(50) + 1) - base
    q60 = _windows(x, s, -ms(60), ms(60) + 1) - base
    q120 = _windows(x, s, -ms(120), ms(120) + 1) - base
    peak = np.abs(q50).max(axis=1)
    f["r_amp"] = x[s]
    f["qrs_ptp"] = np.ptp(q50, axis=1)
    f["qrs_polarity"] = np.sign(_signed_extreme(q50))
    f["qrs_width"] = (np.abs(q120) >= 0.5 * peak[:, None]).sum(axis=1) * 1000 / fs
    f["qrs_area"] = np.abs(q60).sum(axis=1) / fs
    f["qrs_energy_share"] = (q60 ** 2).sum(axis=1) / np.maximum((Bb ** 2).sum(axis=1), 1e-12)
    d60 = np.diff(q60, axis=1) * fs
    f["max_upslope"], f["max_downslope"] = d60.max(axis=1), d60.min(axis=1)
    f["q_depth"] = (_windows(x, s, -ms(60), 0) - base).min(axis=1)
    f["s_depth"] = (_windows(x, s, 1, ms(80) + 1) - base).min(axis=1)
    before, after = np.abs(q60[:, :ms(60)]).sum(axis=1), np.abs(q60[:, ms(60) + 1:]).sum(axis=1)
    f["qrs_asymmetry"] = (after - before) / np.maximum(after + before, 1e-12)
    f["t_amp"] = _signed_extreme(_windows(x, s, ms(150), ms(400)) - base)
    f["t_qrs_ratio"] = f["t_amp"] / np.maximum(f["qrs_ptp"], 1e-9)
    p_seg = _windows(x, s, -ms(250), -ms(80)) - base
    f["p_amp"] = _signed_extreme(p_seg)
    f["p_energy_share"] = (p_seg ** 2).sum(axis=1) / np.maximum((Bb ** 2).sum(axis=1), 1e-12)
    f["st_level"] = (_windows(x, s, ms(80), ms(120)) - base).mean(axis=1)

    # rhythm
    qrs = samples[np.isin(symbols, cfg["qrs_symbols"])]
    rr_pre, rr_post, rr_local = (v.astype(np.float64) for v in rr_features(qrs, s, fs))
    f["rr_pre"], f["rr_post"], f["rr_local"] = rr_pre, rr_post, rr_local
    f["rr_pre_ratio"], f["rr_post_ratio"] = rr_pre / rr_local, rr_post / rr_local
    f["rr_pre_ratio_32"], f["rr_pre_ratio_128"] = (rr_pre / _rr_reference(qrs, s, fs, k) for k in (32, 128))
    f["rr_pre_post"], f["rr_diff"] = rr_pre / rr_post, rr_post - rr_pre
    f["heart_rate"] = 60 / rr_pre
    f["rr_local_std"], f["rr_pre_delta"] = _rr_context(qrs, s, fs)
    q_sorted = np.sort(qrs)
    f["beats_in_window"] = np.searchsorted(q_sorted, s + post) - np.searchsorted(q_sorted, s - pre)

    # spectrum
    freqs = np.fft.rfftfreq(B.shape[1], 1 / fs)
    power = np.abs(np.fft.rfft((B - B.mean(axis=1, keepdims=True)) * sps.windows.hann(B.shape[1]), axis=1)) ** 2
    band = (freqs >= 0.5) & (freqs <= 60)
    freqs, power = freqs[band], power[:, band]
    total = np.maximum(power.sum(axis=1), 1e-20)
    share = power / total[:, None]
    for name, lo, hi in (("bp_low", 0.5, 5), ("bp_mid", 5, 15), ("bp_high", 15, 40)):
        f[name] = share[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
    f["spec_centroid"] = (share * freqs).sum(axis=1)
    f["spec_spread"] = np.sqrt((share * (freqs[None, :] - f["spec_centroid"][:, None]) ** 2).sum(axis=1))
    cumulative = np.cumsum(share, axis=1)
    f["median_freq"] = freqs[(cumulative >= 0.5).argmax(axis=1)]
    f["peak_freq"] = freqs[power.argmax(axis=1)]
    f["edge_95"] = freqs[(cumulative >= 0.95).argmax(axis=1)]
    f["spec_entropy"] = _entropy(share)
    f["spec_flatness"] = np.exp(np.log(np.maximum(power, 1e-20)).mean(axis=1)) / np.maximum(power.mean(axis=1), 1e-20)

    # Haar wavelet energy shares (columns: d1 ... d5, a5)
    dwt = _haar_energy_shares(B)
    f["dwt_d2"], f["dwt_d3"], f["dwt_d4"], f["dwt_d5"], f["dwt_a5"] = dwt[:, 1], dwt[:, 2], dwt[:, 3], dwt[:, 4], dwt[:, 5]

    # relative to the other beats of the record
    seg = _windows(x, s, -ms(100), ms(150)) - base
    template = np.median(seg, axis=0)
    f["template_corr"] = _corr(seg, template)
    f["template_rmse"] = np.sqrt(((seg - template) ** 2).mean(axis=1))
    f["prev_beat_corr"] = np.concatenate([[np.nan], _corr(seg[1:], seg[:-1])])
    f["qrs_ptp_rel"] = f["qrs_ptp"] / np.median(f["qrs_ptp"])
    f["qrs_width_rel"] = f["qrs_width"] / np.median(f["qrs_width"])
    f["energy_rel"] = f["energy"] / np.median(f["energy"])

    # the 512-sample beat window after normalization
    z = (W - W.mean(axis=1, keepdims=True)) / np.maximum(W.std(axis=1, keepdims=True), 1e-6)
    f["win_std"], f["win_ptp"] = W.std(axis=1), np.ptp(W, axis=1)
    f["win_max_z"] = np.abs(z).max(axis=1)
    f["win_clipped_share"] = (np.abs(z) >= cfg["normalization"]["clip"]).mean(axis=1)
    f["win_kurtosis"] = stats.kurtosis(W, axis=1)

    assert set(f) == set(FEATURES), set(f) ^ set(FEATURES)
    out = pd.DataFrame({name: np.asarray(f[name], dtype=np.float64) for name in FEATURES})
    out.insert(0, "y", [mapping[sym] for sym in symbols[keep]])
    out.insert(0, "symbol", symbols[keep])
    out.insert(0, "sample", s)
    return out
