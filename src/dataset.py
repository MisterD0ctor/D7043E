"""PyTorch dataset for the prepared heartbeat windows, with training-only augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from config import load_config, resolve


def load_split(split: str, data_cfg: dict | None = None) -> dict:
    data_cfg = data_cfg or load_config("configs/data.yaml")
    with np.load(resolve(data_cfg["paths"]["processed_dir"]) / f"{split}.npz") as f:
        return {k: f[k] for k in f.files}


class ECGBeatDataset(Dataset):
    """Beat windows of shape (1, L) in [-1, 1] with integer labels N=0, S=1, V=2, F=3.

    Augmentation is only applied when `augment` is given - pass it for the training split only.
    """

    def __init__(self, split: str, augment: dict | None = None, data_cfg: dict | None = None, seed: int = 42):
        d = load_split(split, data_cfg)
        self.X = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(d["y"])
        self.record = d["record"]
        self.classes = [str(c) for c in d["classes"]]
        self.augment = augment if augment and augment.get("enabled") else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = self._augment(x.clone())
        return x, self.y[i]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        a, rng, L = self.augment, self.rng, x.shape[-1]
        if a.get("amplitude_scale"):
            x *= rng.uniform(*a["amplitude_scale"])
        if a.get("max_shift"):
            shift = int(rng.integers(-a["max_shift"], a["max_shift"] + 1))
            x = torch.roll(x, shift, dims=-1)
        if a.get("baseline_wander"):
            bw = a["baseline_wander"]
            t = torch.arange(L, dtype=x.dtype) / bw["fs"]
            freq, phase = rng.uniform(0.05, bw["max_freq_hz"]), rng.uniform(0, 2 * np.pi)
            x += rng.uniform(0, bw["amplitude"]) * torch.sin(2 * np.pi * freq * t + phase)
        if a.get("gaussian_noise_std"):
            x += torch.randn_like(x) * rng.uniform(0, a["gaussian_noise_std"])
        return x.clamp_(-1.0, 1.0)

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y.numpy(), minlength=len(self.classes))

    def class_weights(self, power: float = 1.0) -> torch.Tensor:
        """Inverse-frequency class weights, normalised to mean 1. power=0.5 gives sqrt-inverse weights."""
        counts = np.maximum(self.class_counts(), 1)
        w = (counts.sum() / counts) ** power
        return torch.tensor(w / w.mean(), dtype=torch.float32)

    def balanced_sampler(self) -> WeightedRandomSampler:
        w = 1.0 / np.maximum(self.class_counts(), 1)
        return WeightedRandomSampler(torch.as_tensor(w[self.y.numpy()], dtype=torch.double), len(self.y))
