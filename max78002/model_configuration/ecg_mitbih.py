"""ai8x-training dataset loader for the prepared MIT-BIH beats (data/processed/*.npz).

DRAFT - copy to ai8x-training/datasets/ (see max78002/README.md).

Leakage guard: ai8x-training uses the *test* loader for validation/model selection when
--validation-split 0 is given, so
    ECG_MITBIH      train = DS1 training records, test = DS1 validation records  (development)
    ECG_MITBIH_DS2  train = DS1 training records, test = DS2                     (final evaluation only)
Never train with a non-zero --validation-split: it splits beats randomly, not by record.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ('N', 'S', 'V', 'F')


class ECGBeats(Dataset):
    def __init__(self, path, act_mode_8bit):
        with np.load(path) as f:
            self.X = torch.from_numpy(f['X'])
            self.y = torch.from_numpy(f['y'])
        self.act_mode_8bit = act_mode_8bit

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Inputs are already in [-1, 1]; quantize to 8-bit steps like ai8x.normalize().
        x = self.X[i].mul(128.).round().clamp(min=-128, max=127)
        if not self.act_mode_8bit:
            x = x.div(128.)
        return x, int(self.y[i])


def _get_datasets(data, test_split, load_train=True, load_test=True):
    (data_dir, args) = data
    train = ECGBeats(os.path.join(data_dir, 'train.npz'), args.act_mode_8bit) if load_train else None
    test = ECGBeats(os.path.join(data_dir, f'{test_split}.npz'), args.act_mode_8bit) if load_test else None
    return train, test


def ecg_mitbih_get_datasets(data, load_train=True, load_test=True):
    return _get_datasets(data, 'val', load_train, load_test)


def ecg_mitbih_ds2_get_datasets(data, load_train=True, load_test=True):
    return _get_datasets(data, 'test', load_train, load_test)


datasets = [
    {
        'name': 'ECG_MITBIH',
        'input': (1, 512),
        'output': CLASSES,
        'loader': ecg_mitbih_get_datasets,
    },
    {
        'name': 'ECG_MITBIH_DS2',
        'input': (1, 512),
        'output': CLASSES,
        'loader': ecg_mitbih_ds2_get_datasets,
    },
]
