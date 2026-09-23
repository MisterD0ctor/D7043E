"""ai8x-training port of src/model.py::ECGNet for the MAX78002.

DRAFT - not yet run through ai8x-training. Copy to ai8x-training/models/ (see max78002/README.md).
Keep this file layer-for-layer identical to ECGNet with its default parameters.
"""
from torch import nn

import ai8x


class AI85ECGNet(nn.Module):
    """Input (1, 512) -> 5 fused Conv1d blocks -> MaxPool(4) -> Linear(512, 4)."""

    def __init__(self, num_classes=4, num_channels=1, dimensions=(512, 1), bias=True, **kwargs):
        super().__init__()
        length = dimensions[0]
        self.conv1 = ai8x.FusedConv1dBNReLU(num_channels, 16, 5, stride=1, padding=2, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv1dBNReLU(16, 32, 5, pool_size=2, pool_stride=2,
                                                   stride=1, padding=2, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv1dBNReLU(32, 32, 5, pool_size=2, pool_stride=2,
                                                   stride=1, padding=2, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv1dBNReLU(32, 64, 5, pool_size=2, pool_stride=2,
                                                   stride=1, padding=2, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv1dBNReLU(64, 64, 5, pool_size=2, pool_stride=2,
                                                   stride=1, padding=2, bias=bias, **kwargs)
        self.pool = ai8x.MaxPool1d(kernel_size=4, stride=4, **kwargs)
        self.drop = nn.Dropout(p=0.2)
        self.fc = ai8x.Linear(64 * (length // 64), num_classes, bias=True, wide=True, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ai85ecgnet(pretrained=False, **kwargs):
    assert not pretrained
    return AI85ECGNet(**kwargs)


models = [
    {
        'name': 'ai85ecgnet',
        'min_input': 1,
        'dim': 1,
    },
]
