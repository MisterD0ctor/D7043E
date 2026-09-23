"""Compact 1D CNN for heartbeat classification, restricted to MAX78002-compatible operations.

Each block mirrors an ai8x fused layer so the model can be ported 1:1 to ai8x-training
(see max78002/model_configuration/ai85ecgnet.py):

    block 0     : Conv1d -> BatchNorm -> ReLU                 == ai8x.FusedConv1dBNReLU
    blocks 1..n : MaxPool1d(2) -> Conv1d -> BatchNorm -> ReLU == ai8x.FusedMaxPoolConv1dBNReLU
    head        : MaxPool1d -> flatten -> Linear              == ai8x.MaxPool1d + ai8x.Linear

BatchNorm is folded into the convolution during quantization. Keep kernel_size <= 9,
padding <= 2 and stride 1 - verify limits against the ai8x-synthesis documentation
before changing the architecture.
"""
import torch
from torch import nn


class ECGNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, input_len=512,
                 channels=(16, 32, 32, 64, 64), kernel_size=5, head_pool=4, dropout=0.2):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size // 2 <= 2, "ai8x Conv1d padding is limited to 0..2"
        layers, c_in, length = [], in_channels, input_len
        for i, c_out in enumerate(channels):
            if i > 0:
                layers.append(nn.MaxPool1d(2))
                length //= 2
            layers += [nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(c_out), nn.ReLU()]
            c_in = c_out
        layers.append(nn.MaxPool1d(head_pool))
        length //= head_pool
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in * length, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(self.dropout(torch.flatten(x, 1)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict) -> nn.Module:
    return ECGNet(**cfg.get("model", {}).get("params", {}))


if __name__ == "__main__":
    m = ECGNet()
    n = count_parameters(m)
    print(m)
    print(f"parameters: {n:,}  (~{n / 1024:.1f} KB at INT8, ~{4 * n / 1024:.1f} KB at FP32)")
    print("output shape:", tuple(m(torch.zeros(2, 1, 512)).shape))
