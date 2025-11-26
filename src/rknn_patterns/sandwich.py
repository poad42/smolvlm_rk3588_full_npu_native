import torch.nn as nn


class InputScaler(nn.Module):
    """
    Scales input up by 10x to protect from NPU underflow/noise.
    Part of the Sandwich Quantization pattern.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 10.0


class OutputDescaler(nn.Module):
    """
    Scales output down by 0.1x to restore original magnitude.
    Part of the Sandwich Quantization pattern.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.1
