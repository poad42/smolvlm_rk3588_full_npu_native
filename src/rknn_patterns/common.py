import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeLayerNorm(nn.Module):
    """
    Wraps LayerNorm to ensure compatibility with RKNN export.
    """

    def __init__(self, original_ln):
        super().__init__()
        self.weight = original_ln.weight
        self.bias = original_ln.bias
        self.eps = original_ln.eps
        self.normalized_shape = original_ln.normalized_shape

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class DecomposedGELU(nn.Module):
    """
    Approximates GELU with Sigmoid for better NPU support.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(1.702))

    def forward(self, x):
        return x * torch.sigmoid(x * self.alpha)
