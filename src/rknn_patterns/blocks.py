import torch.nn as nn
from .sandwich import InputScaler, OutputDescaler
from .common import NativeLayerNorm, DecomposedGELU
from .attention import NanoTiledAttention


class UniversalBlock(nn.Module):
    """
    Universal Wrapper that can apply Sandwich Quantization and Tiled Attention
    to any Transformer layer.
    """

    def __init__(self, original_layer, part="attn", use_sandwich=False):
        super().__init__()
        self.part = part
        self.use_sandwich = use_sandwich

        # Create Scalers
        if use_sandwich:
            self.scaler = InputScaler()
            self.descaler = OutputDescaler()

        if part == "attn":
            self.layer_norm = NativeLayerNorm(original_layer.layer_norm1)
            self.op = NanoTiledAttention(original_layer.self_attn)
        else:
            self.layer_norm = NativeLayerNorm(original_layer.layer_norm2)
            self.op = original_layer.mlp

        # Patch Activation
        target_op = self.op
        if hasattr(target_op, "act_fn"):
            target_op.act_fn = DecomposedGELU()
        elif hasattr(target_op, "activation_fn"):
            target_op.activation_fn = DecomposedGELU()
        elif hasattr(target_op, "act"):
            target_op.act = DecomposedGELU()

    def forward(self, hidden_states):
        x_in = hidden_states

        # 1. SCALE UP (Protect from NPU underflow/noise)
        if self.use_sandwich:
            x_in = self.scaler(x_in)

        residual = x_in
        x = self.layer_norm(x_in)

        if self.part == "attn":
            x, _ = self.op(x)
        else:
            x = self.op(x)

        out = residual + x

        # 2. SCALE DOWN (Restore 1x Magnitude for next layer)
        if self.use_sandwich:
            out = self.descaler(out)

        return out
