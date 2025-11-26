import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NanoTiledAttention(nn.Module):
    """
    Memory-efficient Tiled Attention for NPU.
    Splits attention computation into small tiles to avoid OOM and improve cache locality.
    """

    def __init__(self, original, tile_size=32):
        super().__init__()
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.out_proj = original.out_proj
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.tile_size = tile_size

    # Helper to safely transpose large matrices on NPU
    def safe_transpose(self, x):
        B, L, H, D = x.shape
        num_chunks = L // self.tile_size
        x = x.view(B, num_chunks, self.tile_size, H, D)
        x = x.transpose(2, 3)  # Swap Tile <-> Head
        x = x.transpose(1, 2)  # Swap Chunk <-> Head
        x = x.reshape(B, H, L, D)
        return x

    def safe_inverse_transpose(self, x):
        B, H, L, D = x.shape
        num_chunks = L // self.tile_size
        x = x.view(B, H, num_chunks, self.tile_size, D)
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        x = x.reshape(B, L, H, D)
        return x

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        bsz = int(hidden_states.shape[0])
        q_len = int(hidden_states.shape[1])

        # Projections
        q_raw = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        k_raw = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        v_raw = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )

        # Safe Transpose (The fix for "Illegal Transpose" error)
        query_states = self.safe_transpose(q_raw)
        key_states = self.safe_transpose(k_raw)
        value_states = self.safe_transpose(v_raw)

        # Tiled MatMul (The fix for memory spikes)
        num_chunks = max(1, q_len // self.tile_size)
        q_chunks = torch.chunk(query_states, num_chunks, dim=2)

        attn_outputs = []
        for q_chunk in q_chunks:
            # Standard attention on small tiles
            attn_weights = (
                torch.matmul(q_chunk, key_states.transpose(2, 3)) * self.scale
            )
            attn_weights = F.softmax(attn_weights, dim=-1)
            chunk_out = torch.matmul(attn_weights, value_states)
            attn_outputs.append(chunk_out)

        attn_output = torch.cat(attn_outputs, dim=2)
        attn_output = self.safe_inverse_transpose(attn_output)

        return self.out_proj(attn_output.reshape(bsz, q_len, self.embed_dim)), None
