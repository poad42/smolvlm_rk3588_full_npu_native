import torch
import numpy as np
from transformers.modeling_outputs import BaseModelOutput
from .runner import RKNNBlockRunner


class HybridSplitEncoder(torch.nn.Module):
    def __init__(self, original_encoder, shard_dir="./smolvlm_subshards"):
        super().__init__()
        self.config = original_encoder.config
        self.blocks = []

        print(f"Loading 24 NPU Shards (FULL FP16 MODE)...")
        for i in range(12):
            core = i % 3
            self.blocks.append(
                RKNNBlockRunner(f"{shard_dir}/l{i}_attn.rknn", core_id=core)
            )
            self.blocks.append(
                RKNNBlockRunner(f"{shard_dir}/l{i}_mlp.rknn", core_id=core)
            )

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        x_np = inputs_embeds.detach().numpy().astype(np.float32)

        # 1. Global Pre-Scale (Protect all FP16 layers)
        x_np = x_np * 0.1

        # 2. Run All Blocks
        # Since everyone is FP16 Sandwich, they all speak "Scale 0.1" language.
        for runner in self.blocks:
            x_np = runner.run(x_np)

        # 3. Global Restore
        x_np = x_np * 10.0

        # SIGNAL DIAGNOSTIC
        # Expect ~18.0 now (No INT8 loss)
        print(f"[Signal] Output Mean: {np.mean(x_np):.4f} | Std: {np.std(x_np):.4f}")
        return BaseModelOutput(last_hidden_state=torch.from_numpy(x_np))
