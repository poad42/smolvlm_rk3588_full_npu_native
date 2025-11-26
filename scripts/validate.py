import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from smolvlm_infer.runner import RKNNBlockRunner

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
SHARD_DIR = "./smolvlm_subshards"
IMAGE_PATH = "/home/poad42/sml_project/data/vlm_example.jpg"
TARGET_RES = 448


def get_npu_input_data(cpu_tensor, layer_idx):
    """
    Prepares data for NPU.
    CRITICAL: For Layer 0 and 1, we manually shrink data by 0.1
    so it fits through the FP16 'transfer window' without crashing.
    The NPU model (Sandwich Block) will multiply it back by 10x internally.
    """
    data = cpu_tensor.detach().numpy().astype(np.float32)

    is_sandwich_layer = (layer_idx < 2) or (layer_idx >= 5)

    if is_sandwich_layer:
        return data * 0.1  # <--- The Sandwich Pre-Scale
    else:
        return data  # Standard transfer


def main():
    print("=== SMOLVLM-256M ULTIMATE VALIDATION (SANDWICH MODE) ===")

    # Check for image, use dummy if not found (adapted from original)
    if not os.path.exists(IMAGE_PATH):
        print("Image not found, using dummy.")
        image = Image.new("RGB", (448, 448), color=(100, 100, 100))
    else:
        image = Image.open(IMAGE_PATH).convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID).eval()

    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "."}],
            }
        ],
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
        size={"height": TARGET_RES, "width": TARGET_RES},
    )

    print(
        f"{'Layer':<15} | {'CPU Mean':<10} {'CPU Std':<10} | {'NPU Mean':<10} {'NPU Std':<10} | {'Cos Sim':<8} | Status"
    )
    print("-" * 90)

    # 1. Load RKNN Runners
    runners = []
    print("Loading NPU Shards...")
    for i in range(12):
        core = i % 3
        # L{i}_attn and L{i}_mlp
        runners.append(RKNNBlockRunner(f"{SHARD_DIR}/l{i}_attn.rknn", core_id=core))
        runners.append(RKNNBlockRunner(f"{SHARD_DIR}/l{i}_mlp.rknn", core_id=core))

    # 2. Capture Initial Embeddings (Using Hook to be safe)
    captured_embeds = []

    def hook_fn(module, args, output):
        captured_embeds.append(output.detach())

    handle = model.model.vision_model.embeddings.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)  # Run once to trigger hook
    handle.remove()

    if not captured_embeds:
        print("Error: Failed to capture embeddings.")
        return

    # This is the Clean Input for Layer 0
    current_cpu = captured_embeds[0]
    layers = model.model.vision_model.encoder.layers

    # 3. Layer-by-Layer Validation
    with torch.no_grad():
        for i in range(12):

            # ===========================
            # PART A: ATTENTION BLOCK
            # ===========================

            # 1. Run CPU (Ground Truth)
            residual = current_cpu
            x = layers[i].layer_norm1(current_cpu)
            x, _ = layers[i].self_attn(x)
            # Note: In the split architecture, the block output includes the residual add
            target_cpu = residual + x

            # 2. Run NPU (Validation)
            # We feed the Clean CPU Input into the NPU (Teacher Forcing)
            runner = runners[i * 2]
            npu_input_blob = get_npu_input_data(current_cpu, i)
            npu_out = runner.run(npu_input_blob)

            # 3. Compare
            # Flatten for stats
            flat_cpu = target_cpu.flatten()
            flat_npu = torch.from_numpy(npu_out).flatten()
            min_len = min(len(flat_cpu), len(flat_npu))

            cpu_m, cpu_s = flat_cpu.mean().item(), flat_cpu.std().item()
            npu_m, npu_s = flat_npu.mean().item(), flat_npu.std().item()
            cos = F.cosine_similarity(
                flat_cpu[:min_len].unsqueeze(0), flat_npu[:min_len].unsqueeze(0)
            ).item()

            status = "✅" if cos > 0.9 else "❌"
            print(
                f"L{i}_Attn        | {cpu_m:<10.4f} {cpu_s:<10.4f} | {npu_m:<10.4f} {npu_s:<10.4f} | {cos:<8.4f} | {status}"
            )

            # Update CPU stream for next step
            current_cpu = target_cpu

            # ===========================
            # PART B: MLP BLOCK
            # ===========================

            # 1. Run CPU (Ground Truth)
            residual = current_cpu
            x = layers[i].layer_norm2(current_cpu)
            x = layers[i].mlp(x)
            target_cpu = residual + x

            # 2. Run NPU (Validation)
            runner = runners[i * 2 + 1]
            npu_input_blob = get_npu_input_data(current_cpu, i)
            npu_out = runner.run(npu_input_blob)

            # 3. Compare
            flat_cpu = target_cpu.flatten()
            flat_npu = torch.from_numpy(npu_out).flatten()
            min_len = min(len(flat_cpu), len(flat_npu))

            cpu_m, cpu_s = flat_cpu.mean().item(), flat_cpu.std().item()
            npu_m, npu_s = flat_npu.mean().item(), flat_npu.std().item()
            cos = F.cosine_similarity(
                flat_cpu[:min_len].unsqueeze(0), flat_npu[:min_len].unsqueeze(0)
            ).item()

            status = "✅" if cos > 0.9 else "❌"
            print(
                f"L{i}_MLP         | {cpu_m:<10.4f} {cpu_s:<10.4f} | {npu_m:<10.4f} {npu_s:<10.4f} | {cos:<8.4f} | {status}"
            )

            # Update CPU stream for next layer
            current_cpu = target_cpu


if __name__ == "__main__":
    main()
