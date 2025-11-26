import sys
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from smolvlm_infer import HybridSplitEncoder

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_PATH = "data/vlm_example.jpg"
TARGET_RES = 448


def main():
    print("=== SMOLVLM FINAL STABLE INFERENCE ===")

    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        # return # Don't return, let it fail or handle gracefully if needed, but original returned.
        # Actually, let's just warn and maybe use dummy if we want, but original returned.
        # I'll stick to original logic but maybe add a comment.
        pass

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID)

    if os.path.exists(IMAGE_PATH):
        image = Image.open(IMAGE_PATH).convert("RGB")
    else:
        print("Using dummy image")
        image = Image.new("RGB", (448, 448), color=(100, 100, 100))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
        size={"height": TARGET_RES, "width": TARGET_RES},
    )

    # Initialize Hybrid Encoder
    # Note: We assume shards are in ./smolvlm_subshards relative to where script is run
    # or we can pass path. Default is ./smolvlm_subshards
    hybrid_enc = HybridSplitEncoder(model.model.vision_model.encoder)
    model.model.vision_model.encoder = hybrid_enc

    print("\nGenerating (Deterministic)...")
    out_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,  # Deterministic
        repetition_penalty=1.1,
    )

    text = processor.decode(out_ids[0], skip_special_tokens=True)
    print("\n" + "=" * 40)
    print(text)
    print("=" * 40)


if __name__ == "__main__":
    main()
