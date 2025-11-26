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
SHARD_DIR = "./smolvlm_subshards"
IMAGE_PATH = "data/vlm_example.jpg"
TARGET_RES = (
    308  # Reduced to 308 to fit 512 token limit (308/14 = 22, 22^2 = 484 tokens)
)


def main():
    print("=== SMOLVLM FINAL STABLE INFERENCE ===")

    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        # return # Don't return, let it fail or handle gracefully if needed, but original return is safer.
        # Actually, let's just warn and maybe use dummy if we want, but original return is fine.
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

    # Check for RKLLM model
    rkllm_model_path = f"{SHARD_DIR}/smolvlm.rkllm"
    use_rkllm = False
    if os.path.exists(rkllm_model_path):
        print(f"RKLLM model found at {rkllm_model_path}. Using RKLLM for inference.")
        use_rkllm = True
    else:
        print(
            f"RKLLM model not found at {rkllm_model_path}. Using PyTorch CPU inference."
        )

    if use_rkllm:
        print("\nGenerating (RKLLM with RKNN Vision Encoder)...")

        # STEP 1: Process image through RKNN vision encoder
        print("Step 1: Running vision encoder (RKNN)...")

        # Use the working approach: replace encoder and run through model
        hybrid_enc = HybridSplitEncoder(model.model.vision_model.encoder)
        model.model.vision_model.encoder = hybrid_enc

        # Prepare pixel values - remove extra dimension if present
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim == 5:
            pixel_values = pixel_values.squeeze(0)

        # Run the vision model to get image features
        # This handles all the embeddings, position encoding, etc. correctly
        with torch.no_grad():
            vision_outputs = model.model.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            # Get the final hidden states from vision encoder
            image_embeds = vision_outputs.last_hidden_state

        print(f"Vision encoder output shape: {image_embeds.shape}")
        print(
            f"Vision embeddings mean: {image_embeds.mean():.4f}, std: {image_embeds.std():.4f}"
        )

        # STEP 2: Run RKLLM with image embeddings
        print("Step 2: Running language model (RKLLM)...")

        # Define callback to print tokens as they're generated
        def callback(result_ptr, userdata, state):
            if result_ptr.contents.text:
                text = result_ptr.contents.text.decode("utf-8")
                print(text, end="", flush=True)
            return 0  # Continue inference

        # Initialize RKLLM
        try:
            llm.set_chat_template(system_prompt="", prompt_prefix="", prompt_postfix="")

            # Run inference with multimodal input
            print("\n" + "=" * 40)

            # For multimodal input, RKLLM requires <image> tag in the prompt
            simple_prompt = "<image> Describe this image in detail."

            ret = llm.run_multimodal(
                prompt=simple_prompt,
                image_embed=image_embeds.numpy(),
                image_width=TARGET_RES,
                image_height=TARGET_RES,
            )
            print("\n" + "=" * 40)

            if ret != 0:
                print(f"\nWarning: rkllm_run_multimodal returned {ret}")

            llm.destroy()

        except Exception as e:
            print(f"\nError during RKLLM inference: {e}")
            import traceback

            traceback.print_exc()
            print("Falling back to CPU inference.")
            use_rkllm = False

    if not use_rkllm:
        print("RKLLM model not found. Using HybridSplitEncoder (RKNN Vision + CPU LM).")
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
