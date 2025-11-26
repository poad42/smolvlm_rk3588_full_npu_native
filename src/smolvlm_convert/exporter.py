import torch
import onnx
import os
import gc
import numpy as np
from transformers import AutoModelForVision2Seq
from rknn.api import RKNN
from rknn_patterns import UniversalBlock

try:
    from rkllm.api import RKLLM
except ImportError:
    RKLLM = None

# --- CONFIG ---
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_WORK_DIR = "./smolvlm_subshards"
TOTAL_LAYERS = 12


def sanitize_onnx(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    nodes = [n for n in graph.node if n.op_type != "Identity"]
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.save(model, model_path)


def compile_part(layer_idx, part, model_obj, use_fp16, work_dir, custom_dataset=None):
    name = f"l{layer_idx}_{part}"
    print(f"  Processing {name} [{'FP16' if use_fp16 else 'INT8'}]...")

    onnx_path = f"{work_dir}/{name}.onnx"
    rknn_path = f"{work_dir}/{name}.rknn"

    # Dummy input
    # If scaling is used (L0), we simulate the Scaled Down input
    dummy = torch.randn(1, 1024, 768)

    torch.onnx.export(
        model_obj,
        (dummy,),
        onnx_path,
        input_names=["in"],
        output_names=["out"],
        opset_version=12,
        do_constant_folding=True,
    )
    sanitize_onnx(onnx_path)

    rknn = RKNN(verbose=False)

    # Rules
    rules = [
        "fuse_matmul_softmax_matmul_to_sdpa",
        "fuse_transpose_reshape",
        "fuse_channel_shuffle_split_concat_to_gather",
        "squeeze_nd_add",
        "squeeze_to_4d_add",
        "eliminate_add_redundancy_const_dim",
    ]

    if use_fp16:
        # Strict FP16 config
        fp16_rules = rules + ["fuse_exmatmul_mul", "convert_softmax_to_exsoftmax13"]
        rknn.config(
            target_platform="rk3588",
            optimization_level=0,
            float_dtype="float16",
            disable_rules=fp16_rules,
        )
        rknn.load_onnx(model=onnx_path)
        rknn.build(do_quantization=False)
    else:
        # INT8 config
        rknn.config(
            target_platform="rk3588",
            optimization_level=0,
            float_dtype="float16",
            quantized_dtype="asymmetric_quantized-8",
            disable_rules=rules,
        )
        rknn.load_onnx(model=onnx_path)
        ds_path = custom_dataset if custom_dataset else f"{work_dir}/dataset_random.txt"
        rknn.build(do_quantization=True, dataset=ds_path)

    if rknn.export_rknn(rknn_path) != 0:
        print(f"ERROR: {name} failed.")
        exit(1)

    del rknn
    gc.collect()


def export_all(model_id=DEFAULT_MODEL_ID, work_dir=DEFAULT_WORK_DIR):
    print("\n[2/3] Compiling Shards (High-Fidelity Protocol)...")
    full_model = AutoModelForVision2Seq.from_pretrained(model_id).eval()
    orig_layers = full_model.model.vision_model.encoder.layers

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Generate dummy calibration for the few remaining INT8 layers
    np.save(
        f"{work_dir}/calib_random.npy", np.random.randn(1, 1024, 768).astype(np.float32)
    )
    with open(f"{work_dir}/dataset_random.txt", "w") as f:
        f.write(f"{os.path.abspath(f'{work_dir}/calib_random.npy')}\n")

    for i in range(TOTAL_LAYERS):

        # --- LOGIC UPDATE ---
        # GREEN ZONES (Keep INT8): Layers 2, 3, 4
        # RED ZONES (Force FP16): Layers 0, 1 (Head) AND 5-11 (Deep Body)

        use_fp16 = True  # Forcing FP16 as per original script logic which had use_fp16 = True hardcoded inside the loop

        if use_fp16:
            print(f"--- Exporting L{i} in FP16 (Sandwich Mode) ---")
            # FP16 + Sandwich (Protects inputs from driver bug)
            attn = UniversalBlock(orig_layers[i], "attn", use_sandwich=True).eval()
            mlp = UniversalBlock(orig_layers[i], "mlp", use_sandwich=True).eval()
            compile_part(i, "attn", attn, use_fp16=True, work_dir=work_dir)
            compile_part(i, "mlp", mlp, use_fp16=True, work_dir=work_dir)

        else:
            print(f"--- Exporting L{i} in INT8 (Standard Mode) ---")
            # INT8 (L2, L3, L4 are stable enough)
            attn = UniversalBlock(orig_layers[i], "attn", use_sandwich=False).eval()
            mlp = UniversalBlock(orig_layers[i], "mlp", use_sandwich=False).eval()

            # Use Harvested Data if exists, else Random
            calib_file = f"./calibration_data/calib_L{i}.npy"
            if os.path.exists(calib_file):
                ds_path = f"{work_dir}/dataset_L{i}.txt"
                with open(ds_path, "w") as f:
                    f.write(f"{os.path.abspath(calib_file)}\n")
                compile_part(
                    i,
                    "attn",
                    attn,
                    use_fp16=False,
                    work_dir=work_dir,
                    custom_dataset=ds_path,
                )
                compile_part(
                    i,
                    "mlp",
                    mlp,
                    use_fp16=False,
                    work_dir=work_dir,
                    custom_dataset=ds_path,
                )
            else:
                compile_part(i, "attn", attn, use_fp16=False, work_dir=work_dir)
                compile_part(i, "mlp", mlp, use_fp16=False, work_dir=work_dir)


def export_rkllm(model_id=DEFAULT_MODEL_ID, work_dir=DEFAULT_WORK_DIR):
    print("\n[3/3] Compiling LM with RKLLM...")

    if RKLLM is None:
        print("Warning: rkllm not installed. Skipping RKLLM export.")
        return

    model_path = os.path.join(work_dir, "rkllm_model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    llm = RKLLM()

    # Load model
    print(f"Loading model: {model_id}")

    model_path_arg = model_id
    if not os.path.exists(model_id):
        print(
            f"Model path '{model_id}' does not exist locally. Downloading from Hugging Face Hub..."
        )
        try:
            from huggingface_hub import snapshot_download

            # Download the full model to ensure we have the correct config/tokenizer for the VLM
            model_path_arg = snapshot_download(repo_id=model_id)
            print(f"Model downloaded to: {model_path_arg}")
        except ImportError:
            print(
                "Error: huggingface_hub not installed. Please install it to download models."
            )
            return
        except Exception as e:
            print(f"Error downloading model: {e}")
            return

    # Note: RKLLM supports SmolVLM (based on Idefics3/SmolLM2).
    # It typically converts the Language Model part into the .rkllm file.
    # The vision encoder (SigLIP) is handled separately (which matches our split approach).
    # We pass the full model path so RKLLM can read the correct config and tokenizer.
    ret = llm.load_huggingface(model=model_path_arg)
    if ret != 0:
        print("RKLLM Load failed!")
        return

    # Build
    print("Building RKLLM model...")
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8",
        target_platform="rk3588",
        num_npu_core=3,
    )
    if ret != 0:
        print("RKLLM Build failed!")
        return

    # Export
    export_path = f"{work_dir}/smolvlm.rkllm"
    print(f"Exporting to {export_path}...")
    ret = llm.export_rkllm(export_path)
    if ret != 0:
        print("RKLLM Export failed!")
        return

    print(f"RKLLM model exported to {export_path}")
