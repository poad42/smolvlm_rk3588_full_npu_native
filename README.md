# SmolVLM on RK3588 NPU

This library provides a high-performance implementation of **SmolVLM-256M** optimized for the Rockchip RK3588 NPU. It features novel quantization patterns and memory management techniques to enable stable FP16/INT8 hybrid inference on edge devices.

## 🌟 Key Features & Patterns

### 1. Sandwich Quantization Pattern
**Problem**: The RK3588 NPU can suffer from underflow or signal noise when processing raw FP16 activations in certain deep layers, leading to "garbage" output.
**Solution**: We wrap sensitive layers in a "Sandwich" of scalers:
- **`InputScaler`**: Multiplies input by 10x before entering the NPU block.
- **`OutputDescaler`**: Divides output by 10x after leaving the NPU block.
This keeps the signal magnitude in a safe range for the NPU's internal FP16 compute units.

### 2. NanoTiled Attention
**Problem**: Standard Self-Attention requires transposing large matrices ($1024 \times 1024$), which triggers "Illegal Transpose" errors or massive memory spikes on the NPU.
**Solution**: We decompose attention into small $32 \times 32$ tiles.
- **Safe Transpose**: Transposes are performed on small chunks, which the NPU compiler can handle efficiently.
- **Tiled MatMul**: Attention scores are computed block-by-block, drastically reducing peak memory usage.

### 3. Hybrid Split Architecture
**Problem**: Running the entire Vision Encoder as a single graph exceeds the NPU's memory limits and causes core contention.
**Solution**: We split the model into **24 separate shards** (Attention and MLP blocks for each of the 12 layers).
- **Core Balancing**: Shards are distributed across NPU Cores 0, 1, and 2 in a round-robin fashion.
- **Pipeline Execution**: The host CPU orchestrates the data flow between shards, allowing for fine-grained control and debugging.

## 📦 Library Structure

The project is organized into three main packages under `src/`:

### `src/rknn_patterns/`
Contains the reusable design patterns that can be applied to other models.
- `sandwich.py`: Implementation of Input/Output scalers.
- `attention.py`: `NanoTiledAttention` implementation.
- `blocks.py`: `UniversalBlock` wrapper that applies these patterns to HuggingFace layers.

### `src/smolvlm_convert/`
Tools for converting the HuggingFace model to RKNN format.
- **Requires**: `rknn-toolkit2` (x86 Host).
- **Usage**: `python scripts/convert.py`

### `src/smolvlm_infer/`
Runtime engine for executing the model on the RK3588.
- **Requires**: `rknnlite2` (Edge Device).
- **Usage**: `python scripts/run_inference.py`

## 🚀 Getting Started

### 1. Conversion (On x86 Host)
**Requirement**: Python 3.10 is required for `rknn-toolkit2`.

Install dependencies:
```bash
pip install -r requirements_convert.txt
```

Run the conversion script:
```bash
python scripts/convert.py
```
This will generate `.rknn` files in `smolvlm_subshards/`.

### 2. Inference (On RK3588)
Transfer the `smolvlm_subshards/` directory and the `src/` folder to your board.

Install dependencies:
```bash
pip install -r requirements_infer.txt
```

Run inference:
```bash
python scripts/run_inference.py
```

## 🛠️ Validation
To validate the numerical accuracy of the NPU execution against the CPU (PyTorch) baseline:
```bash
python scripts/validate.py
```
This script runs a layer-by-layer cosine similarity check to ensure the NPU is producing correct results.
