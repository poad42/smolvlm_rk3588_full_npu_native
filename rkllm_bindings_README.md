# RKLLM Python Bindings

Python bindings for the RKLLM Runtime library (`librkllmrt.so`) using `ctypes`.

## Installation

### From Wheel (Recommended for Orange Pi 5 Max)

```bash
pip install rkllm_bindings-0.1.0-py3-none-any.whl
```

### From Source

```bash
cd /path/to/smolvlm_con_scripts
pip install -e .
```

## Prerequisites

**librkllmrt.so** is now **bundled with the wheel** - no separate installation required!

The library is automatically loaded from the package installation directory.

## Quick Start

```python
from rkllm_bindings import RKLLM

# Define a callback to handle generated tokens
def callback(result_ptr, userdata, state):
    if result_ptr.contents.text:
        text = result_ptr.contents.text.decode('utf-8')
        print(text, end='', flush=True)
    return 0  # Return 0 to continue, 1 to pause

# Initialize and run
with RKLLM(model_path="model.rkllm", callback=callback) as llm:
    llm.run("Hello, how are you?")

# Multimodal Inference (Vision-Language)
# 1. Set chat template (required for some models)
llm.set_chat_template(system_prompt="", prompt_prefix="", prompt_postfix="")

# 2. Run with image embeddings
llm.run_multimodal(
    prompt="<image> Describe this image",
    image_embed=embeddings_array, # numpy array
    image_width=308,
    image_height=308
)
```

## Building the Wheel

To build a wheel for distribution:

```bash
# Install build tools
pip install build

# Build the wheel
python -m build --wheel

# The wheel will be in dist/
# rkllm_bindings-0.1.0-py3-none-any.whl
```

The wheel is platform-independent (pure Python) and will work on any Linux aarch64 system with Python 3.8+.

## API Reference

See the [full documentation](src/rkllm_bindings/README.md) for detailed API reference.

## License

The Python bindings are licensed under the MIT License.

The bundled `librkllmrt.so` library is Copyright (c) Rockchip Electronics Co., Ltd. and is redistributed under their BSD-3-Clause style license. See the LICENSE file for full details.

**Note**: This package is not endorsed by or affiliated with Rockchip Electronics Co., Ltd. The library is redistributed in accordance with their license terms.
