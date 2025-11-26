# RKLLM Python Bindings

Python bindings for the RKLLM Runtime library (`librkllmrt.so`) using `ctypes`.

## Installation

The library `librkllmrt.so` is **bundled with the package**. No separate installation is required.

## Usage

```python
from rkllm_bindings import RKLLM

# ... callback definition ...

llm = RKLLM(model_path="smolvlm.rkllm", callback=callback)

# Multimodal Inference
llm.set_chat_template(system_prompt="", prompt_prefix="", prompt_postfix="")
llm.run_multimodal(
    prompt="<image> Describe this image",
    image_embed=embeddings,
    image_width=308,
    image_height=308
)

llm.destroy()
```

## API Reference

### RKLLM Class

**Methods:**
- `run(prompt, mode=RKLLM_INFER_GENERATE, keep_history=1)`: Run text-only inference
- `run_multimodal(prompt, image_embed, image_width, image_height, ...)`: Run vision-language inference
- `set_chat_template(system_prompt, prompt_prefix, prompt_postfix)`: Configure chat template
- `abort()`: Abort ongoing inference
- `is_running()`: Check if inference is running
- `destroy()`: Clean up resources

## Notes

- The bindings automatically load `librkllmrt.so` from system paths
- Make sure the library is in your `LD_LIBRARY_PATH` if not in standard locations
- The callback function receives `(result_ptr, userdata, state)` and should return 0 to continue or 1 to pause
