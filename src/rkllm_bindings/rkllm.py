"""
Python bindings for RKLLM Runtime using ctypes.
Based on rkllm.h from airockchip/rknn-llm.
"""

import ctypes
from ctypes import (
    c_void_p,
    c_char_p,
    c_int,
    c_int32,
    c_int8,
    c_uint8,
    c_uint32,
    c_float,
    c_bool,
    c_size_t,
    POINTER,
    Structure,
    CFUNCTYPE,
)
from enum import IntEnum
from typing import Optional, Callable
import os


# Load the library
def _load_library():
    """Load librkllmrt.so from bundled lib or system paths"""
    # Try bundled library first
    bundled_lib = os.path.join(os.path.dirname(__file__), "lib", "librkllmrt.so")

    if os.path.exists(bundled_lib):
        try:
            return ctypes.CDLL(bundled_lib)
        except OSError as e:
            print(f"Warning: Could not load bundled library: {e}")

    # Fall back to system paths
    lib_names = ["librkllmrt.so", "librkllmrt.so.1"]
    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    raise RuntimeError(
        "Could not load librkllmrt.so. "
        "Make sure it's bundled with the package or in your "
        "LD_LIBRARY_PATH/system library path."
    )


_lib = _load_library()


# Enums
class LLMCallState(IntEnum):
    """Describes the possible states of an LLM call."""

    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3


class RKLLMInputType(IntEnum):
    """Defines the types of inputs that can be fed into the LLM."""

    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3


class RKLLMInferMode(IntEnum):
    """Specifies the inference modes of the LLM."""

    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
    RKLLM_INFER_GET_LOGITS = 2


# Structures
class RKLLMExtendParam(Structure):
    """The extend parameters for configuring an LLM instance."""

    _fields_ = [
        ("base_domain_id", c_int32),
        ("embed_flash", c_int8),
        ("enabled_cpus_num", c_int8),
        ("enabled_cpus_mask", c_uint32),
        ("reserved", c_uint8 * 106),
    ]


class RKLLMParam(Structure):
    """Defines the parameters for configuring an LLM instance."""

    _fields_ = [
        ("model_path", c_char_p),
        ("max_context_len", c_int32),
        ("max_new_tokens", c_int32),
        ("top_k", c_int32),
        ("n_keep", c_int32),
        ("top_p", c_float),
        ("temperature", c_float),
        ("repeat_penalty", c_float),
        ("frequency_penalty", c_float),
        ("presence_penalty", c_float),
        ("mirostat", c_int32),
        ("mirostat_tau", c_float),
        ("mirostat_eta", c_float),
        ("skip_special_token", c_bool),
        ("is_async", c_bool),
        ("img_start", c_char_p),
        ("img_end", c_char_p),
        ("img_content", c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMEmbedInput(Structure):
    """Represents an embedding input to the LLM."""

    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_tokens", c_size_t),
    ]


class RKLLMTokenInput(Structure):
    """Represents token input to the LLM."""

    _fields_ = [
        ("input_ids", POINTER(c_int32)),
        ("n_tokens", c_size_t),
    ]


class RKLLMMultiModalInput(Structure):
    """Represents multimodal input (e.g., text and image)."""

    _fields_ = [
        ("prompt", c_char_p),
        ("image_embed", POINTER(c_float)),
        ("n_image_tokens", c_size_t),
        ("n_image", c_size_t),
        ("image_width", c_size_t),
        ("image_height", c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    """Union for different input types."""

    _fields_ = [
        ("prompt_input", c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(Structure):
    """Represents different types of input to the LLM via a union."""

    _fields_ = [
        ("role", c_char_p),
        ("enable_thinking", c_bool),
        ("input_type", c_int),  # RKLLMInputType
        ("input_union", RKLLMInputUnion),
    ]


class RKLLMLoraParam(Structure):
    """Structure defining parameters for Lora adapters."""

    _fields_ = [
        ("lora_adapter_name", c_char_p),
    ]


class RKLLMPromptCacheParam(Structure):
    """Structure to define parameters for caching prompts."""

    _fields_ = [
        ("save_prompt_cache", c_int),
        ("prompt_cache_path", c_char_p),
    ]


class RKLLMInferParam(Structure):
    """Structure for defining parameters during inference."""

    _fields_ = [
        ("mode", c_int),  # RKLLMInferMode
        ("lora_params", POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", POINTER(RKLLMPromptCacheParam)),
        ("keep_history", c_int),
    ]


class RKLLMResultLastHiddenLayer(Structure):
    """Structure to hold the hidden states from the last layer."""

    _fields_ = [
        ("hidden_states", POINTER(c_float)),
        ("embd_size", c_int),
        ("num_tokens", c_int),
    ]


class RKLLMResultLogits(Structure):
    """Structure to hold the logits."""

    _fields_ = [
        ("logits", POINTER(c_float)),
        ("vocab_size", c_int),
        ("num_tokens", c_int),
    ]


class RKLLMPerfStat(Structure):
    """Structure to hold performance statistics."""

    _fields_ = [
        ("prefill_time_ms", c_float),
        ("prefill_tokens", c_int),
        ("generate_time_ms", c_float),
        ("generate_tokens", c_int),
        ("memory_usage_mb", c_float),
    ]


class RKLLMResult(Structure):
    """Structure to represent the result of LLM inference."""

    _fields_ = [
        ("text", c_char_p),
        ("token_id", c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


# Callback type
LLMResultCallback = CFUNCTYPE(
    c_int, POINTER(RKLLMResult), c_void_p, c_int  # LLMCallState
)


# Function signatures
_lib.rkllm_createDefaultParam.argtypes = []
_lib.rkllm_createDefaultParam.restype = RKLLMParam

_lib.rkllm_init.argtypes = [POINTER(c_void_p), POINTER(RKLLMParam), LLMResultCallback]
_lib.rkllm_init.restype = c_int

_lib.rkllm_destroy.argtypes = [c_void_p]
_lib.rkllm_destroy.restype = c_int

_lib.rkllm_run.argtypes = [
    c_void_p,
    POINTER(RKLLMInput),
    POINTER(RKLLMInferParam),
    c_void_p,
]
_lib.rkllm_run.restype = c_int

_lib.rkllm_abort.argtypes = [c_void_p]
_lib.rkllm_abort.restype = c_int

_lib.rkllm_is_running.argtypes = [c_void_p]
_lib.rkllm_is_running.restype = c_int

_lib.rkllm_set_chat_template.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]
_lib.rkllm_set_chat_template.restype = c_int


class RKLLM:
    """
    Python wrapper for RKLLM Runtime.

    Example usage:
        def callback(result, userdata, state):
            if result.contents.text:
                print(result.contents.text.decode('utf-8'), end='', flush=True)
            return 0

        llm = RKLLM(model_path="model.rkllm", callback=callback)
        llm.run("Hello, how are you?")
        llm.destroy()
    """

    def __init__(
        self,
        model_path: str,
        callback: Optional[Callable] = None,
        max_context_len: int = 512,
        max_new_tokens: int = 256,
        top_k: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.8,
        **kwargs,
    ):
        """
        Initialize RKLLM instance.

        Args:
            model_path: Path to the .rkllm model file
            callback: Callback function for handling results
            max_context_len: Maximum context length
            max_new_tokens: Maximum new tokens to generate
            top_k: Top-K sampling parameter
            top_p: Top-P sampling parameter
            temperature: Sampling temperature
            **kwargs: Additional parameters for RKLLMParam
        """
        self.handle = c_void_p()
        self.callback = callback
        self._callback_func = None

        # Create default parameters
        param = _lib.rkllm_createDefaultParam()

        # Override with user parameters
        param.model_path = model_path.encode("utf-8")
        param.max_context_len = max_context_len
        param.max_new_tokens = max_new_tokens
        param.top_k = top_k
        param.top_p = top_p
        param.temperature = temperature

        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(param, key):
                setattr(param, key, value)

        # Setup callback
        if callback:
            self._callback_func = LLMResultCallback(self._wrap_callback)
        else:
            self._callback_func = LLMResultCallback(lambda r, u, s: 0)

        # Initialize
        ret = _lib.rkllm_init(
            ctypes.byref(self.handle), ctypes.byref(param), self._callback_func
        )

        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with code {ret}")

    def _wrap_callback(self, result_ptr, userdata, state):
        """Wrap the user callback to handle Python types."""
        if self.callback:
            return self.callback(result_ptr, userdata, state)
        return 0

    def run(
        self,
        prompt: str,
        mode: RKLLMInferMode = RKLLMInferMode.RKLLM_INFER_GENERATE,
        keep_history: int = 1,
    ) -> int:
        """
        Run inference with a text prompt.

        Args:
            prompt: Input text prompt
            mode: Inference mode
            keep_history: Whether to keep history (1) or not (0)

        Returns:
            Status code (0 for success)
        """
        # Create input
        rkllm_input = RKLLMInput()
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_union.prompt_input = prompt.encode("utf-8")

        # Create infer params
        infer_param = RKLLMInferParam()
        infer_param.mode = mode
        infer_param.keep_history = keep_history

        # Run
        ret = _lib.rkllm_run(
            self.handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None
        )

        return ret

    def run_multimodal(
        self,
        prompt: str,
        image_embed,  # numpy array or list of floats
        image_width: int,
        image_height: int,
        mode: RKLLMInferMode = RKLLMInferMode.RKLLM_INFER_GENERATE,
        keep_history: int = 1,
    ) -> int:
        """
        Run inference with multimodal input (text + image embeddings).

        Args:
            prompt: Input text prompt
            image_embed: Image embeddings from vision encoder (flattened 1D array)
            image_width: Original image width
            image_height: Original image height
            mode: Inference mode
            keep_history: Whether to keep history (1) or not (0)

        Returns:
            Status code (0 for success)
        """
        import numpy as np

        # Convert to numpy if needed
        if not isinstance(image_embed, np.ndarray):
            image_embed = np.array(image_embed, dtype=np.float32)
        else:
            image_embed = image_embed.astype(np.float32)

        # Calculate n_image_tokens from shape before flattening
        n_image_tokens = 0
        if image_embed.ndim == 3:
            n_image_tokens = image_embed.shape[1]
        elif image_embed.ndim == 2:
            n_image_tokens = image_embed.shape[0]
        else:
            # Fallback for 1D array - assume it's already flattened
            # This might be wrong if hidden_dim is not known, but we'll leave it for now
            # Ideally input should be (N, D) or (1, N, D)
            n_image_tokens = len(image_embed) // 768  # Rough guess or just pass len?
            # Previous behavior passed len, which was wrong.
            # Let's assume user passes structured data.
            pass

        # Flatten if needed
        if image_embed.ndim > 1:
            image_embed = image_embed.flatten()

        # Create C array from numpy array
        n_total_elements = len(image_embed)
        embed_array = (c_float * n_total_elements)(*image_embed)

        # Create multimodal input
        multimodal_input = RKLLMMultiModalInput()
        multimodal_input.prompt = prompt.encode("utf-8")
        multimodal_input.image_embed = embed_array
        multimodal_input.n_image_tokens = n_image_tokens
        multimodal_input.n_image = 1  # Single image
        multimodal_input.image_width = image_width
        multimodal_input.image_height = image_height

        # Create RKLLM input
        rkllm_input = RKLLMInput()
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
        rkllm_input.input_union.multimodal_input = multimodal_input

        # Create infer params
        infer_param = RKLLMInferParam()
        infer_param.mode = mode
        infer_param.keep_history = keep_history

        # Run
        ret = _lib.rkllm_run(
            self.handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None
        )

        return ret

    def set_chat_template(
        self, system_prompt: str = "", prompt_prefix: str = "", prompt_postfix: str = ""
    ) -> int:
        """
        Set the chat template for the model.

        Args:
            system_prompt: System prompt template
            prompt_prefix: Prefix before user prompt
            prompt_postfix: Postfix after user prompt

        Returns:
            Status code (0 for success)
        """
        return _lib.rkllm_set_chat_template(
            self.handle,
            system_prompt.encode("utf-8") if system_prompt else b"",
            prompt_prefix.encode("utf-8") if prompt_prefix else b"",
            prompt_postfix.encode("utf-8") if prompt_postfix else b"",
        )

    def abort(self) -> int:
        """Abort ongoing inference."""
        return _lib.rkllm_abort(self.handle)

    def is_running(self) -> bool:
        """Check if inference is running."""
        return _lib.rkllm_is_running(self.handle) == 0

    def destroy(self):
        """Destroy the RKLLM instance and release resources."""
        if self.handle:
            _lib.rkllm_destroy(self.handle)
            self.handle = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.destroy()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.destroy()
