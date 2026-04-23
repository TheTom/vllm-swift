# SPDX-License-Identifier: Apache-2.0
"""C FFI bridge to the Swift mlx-swift-lm inference engine.

Loads the compiled Swift dylib and wraps the C API defined in
swift/bridge.h. All GPU compute happens in Swift/Metal — Python
only drives the vLLM scheduling protocol.
"""

import ctypes
import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

# Path to compiled Swift engine dylib
_LIB_PATH = os.environ.get(
    "VLLM_SWIFT_METAL_LIB",
    str(Path(__file__).parent.parent / "swift" / "libvllm_swift_metal.dylib"),
)

_lib = None


class PerfStats(ctypes.Structure):
    """Mirrors vsm_perf_stats_t from bridge.h."""

    _fields_ = [
        ("prefill_tokens_per_sec", ctypes.c_double),
        ("decode_tokens_per_sec", ctypes.c_double),
        ("peak_memory_bytes", ctypes.c_int64),
        ("total_tokens_generated", ctypes.c_int32),
        ("total_decode_time_sec", ctypes.c_double),
    ]


def _get_lib():
    """Load the Swift engine dylib (cached)."""
    global _lib
    if _lib is not None:
        return _lib

    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(
            f"Swift Metal engine not found at {_LIB_PATH}. "
            "Build the Swift engine first: cd swift && swift build -c release"
        )

    _lib = ctypes.CDLL(_LIB_PATH)

    # Bind function signatures
    _lib.vsm_engine_create.restype = ctypes.c_void_p
    _lib.vsm_engine_create.argtypes = [
        ctypes.c_char_p,  # model_path
        ctypes.c_char_p,  # dtype
        ctypes.c_int32,   # max_kv_size
        ctypes.c_char_p,  # kv_scheme
        ctypes.c_int32,   # kv_bits
        ctypes.c_float,   # memory_fraction
    ]

    _lib.vsm_engine_destroy.restype = None
    _lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

    _lib.vsm_engine_vocab_size.restype = ctypes.c_int32
    _lib.vsm_engine_vocab_size.argtypes = [ctypes.c_void_p]

    _lib.vsm_engine_num_layers.restype = ctypes.c_int32
    _lib.vsm_engine_num_layers.argtypes = [ctypes.c_void_p]

    _lib.vsm_engine_head_dim.restype = ctypes.c_int32
    _lib.vsm_engine_head_dim.argtypes = [ctypes.c_void_p]

    _lib.vsm_engine_model_memory_bytes.restype = ctypes.c_int64
    _lib.vsm_engine_model_memory_bytes.argtypes = [ctypes.c_void_p]

    _lib.vsm_engine_prefill.restype = ctypes.c_int32
    _lib.vsm_engine_prefill.argtypes = [
        ctypes.c_void_p,                          # engine
        ctypes.POINTER(ctypes.c_int32),           # prompt_tokens
        ctypes.c_int32,                           # num_tokens
        ctypes.c_float,                           # temperature
        ctypes.c_float,                           # top_p
    ]

    _lib.vsm_engine_decode_step.restype = ctypes.c_int32
    _lib.vsm_engine_decode_step.argtypes = [
        ctypes.c_void_p,  # engine
        ctypes.c_float,   # temperature
        ctypes.c_float,   # top_p
    ]

    _lib.vsm_engine_decode_batch.restype = ctypes.c_int32
    _lib.vsm_engine_decode_batch.argtypes = [
        ctypes.c_void_p,                          # engine
        ctypes.c_int32,                           # max_tokens
        ctypes.c_float,                           # temperature
        ctypes.c_float,                           # top_p
        ctypes.POINTER(ctypes.c_int32),           # output_tokens
        ctypes.c_int32,                           # output_capacity
    ]

    _lib.vsm_engine_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    _lib.vsm_engine_get_logits.argtypes = [
        ctypes.c_void_p,                          # engine
        ctypes.POINTER(ctypes.c_int32),           # out_vocab_size
    ]

    _lib.vsm_engine_reset.restype = None
    _lib.vsm_engine_reset.argtypes = [ctypes.c_void_p]

    _lib.vsm_engine_get_stats.restype = None
    _lib.vsm_engine_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PerfStats),
    ]

    logger.info("Loaded Swift Metal engine from %s", _LIB_PATH)
    return _lib


class SwiftInferenceEngine:
    """Python wrapper around the Swift inference engine via C FFI.

    All heavy compute runs in Swift/Metal. This class handles lifecycle
    and provides a Pythonic API for the vLLM worker to call.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "float16",
        max_kv_size: int = 0,
        kv_scheme: str | None = None,
        kv_bits: int = 0,
        memory_fraction: float = 0.9,
    ):
        lib = _get_lib()
        self._lib = lib
        self._handle = lib.vsm_engine_create(
            model_path.encode(),
            dtype.encode(),
            max_kv_size,
            kv_scheme.encode() if kv_scheme else None,
            kv_bits,
            memory_fraction,
        )
        if not self._handle:
            raise RuntimeError(f"Failed to create Swift engine for {model_path}")

        self.vocab_size = lib.vsm_engine_vocab_size(self._handle)
        self.num_layers = lib.vsm_engine_num_layers(self._handle)
        self.head_dim = lib.vsm_engine_head_dim(self._handle)
        self.model_memory_bytes = lib.vsm_engine_model_memory_bytes(self._handle)

        logger.info(
            "Swift engine: vocab=%d, layers=%d, head_dim=%d, model=%.1fGB",
            self.vocab_size, self.num_layers, self.head_dim,
            self.model_memory_bytes / 1e9,
        )

    def prefill(
        self,
        prompt_tokens: list[int],
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """Process prompt and return first generated token."""
        arr = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)
        return self._lib.vsm_engine_prefill(
            self._handle, arr, len(prompt_tokens), temperature, top_p
        )

    def decode_step(
        self, temperature: float = 0.0, top_p: float = 1.0
    ) -> int:
        """Generate next token from current KV cache state."""
        return self._lib.vsm_engine_decode_step(
            self._handle, temperature, top_p
        )

    def decode_batch(
        self,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[int]:
        """Generate up to max_tokens, return list of token IDs."""
        buf = (ctypes.c_int32 * max_tokens)()
        n = self._lib.vsm_engine_decode_batch(
            self._handle, max_tokens, temperature, top_p, buf, max_tokens
        )
        return list(buf[:n])

    def get_logits(self) -> tuple[ctypes.Array, int]:
        """Get logits from last forward pass. Returns (pointer, vocab_size)."""
        vocab_size = ctypes.c_int32()
        ptr = self._lib.vsm_engine_get_logits(
            self._handle, ctypes.byref(vocab_size)
        )
        return ptr, vocab_size.value

    def reset(self) -> None:
        """Clear KV cache for new conversation."""
        self._lib.vsm_engine_reset(self._handle)

    def get_stats(self) -> PerfStats:
        """Get performance statistics."""
        stats = PerfStats()
        self._lib.vsm_engine_get_stats(self._handle, ctypes.byref(stats))
        return stats

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.vsm_engine_destroy(self._handle)
            self._handle = None
