# SPDX-License-Identifier: Apache-2.0
"""C FFI bridge to the Swift mlx-swift-lm inference engine."""

import ctypes
import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_LIB_PATH = os.environ.get(
    "VLLM_SWIFT_METAL_LIB",
    str(Path(__file__).parent.parent / "swift" / "libvllm_swift_metal.dylib"),
)

_lib = None


class PerfStats(ctypes.Structure):
    _fields_ = [
        ("prefill_tokens_per_sec", ctypes.c_double),
        ("decode_tokens_per_sec", ctypes.c_double),
        ("peak_memory_bytes", ctypes.c_int64),
        ("total_tokens_generated", ctypes.c_int32),
        ("total_decode_time_sec", ctypes.c_double),
    ]


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib

    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(
            f"Swift Metal engine not found at {_LIB_PATH}. "
            "Build first: cd swift && swift build -c release"
        )

    _lib = ctypes.CDLL(_LIB_PATH)

    # Engine lifecycle
    _lib.vsm_engine_create.restype = ctypes.c_void_p
    _lib.vsm_engine_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    _lib.vsm_engine_destroy.restype = None
    _lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

    # Model info
    for fn in ["vsm_engine_vocab_size", "vsm_engine_num_layers", "vsm_engine_head_dim"]:
        getattr(_lib, fn).restype = ctypes.c_int32
        getattr(_lib, fn).argtypes = [ctypes.c_void_p]
    _lib.vsm_engine_model_memory_bytes.restype = ctypes.c_int64
    _lib.vsm_engine_model_memory_bytes.argtypes = [ctypes.c_void_p]

    # Inference
    _lib.vsm_engine_prefill.restype = ctypes.c_int32
    _lib.vsm_engine_prefill.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _lib.vsm_engine_decode_step.restype = ctypes.c_int32
    _lib.vsm_engine_decode_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _lib.vsm_engine_decode_batch.restype = ctypes.c_int32
    _lib.vsm_engine_decode_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]

    # Logits access
    _lib.vsm_engine_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    _lib.vsm_engine_get_logits.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ]

    # Multi-request API
    _lib.vsm_engine_prefill_req.restype = ctypes.c_int32
    _lib.vsm_engine_prefill_req.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _lib.vsm_engine_decode_step_req.restype = ctypes.c_int32
    _lib.vsm_engine_decode_step_req.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    _lib.vsm_engine_finish_req.restype = None
    _lib.vsm_engine_finish_req.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _lib.vsm_engine_active_requests.restype = ctypes.c_int32
    _lib.vsm_engine_active_requests.argtypes = [ctypes.c_void_p]

    # State management
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
    """Python wrapper around the Swift inference engine via C FFI."""

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

    def prefill(
        self, prompt_tokens: list[int], temperature: float = 0.0, top_p: float = 1.0
    ) -> int:
        arr = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)
        return self._lib.vsm_engine_prefill(
            self._handle, arr, len(prompt_tokens), temperature, top_p
        )

    def decode_step(self, temperature: float = 0.0, top_p: float = 1.0) -> int:
        return self._lib.vsm_engine_decode_step(self._handle, temperature, top_p)

    def decode_batch(
        self, max_tokens: int, temperature: float = 0.0, top_p: float = 1.0
    ) -> list[int]:
        buf = (ctypes.c_int32 * max_tokens)()
        n = self._lib.vsm_engine_decode_batch(
            self._handle, max_tokens, temperature, top_p, buf, max_tokens
        )
        return list(buf[:n])

    def prefill_req(
        self, req_id: str, prompt_tokens: list[int], temperature: float = 0.0, top_p: float = 1.0
    ) -> int:
        arr = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)
        return self._lib.vsm_engine_prefill_req(
            self._handle, req_id.encode(), arr, len(prompt_tokens), temperature, top_p
        )

    def decode_step_req(self, req_id: str) -> int:
        return self._lib.vsm_engine_decode_step_req(self._handle, req_id.encode())

    def finish_req(self, req_id: str) -> None:
        self._lib.vsm_engine_finish_req(self._handle, req_id.encode())

    def active_requests(self) -> int:
        return self._lib.vsm_engine_active_requests(self._handle)

    def reset(self) -> None:
        self._lib.vsm_engine_reset(self._handle)

    def get_stats(self) -> PerfStats:
        stats = PerfStats()
        self._lib.vsm_engine_get_stats(self._handle, ctypes.byref(stats))
        return stats

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.vsm_engine_destroy(self._handle)
            self._handle = None
