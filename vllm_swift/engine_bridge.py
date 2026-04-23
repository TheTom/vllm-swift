# SPDX-License-Identifier: Apache-2.0
"""C FFI bridge to the Swift mlx-swift-lm inference engine."""

import ctypes
import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def _find_lib_path() -> str:
    """Find the Swift bridge dylib, checking multiple locations."""
    if env_path := os.environ.get("VLLM_SWIFT_METAL_LIB"):
        return env_path

    base = Path(__file__).parent.parent / "swift"
    candidates = [
        base / ".build" / "arm64-apple-macosx" / "release" / "libVLLMBridge.dylib",
        base / ".build" / "arm64-apple-macosx" / "debug" / "libVLLMBridge.dylib",
        base / "libvllm_swift.dylib",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    return str(candidates[0])  # default to release path


_LIB_PATH = _find_lib_path()

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
    _lib.vsm_engine_decode_all.restype = ctypes.c_int32
    _lib.vsm_engine_decode_all.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]
    _lib.vsm_engine_decode_all_logprobs.restype = ctypes.c_int32
    _lib.vsm_engine_decode_all_logprobs.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]

    # VLM support
    _lib.vsm_engine_prefill_vlm.restype = ctypes.c_int32
    _lib.vsm_engine_prefill_vlm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
    ]

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

    def decode_all(self, max_reqs: int = 64) -> list[tuple[str, int]]:
        """Decode one step for ALL active sessions in a single call.

        Returns list of (req_id, token) pairs. Token=-1 means EOS.
        """
        req_ids_buf = (ctypes.c_char_p * max_reqs)()
        tokens_buf = (ctypes.c_int32 * max_reqs)()
        n = self._lib.vsm_engine_decode_all(self._handle, req_ids_buf, tokens_buf, max_reqs)
        results = []
        for i in range(n):
            rid = req_ids_buf[i].decode() if req_ids_buf[i] else ""
            results.append((rid, int(tokens_buf[i])))
        return results

    def prefill_vlm(
        self,
        req_id: str,
        prompt_tokens: list[int],
        pixels: list[float] | None = None,
        pixel_shape: list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """Prefill with tokens + preprocessed pixel tensor for VLM models."""
        arr = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)
        if pixels and pixel_shape:
            pix_arr = (ctypes.c_float * len(pixels))(*pixels)
            dims_arr = (ctypes.c_int32 * len(pixel_shape))(*pixel_shape)
            return self._lib.vsm_engine_prefill_vlm(
                self._handle,
                req_id.encode(),
                arr,
                len(prompt_tokens),
                pix_arr,
                len(pixels),
                dims_arr,
                len(pixel_shape),
                temperature,
                top_p,
            )
        return self._lib.vsm_engine_prefill_req(
            self._handle,
            req_id.encode(),
            arr,
            len(prompt_tokens),
            temperature,
            top_p,
        )

    def decode_all_logprobs(self, max_reqs: int = 64) -> list[tuple[str, int, float]]:
        """Decode with logprobs. Returns (req_id, token, logprob) tuples."""
        req_ids_buf = (ctypes.c_char_p * max_reqs)()
        tokens_buf = (ctypes.c_int32 * max_reqs)()
        logprobs_buf = (ctypes.c_float * max_reqs)()
        n = self._lib.vsm_engine_decode_all_logprobs(
            self._handle, req_ids_buf, tokens_buf, logprobs_buf, max_reqs
        )
        results = []
        for i in range(n):
            rid = req_ids_buf[i].decode() if req_ids_buf[i] else ""
            results.append((rid, int(tokens_buf[i]), float(logprobs_buf[i])))
        return results

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
