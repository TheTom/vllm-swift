# SPDX-License-Identifier: Apache-2.0
"""Swift Metal worker — thin wrapper that delegates to Swift engine via C FFI.

No Python-side model execution, no MLX graph building, no per-layer Python
loops. All compute happens in Swift/Metal at native speed (167+ tok/s decode).
"""

from vllm.logger import init_logger

from vllm_swift_metal.engine_bridge import SwiftInferenceEngine

logger = init_logger(__name__)


class SwiftMetalWorker:
    """Worker that drives the Swift mlx-swift-lm engine.

    The vLLM scheduler calls execute_model() which translates the
    SchedulerOutput into Swift engine calls (prefill/decode_step).
    No tokenization here — vLLM handles that in the main process.
    """

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        max_model_len: int = 0,
        kv_scheme: str | None = None,
        kv_bits: int = 0,
        memory_fraction: float = 0.9,
    ):
        self.engine = SwiftInferenceEngine(
            model_path=model,
            dtype=dtype,
            max_kv_size=max_model_len,
            kv_scheme=kv_scheme,
            kv_bits=kv_bits,
            memory_fraction=memory_fraction,
        )
        # Per-request KV cache state tracked by request ID
        # TODO: multi-request support with paged KV cache
        self._active_request: str | None = None

    def prefill(
        self,
        request_id: str,
        prompt_tokens: list[int],
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """Prefill prompt and return first token."""
        if self._active_request and self._active_request != request_id:
            self.engine.reset()
        self._active_request = request_id
        return self.engine.prefill(prompt_tokens, temperature, top_p)

    def decode_step(
        self,
        request_id: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """Generate next token for active request."""
        return self.engine.decode_step(temperature, top_p)

    def finish_request(self, request_id: str) -> None:
        """Clean up after request completion."""
        if self._active_request == request_id:
            self.engine.reset()
            self._active_request = None

    @property
    def vocab_size(self) -> int:
        return self.engine.vocab_size

    @property
    def model_memory_bytes(self) -> int:
        return self.engine.model_memory_bytes
