# SPDX-License-Identifier: Apache-2.0
"""Swift Metal Worker — vLLM WorkerBase that delegates to Swift engine.

No Python-side model execution, no MLX graph building, no per-layer
loops. The Swift engine runs the entire forward pass at native Metal
speed (100+ tok/s on 4B models).
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING, Any

import psutil
import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.outputs import ModelRunnerOutput

from vllm_swift.engine_bridge import SwiftInferenceEngine

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


def _resolve_model_path(model_name: str) -> str:
    """Resolve model name to local directory with MLX-format weights.

    Search order:
      1. Direct path (if it exists as a directory)
      2. ~/models/{name} or ~/models/{short_name}
      3. HuggingFace cache (~/.cache/huggingface/hub)
      4. Auto-download from HuggingFace Hub
    """
    if os.path.isdir(model_name):
        return model_name

    # ~/models/ with full or short name
    short_name = model_name.split("/")[-1]
    for candidate in [
        os.path.expanduser(f"~/models/{model_name}"),
        os.path.expanduser(f"~/models/{short_name}"),
    ]:
        if os.path.isdir(candidate):
            return candidate

    # HF cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(hf_cache, f"models--{model_name.replace('/', '--')}")
    if os.path.isdir(model_dir):
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots):
            snaps = sorted(os.listdir(snapshots))
            if snaps:
                return os.path.join(snapshots, snaps[-1])

    # Auto-download from HuggingFace Hub
    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading model %s from HuggingFace Hub...", model_name)
        local_dir = os.path.expanduser(f"~/models/{short_name}")
        path = snapshot_download(model_name, local_dir=local_dir)
        logger.info("Downloaded to %s", path)
        return path
    except Exception as e:
        logger.warning("Failed to download %s: %s", model_name, e)

    return model_name


class SwiftMetalWorker:
    """Worker that drives inference via the Swift mlx-swift-lm engine.

    Implements the vLLM WorkerBase protocol. The Swift engine handles
    model loading, KV cache, attention, and token generation — this
    worker just translates between vLLM's scheduler protocol and the
    Swift C bridge.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str = "",
        is_driver_worker: bool = True,
        **kwargs: Any,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        self.engine: SwiftInferenceEngine | None = None
        self.device = torch.device("cpu")

        # Request state
        self._active_requests: dict[str, list[int]] = {}
        self._request_params: dict[str, dict] = {}
        self._batched_initialized = False

        # Extract TurboQuant config from additional_config
        add = getattr(vllm_config, "additional_config", None) or {}
        self._kv_scheme = add.get("kv_scheme")
        self._kv_bits = int(add.get("kv_bits", 0))

    def init_device(self) -> None:
        """Initialize device and distributed environment."""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )
        set_random_seed(self.model_config.seed)

    def load_model(self) -> None:
        """Load model via Swift engine."""
        model_path = _resolve_model_path(self.model_config.model)
        logger.info("Loading model via Swift engine: %s", model_path)

        self.engine = SwiftInferenceEngine(
            model_path=model_path,
            dtype="float16",
            max_kv_size=self.model_config.max_model_len,
            kv_scheme=self._kv_scheme,
            kv_bits=self._kv_bits,
            memory_fraction=self.cache_config.gpu_memory_utilization,
        )
        logger.info(
            "Swift engine loaded: layers=%d, head_dim=%d, memory=%.1fGB",
            self.engine.num_layers,
            self.engine.head_dim,
            self.engine.model_memory_bytes / 1e9,
        )

    def determine_available_memory(self) -> int:
        """Report available memory for KV cache."""
        total = psutil.virtual_memory().total
        fraction = self.cache_config.gpu_memory_utilization
        return int(total * fraction)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Return KV cache spec for the scheduler.

        Swift engine manages the real KV cache. We report a uniform spec
        so vLLM's scheduler can track capacity and block allocation.
        """
        num_layers = self.engine.num_layers if self.engine else 28
        head_dim = self.engine.head_dim if self.engine else 128
        block_size = self.cache_config.block_size
        specs = {}
        for i in range(num_layers):
            layer_name = f"layers.{i}.self_attn"
            specs[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=head_dim,
                dtype=torch.float16,
            )
        return specs

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize KV cache (Swift engine manages this internally)."""
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize from KV cache config (no-op, Swift manages cache)."""
        pass

    def compile_or_warm_up_model(self) -> None:
        """Warm up the model."""
        if self.engine is None:
            return
        # Run a tiny prefill to warm up Metal kernels
        warmup_tokens = [1, 2, 3]
        self.engine.prefill(warmup_tokens, temperature=0.0)
        self.engine.reset()
        logger.info("Swift engine warmed up")

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        """Execute model for scheduled requests.

        Translates vLLM's SchedulerOutput into Swift engine calls:
        - New requests → prefill
        - Cached requests → decode_step
        """
        if self.engine is None:
            raise RuntimeError("Swift engine not loaded")

        sampled_token_ids: list[list[int]] = []
        req_ids: list[str] = []

        # Handle new requests (prefill) — each gets its own session
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            prompt_tokens = list(new_req.prompt_token_ids)
            sp = new_req.sampling_params
            temp = getattr(sp, "temperature", 0.0)
            top_p = getattr(sp, "top_p", 1.0)

            # Check for multimodal (VLM) inputs
            # vLLM preprocesses images Python-side — mm_features has ready pixels
            mm_features = getattr(new_req, "mm_features", None)
            if mm_features and hasattr(mm_features, "pixel_values"):
                pv = mm_features.pixel_values
                if hasattr(pv, "numpy"):
                    pv = pv.numpy()
                pixel_list = pv.flatten().tolist()
                pixel_shape = list(pv.shape)
                # Extract image_grid_thw if available
                grid_thw = None
                if hasattr(mm_features, "image_grid_thw"):
                    g = mm_features.image_grid_thw
                    if hasattr(g, "numpy"):
                        g = g.numpy()
                    grid_thw = g.flatten().tolist()[:3]
                first_token = self.engine.prefill_vlm(
                    req_id,
                    prompt_tokens,
                    pixels=pixel_list,
                    pixel_shape=pixel_shape,
                    grid_thw=grid_thw,
                    temperature=temp,
                    top_p=top_p,
                )
            else:
                first_token = self.engine.prefill_req(
                    req_id, prompt_tokens, temperature=temp, top_p=top_p
                )

            self._active_requests[req_id] = [first_token]
            self._request_params[req_id] = {
                "temperature": temp,
                "top_p": top_p,
                "logprobs": getattr(sp, "logprobs", None) is not None,
            }
            req_ids.append(req_id)
            sampled_token_ids.append([first_token])

        # Batch decode all active sessions
        cached = scheduler_output.scheduled_cached_reqs
        cached_req_ids = list(cached.req_ids)
        if cached_req_ids:
            if not self._batched_initialized:
                # First time: full init from all per-request caches
                self.engine.init_batched()
                self._batched_initialized = True
            else:
                # Incremental updates: remove finished, add new
                for rid in scheduler_output.finished_req_ids:
                    self.engine.remove_batch_slot(rid)
                for new_req in scheduler_output.scheduled_new_reqs:
                    self.engine.add_batch_slot(new_req.req_id)

            wants_logprobs = any(
                self._request_params.get(rid, {}).get("logprobs", False) for rid in cached_req_ids
            )
            if wants_logprobs:
                lp_results = self.engine.decode_all_logprobs(max_reqs=len(cached_req_ids))
                result_map = {rid: tok for rid, tok, _ in lp_results}
            else:
                batch_results = self.engine.decode_all(max_reqs=len(cached_req_ids))
                result_map = {rid: tok for rid, tok in batch_results}
            for req_id in cached_req_ids:
                token = result_map.get(req_id, -1)
                if token >= 0:
                    self._active_requests.setdefault(req_id, []).append(token)
                    sampled_token_ids.append([token])
                else:
                    sampled_token_ids.append([])
                req_ids.append(req_id)

        # Clean up finished requests (free Swift KV cache)
        for req_id in scheduler_output.finished_req_ids:
            self._active_requests.pop(req_id, None)
            self._request_params.pop(req_id, None)
            self.engine.finish_req(req_id)

        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
        )

        # If tokens were generated, use the two-phase pattern
        # (execute_model returns None, sample_tokens returns output).
        # If this is a cleanup-only step (no tokens), return directly
        # so the batch queue doesn't call sample_tokens.
        if req_ids:
            self._pending_output = output
            return None
        return output

    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput | None:
        """Return the output computed by execute_model."""
        output = getattr(self, "_pending_output", None)
        self._pending_output = None
        return output

    def get_model(self) -> Any:
        return self.engine

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len

    def get_cache_block_size_bytes(self) -> int:
        head_dim = self.engine.head_dim if self.engine else 128
        # Approximate: 2 (K+V) * head_dim * 2 (fp16) * block_size
        return 2 * head_dim * 2 * self.cache_config.block_size

    def add_lora(self, lora_request: LoRARequest) -> bool:
        logger.warning("LoRA not supported on Swift Metal")
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep not supported on Swift Metal")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Sleep not supported on Swift Metal")

    def reset_mm_cache(self) -> None:
        pass

    def reset_prefix_cache(self) -> bool:
        return True

    def reset_encoder_cache(self) -> None:
        pass

    def check_health(self) -> None:
        if self.engine is None:
            raise RuntimeError("Swift engine not initialized")

    def shutdown(self) -> None:
        if self.engine is not None:
            del self.engine
            self.engine = None
        self._active_requests.clear()
        self._request_params.clear()
        gc.collect()
        logger.info("Swift Metal worker shutdown complete")
