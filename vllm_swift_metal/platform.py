# SPDX-License-Identifier: Apache-2.0
"""Swift Metal Platform implementation for vLLM.

Registers as an out-of-tree platform plugin. All model execution
is delegated to the Swift mlx-swift-lm engine via C FFI — no Python
in the GPU hot path.
"""

import logging
import platform as py_platform
from typing import TYPE_CHECKING

import psutil
import torch
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class SwiftMetalPlatform(Platform):
    """Platform for Apple Silicon using Swift/MLX inference engine."""

    _enum: PlatformEnum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Apple Silicon (Swift Metal)"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().available

    @classmethod
    def is_available(cls) -> bool:
        if py_platform.machine() != "arm64":
            return False
        if py_platform.system() != "Darwin":
            return False
        try:
            import mlx.core as mx

            return bool(mx.metal.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    def set_device(cls, device_id: int) -> None:
        pass

    @classmethod
    def current_device(cls) -> int:
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        try:
            import mlx.core as mx

            mx.synchronize()
        except (ImportError, AttributeError):
            pass

    @classmethod
    def get_torch_device(cls, device_id: int = 0) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config

        # Use our Swift worker
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_swift_metal.worker.SwiftMetalWorker"

        # Single-process — no IPC overhead
        if parallel_config.distributed_executor_backend in ("auto", None):
            parallel_config.distributed_executor_backend = "uni"

        parallel_config.disable_custom_all_reduce = True

        # Disable chunked prefill — Swift engine handles full sequences
        if getattr(scheduler_config, "enable_chunked_prefill", False):
            scheduler_config.enable_chunked_prefill = False

        # Ensure scheduler can handle full prompt in one step
        if model_config is not None:
            model_max = model_config.max_model_len
            if scheduler_config.max_num_batched_tokens < model_max:
                scheduler_config.max_num_batched_tokens = model_max

        logger.info("Swift Metal platform configured (uni-proc, Swift engine)")


class SwiftMetalPlatformPlugin:
    """Plugin entry point for vLLM platform system."""

    @staticmethod
    def register() -> str | None:
        if SwiftMetalPlatform.is_available():
            logger.info("Swift Metal platform plugin activated")
            return "vllm_swift_metal.platform:SwiftMetalPlatform"
        return None
