# SPDX-License-Identifier: Apache-2.0
"""Test fixtures and vLLM mock setup."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# Stub vllm modules if not installed (allows unit testing without full vLLM)
_VLLM_STUBS = [
    "vllm",
    "vllm.logger",
    "vllm.config",
    "vllm.envs",
    "vllm.platforms",
    "vllm.platforms.interface",
    "vllm.distributed",
    "vllm.lora",
    "vllm.lora.request",
    "vllm.tasks",
    "vllm.utils",
    "vllm.utils.torch_utils",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.output",
    "vllm.v1.outputs",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.worker",
    "vllm.v1.worker.worker_base",
    "vllm.v1.attention",
    "vllm.v1.attention.backends",
    "vllm.v1.attention.backends.registry",
    "vllm.v1.attention.selector",
]

for mod_name in _VLLM_STUBS:
    if mod_name not in sys.modules:
        stub = ModuleType(mod_name)
        # Provide commonly used names
        if mod_name == "vllm.logger":
            stub.init_logger = lambda name: MagicMock()
        elif mod_name == "vllm.config":
            stub.VllmConfig = MagicMock
        elif mod_name == "vllm.platforms.interface":
            stub.Platform = type("Platform", (), {})
            stub.PlatformEnum = MagicMock()
            stub.PlatformEnum.OOT = "OOT"
            stub.DeviceCapability = type(
                "DeviceCapability",
                (),
                {
                    "__init__": lambda self, major=0, minor=0: (
                        setattr(self, "major", major) or setattr(self, "minor", minor)
                    )
                },
            )
        elif mod_name == "vllm.tasks":
            stub.SupportedTask = MagicMock()
            stub.SupportedTask.generate = "generate"
        elif mod_name == "vllm.v1.outputs":
            from dataclasses import dataclass, field

            @dataclass
            class _ModelRunnerOutput:
                req_ids: list = field(default_factory=list)
                req_id_to_index: dict = field(default_factory=dict)
                sampled_token_ids: list = field(default_factory=list)
                spec_token_ids: list = field(default_factory=list)
                logprob_token_ids: object = None
                logprobs: object = None
                prompt_logprob_token_ids: object = None
                prompt_logprobs: object = None
                num_nans_in_logits: int = 0

            stub.ModelRunnerOutput = _ModelRunnerOutput
        elif mod_name == "vllm.v1.kv_cache_interface":
            stub.FullAttentionSpec = MagicMock()
            stub.KVCacheConfig = MagicMock()
            stub.KVCacheSpec = MagicMock()
        elif mod_name == "vllm.v1.core.sched.output":
            stub.GrammarOutput = MagicMock()
            stub.SchedulerOutput = MagicMock()
        elif mod_name == "vllm.distributed":
            stub.init_distributed_environment = MagicMock()
            stub.ensure_model_parallel_initialized = MagicMock()
        elif mod_name == "vllm.utils.torch_utils":
            stub.set_random_seed = MagicMock()
        elif mod_name == "vllm.lora.request":
            stub.LoRARequest = MagicMock()
        sys.modules[mod_name] = stub
