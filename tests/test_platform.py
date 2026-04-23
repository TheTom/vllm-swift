# SPDX-License-Identifier: Apache-2.0
"""Tests for Swift Metal platform detection and configuration."""

import platform
from unittest.mock import MagicMock, patch


class TestSwiftMetalPlatform:
    def test_is_available_on_apple_silicon(self):
        from vllm_swift.platform import SwiftMetalPlatform

        with (
            patch("platform.machine", return_value="arm64"),
            patch("platform.system", return_value="Darwin"),
        ):
            # Depends on actual MLX availability — skip if not macOS
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                assert SwiftMetalPlatform.is_available()

    def test_not_available_on_x86(self):
        from vllm_swift.platform import SwiftMetalPlatform

        with patch("vllm_swift.platform.py_platform") as mock_plat:
            mock_plat.machine.return_value = "x86_64"
            mock_plat.system.return_value = "Linux"
            assert not SwiftMetalPlatform.is_available()

    def test_not_available_on_linux_arm(self):
        from vllm_swift.platform import SwiftMetalPlatform

        with patch("vllm_swift.platform.py_platform") as mock_plat:
            mock_plat.machine.return_value = "arm64"
            mock_plat.system.return_value = "Linux"
            assert not SwiftMetalPlatform.is_available()

    def test_device_count_is_one(self):
        from vllm_swift.platform import SwiftMetalPlatform

        assert SwiftMetalPlatform.get_device_count() == 1

    def test_current_device_is_zero(self):
        from vllm_swift.platform import SwiftMetalPlatform

        assert SwiftMetalPlatform.current_device() == 0

    def test_device_name(self):
        from vllm_swift.platform import SwiftMetalPlatform

        assert "Swift Metal" in SwiftMetalPlatform.get_device_name()

    def test_device_capability(self):
        from vllm_swift.platform import SwiftMetalPlatform

        cap = SwiftMetalPlatform.get_device_capability()
        assert cap.major >= 8

    def test_total_memory_positive(self):
        from vllm_swift.platform import SwiftMetalPlatform

        mem = SwiftMetalPlatform.get_device_total_memory()
        assert mem > 0

    def test_available_memory_positive(self):
        from vllm_swift.platform import SwiftMetalPlatform

        mem = SwiftMetalPlatform.get_device_available_memory()
        assert mem > 0

    def test_check_and_update_config(self):
        from vllm_swift.platform import SwiftMetalPlatform

        vllm_config = MagicMock()
        vllm_config.parallel_config.worker_cls = "auto"
        vllm_config.parallel_config.distributed_executor_backend = "auto"
        vllm_config.scheduler_config.enable_chunked_prefill = True
        vllm_config.scheduler_config.max_num_batched_tokens = 1024
        vllm_config.model_config.max_model_len = 4096
        vllm_config.cache_config.enable_prefix_caching = True

        SwiftMetalPlatform.check_and_update_config(vllm_config)

        assert vllm_config.parallel_config.worker_cls == ("vllm_swift.worker.SwiftMetalWorker")
        assert vllm_config.parallel_config.distributed_executor_backend == "uni"
        assert not vllm_config.scheduler_config.enable_chunked_prefill
        assert vllm_config.scheduler_config.max_num_batched_tokens == 4096

    def test_synchronize_no_crash(self):
        from vllm_swift.platform import SwiftMetalPlatform

        # Should not raise regardless of MLX availability
        SwiftMetalPlatform.synchronize()

    def test_set_device_zero_no_crash(self):
        from vllm_swift.platform import SwiftMetalPlatform

        SwiftMetalPlatform.set_device(0)

    def test_get_torch_device(self):
        import torch

        from vllm_swift.platform import SwiftMetalPlatform

        device = SwiftMetalPlatform.get_torch_device()
        assert isinstance(device, torch.device)


class TestPlatformPlugin:
    def test_register_returns_class_when_available(self):
        from vllm_swift.platform import SwiftMetalPlatformPlugin

        with patch("vllm_swift.platform.SwiftMetalPlatform.is_available", return_value=True):
            result = SwiftMetalPlatformPlugin.register()
            assert result == "vllm_swift.platform.SwiftMetalPlatform"

    def test_register_returns_none_on_unsupported(self):
        from vllm_swift.platform import SwiftMetalPlatformPlugin

        with patch("vllm_swift.platform.SwiftMetalPlatform.is_available", return_value=False):
            assert SwiftMetalPlatformPlugin.register() is None
