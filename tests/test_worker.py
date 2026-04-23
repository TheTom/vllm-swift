# SPDX-License-Identifier: Apache-2.0
"""Tests for the Swift Metal worker."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestResolveModelPath:
    def test_local_models_dir(self, tmp_path):
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()

        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            from vllm_swift_metal.worker import _resolve_model_path

            # Direct path
            assert _resolve_model_path(str(model_dir)) == str(model_dir)

    def test_short_name_lookup(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model = models_dir / "Qwen3-4B-4bit"
        model.mkdir()

        from vllm_swift_metal.worker import _resolve_model_path

        with patch("os.path.expanduser", side_effect=lambda p: str(tmp_path / p.lstrip("~/"))):
            result = _resolve_model_path("org/Qwen3-4B-4bit")
            assert result == str(model)

    def test_fallback_to_raw_name(self):
        from vllm_swift_metal.worker import _resolve_model_path

        result = _resolve_model_path("/some/direct/path")
        assert result == "/some/direct/path"


class TestWorkerLifecycle:
    def test_load_model_creates_engine(self):
        from vllm_swift_metal.worker import SwiftMetalWorker

        vllm_config = MagicMock()
        vllm_config.model_config.model = "/tmp/fake-model"
        vllm_config.model_config.max_model_len = 4096
        vllm_config.model_config.seed = 42
        vllm_config.cache_config.gpu_memory_utilization = 0.9
        vllm_config.cache_config.block_size = 16
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.additional_config = {}

        worker = SwiftMetalWorker(vllm_config=vllm_config)

        mock_engine = MagicMock()
        mock_engine.num_layers = 28
        mock_engine.head_dim = 128
        mock_engine.model_memory_bytes = 4_000_000_000

        with patch("vllm_swift_metal.worker.SwiftInferenceEngine", return_value=mock_engine):
            worker.load_model()
            assert worker.engine is mock_engine

    def test_init_device_sets_device(self):
        from vllm_swift_metal.worker import SwiftMetalWorker

        vllm_config = MagicMock()
        vllm_config.model_config.model = "test"
        vllm_config.model_config.max_model_len = 4096
        vllm_config.model_config.seed = 42
        vllm_config.cache_config.gpu_memory_utilization = 0.9
        vllm_config.cache_config.block_size = 16
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.additional_config = {}

        worker = SwiftMetalWorker(vllm_config=vllm_config)

        with (
            patch("vllm_swift_metal.worker.init_distributed_environment"),
            patch("vllm_swift_metal.worker.ensure_model_parallel_initialized"),
            patch("vllm_swift_metal.worker.set_random_seed"),
        ):
            worker.init_device()
            assert worker.device is not None

    def test_check_health_passes_with_engine(self):
        from vllm_swift_metal.worker import SwiftMetalWorker

        vllm_config = MagicMock()
        vllm_config.model_config.model = "test"
        vllm_config.model_config.max_model_len = 4096
        vllm_config.model_config.seed = 42
        vllm_config.cache_config.gpu_memory_utilization = 0.9
        vllm_config.cache_config.block_size = 16
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.additional_config = {}

        worker = SwiftMetalWorker(vllm_config=vllm_config)
        worker.engine = MagicMock()
        worker.check_health()  # should not raise


class TestSwiftMetalWorker:
    def _make_worker(self):
        from vllm_swift_metal.worker import SwiftMetalWorker

        vllm_config = MagicMock()
        vllm_config.model_config.model = "test-model"
        vllm_config.model_config.max_model_len = 4096
        vllm_config.model_config.seed = 42
        vllm_config.cache_config.gpu_memory_utilization = 0.9
        vllm_config.cache_config.block_size = 16
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.additional_config = {}

        return SwiftMetalWorker(vllm_config=vllm_config)

    def test_init_stores_config(self):
        worker = self._make_worker()
        assert worker.engine is None
        assert worker.is_driver_worker is True

    def test_determine_available_memory_positive(self):
        worker = self._make_worker()
        mem = worker.determine_available_memory()
        assert mem > 0

    def test_get_kv_cache_spec_without_engine(self):
        worker = self._make_worker()
        specs = worker.get_kv_cache_spec()
        assert len(specs) == 28  # default num_layers

    def test_initialize_cache_sets_blocks(self):
        worker = self._make_worker()
        worker.initialize_cache(num_gpu_blocks=100, num_cpu_blocks=0)
        assert worker.cache_config.num_gpu_blocks == 100

    def test_lora_not_supported(self):
        worker = self._make_worker()
        assert worker.add_lora(MagicMock()) is False
        assert worker.list_loras() == set()

    def test_supported_tasks(self):
        worker = self._make_worker()
        tasks = worker.get_supported_tasks()
        assert len(tasks) == 1

    def test_check_health_raises_without_engine(self):
        worker = self._make_worker()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.check_health()

    def test_shutdown_clears_state(self):
        worker = self._make_worker()
        worker._active_requests["req1"] = [1, 2, 3]
        worker.shutdown()
        assert len(worker._active_requests) == 0
        assert worker.engine is None

    def test_execute_model_with_mock_engine(self):
        worker = self._make_worker()
        mock_engine = MagicMock()
        mock_engine.prefill.return_value = 42
        mock_engine.num_layers = 28
        mock_engine.head_dim = 128
        worker.engine = mock_engine

        scheduler_output = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "req-001"
        new_req.prompt_token_ids = [1, 2, 3, 4]
        new_req.sampling_params.temperature = 0.0
        new_req.sampling_params.top_p = 1.0
        cached_data = MagicMock()
        cached_data.req_ids = []
        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.scheduled_cached_reqs = cached_data
        scheduler_output.finished_req_ids = []

        assert worker.execute_model(scheduler_output) is None
        output = worker.sample_tokens(None)
        assert output is not None
        assert output.req_ids == ["req-001"]
        assert output.sampled_token_ids == [[42]]
        mock_engine.reset.assert_called_once()
        mock_engine.prefill.assert_called_once()

    def test_execute_model_decode_step(self):
        worker = self._make_worker()
        mock_engine = MagicMock()
        mock_engine.decode_step.return_value = 99
        worker.engine = mock_engine
        worker._active_requests["req-001"] = [42]
        worker._request_params["req-001"] = {"temperature": 0.0, "top_p": 1.0}

        cached_data = MagicMock()
        cached_data.req_ids = ["req-001"]
        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_data
        scheduler_output.finished_req_ids = []

        assert worker.execute_model(scheduler_output) is None
        output = worker.sample_tokens(None)
        assert output.sampled_token_ids == [[99]]

    def test_execute_model_raises_without_engine(self):
        worker = self._make_worker()
        cached_data = MagicMock()
        cached_data.req_ids = []
        with pytest.raises(RuntimeError, match="not loaded"):
            worker.execute_model(
                MagicMock(
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=cached_data,
                    finished_req_ids=[],
                )
            )

    def test_sample_tokens_returns_pending(self):
        worker = self._make_worker()
        # No pending output
        assert worker.sample_tokens(None) is None

    def test_get_model_returns_engine(self):
        worker = self._make_worker()
        mock_engine = MagicMock()
        worker.engine = mock_engine
        assert worker.get_model() is mock_engine

    def test_update_max_model_len(self):
        worker = self._make_worker()
        worker.update_max_model_len(8192)
        assert worker.model_config.max_model_len == 8192

    def test_get_cache_block_size_bytes(self):
        worker = self._make_worker()
        mock_engine = MagicMock()
        mock_engine.head_dim = 128
        worker.engine = mock_engine
        size = worker.get_cache_block_size_bytes()
        assert size == 2 * 128 * 2 * 16  # K+V * head_dim * fp16 * block_size

    def test_sleep_and_wake_no_crash(self):
        worker = self._make_worker()
        worker.sleep()
        worker.wake_up()

    def test_remove_pin_lora(self):
        worker = self._make_worker()
        assert worker.remove_lora(1) is False
        assert worker.pin_lora(1) is False

    def test_initialize_from_config_noop(self):
        worker = self._make_worker()
        worker.initialize_from_config(MagicMock())  # should not raise

    def test_compile_warmup_without_engine(self):
        worker = self._make_worker()
        worker.compile_or_warm_up_model()  # no engine, should be no-op

    def test_compile_warmup_with_engine(self):
        worker = self._make_worker()
        mock_engine = MagicMock()
        worker.engine = mock_engine
        worker.compile_or_warm_up_model()
        mock_engine.prefill.assert_called_once()
        mock_engine.reset.assert_called_once()

    def test_kv_scheme_from_additional_config(self):
        from vllm_swift_metal.worker import SwiftMetalWorker

        vllm_config = MagicMock()
        vllm_config.model_config.model = "test"
        vllm_config.model_config.max_model_len = 4096
        vllm_config.model_config.seed = 42
        vllm_config.cache_config.gpu_memory_utilization = 0.9
        vllm_config.cache_config.block_size = 16
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.additional_config = {"kv_scheme": "turbo3", "kv_bits": 3}

        worker = SwiftMetalWorker(vllm_config=vllm_config)
        assert worker._kv_scheme == "turbo3"
        assert worker._kv_bits == 3

    def test_decode_step_negative_token(self):
        worker = self._make_worker()
        mock_engine = MagicMock()
        mock_engine.decode_step.return_value = -1
        worker.engine = mock_engine
        worker._active_requests["req-eos"] = [42]
        worker._request_params["req-eos"] = {}

        cached_data = MagicMock()
        cached_data.req_ids = ["req-eos"]
        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_data
        scheduler_output.finished_req_ids = []

        assert worker.execute_model(scheduler_output) is None
        output = worker.sample_tokens(None)
        assert output.sampled_token_ids == [[]]

    def test_empty_scheduler_step(self):
        worker = self._make_worker()
        worker.engine = MagicMock()

        cached_data = MagicMock()
        cached_data.req_ids = []
        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_data
        scheduler_output.finished_req_ids = set()

        result = worker.execute_model(scheduler_output)
        assert result is not None  # returns empty output directly
        assert result.req_ids == []

    def test_reset_methods_no_crash(self):
        worker = self._make_worker()
        worker.reset_mm_cache()
        assert worker.reset_prefix_cache() is True
        worker.reset_encoder_cache()

    def test_finished_requests_cleaned_up(self):
        worker = self._make_worker()
        worker.engine = MagicMock()
        worker._active_requests["req-done"] = [1, 2, 3]
        worker._request_params["req-done"] = {}

        cached_data = MagicMock()
        cached_data.req_ids = []
        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_data
        scheduler_output.finished_req_ids = ["req-done"]

        worker.execute_model(scheduler_output)
        assert "req-done" not in worker._active_requests
