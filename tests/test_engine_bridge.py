# SPDX-License-Identifier: Apache-2.0
"""Tests for the Swift engine C FFI bridge."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_swift_metal.engine_bridge import PerfStats, SwiftInferenceEngine


class TestGetLib:
    def test_raises_on_missing_file(self):
        import vllm_swift_metal.engine_bridge as eb

        old_path = eb._LIB_PATH
        old_lib = eb._lib
        eb._LIB_PATH = "/nonexistent/lib.dylib"
        eb._lib = None
        try:
            with pytest.raises(FileNotFoundError):
                eb._get_lib()
        finally:
            eb._LIB_PATH = old_path
            eb._lib = old_lib

    def test_returns_cached_lib(self):
        import vllm_swift_metal.engine_bridge as eb

        sentinel = object()
        old_lib = eb._lib
        eb._lib = sentinel
        try:
            assert eb._get_lib() is sentinel
        finally:
            eb._lib = old_lib

    def test_loads_real_dylib_if_exists(self):
        import os

        import vllm_swift_metal.engine_bridge as eb

        dylib = os.path.expanduser("~/dev/vllm-swift-metal/swift/libvllm_swift_metal.dylib")
        if not os.path.exists(dylib):
            pytest.skip("dylib not built")
        old_lib = eb._lib
        old_path = eb._LIB_PATH
        eb._lib = None
        eb._LIB_PATH = dylib
        try:
            lib = eb._get_lib()
            assert lib is not None
            assert hasattr(lib, "vsm_engine_create")
        finally:
            eb._lib = old_lib
            eb._LIB_PATH = old_path


class TestPerfStats:
    def test_struct_fields(self):
        stats = PerfStats()
        assert hasattr(stats, "prefill_tokens_per_sec")
        assert hasattr(stats, "decode_tokens_per_sec")
        assert hasattr(stats, "peak_memory_bytes")
        assert hasattr(stats, "total_tokens_generated")
        assert hasattr(stats, "total_decode_time_sec")

    def test_struct_defaults_to_zero(self):
        stats = PerfStats()
        assert stats.prefill_tokens_per_sec == 0.0
        assert stats.decode_tokens_per_sec == 0.0
        assert stats.peak_memory_bytes == 0
        assert stats.total_tokens_generated == 0
        assert stats.total_decode_time_sec == 0.0


class TestSwiftInferenceEngine:
    def test_init_raises_on_missing_dylib(self):
        with patch("vllm_swift_metal.engine_bridge._LIB_PATH", "/nonexistent/path.dylib"):
            with pytest.raises(FileNotFoundError, match="Swift Metal engine"):
                SwiftInferenceEngine(model_path="/tmp/model")

    def test_init_raises_on_null_handle(self):
        mock_lib = MagicMock()
        mock_lib.vsm_engine_create.return_value = None

        with patch("vllm_swift_metal.engine_bridge._get_lib", return_value=mock_lib):
            with pytest.raises(RuntimeError, match="Failed to create"):
                SwiftInferenceEngine(model_path="/tmp/model")

    def test_prefill_converts_tokens(self):
        mock_lib = MagicMock()
        mock_lib.vsm_engine_create.return_value = 0xDEAD
        mock_lib.vsm_engine_vocab_size.return_value = 32000
        mock_lib.vsm_engine_num_layers.return_value = 28
        mock_lib.vsm_engine_head_dim.return_value = 128
        mock_lib.vsm_engine_model_memory_bytes.return_value = 4_000_000_000
        mock_lib.vsm_engine_prefill.return_value = 42

        with patch("vllm_swift_metal.engine_bridge._get_lib", return_value=mock_lib):
            engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
            engine._lib = mock_lib
            engine._handle = 0xDEAD
            engine.vocab_size = 32000
            engine.num_layers = 28
            engine.head_dim = 128
            engine.model_memory_bytes = 4_000_000_000

            token = engine.prefill([1, 2, 3, 4], temperature=0.0)
            assert token == 42
            mock_lib.vsm_engine_prefill.assert_called_once()

    def test_decode_step_returns_token(self):
        mock_lib = MagicMock()
        mock_lib.vsm_engine_decode_step.return_value = 99

        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0xBEEF

        token = engine.decode_step(temperature=0.6, top_p=0.9)
        assert token == 99

    def test_decode_batch_returns_list(self):
        mock_lib = MagicMock()

        def fake_batch(handle, max_tokens, temp, top_p, buf, capacity):
            for i in range(3):
                buf[i] = 10 + i
            return 3

        mock_lib.vsm_engine_decode_batch.side_effect = fake_batch

        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0xCAFE

        tokens = engine.decode_batch(max_tokens=5)
        assert tokens == [10, 11, 12]

    def test_reset_calls_bridge(self):
        mock_lib = MagicMock()
        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0xF00D

        engine.reset()
        mock_lib.vsm_engine_reset.assert_called_once_with(0xF00D)

    def test_get_stats(self):
        mock_lib = MagicMock()
        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0xBEAD

        stats = engine.get_stats()
        assert isinstance(stats, PerfStats)
        mock_lib.vsm_engine_get_stats.assert_called_once()

    def test_destroy_on_del(self):
        mock_lib = MagicMock()
        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0xDEAD

        engine.__del__()
        mock_lib.vsm_engine_destroy.assert_called_once_with(0xDEAD)
        assert engine._handle is None

    def test_prefill_req(self):
        mock_lib = MagicMock()
        mock_lib.vsm_engine_prefill_req.return_value = 77
        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0x1234
        assert engine.prefill_req("req-1", [1, 2, 3]) == 77

    def test_decode_step_req(self):
        mock_lib = MagicMock()
        mock_lib.vsm_engine_decode_step_req.return_value = 88
        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0x1234
        assert engine.decode_step_req("req-1") == 88

    def test_finish_req(self):
        mock_lib = MagicMock()
        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0x1234
        engine.finish_req("req-1")
        mock_lib.vsm_engine_finish_req.assert_called_once()

    def test_active_requests(self):
        mock_lib = MagicMock()
        mock_lib.vsm_engine_active_requests.return_value = 5
        engine = SwiftInferenceEngine.__new__(SwiftInferenceEngine)
        engine._lib = mock_lib
        engine._handle = 0x1234
        assert engine.active_requests() == 5

    def test_kv_scheme_passed_to_bridge(self):
        mock_lib = MagicMock()
        mock_lib.vsm_engine_create.return_value = 0x1
        mock_lib.vsm_engine_vocab_size.return_value = 32000
        mock_lib.vsm_engine_num_layers.return_value = 28
        mock_lib.vsm_engine_head_dim.return_value = 128
        mock_lib.vsm_engine_model_memory_bytes.return_value = 1_000_000

        with patch("vllm_swift_metal.engine_bridge._get_lib", return_value=mock_lib):
            engine = SwiftInferenceEngine(
                model_path="/tmp/model",
                kv_scheme="turbo3",
                kv_bits=3,
            )
            call_args = mock_lib.vsm_engine_create.call_args
            assert call_args[0][3] == b"turbo3"  # kv_scheme
            assert call_args[0][4] == 3  # kv_bits
            engine._handle = None  # prevent __del__ crash
