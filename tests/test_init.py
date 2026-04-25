# SPDX-License-Identifier: Apache-2.0
"""Tests for plugin registration."""

from unittest.mock import patch


def test_register_delegates_to_plugin():
    from vllm_swift import register

    with patch("vllm_swift.platform.SwiftMetalPlatformPlugin.register", return_value="test.path"):
        result = register()
        assert result == "test.path"


def test_register_returns_none_when_unavailable():
    from vllm_swift import register

    with patch("vllm_swift.platform.SwiftMetalPlatformPlugin.register", return_value=None):
        assert register() is None


def test_version():
    from vllm_swift import __version__

    assert __version__ == "0.2.0"
