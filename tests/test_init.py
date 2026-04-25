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

    # Verify version is a valid semver string, not a specific value
    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected semver x.y.z, got {__version__}"
    assert all(p.isdigit() for p in parts), f"Non-numeric version parts: {__version__}"
