# SPDX-License-Identifier: Apache-2.0
"""Tests for plugin registration."""

import os
import sys
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


# ---------------------------------------------------------------------------
# _add_bundled_lib_to_dyld()
#
# Regression coverage for the failure mode where a stale `_lib/` dylib silently
# shadowed a freshly-built one in `swift/.build/`. The fix flips the path
# injection from PREPEND to APPEND so a user-set DYLD_LIBRARY_PATH (e.g. from
# source-tree `activate.sh`) wins over the bundled fallback.
# ---------------------------------------------------------------------------


def _bundled_lib_path() -> str:
    import vllm_swift

    return os.path.join(os.path.dirname(vllm_swift.__file__), "_lib")


def _run_with_clean_env(dyld_value, platform="darwin"):
    """Invoke _add_bundled_lib_to_dyld() with a controlled env and platform.

    Returns the resulting DYLD_LIBRARY_PATH value (or None if unset).
    """
    from vllm_swift import _add_bundled_lib_to_dyld

    saved_env = os.environ.get("DYLD_LIBRARY_PATH")
    saved_platform = sys.platform
    try:
        if dyld_value is None:
            os.environ.pop("DYLD_LIBRARY_PATH", None)
        else:
            os.environ["DYLD_LIBRARY_PATH"] = dyld_value
        # Patch sys.platform for the non-darwin test path.
        with patch.object(sys, "platform", platform):
            _add_bundled_lib_to_dyld()
        return os.environ.get("DYLD_LIBRARY_PATH")
    finally:
        if saved_env is None:
            os.environ.pop("DYLD_LIBRARY_PATH", None)
        else:
            os.environ["DYLD_LIBRARY_PATH"] = saved_env
        # sys.platform restored by patch context manager exit.
        assert sys.platform == saved_platform


def test_add_bundled_lib_uses_lib_when_dyld_unset():
    """With no existing DYLD_LIBRARY_PATH, the bundled `_lib/` becomes the only entry."""
    lib = _bundled_lib_path()
    if not os.path.isdir(lib):
        # No bundled dylib in source checkouts that haven't run install.sh.
        # _add_bundled_lib_to_dyld() short-circuits in that case; verify.
        result = _run_with_clean_env(None)
        assert result is None
        return
    result = _run_with_clean_env(None)
    assert result == lib, f"Expected DYLD_LIBRARY_PATH={lib}, got {result!r}"


def test_add_bundled_lib_appends_when_user_set():
    """A user-set DYLD_LIBRARY_PATH wins — the bundled `_lib/` is appended last.

    This is the v0.6.x bug fix: previously the bundled path was PREPENDED, so a
    stale dylib in `_lib/` shadowed the freshly-built one a developer was trying
    to load via `activate.sh`.
    """
    lib = _bundled_lib_path()
    if not os.path.isdir(lib):
        # Without `_lib/`, the function is a no-op past the isdir check; the
        # user's path is preserved verbatim. Cover that branch too.
        user_path = "/tmp/fake-build-dir"
        result = _run_with_clean_env(user_path)
        assert result == user_path
        return
    user_path = "/tmp/fake-build-dir"
    result = _run_with_clean_env(user_path)
    assert result is not None
    # User's entry must come first; `_lib/` must come after.
    parts = result.split(":")
    assert parts[0] == user_path, f"User-set DYLD entry must come first; got order: {parts}"
    assert lib in parts, f"Bundled `_lib/` missing from final DYLD path: {parts}"
    assert parts.index(user_path) < parts.index(lib), (
        f"`_lib/` must be appended after user path, got {parts}"
    )


def test_add_bundled_lib_is_idempotent():
    """Calling twice must not duplicate `_lib/` in DYLD_LIBRARY_PATH."""
    lib = _bundled_lib_path()
    if not os.path.isdir(lib):
        return
    from vllm_swift import _add_bundled_lib_to_dyld

    saved = os.environ.get("DYLD_LIBRARY_PATH")
    try:
        os.environ.pop("DYLD_LIBRARY_PATH", None)
        _add_bundled_lib_to_dyld()
        first = os.environ["DYLD_LIBRARY_PATH"]
        _add_bundled_lib_to_dyld()
        second = os.environ["DYLD_LIBRARY_PATH"]
        assert first == second, f"Second call duplicated the entry: {first!r} -> {second!r}"
        assert second.count(lib) == 1, f"_lib appears {second.count(lib)} times in {second!r}"
    finally:
        if saved is None:
            os.environ.pop("DYLD_LIBRARY_PATH", None)
        else:
            os.environ["DYLD_LIBRARY_PATH"] = saved


def test_add_bundled_lib_is_noop_off_darwin():
    """No DYLD munging on Linux / Windows — the env var is macOS-specific."""
    user_path = "/some/path"
    result = _run_with_clean_env(user_path, platform="linux")
    assert result == user_path, f"Non-darwin should not touch DYLD_LIBRARY_PATH; got {result!r}"
