# SPDX-License-Identifier: Apache-2.0
"""vLLM Swift Metal plugin — Apple Silicon inference via mlx-swift."""

import os
import sys

__version__ = "0.6.2"


def _apply_macos_defaults() -> None:
    """Apply safe defaults for macOS multiprocessing.

    vLLM V1 launches a worker subprocess. On macOS, fork() with an
    initialized Objective-C runtime crashes the child process. Using
    spawn starts a fresh interpreter and avoids this.
    """
    if sys.platform != "darwin":
        return
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is not None:
        return
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _add_bundled_lib_to_dyld() -> None:
    """Add the package's `_lib/` to DYLD_LIBRARY_PATH so the bundled Swift
    bridge dylib is discoverable for pip-wheel and Homebrew-bottle installs
    that ship a prebuilt dylib next to the Python package.

    The path is APPENDED, not prepended. If the user has set
    DYLD_LIBRARY_PATH themselves (e.g. via `source activate.sh` from a
    source install that points at `swift/.build/<config>/`), their entry
    takes precedence over the bundled fallback. This avoids the failure
    mode where a stale `_lib/` dylib silently shadows a freshly-built one
    in `swift/.build/`, surfacing as `dlsym(..., <new_symbol>): symbol not
    found` on import.
    """
    if sys.platform != "darwin":
        return
    lib = os.path.join(os.path.dirname(__file__), "_lib")
    if not os.path.isdir(lib):
        return
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if lib in existing.split(":"):
        return
    os.environ["DYLD_LIBRARY_PATH"] = f"{existing}:{lib}" if existing else lib


def register() -> str | None:
    _apply_macos_defaults()
    _add_bundled_lib_to_dyld()
    from vllm_swift.platform import SwiftMetalPlatformPlugin

    return SwiftMetalPlatformPlugin.register()
