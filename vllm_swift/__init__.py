# SPDX-License-Identifier: Apache-2.0
"""vLLM Swift Metal plugin — Apple Silicon inference via mlx-swift."""

import os
import sys

__version__ = "0.3.1"


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
    """Prepend the package's _lib/ to DYLD_LIBRARY_PATH so the Swift bridge
    dylib is discoverable when the user invokes `vllm-swift serve` directly
    or imports the plugin from a non-Homebrew install (e.g. pip wheel).
    """
    if sys.platform != "darwin":
        return
    lib = os.path.join(os.path.dirname(__file__), "_lib")
    if not os.path.isdir(lib):
        return
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if lib in existing.split(":"):
        return
    os.environ["DYLD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else lib


def register() -> str | None:
    _apply_macos_defaults()
    _add_bundled_lib_to_dyld()
    from vllm_swift.platform import SwiftMetalPlatformPlugin

    return SwiftMetalPlatformPlugin.register()
