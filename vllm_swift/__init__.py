# SPDX-License-Identifier: Apache-2.0
"""vLLM Swift Metal plugin — Apple Silicon inference via mlx-swift."""

import os
import sys

__version__ = "0.1.0"


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


def register() -> str | None:
    _apply_macos_defaults()
    from vllm_swift.platform import SwiftMetalPlatformPlugin

    return SwiftMetalPlatformPlugin.register()
