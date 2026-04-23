# SPDX-License-Identifier: Apache-2.0
"""vLLM Swift Metal plugin — Apple Silicon inference via mlx-swift."""

__version__ = "0.1.0"


def register() -> str | None:
    from vllm_swift_metal.platform import SwiftMetalPlatformPlugin

    return SwiftMetalPlatformPlugin.register()
