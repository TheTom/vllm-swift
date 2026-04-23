# SPDX-License-Identifier: Apache-2.0
"""vLLM Swift Metal plugin — high-performance Apple Silicon inference via mlx-swift."""


def register():
    """Entry point for vllm.platform_plugins."""
    from vllm_swift_metal.platform import SwiftMetalPlatformPlugin

    return SwiftMetalPlatformPlugin.register()
