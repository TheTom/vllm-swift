# SPDX-License-Identifier: Apache-2.0
"""Swift Metal platform plugin for vLLM."""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class SwiftMetalPlatformPlugin:
    """Platform plugin that activates on macOS with Apple Silicon."""

    @staticmethod
    def register():
        """Called by vLLM plugin system at startup."""
        if not _is_apple_silicon():
            return None

        logger.info("Swift Metal platform plugin registered")
        return "vllm_swift_metal.platform:SwiftMetalPlatform"


def _is_apple_silicon() -> bool:
    """Check if running on macOS with Metal GPU support."""
    import platform

    if platform.system() != "Darwin":
        return False
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False
