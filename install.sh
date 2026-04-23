#!/bin/bash
# vllm-swift install script
set -euo pipefail

echo "=== vllm-swift installer ==="
echo ""

# Check platform
if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "Error: vllm-swift requires macOS on Apple Silicon (arm64)"
    exit 1
fi

# Check Swift
if ! command -v swift &>/dev/null; then
    echo "Error: Swift not found. Install Xcode or Swift toolchain."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Build Swift bridge
echo "Building Swift bridge (release)..."
cd swift
swift build -c release 2>&1 | tail -3
cd ..
echo "✓ Swift bridge built"

# Copy metallib if needed
RELEASE_DIR="swift/.build/arm64-apple-macosx/release"
if [[ ! -f "$RELEASE_DIR/mlx.metallib" ]]; then
    DEBUG_METALLIB="swift/.build/arm64-apple-macosx/debug/mlx.metallib"
    if [[ -f "$DEBUG_METALLIB" ]]; then
        cp "$DEBUG_METALLIB" "$RELEASE_DIR/"
    fi
fi

# Create venv if needed
VENV_DIR="${VLLM_SWIFT_VENV:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing Python dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true
pip install -q vllm-metal 2>/dev/null || true
pip install -q -e . 2>/dev/null

echo ""
echo "✓ Installation complete!"
echo ""
echo "Usage:"
echo "  source $VENV_DIR/bin/activate"
echo "  VLLM_PLUGINS=swift DYLD_LIBRARY_PATH=$RELEASE_DIR \\"
echo "    vllm serve ~/models/Qwen3-4B-4bit --dtype float16 --max-model-len 2048"
echo ""
echo "Or test directly:"
echo "  DYLD_LIBRARY_PATH=$RELEASE_DIR python test_bridge.py"
