#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# vllm-swift install script
#
# Builds the Swift bridge, installs the Python plugin, and sets up
# the metallib so MLX can find it at runtime.
#
# Usage: ./scripts/install.sh [--release|--debug]

set -euo pipefail

BUILD_CONFIG="${1:---release}"
case "$BUILD_CONFIG" in
    --release) CONFIG="release" ;;
    --debug)   CONFIG="debug" ;;
    *)
        echo "Usage: $0 [--release|--debug]"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SWIFT_DIR="$PROJECT_DIR/swift"
BUILD_DIR="$SWIFT_DIR/.build/arm64-apple-macosx/$CONFIG"

echo "=== vllm-swift installer ==="
echo "Config: $CONFIG"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v swift &>/dev/null; then
    echo "ERROR: Swift toolchain not found. Install Xcode or Swift from swift.org"
    exit 1
fi

SWIFT_VERSION=$(swift --version 2>&1 | head -1)
echo "  Swift: $SWIFT_VERSION"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Python: $PYTHON_VERSION"

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: vllm-swift requires Apple Silicon (arm64)"
    exit 1
fi
echo "  Architecture: arm64 (Apple Silicon)"
echo ""

# Build Swift bridge
echo "Building Swift bridge ($CONFIG)..."
cd "$SWIFT_DIR"
swift build -c "$CONFIG" 2>&1 | tail -3

DYLIB="$BUILD_DIR/libVLLMBridge.dylib"
if [ ! -f "$DYLIB" ]; then
    echo "ERROR: Build failed — dylib not found at $DYLIB"
    exit 1
fi
echo "  Built: $DYLIB"
echo ""

# Find and copy MLX metallib
echo "Setting up MLX metallib..."
MLX_METALLIB=""

# Check common locations for the metallib
for candidate in \
    "$BUILD_DIR/mlx.metallib" \
    "$SWIFT_DIR/.build/artifacts/mlx-swift/mlxc.artifactbundle/"*"/mlx.metallib" \
    "$(python3 -c 'import mlx; import os; print(os.path.join(os.path.dirname(mlx.__file__), "lib", "mlx.metallib"))' 2>/dev/null || echo '')" \
    "$HOME/Library/Developer/Xcode/DerivedData/"*"/Build/Products/"*"/mlx.metallib"
do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        MLX_METALLIB="$candidate"
        break
    fi
done

if [ -n "$MLX_METALLIB" ]; then
    if [ "$MLX_METALLIB" != "$BUILD_DIR/mlx.metallib" ]; then
        cp "$MLX_METALLIB" "$BUILD_DIR/mlx.metallib"
        echo "  Copied metallib from: $MLX_METALLIB"
    else
        echo "  Metallib already in place: $MLX_METALLIB"
    fi
else
    echo "  WARNING: mlx.metallib not found. MLX will compile kernels at runtime (slower first run)."
    echo "  This is normal for first builds — MLX generates it on first use."
fi
echo ""

# Install Python plugin
echo "Installing Python plugin..."
cd "$PROJECT_DIR"
pip install -e . --quiet 2>&1 | tail -3
echo "  Installed: vllm-swift (editable)"
echo ""

# Create activation script
ACTIVATE_SCRIPT="$PROJECT_DIR/activate.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
# Source this file to set up vllm-swift environment
# Usage: source activate.sh
export DYLD_LIBRARY_PATH="$BUILD_DIR:\${DYLD_LIBRARY_PATH:-}"
echo "vllm-swift activated (DYLD_LIBRARY_PATH set)"
EOF
echo "Created: activate.sh (source this before running vllm serve)"
echo ""

# Verify installation
echo "Verifying installation..."
if python3 -c "from vllm_swift import register; print('  Plugin loads OK')" 2>&1; then
    echo ""
else
    echo "  WARNING: Plugin import failed. Check Python environment."
fi

echo "=== Installation complete ==="
echo ""
echo "Quick start:"
echo "  source activate.sh"
echo "  vllm serve ~/models/Qwen3-0.6B-4bit --max-model-len 2048"
echo ""
