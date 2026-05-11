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

# Build MLX metallib (custom kernels: GatedDelta, TurboFlash, etc.)
#
# SPM does not compile .metal sources, so the bottle pipeline (and we) must
# invoke mlx-swift-lm's build-metallib.sh against the SPM-resolved checkouts.
# The script discovers metal sources at ../mlx-swift/Source/Cmlx/mlx-generated/metal
# relative to its own location, which lines up with the SPM checkout layout.
#
# Without this, custom kernels (gated_delta_step_fused_*, TurboFlash, etc.)
# are missing at runtime and Qwen3Next-family / TurboFlash models fail to load.
# See: https://github.com/TheTom/vllm-swift/issues/7
echo "Building MLX metallib (custom kernels)..."

MLX_LM_CHECKOUT="$SWIFT_DIR/.build/checkouts/mlx-swift-lm"
METAL_BUILD_SCRIPT="$MLX_LM_CHECKOUT/scripts/build-metallib.sh"
MLX_SWIFT_CHECKOUT="$SWIFT_DIR/.build/checkouts/mlx-swift"

if [ ! -f "$METAL_BUILD_SCRIPT" ]; then
    echo "ERROR: mlx-swift-lm checkout missing build-metallib.sh at:"
    echo "  $METAL_BUILD_SCRIPT"
    echo "  swift build should have populated .build/checkouts/. Try: swift package resolve"
    exit 1
fi

if [ ! -d "$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx-generated/metal" ]; then
    echo "ERROR: mlx-swift checkout missing Metal sources at:"
    echo "  $MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx-generated/metal"
    exit 1
fi

# Run the script — it writes to its own .build/arm64-apple-macosx/$CONFIG/mlx.metallib.
# Tail output to keep installer noise low; full log on failure.
METAL_LOG="$(mktemp)"
trap 'rm -f "$METAL_LOG"' EXIT
if ! bash "$METAL_BUILD_SCRIPT" "$CONFIG" >"$METAL_LOG" 2>&1; then
    echo "ERROR: build-metallib.sh failed. Last 30 lines of output:"
    tail -30 "$METAL_LOG"
    exit 1
fi
tail -3 "$METAL_LOG"

GENERATED_METALLIB="$MLX_LM_CHECKOUT/.build/arm64-apple-macosx/$CONFIG/mlx.metallib"
if [ ! -f "$GENERATED_METALLIB" ]; then
    echo "ERROR: build-metallib.sh ran but produced no metallib at:"
    echo "  $GENERATED_METALLIB"
    echo "Full log:"
    cat "$METAL_LOG"
    exit 1
fi

# Place the metallib next to the dylib — MLX's Metal device looks alongside
# the loaded dylib for mlx.metallib at runtime.
cp "$GENERATED_METALLIB" "$BUILD_DIR/mlx.metallib"
echo "  Installed: $BUILD_DIR/mlx.metallib"
echo ""

# Verification — fail loudly if the metallib is missing or empty.
# This is the exact condition that issue #7 was filed about: source builds
# silently shipping without mlx.metallib and failing later at model load.
if [ ! -s "$BUILD_DIR/mlx.metallib" ]; then
    echo "ERROR: mlx.metallib was not installed alongside libVLLMBridge.dylib."
    echo "  Expected: $BUILD_DIR/mlx.metallib"
    echo "  Without this file, GatedDelta / TurboFlash kernels are missing and"
    echo "  models like Qwen3Next-* will fail to load at runtime."
    echo "  See: https://github.com/TheTom/vllm-swift/issues/7"
    exit 1
fi
echo ""

# Also refresh the bundled artifacts inside the Python package's `_lib/`.
#
# The package's `vllm_swift/__init__.py:_add_bundled_lib_to_dyld()` appends
# `vllm_swift/_lib/` to DYLD_LIBRARY_PATH at import time. User-set values
# (e.g. `activate.sh` pointing at $BUILD_DIR) win, but if nothing is set the
# bundled `_lib/` is the only path the loader has. A stale dylib there then
# loads at runtime with a stale ABI: launching `vllm serve` after a clean
# source build dies with `dlsym(..., <new_symbol>): symbol not found`.
#
# Keep `_lib/` in sync with the build output so source installs, pip-editable
# installs, and bottle installs all see the same artifacts.
LIB_DIR="$PROJECT_DIR/vllm_swift/_lib"
mkdir -p "$LIB_DIR"
cp "$DYLIB" "$LIB_DIR/libVLLMBridge.dylib"
cp "$BUILD_DIR/mlx.metallib" "$LIB_DIR/mlx.metallib"
echo "  Refreshed: $LIB_DIR/libVLLMBridge.dylib"
echo "  Refreshed: $LIB_DIR/mlx.metallib"
echo ""

# Sanity check: a known-good source build of feature/m5-baseline exports
# vsm_engine_get_batch_tokens. If it's missing here, either the build didn't
# update the dylib or the copy above silently failed.
if ! nm "$LIB_DIR/libVLLMBridge.dylib" 2>/dev/null | grep -q "_vsm_engine_get_batch_tokens"; then
    echo "WARNING: _lib/libVLLMBridge.dylib is missing the vsm_engine_get_batch_tokens"
    echo "  symbol expected by vllm_swift/engine_bridge.py. The dylib may be stale or"
    echo "  the build did not include the batched-uniform prefill stash. Aborting."
    exit 1
fi
echo ""

# Find Python 3.10-3.13 (vLLM doesn't support 3.14+)
_find_python() {
    for p in python3.13 python3.12 python3.11 python3.10; do
        if command -v "$p" &>/dev/null; then echo "$p"; return; fi
        for dir in /opt/homebrew/bin /usr/local/bin; do
            if [ -x "$dir/$p" ]; then echo "$dir/$p"; return; fi
        done
    done
    local ver=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    if [ "${ver:-0}" -ge 10 ] && [ "${ver:-99}" -le 13 ]; then echo "python3"; return; fi
    echo ""
}

PYTHON=$(_find_python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10-3.13 required (vLLM doesn't support 3.14+ yet)."
    echo "  Install via: brew install python@3.13"
    echo "  or: https://www.python.org/downloads/"
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

# Create venv and install Python plugin
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
echo "Installing Python plugin..."
cd "$PROJECT_DIR"
"$VENV_DIR/bin/pip" install -e . 2>&1 | tail -5
echo "  Installed: vllm-swift (editable) in .venv"
echo ""

# Install vLLM if not already present
if ! "$VENV_DIR/bin/python3" -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM (this may take a few minutes)..."
    # Apple Clang errors on chained comparisons in vLLM's C++ code.
    # Same workaround used by vllm-metal's install.sh.
    CFLAGS="-Wno-parentheses" CXXFLAGS="-Wno-parentheses" "$VENV_DIR/bin/pip" install "vllm>=0.19.0" 2>&1 | tail -10 || true
    if ! "$VENV_DIR/bin/python3" -c "import vllm" 2>/dev/null; then
        echo ""
        echo "WARNING: vLLM installation failed. You may need to install it manually:"
        echo "  source .venv/bin/activate"
        echo "  pip install vllm"
        echo ""
        echo "activate.sh will still be created so you can set up vLLM yourself."
    fi
fi
echo ""

# Create activation script
ACTIVATE_SCRIPT="$PROJECT_DIR/activate.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
# Source this file to set up vllm-swift environment
# Usage: source activate.sh
source "$VENV_DIR/bin/activate"
export DYLD_LIBRARY_PATH="$BUILD_DIR:\${DYLD_LIBRARY_PATH:-}"
echo "vllm-swift activated (venv + DYLD_LIBRARY_PATH set)"
EOF
echo "Created: activate.sh (source this before running vllm serve)"
echo ""

# Verify installation
echo "Verifying installation..."
if "$VENV_DIR/bin/python3" -c "from vllm_swift import register; print('  Plugin loads OK')" 2>&1; then
    echo ""
else
    echo "  WARNING: Plugin import failed. Check Python environment."
fi

echo "=== Installation complete ==="
echo ""
echo "Quick start:"
echo "  cd $PROJECT_DIR"
echo "  source activate.sh"
echo "  vllm serve ~/models/Qwen3-4B-4bit --max-model-len 4096"
echo ""
