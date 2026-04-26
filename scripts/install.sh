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
CHECKOUT_MLXLM_DIR="$SWIFT_DIR/.build/checkouts/mlx-swift-lm"
CHECKOUT_METALLIB_SCRIPT="$CHECKOUT_MLXLM_DIR/scripts/build-metallib.sh"
CHECKOUT_METALLIB_PATH="$CHECKOUT_MLXLM_DIR/.build/arm64-apple-macosx/$CONFIG/mlx.metallib"

_metallib_has_gdn_kernels() {
    local metallib="$1"
    [ -f "$metallib" ] || return 1
    # Use grep directly on binary data (-a) to avoid pipefail/SIGPIPE false negatives
    # from `strings | grep -q` pipelines.
    LC_ALL=C grep -aq "gated_delta_step_fused_" "$metallib"
}

_metal_compiler_available() {
    xcrun metal -v >/dev/null 2>&1
}

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
SWIFT_BUILD_LOG="$(mktemp)"
if ! swift build -c "$CONFIG" >"$SWIFT_BUILD_LOG" 2>&1; then
    echo "ERROR: Swift build failed. Last 80 lines:"
    tail -80 "$SWIFT_BUILD_LOG"
    rm -f "$SWIFT_BUILD_LOG"
    exit 1
fi
tail -3 "$SWIFT_BUILD_LOG"
rm -f "$SWIFT_BUILD_LOG"

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
MLX_METALLIB_FALLBACK=""

if [ -f "$CHECKOUT_METALLIB_SCRIPT" ]; then
    echo "  Attempting metallib build from mlx-swift-lm checkout..."
    if _metal_compiler_available; then
        if bash "$CHECKOUT_METALLIB_SCRIPT" "$CONFIG"; then
            if [ -f "$CHECKOUT_METALLIB_PATH" ]; then
                cp "$CHECKOUT_METALLIB_PATH" "$BUILD_DIR/mlx.metallib"
                echo "  Built and copied checkout metallib: $CHECKOUT_METALLIB_PATH"
            fi
        else
            echo "  WARNING: Checkout metallib build failed. Will fall back to existing metallib candidates."
        fi
    else
        echo "  WARNING: Metal compiler is not runnable in this Xcode setup."
        echo "           Install/update Metal Toolchain, then rerun install:"
        echo "           xcodebuild -downloadComponent MetalToolchain"
    fi
fi

# Check common locations for the metallib
for candidate in \
    "$BUILD_DIR/mlx.metallib" \
    "$CHECKOUT_METALLIB_PATH" \
    "$SWIFT_DIR/.build/artifacts/mlx-swift/mlxc.artifactbundle/"*"/mlx.metallib" \
    "$(python3 -c 'import mlx; import os; print(os.path.join(os.path.dirname(mlx.__file__), "lib", "mlx.metallib"))' 2>/dev/null || echo '')" \
    "$HOME/Library/Developer/Xcode/DerivedData/"*"/Build/Products/"*"/mlx.metallib"
do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        if _metallib_has_gdn_kernels "$candidate"; then
            MLX_METALLIB="$candidate"
            break
        fi
        if [ -z "$MLX_METALLIB_FALLBACK" ]; then
            MLX_METALLIB_FALLBACK="$candidate"
        fi
    fi
done

if [ -z "$MLX_METALLIB" ] && [ -n "$MLX_METALLIB_FALLBACK" ]; then
    MLX_METALLIB="$MLX_METALLIB_FALLBACK"
fi

if [ -n "$MLX_METALLIB" ]; then
    if [ "$MLX_METALLIB" != "$BUILD_DIR/mlx.metallib" ]; then
        cp "$MLX_METALLIB" "$BUILD_DIR/mlx.metallib"
        echo "  Copied metallib from: $MLX_METALLIB"
    else
        echo "  Metallib already in place: $MLX_METALLIB"
    fi
else
    echo "  WARNING: mlx.metallib not found. Attempting to generate..."
    # Force metallib generation by running a trivial Metal op
    python3 -c "
try:
    import mlx.core as mx; mx.eval(mx.add(mx.array([1]), mx.array([2])))
    import os; src = os.path.join(os.path.dirname(mx.__file__), 'lib', 'mlx.metallib')
    if os.path.exists(src):
        import shutil; shutil.copy(src, '$BUILD_DIR/mlx.metallib'); print('  Generated and copied metallib')
except: pass
" 2>/dev/null
    if [ ! -f "$BUILD_DIR/mlx.metallib" ]; then
        echo "  WARNING: Could not generate metallib. Some models (GDN/TurboFlash) may fail."
        echo "  To fix: pip install mlx && python3 -c 'import mlx.core; mlx.core.eval(mlx.core.array([1]))'  "
    fi
fi

if [ -f "$BUILD_DIR/mlx.metallib" ]; then
    if _metallib_has_gdn_kernels "$BUILD_DIR/mlx.metallib"; then
        echo "  Verified: gated_delta kernels present in mlx.metallib"
    else
        echo "  ERROR: gated_delta kernels NOT found in $BUILD_DIR/mlx.metallib"
        echo "  Models like Qwen3.6-27B-ConfigI-MLX will fail at runtime."
        if ! _metal_compiler_available; then
            echo "  Metal compiler is unavailable. Install/update Metal Toolchain:"
            echo "    xcodebuild -downloadComponent MetalToolchain"
        fi
        echo "  Verify with: strings $BUILD_DIR/mlx.metallib | grep gated_delta"
        if [ "${VLLM_SWIFT_ALLOW_STOCK_METALLIB:-0}" != "1" ]; then
            echo "  Failing install because required GDN kernels are missing."
            echo "  Override (not recommended): VLLM_SWIFT_ALLOW_STOCK_METALLIB=1 ./scripts/install.sh"
            exit 1
        fi
        echo "  WARNING: continuing because VLLM_SWIFT_ALLOW_STOCK_METALLIB=1"
    fi
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
echo "  # Baseline model"
echo "  vllm serve ~/models/Qwen3-4B-4bit --max-model-len 4096"
echo ""
echo "  # ConfigI model (requires gated_delta kernels)"
echo "  hf download thetom-ai/Qwen3.6-27B-ConfigI-MLX"
echo "  vllm serve thetom-ai/Qwen3.6-27B-ConfigI-MLX --max-model-len 4096"
echo ""
