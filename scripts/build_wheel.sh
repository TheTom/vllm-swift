#!/usr/bin/env bash
# Build a platform wheel for vllm-swift.
#
# Steps:
#   1. Compile the Swift bridge (release) if libVLLMBridge.dylib is missing.
#   2. Stage libVLLMBridge.dylib + mlx.metallib into vllm_swift/_lib/ so they
#      get bundled as package data when the wheel is built.
#   3. Run `python -m build --wheel` to produce
#      dist/vllm_swift-<version>-py3-none-macosx_11_0_arm64.whl
#
# Twine upload (after the build succeeds):
#   twine upload --repository testpypi dist/*    # validate first
#   twine upload dist/*                          # production
#
# Both repositories read tokens from ~/.pypirc.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWIFT_BUILD_DIR="$REPO_ROOT/swift/.build/arm64-apple-macosx/release"
LIB_SRC_DYLIB="$SWIFT_BUILD_DIR/libVLLMBridge.dylib"
LIB_SRC_METALLIB="$SWIFT_BUILD_DIR/mlx.metallib"
PKG_LIB_DIR="$REPO_ROOT/vllm_swift/_lib"

PYTHON="${PYTHON:-python3}"

echo "==> Building Swift bridge if needed..."
if [ ! -f "$LIB_SRC_DYLIB" ]; then
  (cd "$REPO_ROOT/swift" && swift build -c release)
fi

if [ ! -f "$LIB_SRC_DYLIB" ]; then
  echo "ERROR: $LIB_SRC_DYLIB not found after Swift build." >&2
  exit 1
fi
if [ ! -f "$LIB_SRC_METALLIB" ]; then
  echo "ERROR: $LIB_SRC_METALLIB not found." >&2
  echo "       Run scripts/build_metallib.sh or invoke the bottle build script." >&2
  exit 1
fi

echo "==> Staging binaries into vllm_swift/_lib/ ..."
mkdir -p "$PKG_LIB_DIR"
cp "$LIB_SRC_DYLIB" "$PKG_LIB_DIR/libVLLMBridge.dylib"
cp "$LIB_SRC_METALLIB" "$PKG_LIB_DIR/mlx.metallib"

echo "==> Cleaning previous build artifacts..."
rm -rf "$REPO_ROOT/dist" "$REPO_ROOT/build" "$REPO_ROOT"/*.egg-info

echo "==> Building wheel..."
(cd "$REPO_ROOT" && "$PYTHON" -m build --wheel --no-isolation)

echo
echo "==> Built wheel:"
ls -la "$REPO_ROOT/dist/"
echo
echo "Upload steps (manual):"
echo "  twine upload --repository testpypi dist/*    # validate"
echo "  twine upload dist/*                          # production"
