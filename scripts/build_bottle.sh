#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build a Homebrew bottle locally and upload to GitHub Releases.
#
# Usage: ./scripts/build_bottle.sh
#
# Builds the Swift bridge, packages everything into a bottle tarball,
# and uploads to TheTom/homebrew-tap releases. Users then get a
# prebuilt binary — no Swift build needed on their machine.

set -euo pipefail

VERSION="0.6.1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SWIFT_DIR="$PROJECT_DIR/swift"
BOTTLE_DIR="/tmp/vllm-swift-bottle/vllm-swift/$VERSION"

# Map macOS major version to Homebrew bottle tag.
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
case "$MACOS_MAJOR" in
    26) MACOS_TAG="arm64_tahoe" ;;
    15) MACOS_TAG="arm64_sequoia" ;;
    14) MACOS_TAG="arm64_sonoma" ;;
    *)  echo "ERROR: unknown macOS major version $MACOS_MAJOR"; exit 1 ;;
esac
BOTTLE_TAR="vllm-swift-${VERSION}.${MACOS_TAG}.bottle.tar.gz"

echo "=== Building vllm-swift bottle v${VERSION} ==="
echo ""

# Build Swift bridge
echo "Building Swift bridge (release)..."
cd "$SWIFT_DIR"
swift build -c release 2>&1 | tail -3
DYLIB=$(find .build -name "libVLLMBridge.dylib" -print -quit)
if [ -z "$DYLIB" ]; then
    echo "ERROR: dylib not found"
    exit 1
fi
echo "Built: $DYLIB"

# Build MLX metallib (SPM doesn't compile .metal files; mlx-swift-lm ships
# a build-metallib.sh that drives xcrun metal/metallib over the
# SPM-resolved .metal sources). We need this for every bottle build —
# previously this step was implicit (whatever metallib happened to be in
# .build/), which led to v0.6.0 shipping a stale metallib missing the
# `turbo_dequant_rotated_*` family. Re-running the script here guarantees
# a fresh build that matches the current mlx-swift pin.
echo "Building MLX metallib (custom kernels)..."
METAL_BUILD_SCRIPT="$SWIFT_DIR/.build/checkouts/mlx-swift-lm/scripts/build-metallib.sh"
GENERATED_METALLIB="$SWIFT_DIR/.build/checkouts/mlx-swift-lm/.build/arm64-apple-macosx/release/mlx.metallib"
if [ ! -f "$METAL_BUILD_SCRIPT" ]; then
    echo "ERROR: build-metallib.sh missing at $METAL_BUILD_SCRIPT"
    echo "  Did swift build resolve mlx-swift-lm? Try: swift package resolve"
    exit 1
fi
METAL_LOG=$(mktemp)
trap 'rm -f "$METAL_LOG"' EXIT
if ! bash "$METAL_BUILD_SCRIPT" release >"$METAL_LOG" 2>&1; then
    echo "ERROR: build-metallib.sh failed. Last 30 lines:"
    tail -30 "$METAL_LOG"
    exit 1
fi
if [ ! -s "$GENERATED_METALLIB" ]; then
    echo "ERROR: build-metallib.sh produced no metallib at $GENERATED_METALLIB"
    cat "$METAL_LOG"
    exit 1
fi
# Sanity check: every bottle must ship the rotated dequant family. This
# is the exact kernel family v0.6.0 missed; failing loudly here prevents
# the same regression silently slipping through future bottles. Use grep
# -c (no early-exit) to dodge pipefail+SIGPIPE on strings.
KERNEL_COUNT=$(strings "$GENERATED_METALLIB" | grep -c "turbo_dequant_rotated_4_256_bf16" || true)
if [ "$KERNEL_COUNT" -eq 0 ]; then
    echo "ERROR: metallib is missing turbo_dequant_rotated_4_256_bf16."
    echo "  This is the regression v0.6.1 was cut to fix - refusing to publish."
    exit 1
fi
echo "Metallib: $GENERATED_METALLIB ($(du -h "$GENERATED_METALLIB" | cut -f1))"

# Package bottle
echo "Packaging bottle..."
rm -rf /tmp/vllm-swift-bottle
mkdir -p "$BOTTLE_DIR/lib" "$BOTTLE_DIR/libexec/vllm_swift" "$BOTTLE_DIR/libexec/scripts" "$BOTTLE_DIR/bin"

# Dylib
cp "$DYLIB" "$BOTTLE_DIR/lib/"

# Metallib — use the explicit path we just built, not a `find` first-match.
cp "$GENERATED_METALLIB" "$BOTTLE_DIR/lib/"

# Python plugin
cp "$PROJECT_DIR"/vllm_swift/*.py "$BOTTLE_DIR/libexec/vllm_swift/"
cp "$PROJECT_DIR/pyproject.toml" "$BOTTLE_DIR/libexec/"
cp "$PROJECT_DIR"/scripts/*.sh "$BOTTLE_DIR/libexec/scripts/" 2>/dev/null || true
cp "$PROJECT_DIR"/scripts/detect_tool_parser.py "$BOTTLE_DIR/libexec/scripts/" 2>/dev/null || true

# Wrapper script
cat > "$BOTTLE_DIR/bin/vllm-swift" << 'WRAPPER'
#!/usr/bin/env bash
# Resolve symlink to find the real Cellar prefix (where libexec lives)
PREFIX="$(cd "$(dirname "$0")" && cd "$(dirname "$(readlink "$0" 2>/dev/null || echo "$0")")/.." && pwd)"
export DYLD_LIBRARY_PATH="$PREFIX/lib:${DYLD_LIBRARY_PATH:-}"
VENV_DIR="$HOME/.vllm-swift/venv"

_find_python() {
  # vLLM supports 3.10-3.13. Prefer 3.13, avoid 3.14+ (too new).
  for p in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$p" &>/dev/null; then echo "$p"; return; fi
    for dir in /opt/homebrew/bin /usr/local/bin; do
      if [ -x "$dir/$p" ]; then echo "$dir/$p"; return; fi
    done
  done
  # Check system python3 version
  local ver=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
  if [ "${ver:-0}" -ge 10 ]; then echo "python3"; return; fi
  echo ""
}

_ensure_venv() {
  # Check if vLLM is actually working, not just if the dir exists
  if "$VENV_DIR/bin/python3" -c "import vllm" 2>/dev/null; then
    return
  fi
  PYTHON=$(_find_python)
  if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10-3.13 required (vLLM doesn't support 3.14+ yet)."
    echo "  Install via: brew install python@3.13"
    echo "  or: https://www.python.org/downloads/"
    exit 1
  fi
  echo "Setting up vllm-swift Python environment (one time)..."
  echo "Using: $PYTHON ($($PYTHON --version 2>&1))"
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  echo "Installing PyTorch (this may take a minute)..."
  "$VENV_DIR/bin/pip" install --progress-bar on torch --index-url https://download.pytorch.org/whl/cpu
  echo "Installing vLLM (this may take a few minutes)..."
  CFLAGS="-Wno-parentheses" CXXFLAGS="-Wno-parentheses" "$VENV_DIR/bin/pip" install --progress-bar on "vllm>=0.19.0"
  # Install plugin
  echo "Installing vllm-swift plugin..."
  if [ -f "$PREFIX/libexec/pyproject.toml" ]; then
    cd "$PREFIX/libexec" && "$VENV_DIR/bin/pip" install -q . && cd - >/dev/null
  fi
  echo "Setup complete."
}

case "${1:-}" in
  serve)
    _ensure_venv
    shift
    # Delegate to the Python CLI (vllm_swift.cli). It owns the full
    # auto-detect + invisible self-heal stack as of v0.4.0:
    #   - tool + reasoning parser detection (3-layer)
    #   - pre-flight registry validation against vLLM's parser sets
    #   - rewriter proxy for max_tokens rescue, Thinking: split,
    #     plaintext-JSON tool-call recovery (streaming + non-streaming)
    # Keeping the bash wrapper as a thin shim that only sets up
    # DYLD_LIBRARY_PATH + venv routing means the bottle inherits every
    # v0.4.0+ feature automatically without re-implementing parser logic
    # in two places.
    exec "$VENV_DIR/bin/python3" -m vllm_swift.cli serve "$@"
    ;;
  download)
    _ensure_venv
    shift
    MODEL="${1:?Usage: vllm-swift download <model-id>}"
    SHORT="$(basename "$MODEL")"
    exec "$VENV_DIR/bin/python3" -c "
from huggingface_hub import snapshot_download; import os
p = snapshot_download('$MODEL', local_dir=os.path.expanduser('~/models/$SHORT'))
print(f'Downloaded to {p}')
"
    ;;
  setup)
    _ensure_venv
    echo "vllm-swift environment ready at $VENV_DIR"
    ;;
  update)
    echo "Updating vllm-swift..."
    brew untap TheTom/tap 2>/dev/null
    rm -rf "$(brew --cache)/downloads/"*vllm* 2>/dev/null
    rm -rf "$HOME/.vllm-swift/venv" 2>/dev/null
    brew tap TheTom/tap && brew reinstall vllm-swift
    _ensure_venv
    echo "Update complete."
    ;;
  version)
    echo "vllm-swift 0.6.1"
    echo "dylib: $PREFIX/lib/libVLLMBridge.dylib"
    [ -d "$VENV_DIR" ] && "$VENV_DIR/bin/python3" -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null
    ;;
  *)
    echo "vllm-swift — Native Swift/Metal backend for vLLM on Apple Silicon"
    echo ""
    echo "Usage:"
    echo "  vllm-swift serve <model> [args]   Start OpenAI-compatible API server"
    echo "  vllm-swift download <model-id>    Download model from HuggingFace"
    echo "  vllm-swift setup                  Set up Python environment"
    echo "  vllm-swift update                 Update to latest version"
    echo "  vllm-swift version                Show version info"
    echo ""
    echo "Examples:"
    echo "  vllm-swift download mlx-community/Qwen3-4B-4bit"
    echo "  vllm-swift serve ~/models/Qwen3-4B-4bit --max-model-len 4096"
    ;;
esac
WRAPPER
chmod +x "$BOTTLE_DIR/bin/vllm-swift"

# Create tarball
cd /tmp/vllm-swift-bottle
tar czf "/tmp/$BOTTLE_TAR" vllm-swift/
echo "Bottle: /tmp/$BOTTLE_TAR ($(du -h "/tmp/$BOTTLE_TAR" | cut -f1))"

# Upload to GitHub Releases (skip with NO_UPLOAD=1 for prep-only builds)
if [ "${NO_UPLOAD:-0}" = "1" ]; then
    echo ""
    echo "NO_UPLOAD=1 set — skipping GitHub Releases upload (bottle remains at /tmp/$BOTTLE_TAR)."
else
    echo ""
    echo "Uploading to GitHub Releases..."
    gh release create bottles --repo TheTom/homebrew-tap \
        --title "Bottles" --notes "Prebuilt Homebrew bottles for vllm-swift" 2>/dev/null || true
    gh release upload bottles "/tmp/$BOTTLE_TAR" --repo TheTom/homebrew-tap --clobber
fi

# Compute SHA for formula
SHA=$(shasum -a 256 "/tmp/$BOTTLE_TAR" | awk '{print $1}')
echo ""
echo "=== Done ==="
if [ "${NO_UPLOAD:-0}" = "1" ]; then
    echo "Bottle built (not uploaded): /tmp/$BOTTLE_TAR"
else
    echo "Bottle uploaded to: https://github.com/TheTom/homebrew-tap/releases/tag/bottles"
fi
echo ""
echo "Add this to Formula/vllm-swift.rb after 'license' line:"
echo ""
echo "  bottle do"
echo "    root_url \"https://github.com/TheTom/homebrew-tap/releases/download/bottles\""
echo "    sha256 cellar: :any, ${MACOS_TAG}: \"$SHA\""
echo "  end"
