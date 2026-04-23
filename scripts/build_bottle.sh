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

VERSION="0.1.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SWIFT_DIR="$PROJECT_DIR/swift"
BOTTLE_DIR="/tmp/vllm-swift-bottle/vllm-swift/$VERSION"
BOTTLE_TAR="vllm-swift-${VERSION}.arm64_sequoia.bottle.tar.gz"

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

# Package bottle
echo "Packaging bottle..."
rm -rf /tmp/vllm-swift-bottle
mkdir -p "$BOTTLE_DIR/lib" "$BOTTLE_DIR/libexec/vllm_swift" "$BOTTLE_DIR/libexec/scripts" "$BOTTLE_DIR/bin"

# Dylib
cp "$DYLIB" "$BOTTLE_DIR/lib/"

# Metallib
METALLIB=$(find .build -name "mlx.metallib" -print -quit 2>/dev/null)
[ -n "$METALLIB" ] && cp "$METALLIB" "$BOTTLE_DIR/lib/"

# Python plugin
cp "$PROJECT_DIR"/vllm_swift/*.py "$BOTTLE_DIR/libexec/vllm_swift/"
cp "$PROJECT_DIR/pyproject.toml" "$BOTTLE_DIR/libexec/"
cp "$PROJECT_DIR"/scripts/*.sh "$BOTTLE_DIR/libexec/scripts/" 2>/dev/null || true

# Wrapper script
cat > "$BOTTLE_DIR/bin/vllm-swift" << 'WRAPPER'
#!/usr/bin/env bash
PREFIX="$(cd "$(dirname "$0")/.." && pwd)"
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
  "$VENV_DIR/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
  CFLAGS="-Wno-parentheses" "$VENV_DIR/bin/pip" install "vllm>=0.19.0" 2>&1 | tail -5
  # Install plugin
  if [ -f "$PREFIX/libexec/pyproject.toml" ]; then
    cd "$PREFIX/libexec" && "$VENV_DIR/bin/pip" install -q . && cd - >/dev/null
  fi
  echo "Setup complete."
}

case "${1:-}" in
  serve)
    _ensure_venv
    shift
    exec "$VENV_DIR/bin/python3" -m vllm.entrypoints.openai.api_server "$@"
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
  version)
    echo "vllm-swift 0.1.0"
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
    echo "  vllm-swift version                Show version info"
    echo ""
    echo "Examples:"
    echo "  vllm-swift download mlx-community/Qwen3-4B-4bit"
    echo "  vllm-swift serve ~/models/Qwen3-4B-4bit --max-model-len 2048"
    ;;
esac
WRAPPER
chmod +x "$BOTTLE_DIR/bin/vllm-swift"

# Create tarball
cd /tmp/vllm-swift-bottle
tar czf "/tmp/$BOTTLE_TAR" vllm-swift/
echo "Bottle: /tmp/$BOTTLE_TAR ($(du -h "/tmp/$BOTTLE_TAR" | cut -f1))"

# Upload to GitHub Releases
echo ""
echo "Uploading to GitHub Releases..."
gh release create bottles --repo TheTom/homebrew-tap \
    --title "Bottles" --notes "Prebuilt Homebrew bottles for vllm-swift" 2>/dev/null || true
gh release upload bottles "/tmp/$BOTTLE_TAR" --repo TheTom/homebrew-tap --clobber

# Compute SHA for formula
SHA=$(shasum -a 256 "/tmp/$BOTTLE_TAR" | awk '{print $1}')
echo ""
echo "=== Done ==="
echo "Bottle uploaded to: https://github.com/TheTom/homebrew-tap/releases/tag/bottles"
echo ""
echo "Add this to Formula/vllm-swift.rb after 'license' line:"
echo ""
echo "  bottle do"
echo "    root_url \"https://github.com/TheTom/homebrew-tap/releases/download/bottles\""
echo "    sha256 cellar: :any, arm64_sequoia: \"$SHA\""
echo "  end"
