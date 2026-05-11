# Source this file to set up vllm-swift dev environment.
# Usage:  source activate.sh
#
# Resolves paths relative to this file so it works from any clone path —
# do NOT hardcode /Users/<you>/... here, that breaks for every other dev.
VLLM_SWIFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ -f "$VLLM_SWIFT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VLLM_SWIFT_ROOT/.venv/bin/activate"
fi
# Prepend the local source-build dir so a freshly-built dylib wins over the
# package's bundled `_lib/` fallback (see vllm_swift/__init__.py).
export DYLD_LIBRARY_PATH="$VLLM_SWIFT_ROOT/swift/.build/arm64-apple-macosx/release:${DYLD_LIBRARY_PATH:-}"
echo "vllm-swift activated (root: $VLLM_SWIFT_ROOT)"
