#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Test vllm-swift brew wrapper commands.
#
# Usage: ./scripts/test_brew_commands.sh
# Requires: vllm-swift installed via brew

set -euo pipefail

PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== vllm-swift wrapper tests ==="
echo ""

# Test: version command
echo "Testing: vllm-swift version"
VERSION_OUT=$(vllm-swift version 2>&1)
echo "$VERSION_OUT" | grep -q "vllm-swift" && check "version shows vllm-swift" 0 || check "version shows vllm-swift" 1
echo "$VERSION_OUT" | grep -q "dylib:" && check "version shows dylib path" 0 || check "version shows dylib path" 1

# Test: dylib exists
DYLIB=$(echo "$VERSION_OUT" | grep "dylib:" | awk '{print $2}')
[ -f "$DYLIB" ] && check "dylib file exists" 0 || check "dylib file exists" 1

# Test: metallib exists alongside dylib
METALLIB="$(dirname "$DYLIB")/mlx.metallib"
[ -f "$METALLIB" ] && check "metallib exists" 0 || check "metallib exists" 1

# Test: help command
HELP_OUT=$(vllm-swift 2>&1 || true)
echo "$HELP_OUT" | grep -q "serve" && check "help shows serve" 0 || check "help shows serve" 1
echo "$HELP_OUT" | grep -q "update" && check "help shows update" 0 || check "help shows update" 1
echo "$HELP_OUT" | grep -q "download" && check "help shows download" 0 || check "help shows download" 1

# Test: setup creates venv
if [ -d "$HOME/.vllm-swift/venv" ]; then
    check "venv exists" 0

    # Test: plugin installed in venv
    "$HOME/.vllm-swift/venv/bin/python3" -c "from vllm_swift import register" 2>/dev/null \
        && check "plugin importable in venv" 0 || check "plugin importable in venv" 1

    # Test: vLLM installed in venv
    "$HOME/.vllm-swift/venv/bin/python3" -c "import vllm" 2>/dev/null \
        && check "vLLM importable in venv" 0 || check "vLLM importable in venv" 1

    # Test: plugin entry point registered
    EP=$("$HOME/.vllm-swift/venv/bin/python3" -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.platform_plugins')
swift_eps = [e for e in eps if 'swift' in e.name]
print(len(swift_eps))
" 2>/dev/null)
    [ "$EP" = "1" ] && check "plugin entry point registered" 0 || check "plugin entry point registered" 1
else
    check "venv exists" 1
    echo "  (run 'vllm-swift setup' first)"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" = "0" ] && exit 0 || exit 1
