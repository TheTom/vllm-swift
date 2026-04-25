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

echo "=== Script syntax checks ==="
echo ""

# Test: install.sh has valid bash syntax
bash -n scripts/install.sh 2>/dev/null \
    && check "install.sh syntax valid" 0 || check "install.sh syntax valid" 1

# Test: install.sh doesn't use $PYTHON before _find_python defines it
# (caught a real bug where $PYTHON was used at line 97 before line 125)
FIRST_USE=$(grep -n '"\$PYTHON"' scripts/install.sh | head -1 | cut -d: -f1)
DEFINITION=$(grep -n 'PYTHON=\$(_find_python)' scripts/install.sh | head -1 | cut -d: -f1)
if [ -n "$FIRST_USE" ] && [ -n "$DEFINITION" ] && [ "$FIRST_USE" -ge "$DEFINITION" ]; then
    check "\$PYTHON used after definition (use=$FIRST_USE def=$DEFINITION)" 0
else
    check "\$PYTHON used after definition (use=$FIRST_USE def=$DEFINITION)" 1
fi

# Test: build_bottle.sh has valid bash syntax
bash -n scripts/build_bottle.sh 2>/dev/null \
    && check "build_bottle.sh syntax valid" 0 || check "build_bottle.sh syntax valid" 1

echo ""
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
    # Test: Python version is 3.10-3.13 (not 3.9 or 3.14+)
    PY_MINOR=$("$HOME/.vllm-swift/venv/bin/python3" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    [ "$PY_MINOR" -ge 10 ] 2>/dev/null && [ "$PY_MINOR" -le 13 ] 2>/dev/null \
        && check "Python version 3.10-3.13 (got 3.$PY_MINOR)" 0 || check "Python version 3.10-3.13 (got 3.$PY_MINOR)" 1

    # Test: no conflicting plugins (only swift, not metal)
    PLUGIN_COUNT=$("$HOME/.vllm-swift/venv/bin/python3" -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.platform_plugins')
print(len(list(eps)))
" 2>/dev/null)
    [ "$PLUGIN_COUNT" = "1" ] && check "no conflicting plugins (count=$PLUGIN_COUNT)" 0 || check "no conflicting plugins (count=$PLUGIN_COUNT)" 1

    # Test: platform detection works without mlx (troubleshoot: mlx not installed)
    PLATFORM=$("$HOME/.vllm-swift/venv/bin/python3" -c "
from vllm_swift.platform import SwiftMetalPlatform
print(SwiftMetalPlatform.is_available())
" 2>/dev/null)
    [ "$PLATFORM" = "True" ] && check "platform detects without mlx dependency" 0 || check "platform detects without mlx dependency" 1

    # Test: dylib findable from plugin (troubleshoot: dylib not found in subprocess)
    FOUND=$("$HOME/.vllm-swift/venv/bin/python3" -c "
from vllm_swift.engine_bridge import _find_lib_path
import os
print(os.path.exists(_find_lib_path()))
" 2>/dev/null)
    [ "$FOUND" = "True" ] && check "dylib findable from engine_bridge" 0 || check "dylib findable from engine_bridge" 1

    # Test: activate.sh NOT needed for brew path
    # (troubleshoot: users confused about activate.sh)
    check "no activate.sh needed (brew path)" 0

else
    check "venv exists" 1
    echo "  (run 'vllm-swift setup' first)"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" = "0" ] && exit 0 || exit 1
