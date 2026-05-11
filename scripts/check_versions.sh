#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Verify every version-bearing file agrees on a single vllm-swift version.
#
# Runs as a pre-flight from build_wheel.sh, build_bottle.sh, and any release
# script — refuses to continue if the bottle, the wheel, the Python plugin's
# __version__, and the Homebrew formula don't all match. Catches the failure
# mode where v0.6.1's bottle and wheel could have been published with a
# stale __init__.py reporting version 0.5.4.
#
# Usage:
#   ./scripts/check_versions.sh          # require agreement, print version
#   ./scripts/check_versions.sh --quiet  # exit code only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
QUIET=0
if [ "${1:-}" = "--quiet" ]; then
    QUIET=1
fi

# Source of truth: pyproject.toml. Everything else must match it.
pyproject_version=$(grep -E '^version = ' "$REPO_ROOT/pyproject.toml" | head -1 \
    | sed -E 's/^version = "([^"]+)"/\1/')
init_version=$(grep -E '^__version__ = ' "$REPO_ROOT/vllm_swift/__init__.py" | head -1 \
    | sed -E 's/^__version__ = "([^"]+)"/\1/')
bottle_version=$(grep -E '^VERSION=' "$REPO_ROOT/scripts/build_bottle.sh" | head -1 \
    | sed -E 's/^VERSION="([^"]+)"/\1/')
formula_version=$(grep -E '^  version ' "$REPO_ROOT/homebrew/vllm-swift.rb" | head -1 \
    | sed -E 's/^  version "([^"]+)"/\1/')

declare -a CHECKS=(
    "pyproject.toml                  $pyproject_version"
    "vllm_swift/__init__.py          $init_version"
    "scripts/build_bottle.sh         $bottle_version"
    "homebrew/vllm-swift.rb          $formula_version"
)

FAIL=0
for entry in "${CHECKS[@]}"; do
    file=$(echo "$entry" | awk '{print $1}')
    ver=$(echo "$entry" | awk '{print $2}')
    if [ "$ver" != "$pyproject_version" ]; then
        echo "FAIL: $file reports $ver (expected $pyproject_version)"
        FAIL=1
    elif [ "$QUIET" -eq 0 ]; then
        echo "ok:   $file = $ver"
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "Version drift detected. pyproject.toml is the source of truth — bump"
    echo "every file above to $pyproject_version (or bump pyproject.toml to the"
    echo "intended target and re-run)."
    exit 1
fi

if [ "$QUIET" -eq 0 ]; then
    echo ""
    echo "All version-bearing files agree on $pyproject_version."
fi
