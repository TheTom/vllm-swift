#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Sanity-check mlx.metallib has every critical TurboQuant kernel variant
# we ship support for. Run before publishing a bottle, and locally if you
# rebuilt the metallib by hand.
#
# Usage:
#   ./scripts/verify_metallib.sh [path/to/mlx.metallib]
#
# With no arg, defaults to the standard build-metallib.sh output path.

set -euo pipefail

DEFAULT_METALLIB="$(dirname "$0")/../swift/.build/checkouts/mlx-swift-lm/.build/arm64-apple-macosx/release/mlx.metallib"
METALLIB="${1:-$DEFAULT_METALLIB}"

if [ ! -f "$METALLIB" ]; then
    echo "ERROR: metallib not found at $METALLIB"
    echo "  Run: bash swift/.build/checkouts/mlx-swift-lm/scripts/build-metallib.sh release"
    exit 1
fi

SYMS=$(strings "$METALLIB")

FAIL=0

# Family-level coverage. min_count is set to the smallest expected
# (bits × dims × dtypes) cardinality for that family. Tighten as new
# kernels are added.
declare -a FAMILY_CHECKS=(
    "turbo_score                       24"
    "turbo_value                       24"
    "turbo_fused_encode                24"
    "turbo_fused_encode_wht            16"
    "turbo_dequant_rotated             48"
    "turbo_flash_p1                    60"
    "turbo_flash_p2                    6"
)

for entry in "${FAMILY_CHECKS[@]}"; do
    family=$(echo "$entry" | awk '{print $1}')
    min=$(echo "$entry" | awk '{print $2}')
    count=$(printf '%s\n' "$SYMS" | grep -c "^${family}_" || true)
    if [ "$count" -lt "$min" ]; then
        echo "FAIL: family $family — found $count symbols, expected ≥ $min"
        FAIL=1
    else
        echo "ok:   family $family — $count symbols"
    fi
done

# Per-head-dim spot-checks for the models we actively support. Add an
# entry every time a new head_dim joins the supported model matrix.
# Each entry is one canonical symbol from that head_dim's kernel set.
declare -a SPOT_CHECKS=(
    "turbo_dequant_rotated_4_64_bf16    Qwen2.5-1.5B/3B-class (head_dim=64)"
    "turbo_dequant_rotated_4_80_bf16    Qwen3-4B (head_dim=80)"
    "turbo_dequant_rotated_4_128_bf16   Qwen3-8B / Qwen2.5-7B (head_dim=128)"
    "turbo_dequant_rotated_4_256_bf16   Qwen3.5/3.6 27B-class (head_dim=256)"
    "turbo_dequant_rotated_4_512_bf16   Gemma 4 26B-A4B (head_dim=512)"
)

for entry in "${SPOT_CHECKS[@]}"; do
    sym=$(echo "$entry" | awk '{print $1}')
    label=$(echo "$entry" | sed -E 's/^[^[:space:]]+[[:space:]]+//')
    # grep -c (counts, no early exit) avoids pipefail+SIGPIPE when SYMS
    # is large. `grep -q` would close the pipe on first match, kill the
    # printf with SIGPIPE, and pipefail would mis-report it as a failure.
    count=$(printf '%s\n' "$SYMS" | grep -c "^${sym}$" || true)
    if [ "$count" -gt 0 ]; then
        echo "ok:   $sym — $label"
    else
        echo "FAIL: $sym missing — $label"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "Metallib verification failed."
    echo "Likely cause: stale .build artifacts. Try:"
    echo "  rm -rf swift/.build/checkouts/mlx-swift-lm/.build/arm64-apple-macosx"
    echo "  bash swift/.build/checkouts/mlx-swift-lm/scripts/build-metallib.sh release"
    echo "  ./scripts/verify_metallib.sh"
    exit 1
fi

echo ""
echo "All critical kernel symbols present in $METALLIB"
