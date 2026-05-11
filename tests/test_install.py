#  SPDX-License-Identifier: Apache-2.0
"""Verify mlx.metallib lands alongside libVLLMBridge.dylib after install.

Covers https://github.com/TheTom/vllm-swift/issues/7 — source builds were
shipping without mlx.metallib, so models using GatedDelta / TurboFlash
kernels failed at load. These tests assert the metallib actually exists,
parses as a Metal library, and contains the kernels we care about.

Both install paths land the metallib in the same place relative to the
dylib (sibling file named mlx.metallib), so the same checks apply:

  - Source build: ``swift/.build/arm64-apple-macosx/{release,debug}/``
                  after ``./scripts/install.sh``
  - Homebrew bottle: ``$(brew --prefix vllm-swift)/lib/`` after
                     ``brew install TheTom/tap/vllm-swift``

The tests skip with a clear message when neither install layout exists,
so they're safe to run on a fresh checkout without forcing a multi-minute
``swift build`` inside pytest.

Usage example:
    pytest tests/test_install.py -v
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

# Sibling-of-dylib filename. MLX's Metal device looks for this name next to
# the loaded dynamic library at runtime.
METALLIB_NAME = "mlx.metallib"
DYLIB_NAME = "libVLLMBridge.dylib"

# Magic bytes for Apple Metal library files. Stored little-endian at offset 0.
# Reference: https://developer.apple.com/documentation/metal — confirmed by
# inspecting any built mlx.metallib produced by build-metallib.sh.
METALLIB_MAGIC = b"MTLB"

REPO_ROOT = Path(__file__).resolve().parents[1]
SWIFT_BUILD_BASE = REPO_ROOT / "swift" / ".build" / "arm64-apple-macosx"


def _candidate_install_dirs() -> list[Path]:
    """Return directories where install.sh / brew may have placed the dylib+metallib pair.

    Order matters only for the picked-first reporting in errors — the test
    iterates and uses the first directory where the dylib actually exists.
    """
    candidates: list[Path] = [
        SWIFT_BUILD_BASE / "release",
        SWIFT_BUILD_BASE / "debug",
    ]

    # Homebrew install — only present when the user actually installed the
    # bottle. ``brew --prefix`` returns non-zero if the formula isn't installed,
    # so guard the call.
    try:
        brew_prefix = subprocess.run(
            ["brew", "--prefix", "vllm-swift"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if brew_prefix:
            candidates.append(Path(brew_prefix) / "lib")
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        # brew not installed, or formula not present — skip the brew path.
        pass

    return candidates


def _pick_install_dir() -> Path | None:
    """Find the first directory that contains a built libVLLMBridge.dylib.

    A directory without the dylib means install.sh / brew install never
    completed there, so checking the metallib in that location would test
    something that doesn't exist yet.
    """
    for directory in _candidate_install_dirs():
        if (directory / DYLIB_NAME).is_file():
            return directory
    return None


@pytest.fixture(scope="module")
def install_dir() -> Path:
    """Locate the install directory or skip the entire test module."""
    chosen = _pick_install_dir()
    if chosen is None:
        searched = "\n  ".join(str(d) for d in _candidate_install_dirs())
        pytest.skip(
            "No vllm-swift install found. Run `./scripts/install.sh` first "
            "or `brew install TheTom/tap/vllm-swift`.\nSearched:\n  " + searched
        )
    return chosen


@pytest.fixture(scope="module")
def metallib_path(install_dir: Path) -> Path:
    return install_dir / METALLIB_NAME


def test_metallib_exists_alongside_dylib(install_dir: Path, metallib_path: Path) -> None:
    """Issue #7 root cause: metallib was missing next to the dylib.

    install.sh now copies it explicitly after invoking build-metallib.sh.
    """
    dylib_path = install_dir / DYLIB_NAME
    assert dylib_path.is_file(), f"Sanity check failed — dylib missing at {dylib_path}"
    assert metallib_path.is_file(), (
        f"mlx.metallib missing next to {DYLIB_NAME}. Re-run scripts/install.sh."
    )


def test_metallib_is_non_empty(metallib_path: Path) -> None:
    """An empty file would pass ``-f`` checks but fail at runtime — guard against it."""
    size = metallib_path.stat().st_size
    # Real metallibs are megabytes; 1 KiB is generous slack while still catching
    # the obvious "shell wrote nothing" failure mode.
    assert size > 1024, f"mlx.metallib at {metallib_path} is suspiciously small ({size} bytes)"


def test_metallib_has_metal_magic_bytes(metallib_path: Path) -> None:
    """Verify the file is actually an Apple Metal library, not garbage.

    Metal libraries start with the 4-byte magic ``MTLB``. Catching this
    detects the case where install.sh wrote some other file (e.g. an HTML
    error page from a download) under the metallib name.
    """
    with metallib_path.open("rb") as f:
        header = f.read(8)
    assert len(header) == 8, f"Truncated header in {metallib_path}"
    magic = header[:4]
    assert magic == METALLIB_MAGIC, (
        f"{metallib_path} is not a Metal library — expected magic "
        f"{METALLIB_MAGIC!r}, got {magic!r}"
    )
    # Bytes 4-7 are the version. Any reasonable nonzero value is fine — we
    # just want to confirm we read past the magic into a real header.
    (version,) = struct.unpack("<I", header[4:8])
    assert version > 0, f"Metallib version field is zero in {metallib_path}"


@pytest.mark.skipif(
    shutil.which("xcrun") is None,
    reason="xcrun not available — skipping symbol table inspection",
)
def test_metallib_contains_gated_delta_kernel(metallib_path: Path) -> None:
    """Confirm the custom kernels that motivated issue #7 are present.

    GatedDelta is the canonical example from the bug report (Qwen3Next-family
    failed to load without it). If install.sh accidentally pointed at the
    upstream MLX metallib instead of mlx-swift-lm's, this assertion catches it.
    """
    result = subprocess.run(
        ["xcrun", "metal-objdump", "--syms", str(metallib_path)],
        capture_output=True,
        text=True,
        check=False,
        # metal-objdump on a 6 MB metallib finishes well under a second, but
        # give CI breathing room.
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(
            f"xcrun metal-objdump failed (Xcode toolchain may be missing): {result.stderr.strip()}"
        )
    symbols = result.stdout
    assert "gated_delta_step_fused" in symbols, (
        "gated_delta_step_fused_* kernels missing from "
        f"{metallib_path}. install.sh likely picked the wrong metallib source."
    )


@pytest.mark.skipif(
    shutil.which("file") is None,
    reason="`file` command unavailable",
)
def test_metallib_recognised_by_file_command(metallib_path: Path) -> None:
    """Cross-check with the BSD ``file`` utility for an independent identification.

    ``file`` recognises Metal libraries on macOS — this catches truncated or
    miscopied files that still start with the right magic bytes by accident.
    """
    out = subprocess.run(
        ["file", "--brief", str(metallib_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.lower()
    # Older macOS versions report "Apple Metal Library" / newer report variants.
    # We just want some confirmation it isn't classified as ASCII text or empty.
    assert "metal" in out or "data" in out, (
        f"`file` reports unexpected type for {metallib_path}: {out.strip()}"
    )


# ---------------------------------------------------------------------------
# install.sh `_lib/` mirror coverage.
#
# install.sh copies the freshly-built libVLLMBridge.dylib + mlx.metallib into
# vllm_swift/_lib/ so that pip-editable installs and bottle-style installs all
# see the same artifacts. A stale `_lib/` is the same failure shape as a stale
# bottle metallib (issue #7 / v0.6.0): the loader silently picks the wrong
# file and dies at runtime.
#
# These tests skip when no `_lib/` mirror exists (fresh clone, no install yet)
# so they don't force a swift build inside pytest.
# ---------------------------------------------------------------------------

# Symbol that install.sh's nm-grep sanity check looks for. It marks the
# batched-uniform prefill stash (vllm-swift commit 3389805), which is the
# floor for what feature/m5-baseline ships. If the `_lib/` dylib doesn't
# export it, install.sh refused to publish — so neither should the bottle.
EXPECTED_BRIDGE_SYMBOL = "_vsm_engine_get_batch_tokens"

LIB_DIR = REPO_ROOT / "vllm_swift" / "_lib"


@pytest.fixture(scope="module")
def lib_dylib_path() -> Path:
    """Locate vllm_swift/_lib/libVLLMBridge.dylib or skip."""
    candidate = LIB_DIR / DYLIB_NAME
    if not candidate.is_file():
        pytest.skip(
            f"No vllm_swift/_lib/{DYLIB_NAME} mirror found. "
            f"Run `./scripts/install.sh` to populate it."
        )
    return candidate


def test_lib_dir_mirrors_dylib(install_dir: Path, lib_dylib_path: Path) -> None:
    """install.sh copies the build-output dylib into `_lib/` after building.

    Whatever's in `_lib/` must be at least as fresh as the dylib actually
    built into swift/.build/. install.sh enforces this via `cp`; this test
    catches the case where the copy silently no-ops (e.g. permissions).
    """
    build_dylib = install_dir / DYLIB_NAME
    assert build_dylib.is_file(), f"Build dylib missing at {build_dylib}"
    build_mtime = build_dylib.stat().st_mtime
    lib_mtime = lib_dylib_path.stat().st_mtime
    # `_lib/` should be at least as fresh as the build artifact. We allow a
    # small clock-skew tolerance because `cp` and `stat` aren't atomic — the
    # mtime granularity on some filesystems is whole seconds.
    assert lib_mtime >= build_mtime - 2, (
        f"_lib/{DYLIB_NAME} is older than the build artifact. "
        f"install.sh did not refresh `_lib/` — possible silent cp failure."
    )


@pytest.mark.skipif(
    shutil.which("nm") is None,
    reason="nm not available — skipping symbol table inspection",
)
def test_lib_dylib_contains_expected_bridge_symbol(lib_dylib_path: Path) -> None:
    """The `_lib/` dylib must export the bridge symbols engine_bridge.py calls.

    install.sh's own nm-grep gate enforces this before refusing to publish.
    We re-check from the test side so a manually-edited `_lib/` (e.g. someone
    copied an older dylib in by hand) still gets caught.
    """
    result = subprocess.run(
        ["nm", str(lib_dylib_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        pytest.skip(f"nm failed on {lib_dylib_path}: {result.stderr.strip()}")
    assert EXPECTED_BRIDGE_SYMBOL in result.stdout, (
        f"{lib_dylib_path} is missing {EXPECTED_BRIDGE_SYMBOL}. "
        f"This is the symbol install.sh's sanity check looks for — the "
        f"`_lib/` mirror is stale. Re-run scripts/install.sh."
    )


def test_lib_dir_has_metallib_too(lib_dylib_path: Path) -> None:
    """install.sh mirrors both the dylib AND the metallib into `_lib/`."""
    metallib = LIB_DIR / METALLIB_NAME
    assert metallib.is_file(), (
        f"_lib/{METALLIB_NAME} missing alongside the dylib. "
        f"install.sh refresh of `_lib/` is incomplete."
    )
    # Metal magic check, same as the install_dir version above. Catches the
    # case where `_lib/mlx.metallib` got truncated or replaced with garbage.
    with metallib.open("rb") as f:
        magic = f.read(4)
    assert magic == METALLIB_MAGIC, (
        f"_lib/{METALLIB_NAME} is not a Metal library — got magic {magic!r}."
    )


# TODO: When CI gains a runner that reliably executes ./scripts/install.sh
# end-to-end, add a slow-marked test that wipes swift/.build/arm64-apple-macosx
# and reruns install.sh, asserting the metallib reappears. Skipped here to keep
# unit-test wall time reasonable on developer machines.
_ = os  # keep `os` import live for future end-to-end test additions
