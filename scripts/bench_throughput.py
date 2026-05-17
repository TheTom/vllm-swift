#!/usr/bin/env python3
"""Concurrent decode throughput benchmark for vllm-swift bridge.

Measures total output tok/s at various concurrency levels using the
Swift bridge directly (no vLLM scheduler overhead).

Usage:
  DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/release \
    python3 scripts/bench_throughput.py [model_path] [--tokens N]
"""

import ctypes
import os
import sys
import time

from pathlib import Path

# Parse args
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.expanduser("~/models/Qwen3-4B-4bit")
MAX_TOKENS = 50
PROMPT_TOKENS = 0  # 0 = use default short prompt; >0 = pad to N tokens
KV_SCHEME = ""  # "" = fp16 raw, "turbo4" / "turbo8" / etc — enables compression
PREFILL_MODE = "sequential"  # or "batched"
PROMPTS_MODE = "identical"  # or "unique" — identical measures prefix-cache-hit
                            # speed for caching engines (chat/completion-style),
                            # unique measures raw recompute speed (independent
                            # request serving). vllm-metal's prefix cache hides
                            # ~95% of prefill compute when prompts repeat.
for i, arg in enumerate(sys.argv):
    if arg == "--tokens" and i + 1 < len(sys.argv):
        MAX_TOKENS = int(sys.argv[i + 1])
    if arg == "--prompt-tokens" and i + 1 < len(sys.argv):
        PROMPT_TOKENS = int(sys.argv[i + 1])
    if arg == "--prefill-mode" and i + 1 < len(sys.argv):
        PREFILL_MODE = sys.argv[i + 1]
        assert PREFILL_MODE in ("sequential", "batched"), PREFILL_MODE
    if arg == "--prompts" and i + 1 < len(sys.argv):
        PROMPTS_MODE = sys.argv[i + 1]
        assert PROMPTS_MODE in ("identical", "unique"), PROMPTS_MODE
    if arg == "--kv-scheme" and i + 1 < len(sys.argv):
        KV_SCHEME = sys.argv[i + 1]

# BENCH_B env var overrides the default sweep with a single B (used by
# task #126 sparse-decode characterization where B>1 would OOM at 128K).
if os.environ.get("BENCH_B"):
    CONCURRENCY_LEVELS = [int(os.environ["BENCH_B"])]
else:
    CONCURRENCY_LEVELS = [1, 8, 32, 64]

# Load dylib
SWIFT_BUILD = Path(__file__).parent.parent / "swift" / ".build" / "arm64-apple-macosx"
for config in ["release", "debug"]:
    candidate = SWIFT_BUILD / config / "libVLLMBridge.dylib"
    if candidate.exists():
        LIB_PATH = str(candidate)
        break
else:
    LIB_PATH = os.environ.get("VLLM_SWIFT_METAL_LIB", "")

if not os.path.exists(LIB_PATH):
    print(f"ERROR: dylib not found. Build first: cd swift && swift build -c release")
    sys.exit(1)

lib = ctypes.CDLL(LIB_PATH)

# Bind C API
lib.vsm_engine_create.restype = ctypes.c_void_p
lib.vsm_engine_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.c_char_p, ctypes.c_int32, ctypes.c_float,
]
lib.vsm_engine_prefill_req.restype = ctypes.c_int32
lib.vsm_engine_prefill_req.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ctypes.c_float, ctypes.c_float,
]
lib.vsm_engine_decode_all.restype = ctypes.c_int32
lib.vsm_engine_decode_all.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
]
lib.vsm_engine_init_batched.restype = ctypes.c_int32
lib.vsm_engine_init_batched.argtypes = [ctypes.c_void_p]
lib.vsm_engine_prefill_batched_uniform.restype = ctypes.c_int32
lib.vsm_engine_prefill_batched_uniform.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_float, ctypes.c_float,
]
lib.vsm_engine_finish_req.restype = None
lib.vsm_engine_finish_req.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.vsm_engine_reset.restype = None
lib.vsm_engine_reset.argtypes = [ctypes.c_void_p]
lib.vsm_engine_destroy.restype = None
lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

# Load tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

# Create engine
print(f"Model: {MODEL_PATH}")
print(f"Tokens per request: {MAX_TOKENS}")
print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
print()

print("Loading model...")
t0 = time.perf_counter()
kv_scheme_arg = KV_SCHEME.encode() if KV_SCHEME else None
engine = lib.vsm_engine_create(MODEL_PATH.encode(), b"float16", 0, kv_scheme_arg, 0, 0.9)
if KV_SCHEME:
    print(f"KV scheme: {KV_SCHEME}")
load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.1f}s")

if not engine:
    print("FAILED to create engine")
    sys.exit(1)

# Build prompt pool. `identical` mode duplicates one prompt B times — measures
# cache-hit speed for engines with prefix caching. `unique` mode uses B distinct
# prompts — measures raw recompute speed.
SEEDS = [
    "Explain the theory of relativity in detail, covering both special and general relativity:",
    "Describe quantum mechanics, including wave-particle duality and the uncertainty principle:",
    "Walk through the proof of the Pythagorean theorem step by step:",
    "Summarize the causes and consequences of the French Revolution:",
    "Outline how a transformer language model processes a sentence end to end:",
    "Explain how photosynthesis converts sunlight into chemical energy:",
    "Compare and contrast classical conditioning and operant conditioning:",
    "Describe how plate tectonics shapes mountain ranges over geologic time:",
]


def build_prompt_ids(seed_text):
    ids = tok.encode(seed_text)
    if PROMPT_TOKENS > 0:
        while len(ids) < PROMPT_TOKENS:
            ids.extend(ids[: min(len(ids), PROMPT_TOKENS - len(ids))])
        ids = ids[:PROMPT_TOKENS]
    return ids


# For `identical` mode, build a single prompt and reuse arr for every request.
# For `unique` mode, build per-slot prompts at decode-loop time below.
default_seed = SEEDS[0]
input_ids = build_prompt_ids(default_seed)
arr = (ctypes.c_int32 * len(input_ids))(*input_ids)

print(f"Prompt: {len(input_ids)} tokens, prompts mode: {PROMPTS_MODE}")
print()

results = []

for B in CONCURRENCY_LEVELS:
    # Reset engine state
    lib.vsm_engine_reset(engine)

    # Per-slot prompt arrays. `identical` mode reuses one arr for every slot;
    # `unique` mode cycles SEEDS so each slot gets a distinct prefix (any
    # prefix-cache hit between slots is impossible by construction).
    if PROMPTS_MODE == "identical":
        per_slot_ids = [input_ids] * B
        per_slot_arr = [arr] * B
    else:  # unique
        per_slot_ids = [build_prompt_ids(SEEDS[i % len(SEEDS)]) for i in range(B)]
        per_slot_arr = [(ctypes.c_int32 * len(ids))(*ids) for ids in per_slot_ids]

    # All slots same length (PROMPT_TOKENS pad guarantees it).
    T = len(per_slot_ids[0])

    # Prefill all requests — TIMED so we can report end-to-end alongside decode-only
    t_prefill = time.perf_counter()
    if PREFILL_MODE == "sequential":
        for i in range(B):
            rid = f"req-{i}".encode()
            lib.vsm_engine_prefill_req(engine, rid, per_slot_arr[i], T, 0.0, 1.0)
        # init_batched at every B (including B=1) so the fully-batched
        # decode path runs and stays bandwidth-bound at large models —
        # the BatchedKVCache pre-allocation overhead is acceptable in
        # exchange for ~5% B=1 throughput at Qwen2.5-14B.
        lib.vsm_engine_init_batched(engine)
    else:  # batched
        rids = [f"req-{i}".encode() for i in range(B)]
        rid_arr = (ctypes.c_char_p * B)(*rids)
        flat = (ctypes.c_int32 * (B * T))()
        for i in range(B):
            for j, t in enumerate(per_slot_ids[i]):
                flat[i * T + j] = t
        rc = lib.vsm_engine_prefill_batched_uniform(
            engine, rid_arr, flat, B, T, 0.0, 1.0
        )
        assert rc == 0, f"prefill_batched_uniform returned {rc}"
    prefill_elapsed = time.perf_counter() - t_prefill

    # Decode loop
    req_ids_buf = (ctypes.c_char_p * (B + 1))()
    tokens_buf = (ctypes.c_int32 * (B + 1))()
    total_tokens = 0

    # Warmup (2 steps)
    for _ in range(2):
        lib.vsm_engine_decode_all(engine, req_ids_buf, tokens_buf, B + 1)

    # Timed run — decode only
    t0 = time.perf_counter()
    for step in range(MAX_TOKENS):
        n = lib.vsm_engine_decode_all(engine, req_ids_buf, tokens_buf, B + 1)
        total_tokens += n
    elapsed = time.perf_counter() - t0

    tps_decode = total_tokens / elapsed if elapsed > 0 else 0
    # End-to-end mirrors how vllm-metal's bench measures (prefill+decode/total_tokens).
    # Includes prefill cost so it's apples-to-apples with that bench's headline number.
    tps_e2e = total_tokens / (prefill_elapsed + elapsed) if (prefill_elapsed + elapsed) > 0 else 0
    per_req = tps_decode / B if B > 0 else 0

    print(
        f"B={B:3d}: e2e={tps_e2e:,.1f} decode={tps_decode:,.1f} tok/s "
        f"[prefill={prefill_elapsed*1000:.0f}ms, decode={elapsed:.2f}s]"
    )
    results.append((B, tps_e2e, tps_decode, prefill_elapsed))

    # Cleanup requests
    for i in range(B):
        lib.vsm_engine_finish_req(engine, f"req-{i}".encode())

lib.vsm_engine_destroy(engine)

# Summary table
print()
print(f"=== {os.path.basename(MODEL_PATH)} — vllm-swift bridge direct ===")
print(f"{'B':>4} {'E2E tok/s':>11} {'Decode tok/s':>13} {'Prefill ms':>11}")
print("-" * 44)
for B, tps_e2e, tps_decode, prefill in results:
    print(f"{B:>4d} {tps_e2e:>11,.1f} {tps_decode:>13,.1f} {prefill*1000:>11.0f}")
