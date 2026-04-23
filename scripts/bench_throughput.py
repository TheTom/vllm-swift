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
for i, arg in enumerate(sys.argv):
    if arg == "--tokens" and i + 1 < len(sys.argv):
        MAX_TOKENS = int(sys.argv[i + 1])

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
engine = lib.vsm_engine_create(MODEL_PATH.encode(), b"float16", 0, None, 0, 0.9)
load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.1f}s")

if not engine:
    print("FAILED to create engine")
    sys.exit(1)

prompt = "Explain the theory of relativity in detail, covering both special and general relativity:"
input_ids = tok.encode(prompt)
arr = (ctypes.c_int32 * len(input_ids))(*input_ids)

print(f"Prompt: {len(input_ids)} tokens")
print()

results = []

for B in CONCURRENCY_LEVELS:
    # Reset engine state
    lib.vsm_engine_reset(engine)

    # Prefill all requests
    for i in range(B):
        rid = f"req-{i}".encode()
        lib.vsm_engine_prefill_req(engine, rid, arr, len(input_ids), 0.0, 1.0)

    # Init batched KV cache
    lib.vsm_engine_init_batched(engine)

    # Decode loop
    req_ids_buf = (ctypes.c_char_p * (B + 1))()
    tokens_buf = (ctypes.c_int32 * (B + 1))()
    total_tokens = 0

    # Warmup (2 steps)
    for _ in range(2):
        lib.vsm_engine_decode_all(engine, req_ids_buf, tokens_buf, B + 1)

    # Timed run
    t0 = time.perf_counter()
    for step in range(MAX_TOKENS):
        n = lib.vsm_engine_decode_all(engine, req_ids_buf, tokens_buf, B + 1)
        total_tokens += n
    elapsed = time.perf_counter() - t0

    tps = total_tokens / elapsed if elapsed > 0 else 0
    per_req = tps / B if B > 0 else 0

    print(f"B={B:3d}: {tps:,.1f} tok/s total ({per_req:,.1f} per request) [{elapsed:.2f}s]")
    results.append((B, tps, per_req))

    # Cleanup requests
    for i in range(B):
        lib.vsm_engine_finish_req(engine, f"req-{i}".encode())

lib.vsm_engine_destroy(engine)

# Summary table
print()
print(f"=== {os.path.basename(MODEL_PATH)} — vllm-swift bridge direct ===")
print(f"{'Concurrency':>12} {'Total tok/s':>14} {'Per-request':>14}")
print("-" * 44)
for B, tps, per_req in results:
    print(f"{B:>12d} {tps:>14,.1f} {per_req:>14,.1f}")
