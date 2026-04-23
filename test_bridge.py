#!/usr/bin/env python3
"""Quick smoke test for the Swift Metal bridge.

Usage:
  cd swift && swift build
  DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/debug python3 test_bridge.py
"""

import ctypes
import os
import time

from pathlib import Path

# Find the dylib
SWIFT_BUILD = Path(__file__).parent / "swift" / ".build" / "arm64-apple-macosx" / "debug"
LIB_PATH = str(SWIFT_BUILD / "libVLLMBridge.dylib")

if not os.path.exists(LIB_PATH):
    # Try the copied location
    LIB_PATH = str(Path(__file__).parent / "swift" / "libvllm_swift_metal.dylib")

lib = ctypes.CDLL(LIB_PATH)

# Bind C API
lib.vsm_engine_create.restype = ctypes.c_void_p
lib.vsm_engine_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.c_char_p, ctypes.c_int32, ctypes.c_float,
]
lib.vsm_engine_prefill.restype = ctypes.c_int32
lib.vsm_engine_prefill.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32, ctypes.c_float, ctypes.c_float,
]
lib.vsm_engine_decode_step.restype = ctypes.c_int32
lib.vsm_engine_decode_step.argtypes = [
    ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
]
lib.vsm_engine_destroy.restype = None
lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

# Default model path — override with MODEL_PATH env var
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen3-4B-4bit"))

print(f"Model: {MODEL_PATH}")
print("Creating engine...")
t0 = time.perf_counter()
engine = lib.vsm_engine_create(MODEL_PATH.encode(), b"float16", 0, None, 0, 0.9)
t1 = time.perf_counter()
print(f"Engine created in {t1 - t0:.2f}s")

if not engine:
    print("FAILED to create engine")
    exit(1)

# Tokenize with transformers
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt = "Explain quantum computing in simple terms:"
input_ids = tok.encode(prompt)
print(f'Prompt: "{prompt}" ({len(input_ids)} tokens)')

arr = (ctypes.c_int32 * len(input_ids))(*input_ids)

# Prefill
t0 = time.perf_counter()
first_token = lib.vsm_engine_prefill(engine, arr, len(input_ids), 0.0, 1.0)
t1 = time.perf_counter()
prefill_tps = len(input_ids) / (t1 - t0)
print(f"Prefill: {(t1 - t0) * 1000:.1f}ms ({prefill_tps:.1f} tok/s)")

# Decode
all_tokens = [first_token]
t0 = time.perf_counter()
for _ in range(100):
    tok_id = lib.vsm_engine_decode_step(engine, 0.0, 1.0)
    if tok_id < 0:
        break
    all_tokens.append(tok_id)
t1 = time.perf_counter()

n = len(all_tokens)
elapsed = t1 - t0
decode_tps = n / elapsed if elapsed > 0 else 0

print(f"\nDecode: {n} tokens in {elapsed * 1000:.1f}ms = {decode_tps:.1f} tok/s")
print(f"\nOutput:\n{tok.decode(all_tokens, skip_special_tokens=True)}")

lib.vsm_engine_destroy(engine)
