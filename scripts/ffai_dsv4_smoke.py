#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# End-to-end smoke test for the vllm-swift → FFAI → metaltile path:
# Python ctypes → libVLLMBridge.dylib (C ABI) → Swift VLLMBridge →
# FFAI DSv4 engine → metaltile Metal kernels. Loads the GGUF DSv4-Flash
# model, prefills a tiny prompt, decodes N tokens, and times decode tps —
# exactly the call sequence vllm_swift/engine_bridge.py uses.
import ctypes
import os
import sys
import time

DYLIB = os.environ.get(
    "VLLM_SWIFT_METAL_LIB",
    os.path.expanduser("~/dev/vllm-swift/swift/.build/release/libVLLMBridge.dylib"),
)
MODEL = os.environ.get("DS4", os.path.expanduser("~/models/ds4-model"))
NDECODE = int(os.environ.get("NDECODE", "12"))

lib = ctypes.CDLL(DYLIB)

lib.vsm_engine_create.restype = ctypes.c_void_p
lib.vsm_engine_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p,
    ctypes.c_int32, ctypes.c_float, ctypes.c_int32,
]
lib.vsm_engine_vocab_size.restype = ctypes.c_int32
lib.vsm_engine_vocab_size.argtypes = [ctypes.c_void_p]
lib.vsm_engine_num_layers.restype = ctypes.c_int32
lib.vsm_engine_num_layers.argtypes = [ctypes.c_void_p]
lib.vsm_engine_prefill.restype = ctypes.c_int32
lib.vsm_engine_prefill.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ctypes.c_float, ctypes.c_float,
]
lib.vsm_engine_decode_step.restype = ctypes.c_int32
lib.vsm_engine_decode_step.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
lib.vsm_engine_get_logits.restype = ctypes.POINTER(ctypes.c_float)
lib.vsm_engine_get_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.vsm_engine_destroy.argtypes = [ctypes.c_void_p]

print(f"[smoke] dylib={DYLIB}")
print(f"[smoke] model={MODEL}")
os.environ.setdefault("FFAI_DSV4_GATHER", "1")

h = lib.vsm_engine_create(MODEL.encode(), b"f16", 0, None, 0, 0.0, 1)
if not h:
    print("[smoke] FAIL: vsm_engine_create returned null")
    sys.exit(1)
print(f"[smoke] engine handle={hex(h)}")
print(f"[smoke] vocab_size={lib.vsm_engine_vocab_size(h)} num_layers={lib.vsm_engine_num_layers(h)}")

# Prefill a small prompt (bos + a couple tokens).
prompt = [0, 100, 200, 300]
arr = (ctypes.c_int32 * len(prompt))(*prompt)
t0 = time.time()
first = lib.vsm_engine_prefill(h, arr, len(prompt), 0.0, 1.0)
print(f"[smoke] prefill({len(prompt)} toks) -> first token {first} in {time.time()-t0:.2f}s")

# verify get_logits returns a real buffer
vs = ctypes.c_int32(0)
lp = lib.vsm_engine_get_logits(h, ctypes.byref(vs))
if lp:
    print(f"[smoke] get_logits: vocab={vs.value}, logits[{first}]={lp[first]:.3f}")
else:
    print("[smoke] get_logits returned null")

# Decode loop + tps.
toks, times = [], []
for i in range(NDECODE):
    t = time.time()
    tok = lib.vsm_engine_decode_step(h, 0.0, 1.0)
    dt = time.time() - t
    toks.append(tok)
    if i > 0:
        times.append(dt)
    print(f"[smoke] decode {i}: token={tok} ({1.0/dt:.2f} tps)")

if times:
    mean = sum(times) / len(times)
    print(f"[smoke] sustained decode (tok 1+): {1.0/mean:.2f} tps")
print(f"[smoke] tokens: {toks}")

lib.vsm_engine_destroy(h)
print("[smoke] OK — vllm-swift → FFAI → metaltile end-to-end")
