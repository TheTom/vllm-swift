#!/usr/bin/env python3
"""Concurrent decode throughput benchmark for vllm-metal (Python/MLX).

Measures total output tok/s at various concurrency levels using vllm-metal's
Python/MLX inference path via the vLLM offline LLM API.

Usage:
  cd /Users/tom/dev/vllm-metal
  source .venv-vllm-metal/bin/activate
  python3 /Users/tom/dev/vllm-swift/scripts/bench_vllm_metal.py [model_path]
"""

import gc
import os
import sys
import time

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3-4B-4bit")
MAX_TOKENS = 50
CONCURRENCY_LEVELS = [1, 8, 32, 64]

print(f"Model: {MODEL_PATH}")
print(f"Max tokens: {MAX_TOKENS}")
print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
print()

from vllm import LLM, SamplingParams

prompt = "Explain the theory of relativity in detail, covering both special and general relativity:"
results = []

for B in CONCURRENCY_LEVELS:
    print(f"\n--- B={B} ---")
    # Create fresh engine per concurrency level to avoid subprocess issues
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        disable_log_stats=True,
    )

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    prompts = [prompt] * B

    # Warmup
    llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=5))

    # Timed run
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    per_req = tps / B if B > 0 else 0

    print(f"B={B:3d}: {tps:,.1f} tok/s total ({per_req:,.1f} per request) [{elapsed:.2f}s, {total_tokens} tokens]")
    results.append((B, tps, per_req))

    # Cleanup to free GPU memory
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    time.sleep(1)

# Summary table
print()
print(f"=== {os.path.basename(MODEL_PATH)} — vllm-metal (Python/MLX) ===")
print(f"{'Concurrency':>12} {'Total tok/s':>14} {'Per-request':>14}")
print("-" * 44)
for B, tps, per_req in results:
    print(f"{B:>12d} {tps:>14,.1f} {per_req:>14,.1f}")
