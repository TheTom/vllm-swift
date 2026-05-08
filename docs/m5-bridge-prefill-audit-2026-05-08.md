# vllm-swift Bridge Sequential Prefill Bug — Audit

**Date:** 2026-05-08
**Branch:** `feature/m5-baseline`
**Phase:** M5 Max Long-Context Plan, Phase 1 (days 4-7)

## Problem statement (from memory: `project_vllm_swift_sequential_prefill.md`)

> Bridge does B sequential `model()` calls. 20-33× slower than vllm-metal at
> long-ctx high-B. Internal finding, not public correction.

This audit pinpoints the exact code path where the sequential dispatch
happens, identifies what's already in place vs. what's missing, and
sketches the fix.

## Where the sequential pattern lives

### Python side: `vllm_swift/worker.py:239-275`

```python
# Handle new requests (prefill) — each gets its own session
for new_req in scheduler_output.scheduled_new_reqs:
    req_id = new_req.req_id
    prompt_tokens = list(new_req.prompt_token_ids)
    ...
    first_token = self.engine.prefill_req(
        req_id, prompt_tokens, temperature=temp, top_p=top_p
    )
```

For B new requests in a single scheduler step, this loops calling
`prefill_req` B times **sequentially**. Each call enters Swift's
`vsm_engine_prefill_req`, acquires `engineQueue.sync`, instantiates a fresh
`TokenIterator`, and does one forward pass through the model.

**Result at B=64, T=2048: ~23-27 s.** Each per-request prefill is
compute-bound on the per-request forward; B of them serialized through
`engineQueue` is exactly B× the cost of one.

### Swift side: `swift/Sources/VLLMBridge/Bridge.swift:340-385`

```swift
@_cdecl("vsm_engine_prefill_req")
public func vsm_engine_prefill_req(...) -> Int32 {
    return engineQueue.sync { () -> Int32 in
        ...
        var iterator = try TokenIterator(
            input: input,
            model: engine.model,
            parameters: params
        )
        guard let firstToken = iterator.next() else { return -1 }
        ...
    }
}
```

`engineQueue.sync` enforces serialization. Even if Python launched
`prefill_req` calls concurrently, they would queue up here. So the bug is
not just upper-layer iteration — the Swift API itself is single-request.

## What's already fixed in Swift but unused from Python

The batched-uniform prefill path is already implemented in Swift and
referenced by tests, but **has no Python binding**:

### Existing Swift entry: `Bridge.swift:1349-1407`

```swift
/// Replaces the sequential pattern of `B × prefill_req` + `init_batched`
/// with a single `[B, T]` forward through the model. The mlx-lm Python
/// equivalent and vllm-swift sequential both take ~23-27s for B=64/T=2048
/// on 4B (compute-bound on per-request prefill). This path collapses the
/// 64 sequential forwards into one batched forward.
@_cdecl("vsm_engine_prefill_batched_uniform")
public func vsm_engine_prefill_batched_uniform(
    _ handle: UnsafeMutableRawPointer?,
    reqIds: UnsafePointer<UnsafePointer<CChar>?>?,
    promptTokens: UnsafePointer<Int32>?,
    numReqs: Int32,
    promptLen: Int32,
    temperature: Float,
    topP: Float
) -> Int32 {
    ...
    let inputBatch = MLXArray(tokens).reshaped(B, T)
    if let qwenModel = engine.model as? Qwen3Model {
        return prefillBatchedUniformQwen3(...)
    } else if let hybridModel = engine.model as? any BatchedHybridLLM {
        return prefillBatchedUniformHybrid(...)
    }
    return -2
}
```

Two model-family-specific implementations exist downstream:
- `prefillBatchedUniformQwen3` — dense all-attention models (line 1412+)
- `prefillBatchedUniformHybrid` — attention + GDN hybrid models (line 1525+)

**Constraint flagged in comment (Bridge.swift:1357):** "All requests use
the same prompt length T (variable-length deferred to M4)."

Test-only sequential variant exists for equivalence checking:
`vsm_engine_prefill_seq_uniform_topk` (line 1215) → captures top-K logits
for direct comparison vs. `prefill_batched_uniform`.

### What's missing in Python

`vllm_swift/engine_bridge.py` does **not** bind `vsm_engine_prefill_batched_uniform`.
Grep confirms only these prefill bindings:

```
engine_bridge.py:122:  _lib.vsm_engine_prefill_req.restype  = ctypes.c_int32
engine_bridge.py:230:  def prefill(...)            # → vsm_engine_prefill (single-req default)
engine_bridge.py:250:  def prefill_req(...)        # → vsm_engine_prefill_req
engine_bridge.py:347:  return self._lib.vsm_engine_prefill_req(...)
```

No reference to `prefill_batched_uniform` exists Python-side.

## The fix

Three layers, sequenced:

### Layer 1 — Python ctypes binding

Add to `engine_bridge.py` (around line 130, alongside the existing
`vsm_engine_prefill_req` binding):

```python
_lib.vsm_engine_prefill_batched_uniform.restype = ctypes.c_int32
_lib.vsm_engine_prefill_batched_uniform.argtypes = [
    ctypes.c_void_p,                                   # handle
    ctypes.POINTER(ctypes.c_char_p),                   # reqIds (NULL-terminated array)
    ctypes.POINTER(ctypes.c_int32),                    # promptTokens (flattened B*T)
    ctypes.c_int32,                                    # numReqs
    ctypes.c_int32,                                    # promptLen
    ctypes.c_float,                                    # temperature
    ctypes.c_float,                                    # topP
]
```

Plus a Python wrapper method on the engine class:

```python
def prefill_batched_uniform(
    self, req_ids: list[str], prompt_tokens: list[int],
    prompt_len: int, temperature: float, top_p: float,
) -> int:
    n = len(req_ids)
    arr = (ctypes.c_int32 * (n * prompt_len))(*prompt_tokens)
    rid_ptrs = (ctypes.c_char_p * n)(*[r.encode() for r in req_ids])
    return self._lib.vsm_engine_prefill_batched_uniform(
        self._handle, rid_ptrs, arr, n, prompt_len, temperature, top_p
    )
```

### Layer 2 — Worker dispatch

Modify `worker.py:239-275` to group new requests by prompt length and use
the batched path when possible:

```python
# Group new requests by prompt length
from collections import defaultdict
by_len: dict[int, list] = defaultdict(list)
singletons: list = []
vlm_reqs: list = []
for new_req in scheduler_output.scheduled_new_reqs:
    if has_multimodal(new_req):
        vlm_reqs.append(new_req)
    else:
        by_len[len(new_req.prompt_token_ids)].append(new_req)

# Batched path: groups of >= 2 with same prompt length
for plen, reqs in by_len.items():
    if len(reqs) >= 2:
        # Common temperature and top_p assumed; if they differ, fall back
        if all_same_sampling_params(reqs):
            flat_tokens = [t for r in reqs for t in r.prompt_token_ids]
            req_ids = [r.req_id for r in reqs]
            self.engine.prefill_batched_uniform(
                req_ids, flat_tokens, plen, temp=reqs[0].temp, top_p=reqs[0].top_p
            )
            # Pull first tokens via existing logits/sampling API
            ...
        else:
            singletons.extend(reqs)
    else:
        singletons.extend(reqs)

# Singletons + VLM keep the per-request loop
for new_req in singletons:
    first_token = self.engine.prefill_req(...)
for new_req in vlm_reqs:
    first_token = self.engine.prefill_vlm(...)
```

### Layer 3 — Variable-length deferred to M4 (Bridge comment, line 1357)

Variable-length batched prefill (different `T` per request, packed into a
single forward via per-row attention masks or padded-with-mask) is a
larger project. The M1 fix unblocks the common case (uniform-length
prefills from the scheduler) and is what gives most of the 20-33× win
quoted in the memory.

For our M5 long-context push, M1 is sufficient: the scheduler usually
batches new requests of comparable length together, and even if not, the
fix improves any group of 2+ uniform-length requests, which is the
common case under load.

## Validation plan

1. **Equivalence:** call `vsm_engine_prefill_seq_uniform_topk` (sequential
   reference) and `vsm_engine_prefill_batched_uniform` (new path) on the
   same inputs. Assert top-K logits match within bf16 tolerance. Test
   exists in `tests/test_engine_bridge.py` already; verify it covers the
   new Python binding once added.

2. **Throughput:** bench at B=4, 8, 16, 32, 64 with T=512, 1024, 2048,
   4096. Compare:
   - Old: B sequential `prefill_req` calls
   - New: one `prefill_batched_uniform` call
   - Target: 5-20× speedup on B=64 cases (per Bridge.swift:1351 comment)

3. **Long-context check:** at T=8192, 16384, 32768 with B=2-8, verify
   correctness and measure speedup. This is where the M5 long-context
   work needs the fix — long-ctx is exactly when sequential overhead
   compounds worst.

4. **Mixed batch fallback:** verify the worker correctly falls back to
   per-request `prefill_req` when sampling params differ or prompt
   lengths vary.

## Out of scope (for this audit, deferred)

- Variable-length batched prefill (Bridge comment "deferred to M4")
- VLM (multimodal) batched prefill — has its own pipeline (`prefill_vlm`)
- Streaming TQ K8V4 quantization during prefill — separate phase of
  M5-longctx plan, builds on top of the batched path

## Sources

- `swift/Sources/VLLMBridge/Bridge.swift` lines 340 (single-req entry),
  1349 (batched-uniform entry), 1409 (Qwen3 impl), 1525 (hybrid impl)
- `vllm_swift/engine_bridge.py` lines 122-125 (current bindings)
- `vllm_swift/worker.py` lines 239-275 (sequential prefill loop)
- Memory: `project_vllm_swift_sequential_prefill.md` (problem statement)
- M5 Max baseline: `~/dev/mlx-swift-lm/benchmarks/m5-max-128gb-2026-05-08-v0-baseline.md`

## Next concrete step

Add the ctypes binding for `vsm_engine_prefill_batched_uniform` in
`engine_bridge.py` and a thin Python wrapper. Validate via existing
test infrastructure (`tests/test_engine_bridge.py`). Once green, modify
the worker dispatch to use the batched path opportunistically. Bench
against baseline at B=8/16/32 to quantify the win.
