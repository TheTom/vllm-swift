# Release History

## v0.6.1

**Bugfix: ship the `turbo_dequant_rotated_*` kernel family in the bottle.**

The v0.6.0 bottle's `mlx.metallib` was built from a stale `.build`
checkout that predated the `turbo_dequant_rotated` Metal kernel family
(added upstream in mlx-swift commit `8a5a74e`). Models with
`head_dim=256` (e.g. Qwen3.5/3.6 27B-class) failed at engine init with:

```
Fatal error: [metal::Device] Unable to load kernel turbo_dequant_rotated_4_256_bf16
```

This release does a clean rebuild of the metallib so every
`(bits ∈ {2,3,4,8}, dim ∈ {64,80,96,128,256,512}, dtype ∈ {bf16,f16})`
variant is present. `head_dim=128` models on v0.6.0 (Qwen3-4B, Qwen3-8B,
Qwen2.5-7B) were unaffected and continue to work.

No source code change beyond version bumps.

Reported by @heykb on M4 Pro 48G.

## v0.6.0

**TriAttention V3 + longctx ChatSession integration; Gemma 4 MTP drafter.**

This release rebases the bundled `mlx-swift-lm` to v3.33.0-alpha, which
brings in the V3+longctx rescue path (validated end-to-end via 256K
NIAH on Apple Silicon) and the first Swift-native port of Google's
Gemma 4 Multi-Token Prediction drafter for speculative decoding.

### What's new

- **V3 + longctx via ChatSession.** The Tier-3 auto-rehydrate hook
  (commit `fe1a3b0` in mlx-swift-lm) fires before each turn's prefill
  using the user's question text as the retrieval query, prepending
  recovered chunks as a system message. Empirical receipts on
  Qwen3.5-2B-4bit (M5 Max) at the full 32K → 256K ramp:
  V3+longctx ✓HIT every rung; V3-only ✗miss every rung; baseline
  turbo8v4 ✓HIT every rung. V3 alone is unsafe for retrieval
  workloads — we now document this explicitly in the README and in
  `mlx-swift-lm`'s "TriAttention V3 + longctx" section.

- **Gemma 4 MTP drafter (Swift port).** New `Gemma4Assistant` model
  + `MTPSpec` iterator + factory wiring. 41.7 tok/s on Gemma 4 31B
  4-bit at k=2 with the drafter quantized to 4-bit at load time
  (1.50× speedup over no-drafter baseline; tied with community
  Python `mlx-vlm` on absolute throughput). Block-size sweet spot
  matches Google's analysis (block_size 3 = numDraftTokens 2).
  Drafter quant noise alignment with target raises acceptance rate
  from 124% to 132%.

- **Defaults stay safe.** TriAttention V3 is OFF by default. Enable
  via `VLLM_TRIATT_ENABLED=1` AND `LONGCTX_ENDPOINT=http://...`
  together. MTP drafter is opt-in via the harness in the bundled
  benchmark.

### Bumped

- `mlx-swift-lm` (vllm-swift-stable branch) → tip equivalent to
  v3.33.0-alpha
- `pyproject.toml` 0.5.4 → 0.6.0
- `homebrew/vllm-swift.rb` 0.5.4 → 0.6.0 (bottle SHA needs rebuild)

### Reproduction

```bash
# longctx-svc on the same host
longctx-svc serve --host 127.0.0.1 --port 5054

# vllm-swift with V3 + longctx
vllm-swift serve <model> --enable-longctx
```

See `mlx-swift-lm/README.md` "TriAttention V3 + longctx (long-context
rescue)" for full env-var reference and the recommended-defaults
block. Cross-references:

- `turboquant_plus/docs/papers/longctx-1m-and-triattention.md`
- `turboquant_plus/docs/papers/triattention-v3.md` (§10 addendum)

## v0.5.4 — May 7, 2026

**Fix: turbo KV schemes on dense Qwen3 no longer emit degenerate output.**
Defilan reported on the v0.5.3 alpha that `--additional-config
'{"kv_scheme":"turbo4v2","kv_bits":4}'` on Qwen3-4B-4bit produced
`"<think>1\n1\n1..."` — the same garbage on `turbo8v4` and even with
longctx removed at `prompt_tokens=19`. v0.5.3 had wired the scheme into
the **batched-prefill** paths (`prefill_batched_uniform` /
`hybrid init_batched`), but the dense Qwen3 *serve* flow goes through
a different path: `prefill_req` → `init_batched` → `decode_all`. Both
stages drop turbo on dense:

- `init_batched` casts each per-request cache to `KVCacheSimple`, which
  fails when `kvScheme=turbo*` is set (mlx-swift-lm uses
  `RotatingKVCache` for that case). The cast guard returns 0 silently
  and `engine.batchedCaches` stays nil.
- `decode_all`'s Qwen3 fully-batched path skips (no `batchedCaches`).
  The Qwen3 *semi-batched* fallback then runs
  `Qwen3Attention.batchedForward`, whose per-request RoPE+update loop
  corrupts on rotating-window K/V dequant semantics. Decode degenerates.

**Fix:** `decode_all` now gates both Qwen3 batched paths on
`engine.generateParams.kvScheme?.hasPrefix("turbo")`. When turbo is set,
fall through to the sequential `stepAsync` `TokenIterator` path — the
well-tested turbo decode path every standalone mlx-swift-lm consumer
uses. Trade-off: no batched SDPA across concurrent requests on dense
Qwen3 + turbo, so high-concurrency throughput regresses on that
specific cell. Hardening `Qwen3Attention.batchedForward` for turbo K/V
is v0.5.5+ work.

Verified locally on Qwen3-4B-4bit:

| config | output |
|--------|--------|
| `kv_scheme=turbo4v2` short prompt | `<think>\nOkay, the user wants...` clean |
| `kv_scheme=turbo4v2` longer prompt | clean |
| no `kv_scheme` | clean (no regression) |

No mlx-swift-lm changes — pure Bridge.swift gate. Wheel + bottle
rebuild only.

## v0.5.3 — May 7, 2026

**Fix: turbo KV schemes no longer silently bypassed on the batched-decode
path.** Field reports of "Hello." → ".2.2.2.2..." drift on Qwen3.5/3.6
hybrid models with `--additional-config '{"kv_scheme":"turbo4v2"}'`
turned out to be a deeper structural bug than v0.5.2's max_tokens-bump
sweep covered. Tracking it down:

- `BatchedKVCache` (in mlx-swift-lm), the cache class used by Qwen3
  dense + Qwen3.5 / Qwen3.6 / Qwen3Next hybrid models on the batched-
  decode path, only knew how to store raw fp16/bf16 K/V. Whatever
  `kv_scheme` the user set on `--additional-config` was parsed into
  `GenerateParameters.kvScheme` and then dropped on the floor when the
  bridge built the BatchedHybridCache. Decode ran on raw fp16 KV no
  matter what scheme flag was set.
- This meant alpha-tester benches that compared "raw vs turbo4v2" on
  hybrid models were comparing raw vs raw. Buddy's drift wasn't a
  turbo-codec quality issue — it was the raw-KV path interacting badly
  with v0.5.1's max_tokens-bump on a tiny "Hello" prompt that got
  pushed to 16K-20K tokens before hitting an EOS.

**Fix shipped in mlx-swift-lm `vllm-swift-stable` (commit
`d53bfe1`):** `BatchedKVCache` gains an internal `.raw` / `.turbo`
storage variant. Turbo path allocates packed-byte K/V + per-token L2
norms, encodes new tokens via the existing batch-parallel `fusedEncode`
kernel (flatten `[B, kvH, D]` → `[B*kvH, D]`, encode, reshape), and
runs decode-time attention via a new `attention(queries:scale:mask:)`
method that mirrors `TurboQuantKVCache.compressedAttention`'s
dequant-first SDPA (pre-rotate Q with codec's rotation matrix folded
with attention scale, bulk-dequant K/V via `bulkDequantRotated` into
rotated codec space, run MLXFast SDPA in rotated space — score-
preserved under orthogonal rotation `(RQ)·(RK)ᵀ = QKᵀ` — apply inverse
rotation to output).

Bridge.swift threads `engine.generateParams.kvScheme` through
`Qwen35TextModel.newBatchedHybridCache(maxBatch:parameters:turboKeyBits:
turboValueBits:)` and the equivalent Qwen3Next factory. The Qwen3
*dense* prefill→batched K/V copy path still drops `kvScheme` (TODO
v0.5.4 — needs a bulk-encode pass over the prefilled fp16 K/V into the
turbo cache); for hybrid models the fix is complete.

3 model-side callers (Qwen3 / Qwen3.5 / Qwen3Next `fullyBatchedForward`)
migrated to call `cache.attention(...)` instead of inline-slicing
`cache.keys`/`cache.values` + `MLXFast.scaledDotProductAttention`. The
default raw-mode `attention()` impl is bit-identical to the inline
pattern — pure refactor first, additive turbo storage second. **20/20
existing batched-hybrid + Qwen35 lockstep + fullyBatchedDecode slot-
equivalence tests stay green.**

7 new regression tests (`Tests/MLXLMTests/BatchedTurboKVCacheTests.
swift`) lock down: raw vs turbo flag reporting, raw-key mode (V-only
compression with `keyBits == 0`), update slot routing for B=1 and
B=4, and dequant-first attention shape + non-NaN output through the
real Metal kernel chain.

Live repro vs buddy's exact failing config (Qwen3.5-35B-A3B-4bit +
turbo4v2 + max-num-seqs=1 + "Say hello in one short sentence."):
clean output, `Hello!`, `finish_reason=stop`, no drift, no `.2.2.2.2`
runaway. **The fix works.**

If you were running with `--additional-config '{"kv_scheme":
"turbo4v2"}'` before v0.5.3, your tok/s and memory-footprint numbers
on the batched-decode path were raw-fp16 numbers wearing a turbo
label. v0.5.3 onward, the flag actually compresses.

`pip install vllm-swift==0.5.3` and rebuilt bottle ship the fix.

## v0.5.2 — May 7, 2026

**Patch: alpha-tester regression sweep.** Field reports from the v0.5.1 alpha
surfaced 5 obvious bugs and 2 non-obvious ones (Metal-side; tracked separately).
This release fixes the obvious ones and locks them down with regression tests.

- **vllm not declared as a runtime dep** (#1). `pip install vllm-swift==0.5.1` left
  users staring at `ModuleNotFoundError: No module named 'vllm'` on first
  `vllm-swift serve`. `pyproject.toml` now declares `vllm>=0.10` directly. As a
  bonus this narrows pip's resolver window and stops it pulling rc/dev versions
  of safetensors/tokenizers/transformers under `--pre`.

- **Reasoning-budget bump clobbered explicit small `max_tokens`** (#3). A client
  sending `max_tokens=64` got `completion_tokens=20480` because
  `rewrite_request` unconditionally bumped any `max_tokens < 16384` (the
  starvation-prevention floor). Now respected when the client sets <1024 —
  that's clearly intentional (curl smoke tests, "say hello", token-count
  probes). The OpenCode/Hermes 4K-8K starvation case still bumps as before.

- **`message.reasoning` not normalized to `reasoning_content`** (#7). Some vLLM
  versions emit `message.reasoning` (their newer naming) instead of the
  OpenAI-standard `message.reasoning_content`. `rewrite_chat_completion` now
  copies `reasoning` → `reasoning_content` when the standard field is missing,
  preserving the original for back-compat. OpenAI clients (Hermes,
  openai-python) see the field they expect.

- **longctx splice spammed 8 chunks regardless of relevance** (#6). A trivial
  "say hello" query produced `prompt_tokens=5423` because every retrieve
  request returned 8 chunks, splice-cap or not. Added a relevance floor
  (default cosine score >= 0.20, env-tunable via `LONGCTX_RELEVANCE_FLOOR`)
  that drops noise chunks before splicing.

- **`--max-model-len` exceeding the model's `max_position_embeddings`** (#2).
  Pre-flight now reads the model's `config.json` and warns with the actual
  numbers ("65536 exceeds 40960; recommend --max-model-len 40960"), instead of
  letting vLLM reject prompts later with a less specific error.

8 new tests in `tests/test_longctx_endpoint.py` pin all five behaviors:

- `test_enrich_filters_chunks_below_relevance_floor` (#6)
- `test_enrich_keeps_chunks_at_or_above_floor` (#6)
- `test_enrich_relevance_floor_overridable_by_env` (#6)
- `test_rewrite_request_honors_explicit_small_max_tokens` (#3)
- `test_rewrite_request_still_bumps_default_starvation_budget` (#3)
- `test_rewrite_chat_completion_normalizes_reasoning_field` (#7)
- `test_rewrite_chat_completion_leaves_reasoning_content_alone` (#7)
- `test_warn_when_max_model_len_exceeds_model_cap` + 2 sibling cases (#2)

Plus a CI-fixing pass: `tests/test_longctx_endpoint.py` had stale `import json`
+ unused imports flagged by ruff F811/F401 (the v0.5.1 commit's CI failed on
this). All ruff lint clean now. **502/502 tests pass.**

**NOT fixed in this release** (separate Metal-kernel investigation):

- **#4 KV-cache corruption signature under turbo4v2 4-bit + sustained decode.**
  Buddy reported that for "Say hello in one short sentence." the model produced
  `Hello.` then `.2.2.2.2...` for thousands of tokens until cap. Classic
  turbo4v2-4bit drift. Workaround for testers: drop `--additional-config`
  entirely, or bump to `kv_bits: 8` (asymmetric K8/V4) for the same scheme.
- **#5 4× decode throughput decay** (128 → 30 tok/s monotonic) — likely the
  same root cause as #4. Same workaround.

`pip install vllm-swift==0.5.2` and rebuilt bottle ship the obvious fixes.

## v0.5.1 — May 7, 2026

**Patch: BatchedKVCache memory leak under sustained Hermes load.** Tom's Hermes alpha session hit a macOS application-memory OOM after ~14 turns. EngineCore RSS climbed 20 GB → 85 GB at ~4.6 GB/turn while every other knob (`--max-num-seqs 1`, `--gpu-memory-utilization 0.5`, `--no-enable-prefix-caching`) said it shouldn't.

Root cause: `Bridge.swift`'s `init_batched`, `initBatchedHybrid`, and the `prefill_batched_*` ctx-stub paths all hardcoded `max(B, 64)` for the BatchedKVCache slot dimension. With `--max-num-seqs 1` that pre-allocates 64 slots regardless of actual concurrency. At max_seq = 64K, B=1, bf16 K+V across 28 layers, the unused-but-allocated slack is ~60 GB. Worse: `maxSeq` was sized off the first batch's longest prefill (`max(2048, prefill + 512)`), so subsequent longer prefills forced re-grows. On Metal each re-grow stacks new backing storage in the heap before the old one is reclaimed — the per-turn growth.

Fix:

- `InferenceEngine` gains `maxConcurrentRequests` (driven by `vllm_config.scheduler_config.max_num_seqs`) and `maxKVSize` (driven by `model_config.max_model_len`), threaded through `vsm_engine_create` over the C-FFI.
- All three over-alloc sites now use `max(B, engine.maxConcurrentRequests)` instead of `max(B, 64)`.
- `maxSeq` is pinned to `engine.maxKVSize` at init, so subsequent prefills can't re-grow the underlying tensors.

Verified live: 19.5 → 78.6 GB in 3 turns pre-fix; 19.5 ↔ 21–37 GB oscillating band post-fix (Metal heap reuse, no monotonic growth).

Wire-level changes:

- `vsm_engine_create` C signature gains a trailing `Int32 maxNumSeqs` parameter (additive — old callers pass 0 → legacy 64-slot behavior preserved).
- `vllm_swift.engine_bridge.SwiftInferenceEngine.__init__` gains `max_num_seqs: int = 0`.
- `vllm_swift.worker.SwiftMetalWorker.load_model` reads from `vllm_config.scheduler_config.max_num_seqs`.

Bottle SHA cleared — needs rebuild before brew users get the fix. `pip install vllm-swift==0.5.1` carries it on the wheel side.

## v0.5.0 — May 7, 2026

**Feature: optional longctx retrieval companion.** vllm-swift can now wire up `TheTom/longctx` (alpha) so chat-completion prompts get retrieved code chunks spliced in automatically. The companion is **optional everywhere** — flag absent + env unset = bit-for-bit unchanged engine behavior. 487/487 existing tests still green.

Two integration paths:

- **`--retrieval-endpoint URL`** (or `LONGCTX_ENDPOINT` env): point at an already-running `longctx-svc`. The transparent rewriter calls `/retrieve`, splices retrieved chunks into the system message, forwards to vLLM unchanged.
- **`--enable-longctx`** (or `LONGCTX_ENABLE=1` env): one-flag UX. vllm-swift auto-spawns a `longctx-svc` subprocess on a free port, manages its lifecycle, tears it down on shutdown. Requires `pip install longctx-svc` (alpha; ~50 MB plus sentence-transformers / bge-reranker-v2-m3 / watchdog / pathspec).

```bash
# explicit endpoint
vllm-swift serve ~/models/Qwen3-4B-4bit --retrieval-endpoint http://127.0.0.1:8765

# one-flag
vllm-swift serve ~/models/Qwen3-4B-4bit --enable-longctx
```

The rewriter exposes four debug headers on every chat-completion response so testers can see what happened: `x-longctx-session`, `x-longctx-scope`, `x-longctx-chunks-used`, `x-longctx-scope-status`. End-to-end smoke harness in `services/longctx-svc/integration/harness.py` covers proxy + embedded modes; both verified locally on `Qwen3-4B-4bit` and on Mac mini M2 with `Qwen3.5-2B-4bit`.

22 new tests in `tests/test_longctx_endpoint.py` pin flag parsing (=-form / space-form / env fallback / `--no-` opt-out / flag-overrides-env), splice format, network-failure degradation, non-200 degradation, no-chunks no-splice, header case-insensitivity, splice into existing system message vs prepended new system message, OpenAI vision-style content arrays.

`pip install vllm-swift==0.5.0` and `brew upgrade vllm-swift` (after the bottle is rebuilt) ship it. `pip install longctx-svc` is needed alongside if you want to use either flag.

## v0.4.2 — May 5, 2026

**Patch: skip max_tokens bump when max_model_len is too small for reasoning headroom.** Follow-up to 0.4.1's clamp. Mac Mini M2 testing surfaced that even after clamping the bump against `max_model_len`, a small context window (e.g. Qwen3.5-2B with `--max-model-len 4096`) still 400'd: `prompt_tokens + max_tokens > max_model_len`. The 0.4.1 safety margin (256 tokens) wasn't enough headroom for realistic prompts.

- When `max_model_len < 16384` (or when half of it falls below the 8K useful-reasoning threshold), the rewriter now skips the bump entirely instead of clamping into a too-tight ceiling. Trust the client's `max_tokens`; vLLM will surface a clear error if the request truly doesn't fit.
- Reserves half of `max_model_len` for prompt tokens (or 1K minimum) when computing the bump ceiling.
- 1 new test pinning the skip behavior on a 4K context, plus updated tests for the band where clamping still applies (24K context).

`pip install vllm-swift==0.4.2` and rebuilt bottle ship the fix.


## v0.4.1 — May 5, 2026

**Patch: clamp `max_tokens` rescue against the configured `max_model_len`.**
Empirical bug surfaced on Mac Mini M2 testing v0.4.0 against `Qwen3.5-2B-4bit` with `--max-model-len 4096`: the request rewriter bumped a client `max_tokens=256` to the static `_REASONING_MAX_TOKENS_BUMP=32768`, which vLLM then 400'd with `max_tokens cannot be greater than max_model_len=4096`. The bump was correct in spirit (small budget against a reasoning model would have starved the `<think>` block) but didn't know about the server's context ceiling.

- `rewrite_request` now takes an optional `max_model_len` and clamps the bump to `max_model_len - 256` (safety margin for prompt tokens).
- If the clamped bump is below the client's requested value, leave the request alone — never bump *down*.
- CLI extracts `--max-model-len` from passthrough args and threads it through `_serve_with_rewriter` → `run` → `_make_app` so the rewriter has the value when bumping.
- 3 new tests pin the clamp behavior at three boundaries (clamp applies, clamp is no-op when ample, clamp would bump down so skip).

`pip install vllm-swift==0.4.1` and the rebuilt Homebrew bottle for v0.4.1 carry the fix.

## v0.4.0 — May 5, 2026

**Auto-detect tool + reasoning parsers, plus an invisible self-heal layer for the rough edges.** Closes [#13](https://github.com/TheTom/vllm-swift/issues/13). The original triggering case (`mlx-community/Qwen3.6-35B-A3B-8bit` needing a manual `--tool-call-parser qwen3_coder --reasoning-parser qwen3` workaround) now Just Works, and several other footguns get caught by the same plumbing.

### Auto-detection

- Three-layer detection from a model directory: architecture-prefix mapping (40+ families), chat-template marker fallback for unknown architectures, and a directory-name discriminator that catches converted MLX/GGUF builds whose `config.json` lost the specialized arch suffix (Qwen3-Coder MLX, R1 forks, Kimi-K2.6 disguised as DeepSeekV3, etc).
- Capability gate: skip injection on models whose chat template carries no tool fragments (Phi-3-mini, Gemma 2, etc) so they don't get a parser they can't satisfy.
- Pre-flight registry validation: parser names are checked against the running vLLM's `_TOOL_PARSERS_TO_REGISTER` / `_REASONING_PARSERS_TO_REGISTER` before injection. If a name isn't registered (vLLM renamed or removed it, or our detector got ahead of upstream), the injection is skipped with a stderr warning rather than letting vLLM crash with an opaque "unknown parser" error.
- Validated against 18 real local MLX models in CI; the `mlx-community/Qwen3.6-35B-A3B-8bit` case from #13 was independently re-validated by [@Defilan](https://github.com/TheTom/vllm-swift/pull/14#issuecomment-4376186794).

### Empirical correctness fixes (versus the original detector intent)

- All `Qwen3.5+`, `Qwen3.6+`, `Qwen3Next`, and `Qwen3MoeForCausalLM` variants ship the `qwen3_coder` XML tool-call shape in their chat template, not the older `hermes` JSON. Routing fixed; older dense Qwen3 / Qwen3.5-Instruct dense kept on hermes per their actual templates.
- Nemotron H / Cascade-2 routes to `qwen3_coder` (tool) + `nemotron_v3` (reasoning) per NVIDIA's HF discussion #7, not the previous Qwen3-derivative defaults.
- Qwen3-Coder gets reasoning auto-suppressed via a `-Coder-` directory-name rule; the `qwen3` reasoning parser otherwise eats tool calls emitted inside `<think>` blocks (vllm-project/vllm#39056-class race) and clients see `tool_calls=[]`.
- MiMo (Xiaomi) routes to `qwen3` reasoning + `qwen3_xml` tool per the official MiMo-V2-Flash vLLM recipe (the `mimo` parser name our detector previously emitted is not registered in vLLM 0.19.1's parser set and would fail server startup).
- GLM-4.7 routes to `glm45` reasoning as a workaround until vLLM ships a dedicated `glm47` reasoning parser (vllm-project/vllm#33348).
- xLAM family (`Salesforce/xLAM-1b-fc-r`, `Salesforce/Llama-xLAM-2-*-fc-r`) ships as `LlamaForCausalLM` arch but uses the dedicated `xlam` parser; dirname discriminator now handles this.
- LongCat (Meituan `LongCat-Flash-*`) routes to the dedicated `longcat` parser.

### Invisible self-heal layer (response_rewriter)

A transparent proxy that fronts vLLM on the user-facing port and applies these rewrites only when needed (no-op for non-reasoning, non-leaky-parser models — zero overhead path):

- **`max_tokens` rescue.** When a reasoning parser is in play and the client sent `max_tokens` below a reasoning-safe floor (16384), bump it to 32768 in-flight. Prevents the OpenCode/Pi pattern where a hardcoded 8192 budget gets eaten by `<think>`, vLLM truncates, `</think>` never closes, and the parser dumps raw thinking into `content` as a monologue. Empirically validated: takes Nemotron-Cascade-2 + OpenCode from a 4-minute wedge to 4/4-pass on the standard agent test set.
- **Auto-recovery for plaintext-JSON tool-call leaks.** Four shapes detected and re-synthesized into structured `message.tool_calls`: hermes JSON, qwen3_coder XML, phi4 pipe-tag (Microsoft's own model card admits Phi-4-mini emits this shape as text — vllm-project/vllm#14682), and mistral bracket. Both non-streaming and streaming paths covered; streaming uses a per-choice three-state machine (DECIDING / PASSTHROUGH / BUFFERING) so healthy chat traffic still streams delta-by-delta. Conservative ratio gate (≥50% of content) defends against false-positives on responses that legitimately mention tool-call shapes inline.
- **`Thinking:` prefix split.** For models that emit "Thinking:" plaintext instead of `<think>...</think>` tags (notably Nemotron-Cascade-2), the prefix is split out of `content` into `reasoning_content` so the OpenAI-shape contract holds.
- **Streaming usage-chunk preservation.** vLLM emits the final `usage` block in a chunk with `choices: []`; the rewriter passes these through verbatim instead of dropping them (had been swallowing them, visible to users as Hermes' context-token counter never advancing).

### Documentation

- New [`docs/MODEL_COMPATIBILITY.md`](docs/MODEL_COMPATIBILITY.md) — empirical pass / soft-fail / hard-fail across 12 local MLX models with root-cause classification. Updated for v0.4.0: 7/12 PASS now that Phi-4-mini gets caught by auto-recovery.
- New [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — symptom → diagnostic → fix for known failure patterns. Includes the `--default-chat-template-kwargs '{"enable_thinking": false}'` escape hatch originally surfaced by [@Defilan](https://github.com/TheTom/vllm-swift/pull/14#issuecomment-4376186794).

### Tests

460 unit + integration tests, including 8 fixture-based replay tests using anonymized snapshots of the actual agent traffic shapes that triggered the original bugs. Auto-recovery alone has 27 dedicated tests across positive / negative / boundary / replay axes. The healthy-chat false-positive replay is the strongest defense against a future over-eager regex change.

### Compatibility note

The Homebrew bash wrapper at `/opt/homebrew/bin/vllm-swift` will rebuild against this release when the bottle workflow runs against the v0.4.0 tag; the pip wheel is the authoritative path until that lands.

## v0.3.3 — May 5, 2026

**Re-release of v0.3.2 with proper wheel contents.** The 0.3.2 PyPI wheel was built before the `package_data` changes were merged to `main`, so it shipped without the bundled `libVLLMBridge.dylib` + `mlx.metallib`. PyPI release files are immutable, so 0.3.2 is yanked and 0.3.3 is the working release. No source changes vs 0.3.2.

- `pip install vllm-swift==0.3.3` now ships the dylib + metallib via wheel `package_data`.
- Homebrew bottle rebuilt against the same source tree.

## v0.3.2 — May 4, 2026

**Patch release: symlinked model dirs no longer break MLX qwen3 loader.** Reported by @defilan (LLMKube metal-agent integration): passing a symlinked model dir (e.g. `~/models/mlx-community/Qwen3.6-35B-A3B-8bit -> ~/models/Qwen3.6-35B-A3B-8bit`) crashed vllm-swift with `[vsm] Failed to load model: Unsupported model type: qwen3`. The Swift bridge handed the symlinked URL straight to MLX's `LLMModelFactory`, which derives the architecture key from a mix of path components and `config.json` — those disagree on a symlinked path and the qwen3 codepath rejects the mismatch.

- `swift/Sources/VLLMBridge/Bridge.swift` now calls `URL.resolvingSymlinksInPath()` before handing the URL to MLX. No-op on canonical paths; fixes the symlink case.
- Verified locally on M5 Max: canonical `~/.cache/huggingface/hub/.../snapshots/<commit>/` still loads; symlinked dir pointing at the same target now also loads.
- Reported by @defilan in defilantech/LLMKube#393.

## v0.3.1 — May 2, 2026

**Patch release: serve subcommand model-path bug.** The wrapper was forwarding the model path as a positional argument to `vllm.entrypoints.openai.api_server`, where vLLM 0.19.1's argparse maps a stray positional to the deprecated `model_tag` slot rather than `ModelConfig.model`. The path was silently dropped and the engine fell back to the `Qwen/Qwen3-0.6B` placeholder. Reported in #11 (Defilan), surfaced again triaging #4 and #10.

- `vllm-swift serve <path>` now passes `--model <path>` explicitly to vLLM (#12)
- Fixes silent fallback to `Qwen/Qwen3-0.6B` when serving local model directories
- No dylib changes; bottle rebuild ships only the wrapper fix

## v0.3.0 — April 28, 2026

**Stability and throughput on TurboQuant MoE.** Closes a long-standing Metal `Invalid Resource` race that hit concurrent custom-kernel workloads (TurboQuant B-path on MoE B≥8). Removing the swift-side band-aid that was only there to mask the underlying race recovers ~10% throughput on Qwen3.5-35B-A3B at B≥17.

- Metal buffer-aliasing race fixed via `CommandEncoder` retain on first-bind under `MTLResourceHazardTrackingModeUntracked` (mirrors `ml-explore/mlx#3461 / #3462`)
- Removed redundant `stopGradient + asyncEval` boundary in `compressedAttention` — Qwen3.5-35B-A3B B=17 4K decode 108.7 → 119.9 tok/s (+10%)
- TurboQuant compressed-attention path is now the default decode method (B-path) — A-path still selectable via `TURBO_COMPRESSED_ATTENTION=0`
- bf16 kernel output for TurboFlash pass2 + dim=512 instantiation for Gemma 4 31B
- Per-model `prefillStepSize` defaults via protocol (drops the stacked 3-place caller / model / fallback resolution)
- `prepareQueriesScaled` per-layer cache (saves one elementwise multiply per decode step)
- A-path rotation bypass — recovers decode tok/s and matches `--kv none` peak when `TURBO_COMPRESSED_ATTENTION=0`
- Initial foundation for DeepSeek-V4 — `model_type: deepseek_v4` dispatch wired in, weight loading works (DSV4-Flash-2bit-DQ tested on M5 Max). Forward pass not yet production-stable, follow-up in Phase 2.

## v0.2.2 — April 26, 2026

**Batched decode for hybrid models and capacity fixes.** Qwen3.5, Qwen3.6, and Qwen3Next now scale with concurrency instead of staying flat at single-request speed. Long-context high-batch workloads no longer OOM. Source installs now work without manually copying the Metal library.

- Qwen3.5 / Qwen3.6 / Qwen3Next batched decode — 16× total tok/s at B=64 on Qwen3.6-27B (was flat across B)
- Fixed OOM at high batch + long context (4B / B=64 / 8K) by releasing per-request prefill caches as they are copied into the batched cache
- Fixed crash at prompt length ≥ 2048 with batched decode (cache was sized for the wrong dimension)
- `scripts/install.sh` now builds and places `mlx.metallib`; source installs of GatedDelta / TurboFlash models work without manual steps

## v0.2.1 — April 25, 2026

**Performance recovery for small models.** Decode throughput on models with fewer KV heads (0.8B, 2B, 35B-A3B) was 40-60% slower than expected due to an overly aggressive GPU sync barrier. This release replaces it with a lightweight alternative, bringing decode speed back to within 10-17% of uncompressed baseline.

- Faster TurboQuant+ decode on small models (0.8B, 2B, 35B-A3B)
- TurboQuant+ support for NemotronH hybrid models
- Fixed a bug where compressed KV cache slots were being overwritten instead of appended
- Install script fixes for machines without MLX Python installed

## v0.2.0 — April 24, 2026

**KV cache compression and Homebrew install.** TurboQuant+ compresses the KV cache 3-5x with no measurable impact on output quality, enabling longer conversations on memory-constrained devices. Homebrew bottle means no Swift toolchain needed.

- TurboQuant+ KV cache compression (`--additional-config '{"kv_scheme": "turbo4v2"}'`)
- `brew install vllm-swift` with prebuilt bottle
- `vllm-swift update` command
- Decode and prompt logprobs
- Experimental vision-language model support

## v0.1.0 — April 22, 2026

**Initial release.** Native Swift/Metal inference backend for vLLM on Apple Silicon. Up to 2.6x faster decode than Python/MLX at low concurrency by removing Python from the inference hot path.

- OpenAI-compatible API server
- Batched concurrent decode
- Streaming responses
- Auto model download from HuggingFace
