<p align="center">
  <img src="assets/logo.png" alt="vllm-swift" width="400">
</p>

<p align="center">
  Native Swift/Metal backend for vLLM on Apple Silicon. No Python in the inference hot path.
</p>

## Quick Start

```bash
git clone https://github.com/TheTom/vllm-swift.git && cd vllm-swift
./scripts/install.sh
source activate.sh
vllm serve ~/models/Qwen3-4B-4bit --max-model-len 2048
```

For long context, enable TurboQuant+ KV cache compression (3-5x memory savings, no speed penalty):

```bash
vllm serve ~/models/Qwen3-4B-4bit --max-model-len 32768 \
  --additional-config '{"kv_scheme": "turbo3", "kv_bits": 3}'
```

## Performance (M5 Max 128GB)

All numbers measured on the same hardware with release builds. Decode output tok/s.

> **How to read these numbers**: "Bridge direct" = Swift engine called via ctypes FFI, bypassing vLLM's Python scheduling overhead. This is the raw engine speed. "vllm serve" = full OpenAI-compatible API server with vLLM scheduler, batching, and HTTP overhead. Real-world serving uses the vllm serve path.

### Qwen3-0.6B

| | Single | 8 concurrent | 32 concurrent | 64 concurrent |
|---|:---:|:---:|:---:|:---:|
| **vllm-swift** (bridge direct) | **340** | **1,512** | **2,862** | **3,383** |
| vllm-metal (Python/MLX) | 142 | 1,170 | 2,457 | 3,017 |

### Qwen3-4B

| | Single | 8 concurrent | 32 concurrent | 64 concurrent |
|---|:---:|:---:|:---:|:---:|
| **vllm-swift** (bridge direct) | **149** | **479** | **1,166** | **1,519** |
| vllm-metal (Python/MLX) | 105 | 408 | 1,067 | 1,387 |

**Why the speed difference?** The Swift bridge keeps the entire forward pass in compiled Swift/Metal — no Python in the GPU hot path. The gap is largest at low concurrency (2.4x single-request on 0.6B) where per-step overhead dominates, and narrows at high batch sizes where GPU compute becomes the bottleneck.

### TurboQuant+ KV Cache Compression

All numbers from mlx-swift-lm on M5 Max. Same model code used by vllm-swift.

**Qwen3.5 2B (4-bit weights)**

| KV Cache | Compression | PPL @1K | PPL @32K | Prefill @1K | Prefill @32K | Decode @1K | Decode @32K |
|----------|:-----------:|:------:|:-------:|:----------:|:-----------:|:----------:|:-----------:|
| FP16 | 1.0x | 2.72 | 4.40 | 11,173 tok/s | 6,903 tok/s | 264 tok/s | 157 tok/s |
| turbo4v2 | 3.2x | 3.22 | 3.72 | 11,298 tok/s | 6,916 tok/s | 265 tok/s | 157 tok/s |
| turbo3 | 4.6x | 3.95 | 3.89 | 11,348 tok/s | 6,958 tok/s | 264 tok/s | 157 tok/s |

TurboQuant+ compresses KV cache 3-5x with negligible quality impact. Prefill and decode speed are unchanged — compression is free. The 157 tok/s flat across configs at 32K is expected: decode at this model size is bottlenecked by model weight reads, not KV cache.

## Architecture

```
Python (vLLM API, tokenization, scheduling)
  ↓ ctypes FFI
C bridge (bridge.h)
  ↓ @_cdecl
Swift (mlx-swift-lm, BatchedKVCache, batched decode)
  ↓
Metal GPU
```

## Features

- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`)
- Streaming (SSE) responses
- Chat templates (applied by vLLM, model-specific)
- Batched concurrent decode with `BatchedKVCache` (fully batched projections + attention)
- Per-request temperature sampling in batched path
- Auto model download from HuggingFace Hub
- TurboQuant+ KV cache compression (`turbo3`, `turbo4v2`) via mlx-swift-lm
- Decode logprobs (log-probability of sampled tokens)
- Greedy and temperature sampling
- EOS / stop token detection (vLLM scheduler)
- VLM (vision-language model) support (experimental)

## Configuration

### TurboQuant+ KV Cache Compression

Enable KV cache compression to reduce memory usage 3-5x with minimal quality loss:

```bash
vllm-swift serve ~/models/Qwen3-4B-4bit \
  --max-model-len 32768 \
  --additional-config '{"kv_scheme": "turbo3", "kv_bits": 3}'
```

| Scheme | Compression | Best for |
|--------|:-----------:|----------|
| `turbo3` | 4.6x | Maximum compression, long context |
| `turbo4v2` | 3.2x | Balanced quality/compression |

### Common flags

```bash
vllm-swift serve <model> [options]

  --max-model-len N          Max sequence length (default: model config)
  --port PORT                API server port (default: 8000)
  --gpu-memory-utilization F Memory fraction 0.0-1.0 (default: 0.9)
  --dtype float16            Model dtype (default: float16)
  --additional-config JSON   Extra config (kv_scheme, kv_bits)
```

All standard `vllm serve` flags work — vllm-swift is a drop-in backend.

## Known Limitations

- **Prompt logprobs** not yet supported (decode logprobs work)
- **LoRA** not supported (Swift engine limitation)
- **Chunked prefill** disabled (Swift engine handles full sequences)
- **top_p sampling** not supported in batched decode path (temperature works)
- Only **Qwen3** models use the fully batched decode path; other architectures fall back to sequential decode (still functional, just slower at high concurrency)
- Requires macOS on Apple Silicon (no Linux/CUDA)

## Install

### From source (recommended)

```bash
git clone https://github.com/TheTom/vllm-swift.git
cd vllm-swift

# One-step install (builds Swift, installs plugin, sets up metallib)
./scripts/install.sh

# Activate and serve
source activate.sh
vllm serve ~/models/Qwen3-4B-4bit --max-model-len 2048
```

### Manual (full control)

```bash
git clone https://github.com/TheTom/vllm-swift.git
cd vllm-swift

# Build the Swift bridge
cd swift && swift build -c release && cd ..

# Install the plugin
pip install -e .

# Set library path and run
DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/release \
  vllm serve ~/models/Qwen3-4B-4bit --max-model-len 2048
```

### Download a model

```bash
# MLX-format models from HuggingFace
huggingface-cli download mlx-community/Qwen3-4B-4bit --local-dir ~/models/Qwen3-4B-4bit

# Or vllm-swift will auto-download if huggingface_hub is installed
vllm-swift serve mlx-community/Qwen3-4B-4bit --max-model-len 2048
```

## Project Structure

```
vllm_swift/           Python plugin (vLLM WorkerBase)
swift/
  Sources/VLLMBridge/       C bridge (@_cdecl exports)
  bridge.h                  C API (prefill, decode, batched decode)
scripts/
  install.sh                One-step build + install
  integration_test.sh       End-to-end smoke test
homebrew/
  vllm-swift.rb             Homebrew formula
tests/                      82 tests, 97% coverage
```

## Requirements

- macOS 14+ on Apple Silicon
- Xcode 15+ or Swift 6.0+ toolchain (for building from source; Homebrew handles this)
- Python 3.10+
- [vLLM](https://github.com/vllm-project/vllm) 0.19+
- [mlx-swift-lm](https://github.com/TheTom/mlx-swift-lm/tree/vllm-swift-stable) (pulled automatically by Swift Package Manager)

## License

Apache-2.0
