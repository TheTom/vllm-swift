# vllm-swift

Native Swift/Metal backend for vLLM on Apple Silicon. No Python in the inference hot path.

## Performance (M5 Max 128GB)

All numbers measured on the same hardware. vllm-swift uses the Swift/Metal path (release build). vllm-metal uses the Python/MLX path with paged attention Metal kernels.

### Qwen3-0.6B (decode output tok/s)

| | Single | 8 concurrent | 32 concurrent | 64 concurrent |
|---|:---:|:---:|:---:|:---:|
| **vllm-swift** | **575.8** | **1,567** | **2,922** | **3,408** |
| vllm-metal (Python) | 78.3 | 788.5 | 2,367 | — |

### Qwen3-4B (decode output tok/s)

| | Single | 8 concurrent | 32 concurrent | 64 concurrent |
|---|:---:|:---:|:---:|:---:|
| **vllm-swift** | **178.8** | **482.2** | **1,207** | **1,533** |
| vllm-metal (Python) | 5.3* | — | — | — |

\*vllm-metal 7B number shown (no 4B test available).

### TurboQuant+ KV Cache Compression

All numbers from mlx-swift-lm on M5 Max. Same model code used by vllm-swift.

**Qwen3.5 9B (4-bit weights)**

| KV Cache | Compression | PPL @1K | PPL @32K | Decode @1K | Decode @32K |
|----------|:-----------:|:------:|:-------:|:----------:|:-----------:|
| FP16 | 1.0x | 1.87 | 2.25 | 91 tok/s | 80 tok/s |
| turbo4v2 | 3.2x | — | — | — | — |
| turbo3 | 4.6x | 1.96 | 2.12 | 94 tok/s | 79 tok/s |

**Qwen3.5 2B (4-bit weights)**

| KV Cache | Compression | PPL @1K | PPL @32K | Decode @1K | Decode @32K |
|----------|:-----------:|:------:|:-------:|:----------:|:-----------:|
| FP16 | 1.0x | 2.72 | 4.40 | 264 tok/s | 157 tok/s |
| turbo4v2 | 3.2x | 3.22 | 3.72 | 265 tok/s | 157 tok/s |
| turbo3 | 4.6x | 3.95 | 3.89 | 264 tok/s | 157 tok/s |

TurboQuant+ compresses KV cache 3-5x with negligible PPL impact. Decode speed is unchanged — compression is free during memory-bound decode.

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
- Batched concurrent decode with `BatchedKVCache`
- Auto model download from HuggingFace Hub
- TurboQuant KV cache support (`turbo3`, `turbo4`) via mlx-swift-lm
- Greedy and temperature sampling
- EOS / stop token detection (vLLM scheduler)

## Quick Start

```bash
# Build the Swift bridge
cd swift && swift build && cd ..

# Install the plugin
pip install -e .

# Test
DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/debug \
  python test_bridge.py
```

## Project Structure

```
vllm_swift/           Python plugin (vLLM WorkerBase)
swift/
  Sources/VLLMBridge/       C bridge (@_cdecl exports)
  bridge.h                  C API (prefill, decode, batched decode)
tests/                      67 tests, 95%+ coverage
```

## Requirements

- macOS 14+ on Apple Silicon
- Swift 6.0+
- Python 3.10+
- [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm) alpha branch

## License

Apache-2.0
