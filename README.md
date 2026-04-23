# vllm-swift-metal

vLLM plugin for Apple Silicon. Inference runs through a Swift/Metal C bridge — zero Python in the GPU hot path.

## Performance (M5 Max 128GB, debug build)

### Single Request (decode tok/s)

| Model | vllm-swift-metal | vllm-metal (official) |
|-------|:----------------:|:---------------------:|
| Qwen3-0.6B-4bit | **152.6** | 76.5 |
| Qwen3-4B-4bit | **114.8** | — |

### Throughput (output tok/s, concurrent requests)

| Model | B=32 | B=64 |
|-------|:----:|:----:|
| Qwen3-0.6B-4bit | **2,106** | **2,902** |
| Qwen3-4B-4bit | **1,047** | **1,386** |
| vllm-metal 0.6B (reference) | 2,300 | — |

Batched decode uses `BatchedKVCache` — single shared tensor for all requests, vectorized mask, one SDPA call per layer. No per-request loops.

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
vllm_swift_metal/           Python plugin (vLLM WorkerBase)
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
