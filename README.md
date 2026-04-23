# vllm-swift

Native Swift/Metal backend for vLLM on Apple Silicon. No Python in the inference hot path.

## Performance (M5 Max 128GB, debug build)

All numbers measured on the same hardware. vllm-swift uses the Swift/Metal path. vllm-metal uses the Python/MLX path with paged attention Metal kernels.

### Qwen3-0.6B (decode output tok/s)

| | Single request | 8 concurrent | 32 concurrent |
|---|:---:|:---:|:---:|
| **vllm-swift** | **155.6** | **790.9** | **2,075** |
| vllm-metal (Python) | 78.3 | 788.5 | 2,367 |

### Qwen3-4B (decode output tok/s)

| | Single request | 8 concurrent | 32 concurrent |
|---|:---:|:---:|:---:|
| **vllm-swift** | **116.3** | **388.7** | **1,041** |
| vllm-metal (Python) | 5.3* | — | — |

\*vllm-metal 7B number shown (no 4B test available). The Python hot-path overhead dominates at larger model sizes.

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
