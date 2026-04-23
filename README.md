# vllm-swift-metal

vLLM plugin for Apple Silicon. Bypasses Python model execution entirely — inference runs through a Swift/Metal C bridge at native speed.

## Why

The existing vllm-metal plugin hits 4.8 tok/s on 7B models due to 197ms of Python overhead per token (`mx.async_eval` graph compilation). The GPU only needs 12ms. This plugin eliminates that gap by delegating all compute to Swift via FFI.

## Performance (M5 Max, 128GB)

| Model | Decode | Build |
|-------|--------|-------|
| Qwen3-4B-4bit | 113.2 tok/s | debug |
| Qwen2.5-0.5B | 139.7 tok/s | debug |

Release builds pending. TurboQuant KV cache (`turbo3`, `turbo4`) supported via `--kv-scheme`.

## Quick Start

```bash
# 1. Build the Swift bridge
cd swift
swift build
cd ..

# 2. Install the plugin
pip install -e .

# 3. Run
DYLD_LIBRARY_PATH=swift/.build/arm64-apple-macosx/debug \
  python test_bridge.py
```

Set `MODEL_PATH` to point at an MLX-format model directory:

```bash
MODEL_PATH=~/models/Qwen3.5-35B-A3B-4bit python test_bridge.py
```

## Architecture

```
Python (vLLM API, tokenization, scheduling)
  ↓ ctypes FFI
C bridge (bridge.h — 12 functions)
  ↓ @_cdecl
Swift (mlx-swift-lm TokenIterator, KV cache, attention)
  ↓
Metal GPU
```

Zero Python in the GPU hot path. The Swift engine handles model loading, prefill, decode, KV cache management, and sampling.

## Project Structure

```
vllm_swift_metal/           Python plugin
  __init__.py               vLLM platform plugin registration
  platform.py               Apple Silicon detection + config
  engine_bridge.py           ctypes → libVLLMBridge.dylib
  worker.py                 WorkerBase impl (prefill/decode)
swift/
  Package.swift             Swift package (links mlx-swift-lm)
  Sources/VLLMBridge/       Bridge.swift (@_cdecl C exports)
  bridge.h                  C API header
tests/                      61 tests, 95%+ coverage
```

## Requirements

- macOS 14+ on Apple Silicon
- Swift 6.0+
- Python 3.10+
- [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm) (linked locally for dev)

## Status

Early development. Single-request mode only. `vllm serve` integration in progress.

## License

Apache-2.0
