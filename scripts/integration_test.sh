#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Integration test: start vllm serve, send request, verify response.
#
# Usage: ./scripts/integration_test.sh [model_path]
# Default model: ~/models/Qwen3-0.6B-4bit

set -euo pipefail

MODEL="${1:-$HOME/models/Qwen3-0.6B-4bit}"
PORT=8199
PID=""

cleanup() {
    if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== vllm-swift integration test ==="
echo "Model: $MODEL"

# Check model exists
if [ ! -d "$MODEL" ]; then
    echo "SKIP: Model not found at $MODEL"
    echo "Download with: huggingface-cli download mlx-community/Qwen3-0.6B-4bit --local-dir ~/models/Qwen3-0.6B-4bit"
    exit 0
fi

# Check dylib exists
DYLIB="$(dirname "$0")/../swift/.build/arm64-apple-macosx/release/libVLLMBridge.dylib"
if [ ! -f "$DYLIB" ]; then
    DYLIB="$(dirname "$0")/../swift/.build/arm64-apple-macosx/debug/libVLLMBridge.dylib"
fi
if [ ! -f "$DYLIB" ]; then
    echo "SKIP: Swift bridge not built. Run: cd swift && swift build -c release"
    exit 0
fi

# Start vllm serve in background
echo "Starting vllm serve on port $PORT..."
DYLD_LIBRARY_PATH="$(dirname "$DYLIB")" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-model-len 512 \
    --gpu-memory-utilization 0.5 \
    2>&1 &
PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "FAIL: Server process died"
        exit 1
    fi
    sleep 1
done

# Test completions endpoint
echo "Testing /v1/completions..."
RESPONSE=$(curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0
    }')

echo "Response: $RESPONSE"

# Verify response has generated text
if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['text'].strip(), 'empty response'; print('PASS: Got text:', repr(d['choices'][0]['text'][:50]))"; then
    echo ""
else
    echo "FAIL: No valid completion response"
    exit 1
fi

# Test chat endpoint
echo "Testing /v1/chat/completions..."
CHAT_RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "temperature": 0
    }')

if echo "$CHAT_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content'].strip(), 'empty'; print('PASS: Got chat:', repr(d['choices'][0]['message']['content'][:50]))"; then
    echo ""
else
    echo "FAIL: No valid chat response"
    exit 1
fi

# Test streaming
echo "Testing streaming..."
STREAM_RESPONSE=$(curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "Hello",
        "max_tokens": 5,
        "stream": true
    }')

if echo "$STREAM_RESPONSE" | grep -q "data:"; then
    echo "PASS: Streaming works"
else
    echo "FAIL: No streaming data received"
    exit 1
fi

echo ""
echo "=== All integration tests passed ==="
