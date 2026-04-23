// SPDX-License-Identifier: Apache-2.0
// vllm-swift-metal C bridge — Python ↔ Swift inference engine
//
// Exposes a minimal C API for the Python plugin to drive the Swift
// mlx-swift-lm inference engine. All heavy compute stays in Swift/Metal.
// Python only handles vLLM protocol (scheduling, tokenization, API).

#ifndef VLLM_SWIFT_METAL_BRIDGE_H
#define VLLM_SWIFT_METAL_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the Swift inference engine
typedef void* vsm_engine_t;

// Engine lifecycle
vsm_engine_t vsm_engine_create(
    const char* model_path,     // HuggingFace model ID or local path
    const char* dtype,          // "float16", "bfloat16"
    int32_t max_kv_size,        // Max KV cache tokens (0 = unlimited)
    const char* kv_scheme,      // "turbo3", "turbo4", etc. (NULL = none)
    int32_t kv_bits,            // KV quantization bits (0 = none)
    float memory_fraction       // Metal memory fraction (0.0-1.0)
);

void vsm_engine_destroy(vsm_engine_t engine);

// Model info (call after create)
int32_t vsm_engine_vocab_size(vsm_engine_t engine);
int32_t vsm_engine_num_layers(vsm_engine_t engine);
int32_t vsm_engine_head_dim(vsm_engine_t engine);
int64_t vsm_engine_model_memory_bytes(vsm_engine_t engine);

// Prefill: process prompt tokens, return first generated token
// prompt_tokens: array of token IDs
// num_tokens: length of prompt_tokens
// Returns: first generated token ID, or -1 on error
int32_t vsm_engine_prefill(
    vsm_engine_t engine,
    const int32_t* prompt_tokens,
    int32_t num_tokens,
    float temperature,
    float top_p
);

// Decode: generate next token from current KV cache state
// Returns: next token ID, or -1 on error (e.g. EOS)
int32_t vsm_engine_decode_step(
    vsm_engine_t engine,
    float temperature,
    float top_p
);

// Batch decode: generate N tokens, write to output buffer
// Returns: number of tokens actually generated
int32_t vsm_engine_decode_batch(
    vsm_engine_t engine,
    int32_t max_tokens,
    float temperature,
    float top_p,
    int32_t* output_tokens,     // caller-allocated output buffer
    int32_t output_capacity     // size of output buffer
);

// Get logits for the last forward pass (for custom sampling)
// Returns pointer to float32 logits array (vocab_size elements)
// Pointer valid until next prefill/decode call
const float* vsm_engine_get_logits(
    vsm_engine_t engine,
    int32_t* out_vocab_size     // receives vocab size
);

// Reset KV cache (new conversation)
void vsm_engine_reset(vsm_engine_t engine);

// Performance stats
typedef struct {
    double prefill_tokens_per_sec;
    double decode_tokens_per_sec;
    int64_t peak_memory_bytes;
    int32_t total_tokens_generated;
    double total_decode_time_sec;
} vsm_perf_stats_t;

void vsm_engine_get_stats(vsm_engine_t engine, vsm_perf_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif // VLLM_SWIFT_METAL_BRIDGE_H
