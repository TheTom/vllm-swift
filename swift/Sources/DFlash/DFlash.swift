// Copyright 2026 SwiftLM Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Based on DFlash (arXiv:2602.06036)
// vllm-swift DFlash Speculative Decoding Module

/// DFlash: Block-Diffusion Speculative Decoding for Lossless Acceleration
///
/// This module provides speculative decoding capabilities for Apple Silicon
/// using the DFlash algorithm (arXiv:2602.06036).
///
/// ## Overview
///
/// DFlash accelerates LLM inference by using a small draft model to propose
/// multiple tokens at once, which are then verified in parallel by the target
/// model. Unlike traditional speculative decoding which proposes one token at
/// a time, DFlash proposes a block of tokens using block diffusion.
///
/// ## Key Components
///
/// - ``DFlashTargetModel``: Protocol for target models to implement DFlash support
/// - ``DFlashDraftModelProtocol``: Protocol for draft models
/// - ``DFlashRuntime``: Main runtime for DFlash generation
/// - ``DFlashConfiguration``: Configuration options for DFlash

// Core protocols and types
@_exported import MLX
@_exported import MLXLMCommon

// Module version
public let dflashVersion = "1.0.0"
