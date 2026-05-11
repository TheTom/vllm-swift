"""Detect appropriate vLLM reasoning parser from a model directory.

Reads `config.json` for the model architecture and `chat_template.jinja` /
`tokenizer_config.json` for thinking-block markers (e.g. `<think>`,
`<thinking>`, `<channel>`). Maps to a known parser registered in
`vllm.reasoning._REASONING_PARSERS_TO_REGISTER`.

Why architecture + template signal together:
- Architecture string alone is brittle (forks, fine-tunes, future variants).
- Template alone doesn't know which parser format the model emits.
- Combining them avoids both false positives (non-thinking models forced
  through a parser they weren't trained for) and false negatives (a Qwen3
  fine-tune that still uses `<think>` tags).

Used by the vllm-swift `serve` wrapper and the Python CLI to auto-inject
`--reasoning-parser <name>` when the user did not pass it explicitly.
Saves the OpenCode / Hermes / Droid / etc symptom where the model's
chain-of-thought leaks into `message.content` and `message.reasoning_content`
stays null because vLLM has no parser configured.
"""

from __future__ import annotations

import json
import os
import sys

# Architecture prefixes that LOOK like thinking-capable models but ship
# templates with thinking markers ONLY inside conditionals that default
# to off. Auto-injection on these would catch llama.cpp's #21616 / #20809
# class of false positives (Reka Edge, Qwen3-Instruct-2507, etc).
_REASONING_SUPPRESS_PREFIXES: tuple[str, ...] = (
    "Reka",  # Reka Edge templates have <think> markers but model doesn't reason
    "Ring",  # inclusionAI Ring 2.0 — non-thinking variants ship marker-bearing template
    "Bailing",  # inclusionAI Bailing family
    "Ling",  # inclusionAI Ling family
)

# Suffix patterns on model directory names that indicate a non-thinking
# variant of an otherwise thinking-capable family. Defends against #20809
# (Qwen3.5-*-Instruct and Qwen3-*-Instruct-2507 false-positive injection).
_NON_REASONING_NAME_SUFFIXES: tuple[str, ...] = (
    "-Instruct-2507",
    "-Instruct-FP8",
    "-NoThink",
    "-no-think",
)

# Substring patterns on model directory names that should suppress reasoning
# parser injection because the parser+model combo races (vLLM #39056-class).
# Specifically: `qwen3_coder` tool parser + `qwen3` reasoning parser causes
# the model to emit tool calls *inside* `<think>...</think>` blocks, which
# the reasoning parser eats before the tool parser ever sees them. Result:
# `tool_calls=[]` and silent dropped output. Empirically reproduced against
# Qwen3-Coder-30B-A3B-Instruct-MLX-6bit on 2026-05-04: model generated
# tokens but they vanished into reasoning_content, never surfaced as either
# content or tool_calls. Suppressing reasoning makes Qwen3-Coder reliably
# tool-dispatch since the model still thinks internally — we just stop
# trying to extract the thinking server-side.
_REASONING_SUPPRESS_NAME_SUBSTRINGS: tuple[str, ...] = (
    "-Coder-",  # Qwen3-Coder, future Qwen-Coder family
    "-coder-",  # lowercase variants
    # Note 2026-05-11: ConfigI family does NOT need a suppress entry — the
    # proper fix is `rewrite_request` auto-injecting
    # `chat_template_kwargs={"enable_thinking": false}` for the qwen3
    # reasoning parser (see _REASONING_PARSERS_WANT_ENABLE_THINKING_FALSE
    # in response_rewriter.py). Earlier `-ConfigI-` suppress entry was
    # over-broad — it disabled the parser entirely instead of just
    # changing the model's prompt to not start thinking. Removed.
)

# Architecture prefix -> vLLM reasoning parser name. Order matters: longer
# / more specific prefixes must come first so e.g. "DeepseekR1" matches
# before the "Deepseek" generic family. Names match vLLM's
# `_REASONING_PARSERS_TO_REGISTER` registry exactly.
_ARCH_TO_REASONING_PARSER: tuple[tuple[str, str], ...] = (
    # DeepSeek thinking variants — V4 falls into the V3 family for now
    ("DeepseekR1", "deepseek_r1"),
    ("DeepSeekR1", "deepseek_r1"),
    ("DeepseekV4", "deepseek_v3"),
    ("DeepSeekV4", "deepseek_v3"),
    ("DeepseekV32", "deepseek_v3"),
    ("DeepseekV31", "deepseek_v3"),
    ("DeepseekV3", "deepseek_v3"),
    ("DeepseekV2", "deepseek_v3"),
    # Qwen3 family (Qwen3, Qwen3.5, Qwen3.6, Nemotron-Cascade based on Qwen3, MoE variants)
    ("Qwen3Coder", "qwen3"),
    ("Qwen3Moe", "qwen3"),
    ("Qwen3MoE", "qwen3"),
    ("Qwen3_5", "qwen3"),
    ("Qwen3_6", "qwen3"),
    ("Qwen3", "qwen3"),
    # Nemotron-H (Cascade-2, Nemotron-3 Super/Nano) — NVIDIA's purpose-built
    # parser, NOT qwen3. Source: vLLM PR #36393 (Shaun Kotek, NVIDIA) +
    # HF discussion confirming qwen3 is wrong choice:
    # https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/discussions/7
    # nemotron_v3 == DeepSeekR1 (<think>/</think>) + enable_thinking swap.
    ("NemotronH", "nemotron_v3"),
    ("Nemotron", "nemotron_v3"),
    # Gemma 4 has its own native reasoning format
    ("Gemma4", "gemma4"),
    # Mistral reasoning models (Magistral et al)
    ("Magistral", "mistral"),
    ("Mistral", "mistral"),
    # GLM 4.5 / 4.7 / 5.1 thinking. GLM-4.7 specifically has no `glm47`
    # reasoning parser registered in vLLM yet (see vllm-project/vllm#33348);
    # the official deployment guide says to use `--reasoning-parser glm45`
    # as a workaround until vLLM ships glm47 reasoning. Per the same bug
    # report the `right` answer is DeepSeekR1ReasoningParser, but glm45
    # works on Apple Silicon today and shipping the workaround keeps users
    # unblocked.
    ("GlmMoeDsa", "glm45"),  # GLM-5.1
    ("Glm47", "glm45"),  # GLM-4.7 — workaround for vllm-project/vllm#33348
    ("Glm45", "glm45"),
    ("Glm4_5", "glm45"),
    # Granite reasoning
    ("Granite4", "granite"),
    ("Granite", "granite"),
    # MiniMax M2 (M2.5 has its own append-think variant in vLLM,
    # but the standard parser remains the auto-detect default)
    ("MinimaxM2", "minimax_m2"),
    ("MiniMaxM2", "minimax_m2"),
    # Kimi K2 / K2.5 / K2 Thinking
    ("KimiK2Thinking", "kimi_k2"),
    ("KimiK25", "kimi_k2"),
    ("KimiK2", "kimi_k2"),
    ("Kimi", "kimi_k2"),
    # Hunyuan / Step / Olmo / Seed-OSS
    ("HunyuanA13B", "hunyuan_a13b"),
    ("HYV3", "hunyuan_a13b"),  # Hy3-preview (vLLM PR #40681)
    ("HY", "hunyuan_a13b"),
    ("Step3p5", "step3p5"),
    ("Step35", "step3p5"),
    ("Step3", "step3"),
    ("Olmo3", "olmo3"),
    ("SeedOss", "seed_oss"),
    ("SeedOSS", "seed_oss"),
    # Ernie 4.5 thinking — HF ships underscore variants too
    ("Ernie45", "ernie45"),
    ("Ernie4_5_VLMoe", "ernie45"),
    ("Ernie4_5_Moe", "ernie45"),
    ("Ernie4_5", "ernie45"),
    # GPT-OSS / OpenAI open-weights
    ("GptOss", "openai_gptoss"),
    ("OpenaiMoe", "openai_gptoss"),
    ("Holo2", "holo2"),
    # MiMo (Xiaomi) — `mimo` reasoning parser is NOT in vLLM's registered set
    # (verified against vLLM 0.19.1's `_REASONING_PARSERS_TO_REGISTER`); using
    # it would fail server startup. Per Xiaomi's own vLLM recipe for
    # MiMo-V2-Flash and MiMo-V2.5, the recommended pairing is
    # `--reasoning-parser qwen3 --tool-call-parser qwen3_xml`. Route to
    # qwen3 reasoning here; tool-side stays in detect_tool_parser.
    ("MiMo", "qwen3"),
    ("Mimo", "qwen3"),
)


def _arch_to_parser(arch: str) -> str:
    if not arch:
        return ""
    for prefix, parser in _ARCH_TO_REASONING_PARSER:
        if arch.startswith(prefix):
            return parser
    return ""


def _load_arch(model_path: str) -> str:
    cfg = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg):
        return ""
    try:
        with open(cfg) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    archs = data.get("architectures") or []
    if not archs:
        return ""
    return archs[0]


# Markers that indicate the model emits chain-of-thought / thinking content
# the OpenAI client cannot interpret without a `--reasoning-parser`.
_THINKING_MARKERS: tuple[str, ...] = (
    "<think>",
    "</think>",
    "<thinking>",
    "</thinking>",
    "<|channel|>",  # gpt-oss style
    "<|reasoning|>",  # generic
    "<reasoning>",
    "<scratchpad>",
)


def _has_thinking_template(model_path: str) -> bool:
    """True if the model's chat template advertises thinking-block markers.

    Conservative: only returns True when an explicit thinking marker is found
    in `chat_template.jinja`, `chat_template.json`, or the embedded
    `chat_template` field of `tokenizer_config.json`. Avoids false positives
    on non-thinking variants of a thinking-capable architecture (e.g. a
    Qwen3 fine-tune that has CoT disabled in the template).
    """
    candidates = (
        os.path.join(model_path, "chat_template.jinja"),
        os.path.join(model_path, "chat_template.json"),
        os.path.join(model_path, "tokenizer_config.json"),
    )
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                blob = f.read()
        except OSError:
            continue
        for marker in _THINKING_MARKERS:
            if marker in blob:
                return True
    return False


# Layer 2 fallback: thinking-marker patterns -> reasoning parser. When the
# architecture mapping doesn't hit but a thinking marker is present, fall
# back to the most-likely parser for that marker family. Mirrors what
# llama.cpp does for unknown architectures emitting recognizable templates.
_TEMPLATE_PATTERN_TO_REASONING: tuple[tuple[str, str], ...] = (
    # Most-specific markers first
    ("<|tool▁calls▁begin|>", "deepseek_v3"),  # DeepSeek V3 family thinking
    ("<|channel|>", "openai_gptoss"),  # GPT-OSS channel-style reasoning
    ("<scratchpad>", "granite"),  # Granite-style scratchpad
    # Generic <think>/<thinking> markers default to qwen3 (most common
    # in-the-wild thinking format; vLLM's qwen3_reasoning_parser handles
    # the standard <think>...</think> block shape).
    ("<think>", "qwen3"),
    ("<thinking>", "qwen3"),
)


def _pattern_fallback(model_path: str) -> str:
    """Layer 2 fallback: pick a reasoning parser by template content when
    the architecture mapping returned empty."""
    candidates = (
        os.path.join(model_path, "chat_template.jinja"),
        os.path.join(model_path, "chat_template.json"),
        os.path.join(model_path, "tokenizer_config.json"),
    )
    blob_parts: list[str] = []
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                blob_parts.append(f.read())
        except OSError:
            continue
    blob = "\n".join(blob_parts)
    for pattern, parser in _TEMPLATE_PATTERN_TO_REASONING:
        if pattern in blob:
            return parser
    return ""


def _is_suppressed(arch: str, model_dir_name: str) -> bool:
    """True if architecture or model directory name signals a non-thinking
    variant of a thinking-capable family. Defends against false positives
    seen in llama.cpp #20809 / #21616 / #20754 (Qwen3-Instruct-2507,
    Reka Edge, Nemotron-Nano `/no_think` toggles)."""
    if any(arch.startswith(p) for p in _REASONING_SUPPRESS_PREFIXES):
        return True
    name = model_dir_name or ""
    if any(suffix in name for suffix in _NON_REASONING_NAME_SUFFIXES):
        return True
    if any(sub in name for sub in _REASONING_SUPPRESS_NAME_SUBSTRINGS):
        return True
    return False


def _name_discriminator(parser: str, model_dir_name: str) -> str:
    """Refine an architecture-derived reasoning parser using directory
    name signals. Covers cases where a model ships with the parent family's
    arch string but is actually a specialized variant:

      - DeepSeek-R1 (and R1-Distill-* forks) ship `DeepseekV3ForCausalLM`
        arch and get mapped to `deepseek_v3`. Bump to the dedicated
        `deepseek_r1` parser when "R1" or "Distill" appears in dir name.
      - Kimi K2.5/K2.6 ship `DeepseekV3ForCausalLM`. Bump to `kimi_k2`
        for parity with the tool detector.
    """
    name_lower = (model_dir_name or "").lower()
    if parser == "deepseek_v3":
        if "r1" in name_lower.split("-") or "-r1-" in f"-{name_lower}-" or "distill" in name_lower:
            return "deepseek_r1"
        if (
            "kimi" in name_lower
            or "/k2" in name_lower
            or name_lower.startswith("k2")
            or "-k2" in name_lower
        ):
            return "kimi_k2"
    return parser


def detect_parser(model: str) -> str:
    """Return reasoning parser name for the given model path, or '' if none.

    Three-layer detection (defense in depth):
      1. Architecture-prefix mapping (precise, mainstream thinking models).
      2. Template-pattern fallback (catches fine-tunes / unknown archs that
         still emit recognizable thinking markers).
      3. Thinking-template gate (suppress injection when the chat template
         contains no thinking markers, even if architecture matches a
         thinking family — protects non-CoT fine-tunes).
      4. Suppression list (architectures and model name suffixes known to
         produce false positives in llama.cpp's autoparser are explicitly
         excluded — Reka Edge, Ring 2.0, Qwen3-*-Instruct-2507, etc).

    HF model ids without a local directory return '' (auto-injection skipped;
    user can pass --reasoning-parser).
    """
    if not model or not os.path.isdir(model):
        return ""
    # Gate first: no thinking template -> definitely no reasoning parser
    if not _has_thinking_template(model):
        return ""
    arch = _load_arch(model)
    # Suppression check comes BEFORE arch mapping so forks of thinking
    # families (e.g. Qwen3-Instruct-2507 derivatives) get correctly skipped.
    if _is_suppressed(arch, os.path.basename(model.rstrip("/"))):
        return ""
    parser = _arch_to_parser(arch)
    if parser:
        return _name_discriminator(parser, os.path.basename(model.rstrip("/")))
    # Layer 2: architecture didn't map; try template-pattern fallback
    return _pattern_fallback(model)


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    parser = detect_parser(sys.argv[1])
    if parser:
        sys.stdout.write(parser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
