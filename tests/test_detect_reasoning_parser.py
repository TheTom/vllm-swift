# SPDX-License-Identifier: Apache-2.0
"""Tests for the architecture -> reasoning-parser mapping used by `serve`."""

import json
import sys

import pytest

from vllm_swift import detect_reasoning_parser as drp


@pytest.mark.parametrize(
    "arch,expected",
    [
        # Empty / unknown
        ("", ""),
        ("UnknownArchForCausalLM", ""),
        # DeepSeek family
        ("DeepseekR1ForCausalLM", "deepseek_r1"),
        ("DeepSeekR1Distill", "deepseek_r1"),
        ("DeepseekV32ForCausalLM", "deepseek_v3"),
        ("DeepseekV31ForCausalLM", "deepseek_v3"),
        ("DeepseekV3ForCausalLM", "deepseek_v3"),
        ("DeepseekV2ForCausalLM", "deepseek_v3"),
        # Qwen3 family (all generations + variants -> qwen3 parser)
        ("Qwen3ForCausalLM", "qwen3"),
        ("Qwen3MoeForCausalLM", "qwen3"),
        ("Qwen3MoEForCausalLM", "qwen3"),
        ("Qwen3CoderForCausalLM", "qwen3"),
        ("Qwen3_5MoeForConditionalGeneration", "qwen3"),
        ("Qwen3_6ForCausalLM", "qwen3"),
        # Nemotron derivatives use Qwen3-style thinking blocks
        ("NemotronHForCausalLM", "nemotron_v3"),
        ("NemotronForCausalLM", "nemotron_v3"),
        # Gemma 4
        ("Gemma4ForCausalLM", "gemma4"),
        # Mistral reasoning
        ("MagistralForCausalLM", "mistral"),
        ("MistralForCausalLM", "mistral"),
        # GLM 4.5
        ("Glm45ForCausalLM", "glm45"),
        ("Glm4_5ForCausalLM", "glm45"),
        # Granite
        ("Granite4ForCausalLM", "granite"),
        ("GraniteForCausalLM", "granite"),
        # MiniMax M2
        ("MinimaxM2ForCausalLM", "minimax_m2"),
        ("MiniMaxM2ForCausalLM", "minimax_m2"),
        # Misc
        ("KimiK2ForCausalLM", "kimi_k2"),
        ("HunyuanA13BForCausalLM", "hunyuan_a13b"),
        ("Step3p5ForCausalLM", "step3p5"),
        ("Step3ForCausalLM", "step3"),
        ("Olmo3ForCausalLM", "olmo3"),
        ("SeedOssForCausalLM", "seed_oss"),
        ("SeedOSSForCausalLM", "seed_oss"),
        ("Ernie45ForCausalLM", "ernie45"),
        ("GptOssForCausalLM", "openai_gptoss"),
        ("Holo2ForCausalLM", "holo2"),
        # MiMo `mimo` reasoning parser is not in vLLM's registered set; route
        # to qwen3 per Xiaomi's official MiMo-V2-Flash / MiMo-V2.5 vLLM recipe.
        ("MiMoForCausalLM", "qwen3"),
        ("Glm47ForCausalLM", "glm45"),  # workaround for vllm-project/vllm#33348
    ],
)
def test_arch_to_parser(arch, expected):
    assert drp._arch_to_parser(arch) == expected


def test_load_arch_returns_empty_when_missing(tmp_path):
    assert drp._load_arch(str(tmp_path)) == ""


def test_load_arch_returns_empty_on_invalid_json(tmp_path):
    (tmp_path / "config.json").write_text("not json")
    assert drp._load_arch(str(tmp_path)) == ""


def test_load_arch_returns_empty_when_no_architectures(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert drp._load_arch(str(tmp_path)) == ""


def test_load_arch_reads_first_architecture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM", "OtherArch"]})
    )
    assert drp._load_arch(str(tmp_path)) == "Qwen3ForCausalLM"


@pytest.mark.parametrize(
    "marker",
    ["<think>", "</think>", "<thinking>", "</thinking>", "<|channel|>", "<|reasoning|>"],
)
def test_has_thinking_template_true_via_chat_template_jinja(tmp_path, marker):
    (tmp_path / "chat_template.jinja").write_text(f"prefix {marker} suffix")
    assert drp._has_thinking_template(str(tmp_path)) is True


def test_has_thinking_template_true_via_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template": "you may use <think>...</think> blocks"}'
    )
    assert drp._has_thinking_template(str(tmp_path)) is True


def test_has_thinking_template_false_when_no_files(tmp_path):
    assert drp._has_thinking_template(str(tmp_path)) is False


def test_has_thinking_template_false_when_no_marker(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template": "plain chat only"}')
    (tmp_path / "chat_template.jinja").write_text("no thinking blocks here")
    assert drp._has_thinking_template(str(tmp_path)) is False


def test_detect_parser_empty_for_non_directory():
    assert drp.detect_parser("") == ""
    assert drp.detect_parser("/nonexistent/path/xyz") == ""


def test_detect_parser_layer2_fallback_qwen3_for_unknown_arch(tmp_path):
    """Layer 2: unknown architecture but template has <think> markers ->
    fall back to the qwen3 parser (most-common thinking format)."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text("with <think> blocks")
    assert drp.detect_parser(str(tmp_path)) == "qwen3"


def test_detect_parser_layer2_fallback_deepseek_v3(tmp_path):
    """Layer 2: DeepSeek V3 thinking marker detected without arch match."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text("uses <|tool▁calls▁begin|> + <think>x</think>")
    # Layer 2 sees DeepSeek-specific marker before the generic <think>.
    assert drp.detect_parser(str(tmp_path)) == "deepseek_v3"


def test_detect_parser_layer2_fallback_gptoss_channel(tmp_path):
    """Layer 2: GPT-OSS channel marker -> openai_gptoss parser."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text("uses <|channel|>final<|message|>")
    assert drp.detect_parser(str(tmp_path)) == "openai_gptoss"


def test_detect_parser_no_fallback_when_no_thinking_marker(tmp_path):
    """Gate before fallback: no thinking marker -> empty regardless."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["UnknownArch"]}))
    (tmp_path / "chat_template.jinja").write_text("plain chat, no markers")
    assert drp.detect_parser(str(tmp_path)) == ""


def test_detect_parser_empty_when_no_thinking_marker(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text("no thinking markers, plain chat")
    assert drp.detect_parser(str(tmp_path)) == ""


def test_detect_parser_full_match_qwen3(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text("uses <think>...</think>")
    assert drp.detect_parser(str(tmp_path)) == "qwen3"


def test_detect_parser_full_match_deepseek_r1(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DeepseekR1Model"]}))
    (tmp_path / "chat_template.jinja").write_text("emits <think>cot</think>")
    assert drp.detect_parser(str(tmp_path)) == "deepseek_r1"


def test_detect_parser_nemotron_h_resolves_to_nemotron_v3(tmp_path):
    """Nemotron-Cascade is Qwen3.6-derivative but ships its own dedicated
    reasoning parser per NVIDIA (vLLM PR #36393). nemotron_v3 subclasses
    DeepSeekR1 with enable_thinking swap; qwen3 would also fire on the
    <think> markers, but nemotron_v3 is NVIDIA's recommended choice."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["NemotronHForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text(
        "Here's a thinking process: <think>plan</think> answer"
    )
    assert drp.detect_parser(str(tmp_path)) == "nemotron_v3"


def test_main_no_args_returns_0(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["detect_reasoning_parser"])
    assert drp.main() == 0


def test_main_writes_parser_to_stdout(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text("<think>cot</think>")
    monkeypatch.setattr(sys, "argv", ["detect_reasoning_parser", str(tmp_path)])
    rc = drp.main()
    assert rc == 0
    assert capsys.readouterr().out == "qwen3"


def test_main_silent_when_no_parser(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["detect_reasoning_parser", str(tmp_path)])
    rc = drp.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Name-substring suppression
#
# `_REASONING_SUPPRESS_NAME_SUBSTRINGS` skips reasoning-parser auto-injection
# when the model directory name matches certain tokens (e.g. `-Coder-`),
# even though the architecture string would otherwise map to a thinking
# parser. The suppress list previously contained `-ConfigI-`, which was
# the wrong fix: it disabled the reasoning parser entirely. The real fix
# lives in `response_rewriter._REASONING_PARSERS_WANT_ENABLE_THINKING_FALSE`
# (auto-injects `chat_template_kwargs={"enable_thinking": false}`). Pin
# both the kept entries and the absence of the reverted ones here so a
# future merge can't silently regress.
# ---------------------------------------------------------------------------


def _make_thinking_qwen3(tmp_path):
    """Helper: lay down a config.json + chat_template.jinja that would
    otherwise resolve to the qwen3 parser. Reused by suppression tests
    so each one only varies what we're actually testing (dir name)."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3MoeForCausalLM"]}))
    (tmp_path / "chat_template.jinja").write_text("uses <think>cot</think>")


def test_suppress_substrings_contains_coder_kept():
    """The `-Coder-` / `-coder-` entries are still load-bearing for the
    Qwen3-Coder family (chat-only XML tool-call format, no thinking).
    Pin presence so a refactor of the tuple doesn't drop them silently."""
    assert "-Coder-" in drp._REASONING_SUPPRESS_NAME_SUBSTRINGS
    assert "-coder-" in drp._REASONING_SUPPRESS_NAME_SUBSTRINGS


def test_suppress_substrings_does_not_contain_configi():
    """Regression guard: an earlier attempt added `-ConfigI-` here. That
    was over-broad. The correct fix is rewriter-side (auto-inject
    `enable_thinking: false`). If a future commit re-adds ConfigI to
    this tuple without removing the rewriter handling, Aider clients
    will lose `<think>`-aware behavior entirely instead of getting a
    clean content-only response."""
    assert "-ConfigI-" not in drp._REASONING_SUPPRESS_NAME_SUBSTRINGS
    assert "-configi-" not in drp._REASONING_SUPPRESS_NAME_SUBSTRINGS


def test_detect_parser_suppressed_by_coder_in_dir_name(tmp_path):
    """End-to-end coverage of the substring suppression path that was
    previously untested. Without a model_dir_name signal of Coder, the
    same arch + template would return qwen3 (covered by
    `test_detect_parser_full_match_qwen3`)."""
    coder_dir = tmp_path / "Qwen3-Coder-30B-A3B-Instruct"
    coder_dir.mkdir()
    _make_thinking_qwen3(coder_dir)
    assert drp.detect_parser(str(coder_dir)) == ""


def test_detect_parser_returns_qwen3_for_configi_dir(tmp_path):
    """Symmetric to the coder test, but in the opposite direction.
    ConfigI must NOT be suppressed — it still wants the qwen3 parser
    so vLLM can route `<think>` -> reasoning_content. The rewriter
    layer is what makes Aider-compatible by tacking on
    `enable_thinking: false`; this layer just keeps the parser hooked
    up so reasoning-aware clients (who opt back in) still work."""
    configi_dir = tmp_path / "Qwen3.6-35B-A3B-ConfigI-MLX"
    configi_dir.mkdir()
    _make_thinking_qwen3(configi_dir)
    assert drp.detect_parser(str(configi_dir)) == "qwen3"
