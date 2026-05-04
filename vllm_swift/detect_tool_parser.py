"""Detect appropriate vLLM tool-call parser from a model directory or HF id.

Reads config.json for the model architecture and maps to a known parser.
Prints the parser name to stdout, or empty if no mapping. Exits 0 always
(empty stdout means no auto-injection). Used by the vllm-swift wrapper
to provide smart defaults when the user runs `serve` without explicit
--tool-call-parser / --enable-auto-tool-choice flags.
"""
import json
import os
import sys


def _arch_to_parser(arch: str) -> str:
    if not arch:
        return ""
    a = arch
    pairs = [
        ("Qwen3CoderForCausalLM", "qwen3_coder"),
        ("Qwen3", "hermes"),
        ("Qwen2_5", "hermes"),
        ("Qwen2", "hermes"),
        ("NemotronH", "hermes"),
        ("Nemotron", "hermes"),
        ("HermesForCausalLM", "hermes"),
        ("Llama4", "llama4_json"),
        ("Llama", "llama3_json"),
        ("Mistral", "mistral"),
        ("Gemma4", "gemma4"),
        ("Gemma", "gemma4"),
        ("Phi4MiniJson", "phi4_mini_json"),
        ("Phi4", "phi4_mini_json"),
        ("Phi3", "phi4_mini_json"),
        ("Granite4", "granite4"),
        ("Granite", "granite"),
        ("DeepseekV32", "deepseek_v32"),
        ("DeepseekV31", "deepseek_v31"),
        ("DeepseekV3", "deepseek_v3"),
        ("DeepseekV2", "deepseek_v3"),
        ("Glm45", "glm45"),
        ("Glm47", "glm47"),
        ("Glm4", "glm45"),
        ("MinimaxM2", "minimax_m2"),
        ("MiniMaxM2", "minimax_m2"),
        ("MiniMax", "minimax"),
        ("Minimax", "minimax"),
        ("KimiK2", "kimi_k2"),
        ("HunyuanA13B", "hunyuan_a13b"),
        ("Step3", "step3"),
        ("Olmo3", "olmo3"),
        ("InternLM", "internlm"),
        ("Jamba", "jamba"),
        ("Ernie45", "ernie45"),
    ]
    for prefix, parser in pairs:
        if a.startswith(prefix):
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


def _has_tool_template(model_path: str) -> bool:
    """Heuristic: a tool-capable model usually exposes 'tools' in chat_template."""
    for fname in ("tokenizer_config.json", "chat_template.json", "chat_template.jinja"):
        path = os.path.join(model_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                blob = f.read()
        except OSError:
            continue
        if "tools" in blob and ("tool_call" in blob or "function" in blob):
            return True
    return False


def detect_parser(model: str) -> str:
    """Return parser name for the given model path/id, or '' if unmappable.

    Detection is conservative: only returns a parser when both the architecture
    mapping and chat-template tool fields are present. HF model ids without a
    local directory return '' (auto-injection skipped; user can pass --tool-call-parser).
    """
    if not model or not os.path.isdir(model):
        return ""
    arch = _load_arch(model)
    parser = _arch_to_parser(arch)
    if not parser:
        return ""
    if not _has_tool_template(model):
        return ""
    return parser


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    parser = detect_parser(sys.argv[1])
    if parser:
        sys.stdout.write(parser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
