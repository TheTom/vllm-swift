# SPDX-License-Identifier: Apache-2.0
"""Unit tests for response_rewriter.rewrite_request.

Pure-function tests — no aiohttp, no proxy, no vLLM. The proxy plumbing
is exercised by integration tests; this file pins the request-side
budget-rescue rule that prevents the OpenCode/Nemotron monologue regression.

Failure mode being prevented (see docs/`vllm-swift OpenCode max_tokens Trap`):
when a reasoning model receives `max_tokens=8192` from a client like
OpenCode, the `<think>` block can eat the whole budget; vLLM truncates,
`</think>` never closes, and the parser dumps raw thinking into `content`.
The rewriter silently bumps `max_tokens` to a reasoning-safe floor so
the model has headroom for a real answer / tool_call.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from vllm_swift.response_rewriter import (
    _REASONING_MAX_TOKENS_BUMP,
    _REASONING_MAX_TOKENS_FLOOR,
    _REASONING_PARSERS_WANT_ENABLE_THINKING_FALSE,
    _recover_tool_calls_from_content,
    rewrite_chat_completion,
    rewrite_request,
    stream_rewriter,
    stream_tool_recovery,
)

REASONING_PARSER = "nemotron_v3"
NON_REASONING_PARSER = ""


def test_bumps_starved_max_tokens_for_reasoning_parser():
    body = {"max_tokens": 8192, "messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM", reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_BUMP


def test_leaves_generous_max_tokens_alone():
    body = {"max_tokens": 65536, "messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM", reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == 65536


def test_leaves_max_tokens_at_floor_alone():
    """Boundary: floor itself is considered acceptable, not starved."""
    body = {"max_tokens": _REASONING_MAX_TOKENS_FLOOR, "messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM", reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_FLOOR


def test_no_bump_when_max_tokens_absent():
    body = {"messages": []}
    out = rewrite_request(body, arch="NemotronHForCausalLM", reasoning_parser=REASONING_PARSER)
    assert "max_tokens" not in out


def test_no_bump_for_non_reasoning_parser():
    """Tight budget on a non-reasoning model is the client's choice."""
    body = {"max_tokens": 256, "messages": []}
    out = rewrite_request(body, arch="LlamaForCausalLM", reasoning_parser=NON_REASONING_PARSER)
    assert out["max_tokens"] == 256


def test_no_bump_for_unknown_reasoning_parser_name():
    body = {"max_tokens": 256, "messages": []}
    out = rewrite_request(body, arch="WeirdoForCausalLM", reasoning_parser="not_a_real_parser")
    assert out["max_tokens"] == 256


@pytest.mark.parametrize(
    "parser",
    [
        "nemotron_v3",
        "qwen3",
        "deepseek_r1",
        "deepseek_v3",
        "openai_gptoss",
        "gemma4",
        "granite",
        "minimax_m2",
    ],
)
def test_all_known_reasoning_parsers_trigger_bump(parser: str):
    body = {"max_tokens": 8192}
    out = rewrite_request(body, arch="", reasoning_parser=parser)
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_BUMP, (
        f"reasoning parser {parser!r} should trigger budget bump"
    )


def test_zero_max_tokens_left_alone():
    """max_tokens=0 is meaningless; don't paper over it, let vLLM reject."""
    body = {"max_tokens": 0}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == 0


def test_negative_max_tokens_left_alone():
    body = {"max_tokens": -1}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER)
    assert out["max_tokens"] == -1


def test_bump_skipped_when_max_model_len_too_small_for_reasoning_headroom():
    """Empirical bug from Qwen3.5-2B (max_model_len=4096) on M2:
    bump to 32768 hit "max_tokens > max_model_len" 400. Even after
    clamping (4096 - prompt_reserve = ~2K), the leftover headroom is
    less than what reasoning needs — bumping at all just creates the
    follow-on bug "prompt + output > max_model_len". Skip entirely."""
    body = {"max_tokens": 256}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER, max_model_len=4096)
    # No bump — leave client's value alone, log the skip
    assert out["max_tokens"] == 256


def test_bump_unaffected_when_max_model_len_above_default_bump():
    """Large max_model_len (131K) — full bump applies, no clamp needed."""
    body = {"max_tokens": 8192}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER, max_model_len=131072)
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_BUMP


def test_bump_clamps_when_max_model_len_in_useful_band():
    """Mid-range context (e.g. 24K): bump clamps to half (12K) which
    is above the 8K useful threshold — clamp applies instead of skip."""
    body = {"max_tokens": 4096}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER, max_model_len=24576)
    # ceiling = 24576 - 12288 = 12288 (above 8K threshold)
    assert out["max_tokens"] == 12288


def test_bump_skips_when_clamped_value_below_requested():
    """Don't bump *down* — leave the client's value alone if our
    clamped bump is smaller than what they asked for."""
    body = {"max_tokens": 14000}
    out = rewrite_request(body, arch="", reasoning_parser=REASONING_PARSER, max_model_len=24576)
    # ceiling = 12288, but client wants 14000 — leave alone
    assert out["max_tokens"] == 14000


# ---------------------------------------------------------------------------
# Streaming rewriter — usage-chunk preservation
# ---------------------------------------------------------------------------


def _ssestream(events: list[str]) -> bytes:
    """Encode a list of SSE event payloads into a single byte stream."""
    return b"".join(f"data: {e}\n\n".encode() for e in events)


def _drive_rewriter(blob: bytes, arch: str) -> bytes:
    """Run stream_rewriter end-to-end on `blob`, return the joined output.

    Wraps the async machinery in `asyncio.run` so this stays a plain
    synchronous test — no pytest-asyncio dependency.
    """

    async def _async_iter_bytes():
        yield blob

    async def _collect():
        out = []
        async for chunk in stream_rewriter(_async_iter_bytes(), arch):
            out.append(chunk)
        return b"".join(out)

    return asyncio.run(_collect())


def test_stream_rewriter_preserves_usage_chunk():
    """Regression: vLLM emits the final `usage` block in a chunk with
    `choices: []`. Earlier rewriter versions silently dropped these
    metadata-only chunks because the per-choice loop never ran and the
    `if new_choices:` guard skipped the yield. Without this fix, Hermes'
    context counter never advances because token totals never reach the
    client.
    """
    content_chunk = (
        '{"id":"x","object":"chat.completion.chunk","created":1,'
        '"model":"m","choices":[{"index":0,"delta":{"content":"hi"},'
        '"finish_reason":"stop"}]}'
    )
    usage_chunk = (
        '{"id":"x","object":"chat.completion.chunk","created":1,'
        '"model":"m","choices":[],'
        '"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}'
    )
    blob = _ssestream([content_chunk, usage_chunk, "[DONE]"])
    joined = _drive_rewriter(blob, arch="NemotronH").decode()
    assert "usage" in joined, "usage chunk dropped by rewriter"
    assert '"prompt_tokens":10' in joined
    assert '"completion_tokens":2' in joined
    assert "[DONE]" in joined


def test_stream_rewriter_passes_through_for_non_rewrite_arch():
    """Sanity: non-Nemotron arches go through the fast-path passthrough
    branch (no JSON parsing at all)."""
    payload = b"data: anything goes\n\n"
    out = _drive_rewriter(payload, arch="LlamaForCausalLM")
    assert out == payload


# ---------------------------------------------------------------------------
# Tool-call recovery from plaintext-JSON leaks in `content`
# ---------------------------------------------------------------------------


def test_recover_phi4_pipe_tag_leak():
    """Microsoft's own model card admits Phi-4-mini emits this shape as text.
    Recovery should synthesize structured tool_calls and clear the content."""
    content = (
        '<|tool_calls|>[{"name": "get_current_weather", '
        '"arguments": {"location": "Paris", "format": "celsius"}}]'
        "<|/tool_calls|>"
    )
    result = _recover_tool_calls_from_content(content)
    assert result is not None
    calls, residual = result
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_current_weather"
    assert "Paris" in calls[0]["function"]["arguments"]
    assert residual == ""


def test_recover_hermes_block():
    content = '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    result = _recover_tool_calls_from_content(content)
    assert result is not None
    calls, residual = result
    assert calls[0]["function"]["name"] == "bash"
    assert '"command"' in calls[0]["function"]["arguments"]


def test_recover_qwen3_coder_xml():
    """qwen3_coder XML emitted as text instead of parsed."""
    content = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nls -la /tmp\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = _recover_tool_calls_from_content(content)
    assert result is not None
    calls, residual = result
    assert calls[0]["function"]["name"] == "bash"
    assert "ls -la /tmp" in calls[0]["function"]["arguments"]


def test_recover_mistral_bracket():
    content = '[TOOL_CALLS][{"name": "read", "arguments": {"path": "/etc/hosts"}}]'
    result = _recover_tool_calls_from_content(content)
    assert result is not None
    calls, _residual = result
    assert calls[0]["function"]["name"] == "read"


def test_recover_multiple_hermes_blocks():
    content = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    result = _recover_tool_calls_from_content(content)
    assert result is not None
    calls, _residual = result
    assert len(calls) == 2
    assert {c["function"]["name"] for c in calls} == {"a", "b"}


def test_recover_no_match_on_chat_text():
    """Plain chat content with no tool-call shape returns None."""
    content = "I think you should run `ls` to see the files."
    assert _recover_tool_calls_from_content(content) is None


def test_recover_skips_below_ratio_threshold():
    """If the tool-call shape is a tiny fraction of content, skip recovery
    to avoid false positives where the model is *describing* a tool call."""
    big_chat = "Here is what a hermes tool call looks like for reference: "
    big_chat += "x" * 2000
    big_chat += '<tool_call>{"name": "demo", "arguments": {}}</tool_call>'
    big_chat += " That's how the format works."
    # The actual block is ~60 chars in 2K+ of chat — ratio too low
    assert _recover_tool_calls_from_content(big_chat) is None


def test_recover_rejects_malformed_inner_json():
    content = "<tool_call>{not valid json}</tool_call>"
    assert _recover_tool_calls_from_content(content) is None


def test_rewrite_chat_completion_recovers_phi4_leak():
    """End-to-end: a non-streaming response with the leak shape gets
    rewritten — tool_calls populated, content cleared, finish_reason
    bumped to tool_calls."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<|tool_calls|>[{"name": "ls", "arguments": {}}]<|/tool_calls|>',
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
    }
    rewrite_chat_completion(payload)
    msg = payload["choices"][0]["message"]
    assert msg["tool_calls"]
    assert msg["tool_calls"][0]["function"]["name"] == "ls"
    assert msg["content"] in (None, "")
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_rewrite_chat_completion_does_not_clobber_existing_tool_calls():
    """If the message already has structured tool_calls, recovery is a no-op."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<tool_call>{"name": "evil", "arguments": {}}</tool_call>',
                    "tool_calls": [
                        {
                            "id": "real-1",
                            "type": "function",
                            "function": {"name": "real_tool", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    rewrite_chat_completion(payload)
    msg = payload["choices"][0]["message"]
    # Original tool_calls preserved, NOT replaced by recovery
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "real_tool"


# ---------------------------------------------------------------------------
# Edge / boundary / negative tests for recovery (audit-driven)
# ---------------------------------------------------------------------------


def test_recover_returns_none_for_empty_content():
    assert _recover_tool_calls_from_content("") is None


def test_recover_returns_none_for_whitespace_only_content():
    assert _recover_tool_calls_from_content("   \n\t ") is None


def test_recover_at_exact_50pct_boundary_passes():
    """At exactly the ratio floor (>= 50% of content), recovery should fire.
    Crafted so the matched block is exactly half the content length."""
    block = '<tool_call>{"name":"a","arguments":{}}</tool_call>'  # 50 chars
    filler = "x" * len(block)  # equal-length filler — gives exactly 50%
    content = block + filler
    result = _recover_tool_calls_from_content(content)
    assert result is not None, (
        "boundary case: matched_len == 0.5 * len(content) should pass the "
        "`< int(len(content) * RATIO)` gate"
    )


def test_recover_just_below_boundary_skips():
    """Just below 50% should skip — defends the false-positive guard."""
    block = '<tool_call>{"name":"a","arguments":{}}</tool_call>'
    filler = "x" * (len(block) + 10)  # filler 10 chars longer
    content = block + filler
    assert _recover_tool_calls_from_content(content) is None


def test_recover_rejects_phi4_with_malformed_inner_json():
    content = "<|tool_calls|>[not valid json]<|/tool_calls|>"
    assert _recover_tool_calls_from_content(content) is None


def test_recover_rejects_qwen3_coder_with_no_function_tag():
    """qwen3_coder block without `<function=...>` inside is malformed."""
    content = "<tool_call>just text inside, no function</tool_call>"
    # Hermes regex fails (no `{`), qwen3_coder regex fails (no function=).
    # Falls through to other shapes which also miss. Total: None.
    assert _recover_tool_calls_from_content(content) is None


def test_recover_rejects_mistral_with_non_array_payload():
    """Mistral's bracket regex demands an array. Object payload misses."""
    content = '[TOOL_CALLS]{"name":"x","arguments":{}}'
    assert _recover_tool_calls_from_content(content) is None


def test_recover_drops_calls_with_empty_function_name():
    """Validation: name must be a non-empty string. Empty name → call dropped."""
    content = '<tool_call>{"name":"","arguments":{}}</tool_call>'
    assert _recover_tool_calls_from_content(content) is None


def test_recover_drops_calls_with_int_arguments():
    """Validation: arguments must be dict or stringified-JSON-str. Int rejected."""
    content = '<tool_call>{"name":"a","arguments":42}</tool_call>'
    assert _recover_tool_calls_from_content(content) is None


def test_recover_drops_calls_with_null_arguments():
    content = '<tool_call>{"name":"a","arguments":null}</tool_call>'
    assert _recover_tool_calls_from_content(content) is None


def test_recover_accepts_stringified_json_arguments():
    """Some leaks ship arguments already stringified — accept and pass through."""
    content = '<tool_call>{"name":"a","arguments":"{\\"k\\":1}"}</tool_call>'
    result = _recover_tool_calls_from_content(content)
    assert result is not None
    calls, _residual = result
    # Should preserve the stringified form rather than double-encode
    assert calls[0]["function"]["arguments"] == '{"k":1}'


def test_recover_drops_calls_with_unparseable_string_arguments():
    """If arguments is a string but not valid JSON, drop the call."""
    content = '<tool_call>{"name":"a","arguments":"not json"}</tool_call>'
    assert _recover_tool_calls_from_content(content) is None


def test_rewrite_chat_completion_handles_multiple_choices():
    """Recovery iterates over all `choices` independently."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": '<|tool_calls|>[{"name":"a","arguments":{}}]<|/tool_calls|>',
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            },
            {
                "message": {
                    "content": "regular chat in the second choice",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            },
        ],
    }
    rewrite_chat_completion(payload)
    # First choice: recovery fired
    assert payload["choices"][0]["message"]["tool_calls"]
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    # Second choice: untouched
    assert not payload["choices"][1]["message"].get("tool_calls")
    assert payload["choices"][1]["finish_reason"] == "stop"


def test_rewrite_chat_completion_does_not_overwrite_finish_reason_when_already_tool_calls():
    """If finish_reason was already `tool_calls`, leave it alone (don't double-bump)."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": '<|tool_calls|>[{"name":"a","arguments":{}}]<|/tool_calls|>',
                    "tool_calls": None,
                },
                "finish_reason": "tool_calls",  # already set, somehow
            }
        ],
    }
    rewrite_chat_completion(payload)
    # Recovery still fires (tool_calls is None initially), and finish_reason
    # stays "tool_calls" — the "in (stop, length, None)" guard correctly
    # excludes already-set tool_calls.
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"]


def test_rewrite_chat_completion_preserves_other_finish_reasons():
    """`content_filter` and other vLLM finish reasons should NOT be bumped."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": '<|tool_calls|>[{"name":"a","arguments":{}}]<|/tool_calls|>',
                    "tool_calls": None,
                },
                "finish_reason": "content_filter",
            }
        ],
    }
    rewrite_chat_completion(payload)
    # Recovery doesn't bump this — only `stop`, `length`, `None` are bumped.
    assert payload["choices"][0]["finish_reason"] == "content_filter"


def test_recover_residual_preserves_surrounding_text():
    """Text before/after the matched block stays in `residual`."""
    block = '<tool_call>{"name":"a","arguments":{}}</tool_call>'
    content = "OK calling: " + block
    # Make sure ratio passes by keeping prefix small
    result = _recover_tool_calls_from_content(content)
    assert result is not None
    _calls, residual = result
    assert residual == "OK calling:"


# ---------------------------------------------------------------------------
# Fixture-based replay tests — anonymized snapshots of the actual agent
# request/response shapes that triggered the original bugs in this PR.
# These exist so a future change can't silently regress against the
# specific traffic shape that broke things in production.
# ---------------------------------------------------------------------------

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture_json(name: str) -> dict:
    with open(os.path.join(_FIXTURES_DIR, name)) as f:
        return json.load(f)


def _strip_capture_metadata(payload: dict) -> dict:
    """Capture metadata is for human readers; strip before passing to rewriter."""
    payload.pop("_capture_metadata", None)
    return payload


def test_replay_opencode_request_triggers_max_tokens_bump():
    """Original failure: OpenCode hardcodes max_tokens=8192 against a
    reasoning model. Without the bump rescue, the model burns its budget
    inside <think> and never emits final content. The replay confirms the
    rescue fires for this exact request shape."""
    body = _strip_capture_metadata(_load_fixture_json("request_opencode_nemotron.json"))
    assert body["max_tokens"] == 8192, "fixture sanity: should be the starvation value"
    out = rewrite_request(body, arch="NemotronHForCausalLM", reasoning_parser="nemotron_v3")
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_BUMP, (
        "rescue failed to fire on the OpenCode-shaped request that originally wedged"
    )


def test_replay_hermes_uncapped_request_does_not_bump():
    """Hermes leaves max_tokens=null. Bump must NOT fire — defensive against
    accidentally rewriting requests that don't need rescue."""
    body = _strip_capture_metadata(_load_fixture_json("request_hermes_uncapped.json"))
    assert body["max_tokens"] is None, "fixture sanity: should be null"
    out = rewrite_request(body, arch="NemotronHForCausalLM", reasoning_parser="nemotron_v3")
    assert out["max_tokens"] is None, "rewriter accidentally bumped a None max_tokens"


def test_replay_phi4_pipe_leak_response_recovers_to_structured_tool_calls():
    """Original failure: Phi-4-mini emits <|tool_calls|>[{...}]<|/tool_calls|>
    as plain content (per Microsoft's own model card + vllm-project/vllm#14682).
    Recovery should extract the structured call into message.tool_calls."""
    payload = _strip_capture_metadata(_load_fixture_json("response_phi4_pipe_leak.json"))
    msg_before = payload["choices"][0]["message"]
    assert msg_before["tool_calls"] is None, "fixture sanity: leak shape, no structured calls"
    assert "<|tool_calls|>" in msg_before["content"], "fixture sanity: leak shape present"

    rewrite_chat_completion(payload)

    msg_after = payload["choices"][0]["message"]
    assert msg_after["tool_calls"], "recovery failed to extract from phi4 leak shape"
    assert msg_after["tool_calls"][0]["function"]["name"] == "get_current_weather"
    assert "Paris" in msg_after["tool_calls"][0]["function"]["arguments"]
    assert msg_after.get("content") in (None, "")
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_replay_qwen3_coder_xml_leak_response_recovers():
    """Defense-in-depth: even with our detector now routing Qwen3.5+/3.6+
    MoE to qwen3_coder, if a user manually overrides to hermes (or some
    future variant misroutes), the qwen3_coder XML leak should still
    auto-recover into structured tool_calls."""
    payload = _strip_capture_metadata(_load_fixture_json("response_qwen3_coder_xml_leak.json"))
    msg_before = payload["choices"][0]["message"]
    assert not msg_before["tool_calls"], "fixture sanity: leak shape, no structured calls"
    assert "<function=bash>" in msg_before["content"]

    rewrite_chat_completion(payload)

    msg_after = payload["choices"][0]["message"]
    assert msg_after["tool_calls"], "recovery failed to extract from qwen3_coder XML leak"
    assert msg_after["tool_calls"][0]["function"]["name"] == "bash"
    assert "ls -l" in msg_after["tool_calls"][0]["function"]["arguments"]
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_replay_streaming_response_preserves_usage_chunk():
    """Captured streaming SSE shape (vLLM 0.19.1 emits usage in a separate
    chunk with empty `choices`, just before [DONE]). Earlier rewriter
    versions silently dropped this chunk (Hermes context counter never
    advanced). Replay confirms it now survives."""
    with open(os.path.join(_FIXTURES_DIR, "response_streaming_with_usage.txt"), "rb") as f:
        blob = f.read()
    out = _drive_rewriter(blob, arch="NemotronH").decode()
    assert '"usage"' in out, "usage chunk dropped by rewriter on captured stream shape"
    assert '"prompt_tokens":35' in out
    assert '"completion_tokens":50' in out
    assert "[DONE]" in out


def test_replay_hermes_json_leak_response_recovers():
    """Hermes JSON tool-call shape leaked into content (parser misroute /
    future variant). Defensive: recovery must extract even when the
    detector picks the right parser today, in case it doesn't tomorrow."""
    payload = _strip_capture_metadata(_load_fixture_json("response_hermes_json_leak.json"))
    msg_before = payload["choices"][0]["message"]
    assert msg_before["tool_calls"] is None, "fixture sanity: leak shape, no structured calls"
    assert "<tool_call>" in msg_before["content"]

    rewrite_chat_completion(payload)

    msg_after = payload["choices"][0]["message"]
    assert msg_after["tool_calls"], "recovery failed to extract from hermes JSON leak"
    assert msg_after["tool_calls"][0]["function"]["name"] == "bash"
    assert "ls /tmp/sweep-test" in msg_after["tool_calls"][0]["function"]["arguments"]
    assert msg_after.get("content") in (None, "")
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_replay_phi4_healthy_chat_does_not_trigger_recovery():
    """Real Phi-4-mini chat output (markdown code block + explanation, no
    leak shape). Recovery MUST NOT fire on healthy traffic — false
    positives would corrupt legitimate responses by extracting
    nonsense `tool_calls` from natural language.

    This is the strongest defense the test suite has against a future
    over-eager regex change. If someone tightens recovery and it starts
    matching markdown code blocks, this test fails immediately.
    """
    payload = _strip_capture_metadata(_load_fixture_json("response_phi4_healthy_chat.json"))
    original_content = payload["choices"][0]["message"]["content"]
    original_finish = payload["choices"][0]["finish_reason"]

    rewrite_chat_completion(payload)

    msg = payload["choices"][0]["message"]
    # Content must be untouched; tool_calls must remain None/empty;
    # finish_reason must not be bumped to tool_calls.
    assert msg["content"] == original_content, (
        "recovery false-positively rewrote healthy chat content"
    )
    assert not msg.get("tool_calls"), (
        "recovery false-positively synthesized tool_calls from chat text"
    )
    assert payload["choices"][0]["finish_reason"] == original_finish, (
        "recovery false-positively bumped finish_reason on healthy response"
    )


# ---------------------------------------------------------------------------
# Streaming tool-call recovery
# ---------------------------------------------------------------------------


def _drive_recovery(blob: bytes, tool_parser: str) -> bytes:
    """Run stream_tool_recovery end-to-end on `blob`, return joined output."""

    async def _async_iter():
        yield blob

    async def _collect():
        out = []
        async for chunk in stream_tool_recovery(_async_iter(), tool_parser):
            out.append(chunk)
        return b"".join(out)

    return asyncio.run(_collect())


def _sse(events: list[dict | str]) -> bytes:
    """Encode a list of dicts (or the literal '[DONE]') as SSE."""
    out = []
    for e in events:
        if e == "[DONE]":
            out.append("data: [DONE]\n\n")
        else:
            out.append(f"data: {json.dumps(e)}\n\n")
    return "".join(out).encode()


def _content_chunk(content: str, finish: str | None = None, idx: int = 0) -> dict:
    """Compact builder for a single content-delta chunk."""
    return {
        "id": "x",
        "model": "m",
        "choices": [{"index": idx, "delta": {"content": content}, "finish_reason": finish}],
    }


def test_stream_recovery_passthrough_for_non_leaky_parser():
    """Non-leaky parsers get a pure passthrough — no buffering overhead."""
    blob = _sse(
        [
            _content_chunk("hello"),
            _content_chunk(" world", finish="stop"),
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="hermes")
    assert out == blob, "non-leaky parser must be exact passthrough"


def test_stream_recovery_healthy_chat_streams_normally():
    """Phi-4-mini chat content (no leak shape) should reach the client,
    not get buffered until done."""
    blob = _sse(
        [
            _content_chunk("To list "),
            _content_chunk("files use "),
            _content_chunk("ls.", finish="stop"),
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert "To list " in out
    assert "files use" in out
    assert "ls." in out
    assert '"finish_reason": "stop"' in out
    assert "tool_calls" not in out
    assert "[DONE]" in out


def test_stream_recovery_phi4_leak_synthesizes_tool_call():
    """Leak-shaped streaming content gets recovered into a single
    structured tool_calls delta with finish_reason=tool_calls."""
    blob = _sse(
        [
            _content_chunk("<|tool_calls|>"),
            _content_chunk('[{"name": "bash"'),
            _content_chunk(', "arguments": {"command": "ls"}}]'),
            _content_chunk("<|/tool_calls|>", finish="stop"),
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert "<|tool_calls|>" not in out
    assert '"tool_calls"' in out
    assert '"name": "bash"' in out
    assert '"finish_reason": "tool_calls"' in out


def test_stream_recovery_marker_split_across_deltas():
    """Leak opener split across deltas (`<|` then `tool_calls|>`).
    The DECIDING state must keep buffering, not flip to passthrough early."""
    blob = _sse(
        [
            _content_chunk("<|"),
            _content_chunk("tool_calls|>"),
            _content_chunk(
                '[{"name":"a","arguments":{}}]<|/tool_calls|>',
                finish="stop",
            ),
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert '"tool_calls"' in out
    assert '"name":"a"' in out or '"name": "a"' in out
    assert '"finish_reason": "tool_calls"' in out


def test_stream_recovery_truncated_leak_flushes_as_content():
    """If finish_reason=length arrives mid-leak (model hit max_tokens),
    the partial content is flushed as content and finish_reason stays length."""
    blob = _sse(
        [
            _content_chunk("<|tool_calls|>"),
            _content_chunk(
                '[{"name":"a","arguments":{"loc":"Pa',
                finish="length",
            ),
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert "<|tool_calls|>" in out
    assert '"finish_reason": "length"' in out
    assert "tool_calls" not in out or '"finish_reason": "tool_calls"' not in out


def test_stream_recovery_already_structured_tool_calls_passthrough():
    """If the model emits structured tool_calls in a delta (parser DID
    extract correctly), recovery must not interfere."""
    structured = {
        "id": "x",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    finish = {
        "id": "x",
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    blob = _sse([structured, finish, "[DONE]"])
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert '"name": "bash"' in out
    assert '"finish_reason": "tool_calls"' in out


def test_stream_recovery_finish_in_separate_chunk_flushes_deciding_buffer():
    """Real-world vLLM pattern: last content chunk has finish_reason=None,
    then a separate empty-delta chunk has finish_reason=stop. The DECIDING
    buffer must flush at the second chunk, not get silently dropped."""
    blob = _sse(
        [
            _content_chunk("Hi there"),  # finish=None — still deciding
            {
                "id": "x",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            },
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert "Hi there" in out, "DECIDING buffer was dropped on separate finish chunk"
    assert '"finish_reason": "stop"' in out


def test_stream_recovery_done_without_finish_flushes_defensively():
    """Defensive: if upstream sends [DONE] without ever firing finish_reason
    on a buffered choice, the rewriter must still flush rather than swallow."""
    blob = _sse(
        [
            _content_chunk("orphan content"),
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert "orphan content" in out, "buffered content silently dropped at [DONE]"


def test_stream_recovery_metadata_chunks_passthrough():
    """vLLM's usage chunk (choices=[]) and similar must pass through unchanged."""
    usage_chunk = {
        "id": "x",
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    blob = _sse(
        [
            _content_chunk("hi", finish="stop"),
            usage_chunk,
            "[DONE]",
        ]
    )
    out = _drive_recovery(blob, tool_parser="phi4_mini_json").decode()
    assert '"usage"' in out
    assert '"prompt_tokens": 5' in out
    assert "[DONE]" in out


def test_replay_truncated_leak_response_skips_recovery_gracefully():
    """vLLM hit max_tokens mid-emission. The leak shape is partial — no
    closing tag — so the regex shape match fails. Recovery must skip
    cleanly rather than try to half-parse the truncated content.
    finish_reason=length stays as-is (don't promote to tool_calls)."""
    payload = _strip_capture_metadata(_load_fixture_json("response_truncated_leak.json"))
    original_content = payload["choices"][0]["message"]["content"]
    assert payload["choices"][0]["finish_reason"] == "length", "fixture sanity"
    assert "<|tool_calls|>" in original_content, "fixture sanity: partial leak"
    assert "<|/tool_calls|>" not in original_content, "fixture sanity: no closing tag"

    rewrite_chat_completion(payload)

    msg = payload["choices"][0]["message"]
    # Content untouched, no synthesized tool_calls, finish_reason stays length
    assert msg["content"] == original_content
    assert not msg.get("tool_calls")
    assert payload["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# rewrite_request: auto-inject chat_template_kwargs.enable_thinking=False
#
# Qwen3.5+ chat templates default-open `<think>` in the assistant prompt.
# Reasoning-naive clients (Aider, OpenCode chat mode) that don't pass
# `chat_template_kwargs={"enable_thinking": false}` get unbounded thinking
# routed into `reasoning_content` while `content` stays empty -> infinite
# retry loop. Symmetric to the existing tool-parser auto-injection.
#
# Validated 2026-05-11 against Qwen3.6-35B-A3B-ConfigI-MLX via curl: the
# kwarg alone makes `content` populate correctly with clean output.
# ---------------------------------------------------------------------------


def test_enable_thinking_false_qwen3_only():
    """The injection target set must scope tightly to qwen3 for now.

    Other reasoning parsers (deepseek_r1, gpt-oss, gemma4, ...) don't share
    the default-open-think template behavior. Bumping the set without
    matching template evidence would risk silently silencing thinking on
    other families.
    """
    assert _REASONING_PARSERS_WANT_ENABLE_THINKING_FALSE == frozenset({"qwen3"})


def test_injects_enable_thinking_false_for_qwen3_when_kwargs_absent():
    """The headline behavior — qwen3 parser + no chat_template_kwargs ->
    rewriter adds `enable_thinking: false`. This is the Aider/OpenCode
    repro: client sends nothing, model emits thinking forever, content
    stays null. Injection forces a clean answer-only response."""
    body = {"max_tokens": 65536, "messages": []}
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_injects_enable_thinking_false_when_chat_template_kwargs_is_none():
    """`chat_template_kwargs` present but explicitly None — same as absent.
    Some clients always send the key even when they have nothing to put in
    it, so the rewriter must treat None like 'no entry'."""
    body = {"max_tokens": 65536, "messages": [], "chat_template_kwargs": None}
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_injects_enable_thinking_false_when_chat_template_kwargs_not_dict():
    """Defensive: clients that send the wrong type (string, list) must not
    cause a crash, and the rewriter still applies the safe default. The
    isinstance guard in the rewriter is what we're pinning here."""
    body = {"max_tokens": 65536, "messages": [], "chat_template_kwargs": "garbage"}
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_preserves_explicit_enable_thinking_true():
    """Reasoning-aware clients can opt back into thinking by passing
    `enable_thinking: true` explicitly. The rewriter must leave that
    alone — otherwise the auto-inject would silently break workflows
    that genuinely want chain-of-thought."""
    body = {
        "max_tokens": 65536,
        "messages": [],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    assert out["chat_template_kwargs"] == {"enable_thinking": True}


def test_preserves_explicit_enable_thinking_false():
    """Idempotency: if the client already sent the right value, the
    rewriter must not 'helpfully' double-set it or otherwise touch the
    dict."""
    body = {
        "max_tokens": 65536,
        "messages": [],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_preserves_other_chat_template_kwargs_while_injecting():
    """Other kwargs (model-specific template knobs) must survive the
    merge. The injection only adds `enable_thinking` when missing; it
    does not replace the entire dict."""
    body = {
        "max_tokens": 65536,
        "messages": [],
        "chat_template_kwargs": {"system_role": "system", "add_generation_prompt": True},
    }
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    assert out["chat_template_kwargs"] == {
        "system_role": "system",
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_no_injection_for_non_qwen3_reasoning_parser():
    """deepseek_r1 / gpt-oss / gemma4 etc. — their chat templates don't
    default-open `<think>`, so the injection must not fire. Different
    template, different bug; one auto-inject doesn't fit all."""
    body = {"max_tokens": 65536, "messages": []}
    out = rewrite_request(body, arch="DeepseekR1ForCausalLM", reasoning_parser="deepseek_r1")
    assert "chat_template_kwargs" not in out


def test_no_injection_when_no_reasoning_parser():
    """No reasoning parser detected -> nothing to defend against.
    `enable_thinking` is a Qwen3-template concept; passing it to models
    whose templates don't read the kwarg is a silent no-op at best and
    a template error at worst on stricter Jinja envs."""
    body = {"max_tokens": 65536, "messages": []}
    out = rewrite_request(body, arch="LlamaForCausalLM", reasoning_parser="")
    assert "chat_template_kwargs" not in out


def test_injection_fires_before_max_tokens_short_circuit():
    """The headline regression: Aider sends `max_tokens = 16384` exactly
    (the floor), so the budget-bump short-circuits without modifying the
    body. The thinking-injection must NOT live inside the bump branch —
    it has to run on every qwen3 request, regardless of whether the bump
    fires. Pin that ordering here so a future refactor can't move the
    injection inside the bump block."""
    body = {"max_tokens": _REASONING_MAX_TOKENS_FLOOR, "messages": []}
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    # max_tokens unchanged — confirms the bump short-circuited
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_FLOOR
    # …and yet the injection still fired
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_injection_and_bump_both_fire_when_starved():
    """The combined-rewrites path: starved max_tokens (8192) on qwen3 ->
    both the kwarg injection AND the max_tokens bump apply on the same
    request. Catches the case where short-circuiting one rewrite
    accidentally skips the other."""
    body = {"max_tokens": 8192, "messages": []}
    out = rewrite_request(body, arch="Qwen3MoeForCausalLM", reasoning_parser="qwen3")
    assert out["max_tokens"] == _REASONING_MAX_TOKENS_BUMP
    assert out["chat_template_kwargs"] == {"enable_thinking": False}
