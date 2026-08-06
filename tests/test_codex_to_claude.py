"""Tests for the Codex Responses -> Anthropic Messages response translation."""

import json
import re

from claudex_gateway.translate.codex_to_claude import (
    CodexToClaudeStreamTranslator,
    assemble_claude_message,
    estimate_overflow_prompt_tokens,
    rewrite_context_overflow_message,
)

# Mirrors the Claude Code client's contract regex: without a match it falls
# back to trimming 20% per retry instead of the exact actual/limit overflow.
_CLIENT_OVERFLOW_RE = re.compile(
    r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", re.IGNORECASE
)


def _run_stream(
    claude_request: dict, codex_events: list[dict], *, context_window: int | None = None
) -> list[tuple[str, dict]]:
    translator = CodexToClaudeStreamTranslator(claude_request, context_window=context_window)
    events: list[tuple[str, dict]] = []
    for codex_event in codex_events:
        events.extend(translator.translate_event(codex_event))
    return events


def test_full_stream_with_thinking_text_and_tool_call() -> None:
    events = _run_stream(
        {"model": "claude-opus-4-6", "tools": [{"name": "read_file"}]},
        [
            {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.5"}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning"}},
            {"type": "response.reasoning_summary_part.added", "output_index": 0},
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking "},
            {"type": "response.reasoning_summary_text.delta", "delta": "hard"},
            {"type": "response.reasoning_summary_part.done", "output_index": 0},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "reasoning", "encrypted_content": "gAAAAABsig"},
            },
            {"type": "response.content_part.added", "part": {"type": "output_text"}},
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {"type": "response.content_part.done", "part": {"type": "output_text"}},
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {"type": "function_call", "call_id": "call_1", "name": "read_file"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 2,
                "delta": '{"path":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 2,
                "delta": '"/tmp/x"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {"type": "function_call", "call_id": "call_1", "name": "read_file"},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 40,
                        "input_tokens_details": {"cached_tokens": 60},
                    },
                    "output": [],
                },
            },
        ],
    )

    names = [name for name, _ in events]
    assert names == [
        "message_start",
        "content_block_start",  # thinking (index 0)
        "content_block_delta",  # thinking_delta
        "content_block_delta",  # thinking_delta
        "content_block_delta",  # signature_delta
        "content_block_stop",
        "content_block_start",  # text (index 1)
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",  # tool_use (index 2)
        "content_block_delta",  # empty input_json_delta
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    payloads = [payload for _, payload in events]

    # The Anthropic response answers as the model the client requested; the
    # Codex target model is a routing detail that stays off the wire.
    assert payloads[0]["message"]["model"] == "claude-opus-4-6"

    thinking_start = payloads[1]
    assert thinking_start["index"] == 0
    assert thinking_start["content_block"]["type"] == "thinking"
    signature_delta = payloads[4]
    assert signature_delta["delta"] == {"type": "signature_delta", "signature": "gAAAAABsig"}

    text_start = payloads[6]
    assert text_start["index"] == 1
    assert text_start["content_block"]["type"] == "text"
    assert payloads[7]["delta"] == {"type": "text_delta", "text": "Hello "}

    tool_start = payloads[10]
    assert tool_start["index"] == 2
    assert tool_start["content_block"]["type"] == "tool_use"
    assert tool_start["content_block"]["id"] == "call_1"
    assert tool_start["content_block"]["name"] == "read_file"
    tool_args = "".join(
        payload["delta"]["partial_json"]
        for payload in payloads[11:14]
        if payload["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(tool_args) == {"path": "/tmp/x"}

    message_delta = payloads[15]
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert message_delta["usage"] == {
        "input_tokens": 40,
        "output_tokens": 40,
        "cache_read_input_tokens": 60,
    }


def test_message_start_falls_back_to_codex_model_without_a_requested_model() -> None:
    events = _run_stream(
        {},
        [{"type": "response.created", "response": {"id": "resp_x", "model": "gpt-5.5"}}],
    )
    assert events[0][1]["message"]["model"] == "gpt-5.5"


def test_nameless_function_call_is_hydrated_on_done() -> None:
    events = _run_stream(
        {},
        [
            {"type": "response.created", "response": {"id": "resp_2", "model": "gpt-5.5"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_9"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": '{"a":1}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_9", "name": "late_tool"},
            },
            {
                "type": "response.completed",
                "response": {"id": "resp_2", "usage": {}, "output": []},
            },
        ],
    )

    names = [name for name, _ in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start_payload = events[1][1]
    assert start_payload["content_block"]["name"] == "late_tool"
    assert events[2][1]["delta"]["partial_json"] == '{"a":1}'
    assert events[4][1]["delta"]["stop_reason"] == "tool_use"


def test_message_item_fallback_when_no_text_deltas() -> None:
    events = _run_stream(
        {},
        [
            {"type": "response.created", "response": {"id": "resp_3", "model": "gpt-5.5"}},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "full text"}],
                },
            },
            {
                "type": "response.completed",
                "response": {"id": "resp_3", "usage": {}, "output": []},
            },
        ],
    )
    names = [name for name, _ in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "full text"}
    assert events[4][1]["delta"]["stop_reason"] == "end_turn"


def test_error_event_translates_to_claude_error() -> None:
    events = _run_stream(
        {}, [{"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}]
    )
    assert events == [
        (
            "error",
            {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )
    ]


def test_context_overflow_error_is_rewritten_for_claude_compaction() -> None:
    # Claude Code only triggers compaction when the message contains the
    # literal phrase "prompt is too long".
    events = _run_stream(
        {},
        [
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "message": "Your input exceeds the context window of this model.",
                },
            }
        ],
    )
    [(event_name, payload)] = events
    assert event_name == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == (
        "prompt is too long: Your input exceeds the context window of this model."
    )


def test_context_overflow_is_detected_by_message_phrase_without_code() -> None:
    events = _run_stream(
        {},
        [
            {
                "type": "response.failed",
                "response": {"error": {"message": "Request contains too many tokens."}},
            }
        ],
    )
    [(_, payload)] = events
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == "prompt is too long: Request contains too many tokens."


def test_failed_response_without_context_overflow_stays_api_error() -> None:
    events = _run_stream(
        {},
        [{"type": "response.failed", "response": {"error": {"message": "upstream exploded"}}}],
    )
    [(_, payload)] = events
    assert payload["error"] == {"type": "api_error", "message": "upstream exploded"}


def test_prompt_too_long_message_is_not_double_prefixed() -> None:
    events = _run_stream(
        {},
        [
            {
                "type": "error",
                "error": {
                    "code": "context_length_exceeded",
                    "message": "prompt is too long: 300000 tokens > 272000 maximum",
                },
            }
        ],
    )
    [(_, payload)] = events
    assert payload["error"]["message"] == "prompt is too long: 300000 tokens > 272000 maximum"


def test_context_overflow_error_event_synthesizes_numeric_pair_with_context_window() -> None:
    claude_request = {"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]}
    events = _run_stream(
        claude_request,
        [
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "message": "Your input exceeds the context window of this model.",
                },
            }
        ],
        context_window=272000,
    )
    [(_, payload)] = events
    message = payload["error"]["message"]
    match = _CLIENT_OVERFLOW_RE.search(message)
    assert match is not None
    assert int(match.group(1)) > int(match.group(2))
    assert "Your input exceeds the context window of this model." in message


def test_context_overflow_failed_event_synthesizes_numeric_pair_with_context_window() -> None:
    claude_request = {"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]}
    events = _run_stream(
        claude_request,
        [
            {
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "Backend refused: maximum context length exceeded.",
                    }
                },
            }
        ],
        context_window=272000,
    )
    [(_, payload)] = events
    message = payload["error"]["message"]
    match = _CLIENT_OVERFLOW_RE.search(message)
    assert match is not None
    assert int(match.group(1)) > int(match.group(2))
    assert "Backend refused: maximum context length exceeded." in message


def test_context_overflow_clamp_applies_floor_for_small_request() -> None:
    claude_request = {"model": "m"}
    events = _run_stream(
        claude_request,
        [
            {
                "type": "error",
                "error": {"code": "context_length_exceeded", "message": "too many tokens"},
            }
        ],
        context_window=272000,
    )
    [(_, payload)] = events
    match = _CLIENT_OVERFLOW_RE.search(payload["error"]["message"])
    assert match is not None
    assert int(match.group(1)) == (272000 * 110 + 99) // 100


def test_context_overflow_reports_char_based_estimate_above_floor() -> None:
    claude_request = {"model": "m", "messages": [{"role": "user", "content": "x" * 2_000_000}]}
    floor = (272000 * 110 + 99) // 100
    expected = estimate_overflow_prompt_tokens(claude_request)
    assert expected > floor  # sanity check that this fixture exercises the char-based branch

    events = _run_stream(
        claude_request,
        [
            {
                "type": "error",
                "error": {"code": "context_length_exceeded", "message": "too many tokens"},
            }
        ],
        context_window=272000,
    )
    [(_, payload)] = events
    match = _CLIENT_OVERFLOW_RE.search(payload["error"]["message"])
    assert match is not None
    assert int(match.group(1)) == expected


def test_context_overflow_without_context_window_matches_legacy_behavior() -> None:
    message = rewrite_context_overflow_message(
        "context_length_exceeded", "Your input exceeds the context window of this model."
    )
    assert message == "prompt is too long: Your input exceeds the context window of this model."


def test_non_overflow_error_with_context_window_is_unaffected() -> None:
    events = _run_stream(
        {"model": "m"},
        [{"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}],
        context_window=272000,
    )
    assert events == [
        ("error", {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    ]


def test_rewrite_context_overflow_message_is_idempotent_with_enrichment() -> None:
    claude_request = {"model": "m"}
    estimated_tokens = estimate_overflow_prompt_tokens(claude_request)
    first = rewrite_context_overflow_message(
        "context_length_exceeded",
        "too many tokens",
        estimated_tokens=estimated_tokens,
        context_window=272000,
    )
    second = rewrite_context_overflow_message(
        "context_length_exceeded",
        first,
        estimated_tokens=estimated_tokens,
        context_window=272000,
    )
    assert second == first


def test_rewrite_context_overflow_message_is_idempotent_after_neutralization() -> None:
    first = rewrite_context_overflow_message("context_length_exceeded", "100 tokens > 200")
    second = rewrite_context_overflow_message("context_length_exceeded", first)
    assert second == first
    assert _CLIENT_OVERFLOW_RE.search(first) is None


def test_invalid_numeric_pair_with_enrichment_synthesizes_valid_pair() -> None:
    claude_request = {"model": "m"}
    estimated_tokens = estimate_overflow_prompt_tokens(claude_request)
    message = rewrite_context_overflow_message(
        "context_length_exceeded",
        "prompt is too long: 100 tokens > 200",
        estimated_tokens=estimated_tokens,
        context_window=272000,
    )
    match = _CLIENT_OVERFLOW_RE.search(message)
    assert match is not None
    assert int(match.group(1)) > int(match.group(2))


def test_invalid_numeric_pair_without_enrichment_is_neutralized() -> None:
    message = rewrite_context_overflow_message(
        "context_length_exceeded", "prompt is too long: 100 tokens > 200"
    )
    assert _CLIENT_OVERFLOW_RE.search(message) is None
    assert "prompt is too long" in message.lower()


def test_phraseless_invalid_pair_does_not_mint_a_poison_pair() -> None:
    message = rewrite_context_overflow_message("context_length_exceeded", "100 tokens > 200")
    assert "prompt is too long" in message.lower()
    assert _CLIENT_OVERFLOW_RE.search(message) is None


def test_phraseless_valid_pair_is_preserved_after_legacy_prefix() -> None:
    message = rewrite_context_overflow_message("context_length_exceeded", "300 tokens > 200")
    match = _CLIENT_OVERFLOW_RE.search(message)
    assert match is not None
    assert int(match.group(1)) > int(match.group(2))


def test_invalid_pair_followed_by_second_numeric_phrase_has_no_match() -> None:
    message = rewrite_context_overflow_message(
        "context_length_exceeded", "100 tokens > 200 then retried with 300 tokens > 400"
    )
    assert _CLIENT_OVERFLOW_RE.search(message) is None


def test_estimate_overflow_prompt_tokens_matches_ceil_chars_over_3_2() -> None:
    claude_request = {"model": "m", "messages": [{"role": "user", "content": "x" * 100}]}
    chars = len(json.dumps(claude_request, ensure_ascii=False))
    assert estimate_overflow_prompt_tokens(claude_request) == (chars * 5 + 15) // 16


def test_tool_names_are_restored_from_shortened_form() -> None:
    long_name = "mcp__really-long-server-name-that-goes-on-forever__" + "t" * 40
    claude_request = {"tools": [{"name": long_name}]}

    from claudex_gateway.translate.claude_to_codex import build_tool_name_shortening_map

    short_name = build_tool_name_shortening_map(claude_request)[long_name]
    assert short_name != long_name

    events = _run_stream(
        claude_request,
        [
            {"type": "response.created", "response": {"id": "resp_4", "model": "gpt-5.5"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_2", "name": short_name},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_2", "name": short_name},
            },
        ],
    )
    tool_start = next(payload for name, payload in events if name == "content_block_start")
    assert tool_start["content_block"]["name"] == long_name


def test_non_stream_aggregation_from_stream_events() -> None:
    # The Codex backend returns an empty response.output when store=false, so
    # non-streaming responses are assembled from the translated stream events.
    events = _run_stream(
        {"model": "claude-sonnet-4-5", "tools": [{"name": "read_file"}]},
        [
            {"type": "response.created", "response": {"id": "resp_5", "model": "gpt-5.5"}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning"}},
            {"type": "response.reasoning_summary_part.added", "output_index": 0},
            {"type": "response.reasoning_summary_text.delta", "delta": "let me think"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "reasoning", "encrypted_content": "gAAAAABsig2"},
            },
            {"type": "response.output_text.delta", "delta": "the answer"},
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {"type": "function_call", "call_id": "call_3", "name": "read_file"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 2,
                "delta": '{"path": "/tmp/y"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {"type": "function_call", "call_id": "call_3", "name": "read_file"},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_5",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "input_tokens_details": {"cached_tokens": 4},
                    },
                    "output": [],
                },
            },
        ],
    )

    message = assemble_claude_message(events)
    assert message is not None
    assert message["id"] == "resp_5"
    assert message["model"] == "claude-sonnet-4-5"
    assert message["stop_reason"] == "tool_use"
    assert message["usage"] == {
        "input_tokens": 6,
        "output_tokens": 5,
        "cache_read_input_tokens": 4,
    }
    assert message["content"] == [
        {"type": "thinking", "thinking": "let me think", "signature": "gAAAAABsig2"},
        {"type": "text", "text": "the answer"},
        {"type": "tool_use", "id": "call_3", "name": "read_file", "input": {"path": "/tmp/y"}},
    ]


def test_assemble_returns_none_without_message_start() -> None:
    assert assemble_claude_message([]) is None


def test_web_search_call_emits_server_tool_use_and_result() -> None:
    events = _run_stream(
        {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        [
            {"type": "response.created", "response": {"id": "resp_7", "model": "gpt-5.6-sol"}},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "python 3.13 release date"},
                    "results": [
                        {"title": "Python 3.13", "url": "https://docs.python.org/3.13/"},
                        {"title": "", "url": "https://python.org/news"},
                        {"title": "no url entry"},
                    ],
                },
            },
            {"type": "response.output_text.delta", "delta": "Python 3.13 released in 2024."},
            {
                "type": "response.completed",
                "response": {"id": "resp_7", "usage": {}, "output": []},
            },
        ],
    )

    names = [name for name, _ in events]
    assert names == [
        "message_start",
        "content_block_start",  # server_tool_use (index 0)
        "content_block_delta",  # input_json_delta {"query": ...}
        "content_block_stop",
        "content_block_start",  # web_search_tool_result (index 1)
        "content_block_stop",
        "content_block_start",  # text (index 2)
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    tool_use_start = events[1][1]
    assert tool_use_start["index"] == 0
    assert tool_use_start["content_block"] == {
        "type": "server_tool_use",
        "id": "ws_1",
        "name": "web_search",
        "input": {},
    }
    assert json.loads(events[2][1]["delta"]["partial_json"]) == {
        "query": "python 3.13 release date"
    }

    result_start = events[4][1]
    assert result_start["index"] == 1
    assert result_start["content_block"] == {
        "type": "web_search_tool_result",
        "tool_use_id": "ws_1",
        "content": [
            {
                "type": "web_search_result",
                "title": "Python 3.13",
                "url": "https://docs.python.org/3.13/",
                "page_age": None,
            },
            {
                "type": "web_search_result",
                "title": "https://python.org/news",
                "url": "https://python.org/news",
                "page_age": None,
            },
        ],
    }

    # Duplicate done events for the same id must not emit blocks twice.
    assert names.count("message_delta") == 1
    assert events[9][1]["delta"]["stop_reason"] == "end_turn"

    message = assemble_claude_message(events)
    assert message is not None
    assert [block["type"] for block in message["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert message["content"][0]["input"] == {"query": "python 3.13 release date"}


def test_duplicate_web_search_done_is_deduplicated() -> None:
    done_event = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "web_search_call",
            "id": "ws_2",
            "action": {"type": "search", "query": "q"},
        },
    }
    events = _run_stream(
        {},
        [
            {"type": "response.created", "response": {"id": "resp_8", "model": "gpt-5.6-sol"}},
            done_event,
            done_event,
        ],
    )
    block_starts = [payload for name, payload in events if name == "content_block_start"]
    assert len(block_starts) == 2  # one server_tool_use + one result, not four


def test_incomplete_response_maps_max_tokens_stop_reason() -> None:
    events = _run_stream(
        {},
        [
            {"type": "response.created", "response": {"id": "resp_6", "model": "gpt-5.5"}},
            {"type": "response.output_text.delta", "delta": "partial"},
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_6",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {},
                    "output": [],
                },
            },
        ],
    )
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens"
