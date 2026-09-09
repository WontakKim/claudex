"""Tests for the Anthropic Messages -> Codex Responses request translation."""

import json
import re
import urllib.parse
from copy import deepcopy

import pytest

import claudex.translate.claude_to_codex as claude_to_codex
from claudex.translate.claude_to_codex import (
    TranslationError,
    build_tool_name_shortening_map,
    shorten_call_id,
    translate_claude_request_to_codex,
)
from claudex.translate.thought_signature import (
    CARRIER_PREFIX,
    encode_call_signature_carrier,
)


def _find_items(payload: dict, item_type: str) -> list[dict]:
    return [item for item in payload["input"] if item["type"] == item_type]


def _carrier_block(provider: str, call_id: str, signature: str) -> dict:
    carrier = encode_call_signature_carrier(provider, call_id, signature)
    assert carrier is not None
    return {"type": "thinking", "thinking": "...", "signature": carrier}


def _tool_use_block(call_id: str, name: str = "lookup") -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": {}}


def _translate_assistant_content(
    content: list[dict], *, custom_provider: str | None
) -> dict:
    return translate_claude_request_to_codex(
        {"messages": [{"role": "assistant", "content": content}]},
        codex_model="gpt-5.5",
        custom_provider=custom_provider,
    )


def test_basic_request_shape() -> None:
    payload = translate_claude_request_to_codex(
        {
            "model": "claude-sonnet-4-5",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hello"}],
        },
        codex_model="gpt-5.5",
    )

    assert payload["model"] == "gpt-5.5"
    assert payload["instructions"] == ""
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "tools" not in payload
    assert "parallel_tool_calls" not in payload
    assert "service_tier" not in payload

    developer, user = payload["input"]
    assert developer == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "You are helpful."}],
    }
    assert user == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }


def test_service_tier_is_included_when_set() -> None:
    payload = translate_claude_request_to_codex(
        {"messages": []}, codex_model="gpt-5.5", service_tier="priority"
    )

    assert payload["service_tier"] == "priority"


def test_service_tier_is_omitted_by_default() -> None:
    payload = translate_claude_request_to_codex(
        {"messages": []}, codex_model="gpt-5.5"
    )

    assert "service_tier" not in payload


def test_system_attribution_block_is_dropped() -> None:
    payload = translate_claude_request_to_codex(
        {
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: something"},
                {"type": "text", "text": "real system prompt"},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        },
        codex_model="gpt-5.5",
    )
    developer = _find_items(payload, "message")[0]
    assert developer["role"] == "developer"
    assert developer["content"] == [{"type": "input_text", "text": "real system prompt"}]


def test_mid_conversation_system_message_keeps_developer_authority() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "system", "content": "Terse mode enabled."},
                {"role": "user", "content": "second question"},
            ]
        },
        codex_model="gpt-5.5",
    )
    roles = [item["role"] for item in payload["input"]]
    assert roles == ["user", "developer", "user"]
    assert payload["input"][1]["content"] == [
        {"type": "input_text", "text": "Terse mode enabled."}
    ]


def _pdf_document_block(**overrides: object) -> dict:
    block = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
    }
    block.update(overrides)
    return block


def test_base64_pdf_document_becomes_input_file_in_order() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        _pdf_document_block(title="report.pdf"),
                        {"type": "text", "text": "after"},
                    ],
                }
            ]
        },
        codex_model="gpt-5.5",
    )
    assert payload["input"][0]["content"] == [
        {"type": "input_text", "text": "before"},
        {
            "type": "input_file",
            "filename": "report.pdf",
            "file_data": "data:application/pdf;base64,JVBERi0=",
        },
        {"type": "input_text", "text": "after"},
    ]


def test_pdf_document_without_title_gets_a_default_filename() -> None:
    payload = translate_claude_request_to_codex(
        {"messages": [{"role": "user", "content": [_pdf_document_block()]}]},
        codex_model="gpt-5.5",
    )
    assert payload["input"][0]["content"][0]["filename"] == "document.pdf"


@pytest.mark.parametrize(
    "source",
    [
        {"type": "url", "url": "https://example.com/a.pdf"},
        {"type": "file", "file_id": "file_123"},
        {"type": "base64", "media_type": "text/plain", "data": "aGk="},
        {"type": "base64", "media_type": "application/pdf"},
        None,
    ],
)
def test_unsupported_document_sources_are_rejected(source: object) -> None:
    block: dict = {"type": "document"}
    if source is not None:
        block["source"] = source
    with pytest.raises(TranslationError):
        translate_claude_request_to_codex(
            {"messages": [{"role": "user", "content": [block]}]},
            codex_model="gpt-5.5",
        )


def test_assistant_document_is_rejected() -> None:
    with pytest.raises(TranslationError, match="user messages"):
        translate_claude_request_to_codex(
            {"messages": [{"role": "assistant", "content": [_pdf_document_block()]}]},
            codex_model="gpt-5.5",
        )


def test_citation_enabled_document_is_rejected() -> None:
    with pytest.raises(TranslationError, match="citations"):
        translate_claude_request_to_codex(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [_pdf_document_block(citations={"enabled": True})],
                    }
                ]
            },
            codex_model="gpt-5.5",
        )


def test_document_inside_tool_result_is_rejected() -> None:
    with pytest.raises(TranslationError, match="tool_result"):
        translate_claude_request_to_codex(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_03",
                                "content": [_pdf_document_block()],
                            }
                        ],
                    }
                ]
            },
            codex_model="gpt-5.5",
        )


def test_assistant_text_uses_output_text() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        },
        codex_model="gpt-5.5",
    )
    assistant = payload["input"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [{"type": "output_text", "text": "answer"}]


def test_tool_use_and_tool_result_translation() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "running tool"},
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "read_file",
                            "input": {"path": "/tmp/x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": [{"type": "text", "text": "file contents"}],
                        }
                    ],
                },
            ]
        },
        codex_model="gpt-5.5",
    )

    message, function_call, function_call_output = payload["input"]
    assert message["content"] == [{"type": "output_text", "text": "running tool"}]
    assert function_call == {
        "type": "function_call",
        "call_id": "toolu_01",
        "name": "read_file",
        "arguments": '{"path": "/tmp/x"}',
    }
    assert function_call_output == {
        "type": "function_call_output",
        "call_id": "toolu_01",
        "output": [{"type": "input_text", "text": "file contents"}],
    }


def test_tool_result_string_content() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_02", "content": "ok"}
                    ],
                }
            ]
        },
        codex_model="gpt-5.5",
    )
    assert payload["input"][0]["output"] == "ok"


def test_thinking_block_with_gpt_signature_becomes_reasoning_item() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "...", "signature": "gAAAAABabc_123-="},
                        {"type": "text", "text": "done"},
                    ],
                }
            ]
        },
        codex_model="gpt-5.5",
    )
    reasoning_items = _find_items(payload, "reasoning")
    assert reasoning_items == [
        {
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": "gAAAAABabc_123-=",
        }
    ]


def test_thinking_block_with_foreign_signature_is_dropped() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "...", "signature": "EqQBCkgIChAB"},
                        {"type": "text", "text": "done"},
                    ],
                }
            ]
        },
        codex_model="gpt-5.5",
    )
    assert _find_items(payload, "reasoning") == []


def test_carrier_signature_attaches_to_matching_function_call() -> None:
    thought_signature = "opaque-signature+/=한글"
    payload = _translate_assistant_content(
        [
            _carrier_block("gemini", "toolu_01", thought_signature),
            _tool_use_block("toolu_01"),
        ],
        custom_provider="gemini",
    )

    assert _find_items(payload, "function_call") == [
        {
            "type": "function_call",
            "call_id": "toolu_01",
            "name": "lookup",
            "arguments": "{}",
            "extra_content": {
                "google": {"thought_signature": thought_signature}
            },
        }
    ]


def test_carrier_attaches_when_carrier_block_follows_tool_use() -> None:
    payload = _translate_assistant_content(
        [
            _tool_use_block("toolu_01"),
            _carrier_block("gemini", "toolu_01", "signature-after-call"),
        ],
        custom_provider="gemini",
    )

    function_call = _find_items(payload, "function_call")[0]
    assert function_call["extra_content"] == {
        "google": {"thought_signature": "signature-after-call"}
    }


def test_no_extra_content_without_custom_provider() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        _carrier_block("gemini", "toolu_01", "signature"),
                        _tool_use_block("toolu_01"),
                    ],
                }
            ]
        },
        codex_model="gpt-5.5",
    )

    assert _find_items(payload, "function_call") == [
        {
            "type": "function_call",
            "call_id": "toolu_01",
            "name": "lookup",
            "arguments": "{}",
        }
    ]


def test_carrier_provider_mismatch_is_dropped() -> None:
    payload = _translate_assistant_content(
        [
            _carrier_block("other-provider", "toolu_01", "signature"),
            _tool_use_block("toolu_01"),
        ],
        custom_provider="gemini",
    )

    assert "extra_content" not in _find_items(payload, "function_call")[0]


def test_duplicate_carriers_invalidate_call_even_when_identical() -> None:
    carrier = _carrier_block("gemini", "toolu_01", "same-signature")
    payload = _translate_assistant_content(
        [carrier, carrier.copy(), _tool_use_block("toolu_01")],
        custom_provider="gemini",
    )

    assert "extra_content" not in _find_items(payload, "function_call")[0]


def test_conflicting_carriers_invalidate_call() -> None:
    payload = _translate_assistant_content(
        [
            _carrier_block("gemini", "toolu_01", "first-signature"),
            _carrier_block("gemini", "toolu_01", "second-signature"),
            _tool_use_block("toolu_01"),
        ],
        custom_provider="gemini",
    )

    assert "extra_content" not in _find_items(payload, "function_call")[0]


def test_carrier_in_other_message_does_not_attach() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [_carrier_block("gemini", "toolu_01", "signature")],
                },
                {
                    "role": "assistant",
                    "content": [_tool_use_block("toolu_01")],
                },
            ]
        },
        codex_model="gpt-5.5",
        custom_provider="gemini",
    )

    assert "extra_content" not in _find_items(payload, "function_call")[0]


def test_unmatched_and_ambiguous_tool_use_ids_drop_carrier() -> None:
    unmatched_payload = _translate_assistant_content(
        [
            _carrier_block("gemini", "missing-call", "signature"),
            _tool_use_block("other-call"),
        ],
        custom_provider="gemini",
    )
    ambiguous_payload = _translate_assistant_content(
        [
            _carrier_block("gemini", "duplicate-call", "signature"),
            _tool_use_block("duplicate-call", "first_tool"),
            _tool_use_block("duplicate-call", "second_tool"),
        ],
        custom_provider="gemini",
    )

    for payload in (unmatched_payload, ambiguous_payload):
        assert all(
            "extra_content" not in item
            for item in _find_items(payload, "function_call")
        )


def test_parallel_calls_sibling_items_get_no_copy() -> None:
    payload = _translate_assistant_content(
        [
            _carrier_block("gemini", "toolu_01", "first-signature"),
            _tool_use_block("toolu_01", "first_tool"),
            _tool_use_block("toolu_02", "second_tool"),
        ],
        custom_provider="gemini",
    )

    matching_call, sibling_call = _find_items(payload, "function_call")
    assert matching_call["extra_content"] == {
        "google": {"thought_signature": "first-signature"}
    }
    assert "extra_content" not in sibling_call


@pytest.mark.parametrize("custom_provider", [None, "gemini"])
def test_carrier_never_becomes_reasoning_item(custom_provider: str | None) -> None:
    payload = _translate_assistant_content(
        [_carrier_block("gemini", "toolu_01", "signature")],
        custom_provider=custom_provider,
    )

    assert _find_items(payload, "reasoning") == []


def test_fernet_reasoning_replay_unchanged_with_custom_provider() -> None:
    payload = _translate_assistant_content(
        [
            {
                "type": "thinking",
                "thinking": "...",
                "signature": "gAAAAABabc_123-=",
            }
        ],
        custom_provider="gemini",
    )

    assert _find_items(payload, "reasoning") == [
        {
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": "gAAAAABabc_123-=",
        }
    ]


def test_malformed_carrier_dropped_fail_closed() -> None:
    payload = _translate_assistant_content(
        [
            {
                "type": "thinking",
                "thinking": "...",
                "signature": CARRIER_PREFIX + "not*base64",
            },
            _tool_use_block("toolu_01"),
        ],
        custom_provider="gemini",
    )

    assert _find_items(payload, "reasoning") == []
    assert "extra_content" not in _find_items(payload, "function_call")[0]


def test_image_block_becomes_data_url() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",
                            },
                        }
                    ],
                }
            ]
        },
        codex_model="gpt-5.5",
    )
    assert payload["input"][0]["content"] == [
        {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
    ]


def test_tools_are_normalized_and_web_search_translated() -> None:
    long_name = "mcp__some-really-long-server-name-here__" + "tool_" * 10 + "end"
    payload = translate_claude_request_to_codex(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
                {"name": long_name, "input_schema": None},
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "allowed_domains": ["docs.python.org"],
                    "user_location": {"type": "approximate", "country": "KR"},
                },
            ],
        },
        codex_model="gpt-5.5",
    )

    tools = payload["tools"]
    assert len(tools) == 3

    read_file = tools[0]
    assert read_file["type"] == "function"
    assert read_file["strict"] is False
    assert "$schema" not in read_file["parameters"]
    assert read_file["parameters"]["properties"] == {"path": {"type": "string"}}

    shortened = tools[1]
    assert len(shortened["name"]) <= 64
    assert shortened["name"].startswith("mcp__")
    assert shortened["parameters"] == {"type": "object", "properties": {}}

    web_search = tools[2]
    assert web_search == {
        "type": "web_search",
        "filters": {"allowed_domains": ["docs.python.org"]},
        "user_location": {"type": "approximate", "country": "KR"},
    }

    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True


def test_tool_choice_targeting_web_search_tool() -> None:
    payload = translate_claude_request_to_codex(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "tool", "name": "web_search"},
        },
        codex_model="gpt-5.5",
    )
    assert payload["tool_choice"] == {"type": "web_search"}


def test_shortened_names_are_unique() -> None:
    base = "mcp__server__" + "x" * 80
    request = {"tools": [{"name": base}, {"name": base + "y"}]}
    mapping = build_tool_name_shortening_map(request)
    assert len(set(mapping.values())) == 2
    assert all(len(name) <= 64 for name in mapping.values())


def test_tool_choice_mapping() -> None:
    base_request = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "read_file", "input_schema": {"type": "object", "properties": {}}}],
    }

    any_choice = translate_claude_request_to_codex(
        {**base_request, "tool_choice": {"type": "any"}}, codex_model="gpt-5.5"
    )
    assert any_choice["tool_choice"] == "required"

    tool_choice = translate_claude_request_to_codex(
        {**base_request, "tool_choice": {"type": "tool", "name": "read_file"}},
        codex_model="gpt-5.5",
    )
    assert tool_choice["tool_choice"] == {"type": "function", "name": "read_file"}

    disabled_parallel = translate_claude_request_to_codex(
        {**base_request, "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}},
        codex_model="gpt-5.5",
    )
    assert disabled_parallel["parallel_tool_calls"] is False


def test_empty_tools_emit_no_tool_fields() -> None:
    # tools: [] means "no tools"; emitting tool_choice alongside an empty list
    # is a 400 on Grok ("tool_choice set but no tools specified").
    for extra in ({}, {"tool_choice": {"type": "auto"}}):
        payload = translate_claude_request_to_codex(
            {"messages": [{"role": "user", "content": "hi"}], "tools": [], **extra},
            codex_model="gpt-5.5",
        )
        assert "tools" not in payload
        assert "tool_choice" not in payload
        assert "parallel_tool_calls" not in payload


# Claude Code emits ECMAScript u-mode regexes whose \p{...} Unicode property
# escapes the Codex upstream rejects with "'<pattern>' is not a 'regex'". The
# upstream validator behaves like Python's re module (inferred from that
# error message, not verified against its implementation). This is the
# retained Artifact tool regex in a reconstructed propertyNames fixture;
# the original full schema and the field's product role were not captured.
# Codex regex compatibility is opt-in per request
# because the same translation serves custom Responses backends that may
# validate with a JavaScript engine instead.
_ARTIFACT_PROPERTY_NAMES_PATTERN = (
    "^(?!__.*__$)[^\\p{Cc}\\p{Cf}\\p{Zl}\\p{Zp}\"\\\\./[\\]]{1,200}$"
)


def _artifact_property_names_tool() -> dict:
    return {
        "name": "Artifact",
        "description": "Publish a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "data": {
                    "type": "object",
                    "propertyNames": {"pattern": _ARTIFACT_PROPERTY_NAMES_PATTERN},
                },
            },
        },
    }


def _artifact_property_names_pattern(payload: dict) -> str:
    parameters = payload["tools"][0]["parameters"]
    return parameters["properties"]["data"]["propertyNames"]["pattern"]


def _translate_single_pattern(pattern: str) -> str:
    payload = translate_claude_request_to_codex(
        {
            "messages": [],
            "tools": [
                {
                    "name": "tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string", "pattern": pattern}},
                    },
                }
            ],
        },
        codex_model="gpt-5.5",
        codex_regex_compat=True,
    )
    return payload["tools"][0]["parameters"]["properties"]["x"]["pattern"]


def test_artifact_property_names_pattern_is_translated_to_a_python_regex() -> None:
    request = {
        "messages": [{"role": "user", "content": "publish"}],
        "tools": [_artifact_property_names_tool()],
    }
    original = deepcopy(request)

    payload = translate_claude_request_to_codex(
        request, codex_model="gpt-6-astra", codex_regex_compat=True
    )

    assert request == original
    translated = _artifact_property_names_pattern(payload)
    assert translated != _ARTIFACT_PROPERTY_NAMES_PATTERN
    assert "\\p{" not in translated
    # Astral codepoints stay as raw literal characters: the fixed-width \U
    # escape is valid for Python's re but not for JavaScript's u mode.
    assert "\\U" not in translated
    # The u-mode lookahead's dot is rewritten to the exact JS dot class so
    # the translated Python regex excludes the same line terminators, and
    # the $ inside the lookahead gets the same strict end anchor.
    assert "(?!__[^\\n\\r\\u2028\\u2029]*__$(?![\\s\\S]))" in translated
    re.compile(translated)


def test_translated_property_names_pattern_keeps_the_original_semantics() -> None:
    payload = translate_claude_request_to_codex(
        {"messages": [], "tools": [_artifact_property_names_tool()]},
        codex_model="gpt-6-astra",
        codex_regex_compat=True,
    )
    compiled = re.compile(_artifact_property_names_pattern(payload))

    accepted = [
        "hello",
        "My Page",
        "안녕하세요",
        "título",
        "a b",  # U+0020 is Zs; only the Zl/Zp separators are forbidden
        "a",
        "a" * 200,  # upper length boundary, counted in codepoints
        "__ab",  # the reserved rule rejects only __...__ (both ends)
        "a__b__",
        "__",
        "😀😀😀",
        "😀" * 200,  # astral characters count as one codepoint, like the u flag
        "🇰🇷",
        "é",
    ]
    for value in accepted:
        assert compiled.search(value), f"expected to accept {value!r}"

    rejected = [
        "",
        "a" * 201,
        "__reserved__",
        "____",
        "a\tb",  # Cc controls
        "a\nb",
        "a\x7fb",
        "a\x00b",
        "a\u200bb",  # Cf format characters (U+200B, U+00AD, U+FEFF, U+202E)
        "a\xadb",
        "a\ufeffb",
        "a\u202eb",
        "a\u2028b",  # Zl line separator (U+2028)
        "a\u2029b",  # Zp paragraph separator (U+2029)
        "a\U000e0020b",  # astral Cf tag space (U+E0020)
        "a\"b",  # forbidden separators/quotes/backslash/brackets
        "a\\b",
        "a.b",
        "a/b",
        "a[b",
        "a]b",
        # The u-mode original rejects a trailing newline before its end
        # anchor; Python's bare $ would accept one, so the translation must
        # not (a strict end-of-input rewrite, not re's newline leniency).
        "a\n",
        "a" * 200 + "\n",
    ]
    for value in rejected:
        assert not compiled.search(value), f"expected to reject {value!r}"


def test_unicode_property_patterns_are_translated_at_every_schema_position() -> None:
    schema = {
        "type": "object",
        "properties": {
            "plain": {"type": "string", "pattern": "^[a-z]+$"},
            "label": {"type": "string", "pattern": "^[^\\p{Cc}]{1,4}$"},
            "pair": {
                "type": "array",
                "items": {"type": "string", "pattern": "^\\p{Zl}?$"},
            },
            "triple": {
                "type": "array",
                "items": [
                    {"type": "string"},
                    {"type": "string", "pattern": "^\\p{Zp}?$"},
                ],
            },
            "combo": {
                "allOf": [{"type": "string", "pattern": "^\\p{Cc}?$"}],
                "anyOf": [{"type": "string", "pattern": "^\\p{Cf}?$"}],
                "oneOf": [{"type": "string"}],
                "not": {"type": "string", "pattern": "^\\p{Zl}$"},
            },
        },
        "additionalProperties": {"type": "string", "pattern": "^\\p{Zp}$"},
        "patternProperties": {"^[a-z]\\p{Cc}$": {"type": "string"}},
        "definitions": {"name": {"type": "string", "pattern": "^\\p{Cf}$"}},
        "$defs": {"slug": {"type": "string", "pattern": "^\\p{Cc}$"}},
    }
    request = {"messages": [], "tools": [{"name": "mix", "input_schema": schema}]}
    original = deepcopy(request)

    payload = translate_claude_request_to_codex(
        request, codex_model="gpt-5.5", codex_regex_compat=True
    )

    assert request == original
    assert "\\p{" not in json.dumps(payload)

    properties = payload["tools"][0]["parameters"]["properties"]
    assert properties["plain"]["pattern"] == "^[a-z]+$"
    re.compile(properties["label"]["pattern"])
    re.compile(properties["pair"]["items"]["pattern"])
    re.compile(properties["triple"]["items"][1]["pattern"])
    re.compile(properties["combo"]["allOf"][0]["pattern"])
    re.compile(properties["combo"]["anyOf"][0]["pattern"])
    assert "pattern" not in properties["combo"]["oneOf"][0]
    re.compile(properties["combo"]["not"]["pattern"])
    re.compile(payload["tools"][0]["parameters"]["additionalProperties"]["pattern"])
    (translated_key,) = payload["tools"][0]["parameters"]["patternProperties"]
    assert translated_key != "^[a-z]\\p{Cc}$"
    re.compile(translated_key)
    re.compile(payload["tools"][0]["parameters"]["definitions"]["name"]["pattern"])
    re.compile(payload["tools"][0]["parameters"]["$defs"]["slug"]["pattern"])


def test_schemas_without_unicode_escapes_are_unchanged() -> None:
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "pattern": "^[a-z]+$",
                "default": "home",
                "examples": ["home", "work"],
            },
            "count": {"type": "integer", "minimum": 0},
        },
    }
    payload = translate_claude_request_to_codex(
        {"messages": [], "tools": [{"name": "read", "input_schema": schema}]},
        codex_model="gpt-5.5",
        codex_regex_compat=True,
    )
    assert payload["tools"][0]["parameters"] == {
        "type": "object",
        "properties": schema["properties"],
    }


def test_tool_schema_regex_normalization_preserves_native_schema_shape() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "propertyNames": {"pattern": "^[^\\p{Cc}]{1,4}$"},
        "default": {"pattern": "\\p{Cc}"},
        "examples": [{"pattern": "\\p{Zl}"}],
        "x-provider-keyword": {"pattern": "\\p{Cf}"},
    }
    original = deepcopy(schema)

    normalized = claude_to_codex.normalize_tool_schema_regex(schema)

    assert schema == original
    assert normalized["$schema"] == schema["$schema"]
    assert normalized["type"] == "object"
    assert "properties" not in normalized
    assert "\\p{" not in normalized["propertyNames"]["pattern"]
    assert normalized["default"] == schema["default"]
    assert normalized["examples"] == schema["examples"]
    assert normalized["x-provider-keyword"] == schema["x-provider-keyword"]

    plain = {"type": "string", "pattern": "^[a-z]+$"}
    assert claude_to_codex.normalize_tool_schema_regex(plain) is plain
    assert claude_to_codex.normalize_tool_schema_regex(True) is True
    assert claude_to_codex.normalize_tool_schema_regex(False) is False


def test_literal_pattern_shaped_text_is_not_rewritten() -> None:
    # Literal example/default values, and a property that merely happens to be
    # named "pattern", must survive translation verbatim: only a "pattern"
    # string at a schema position is a regex.
    property_schema = {
        "type": "string",
        "enum": ["\\p{Cc}", "\\p{Zl}"],
        "const": "\\p{Cf}",
        "default": "\\p{Zp}",
        "examples": ["\\p{Cc}"],
    }
    schema = {"type": "object", "properties": {"pattern": property_schema}}
    request = {"messages": [], "tools": [{"name": "echo", "input_schema": schema}]}
    original = deepcopy(request)

    payload = translate_claude_request_to_codex(
        request, codex_model="gpt-5.5", codex_regex_compat=True
    )

    assert request == original
    assert payload["tools"][0]["parameters"] == {
        "type": "object",
        "properties": {"pattern": property_schema},
    }


@pytest.mark.parametrize(
    "pattern",
    [
        "^\\p{Lu}+$",  # category too large for a compact expansion
        "\\p{Script=Greek}",  # script property
        "\\P{Cc}",  # negated property escape
        "\\p{Cc",  # unterminated property
        "\\pZ",  # single-letter separator class Z
    ],
)
def test_unsupported_unicode_property_escapes_raise_translation_error(
    pattern: str,
) -> None:
    with pytest.raises(TranslationError, match="Unicode property escape"):
        translate_claude_request_to_codex(
            {
                "messages": [],
                "tools": [
                    {
                        "name": "tool",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "string", "pattern": pattern}
                            },
                        },
                    }
                ],
            },
            codex_model="gpt-5.5",
            codex_regex_compat=True,
        )


def test_escaped_literal_backslash_p_text_is_left_alone() -> None:
    # In JavaScript u mode a "{Zl}" after literal text is an incomplete
    # quantifier, so a real client escapes the braces: an escaped backslash
    # makes the leading "\\p\{Zl\}" literal text, and only the real escape
    # in the class after it must be translated.
    pattern = "\\\\p\\{Zl\\}[\\p{Zp}]"
    translated = _translate_single_pattern(pattern)

    assert translated.startswith("\\\\p\\{Zl\\}")
    assert "\\p{Zp}" not in translated
    compiled = re.compile(translated)
    assert compiled.search("\\p{Zl}\u2029")


def test_codex_regex_compat_is_opt_in_and_off_by_default() -> None:
    # Without the compatibility mode the schema must pass through verbatim:
    # custom Responses backends may validate patterns with a JavaScript
    # engine, and a global rewrite previously turned JS-valid patterns such
    # as \p{Lu} into hard TranslationError rejections for them.
    schema = {
        "type": "object",
        "properties": {
            "upper": {"type": "string", "pattern": "^\\p{Lu}+$"},
            "control": {"type": "string", "pattern": "^[^\\p{Cc}]{1,4}$"},
        },
        "patternProperties": {"^\\p{Zl}$": {"type": "string"}},
    }
    request = {"messages": [], "tools": [{"name": "t", "input_schema": schema}]}
    original = deepcopy(request)

    payload = translate_claude_request_to_codex(request, codex_model="gpt-5.5")

    assert request == original
    parameters = payload["tools"][0]["parameters"]
    assert parameters["properties"]["upper"]["pattern"] == "^\\p{Lu}+$"
    assert parameters["properties"]["control"]["pattern"] == "^[^\\p{Cc}]{1,4}$"
    assert list(parameters["patternProperties"]) == ["^\\p{Zl}$"]


def test_strict_anchor_rewrite_only_touches_unescaped_out_of_class_dollars() -> None:
    # Only a rewritten pattern gets the strict $(?![\s\S]) end anchor; an
    # escaped \$, a $ inside a character class, and any pattern without a
    # property escape keep the original text untouched.
    escaped_dollar = _translate_single_pattern("^\\p{Zl}\\$$")
    # Only the final unescaped $ becomes a strict anchor; the escaped \\
    # literal dollar before it keeps its exact text.
    assert escaped_dollar.count("$(?![\\s\\S])") == 1
    assert "\\$" in escaped_dollar

    in_class = _translate_single_pattern("^\\p{Zl}[$]$")
    assert in_class.count("$(?![\\s\\S])") == 1
    assert "[$]" in in_class

    untouched = _translate_single_pattern("^\\$[a]+$")
    assert untouched == "^\\$[a]+$"

    escaped_brace_dollar = _translate_single_pattern("^\\p{Zl}[\\$]$")
    assert escaped_brace_dollar.count("$(?![\\s\\S])") == 1
    assert "[\\$]" in escaped_brace_dollar


def test_translated_patterns_reject_a_trailing_newline_before_the_end_anchor() -> None:
    # Python's bare $ also matches just before one trailing newline; the
    # JavaScript u-mode original never does. Rewritten patterns must keep
    # the original's strict end-of-input meaning.
    translated = _translate_single_pattern("^\\p{Zl}x*$")
    assert translated != "^\\p{Zl}x*$"
    compiled = re.compile(translated)
    assert compiled.search("\u2028xxx")
    assert not compiled.search("\u2028xxx\n")
    assert not compiled.search("\u2028xxx\r")


def test_divergent_shorthands_and_dots_are_translated_to_equivalents() -> None:
    # \d, \w and their negations are ASCII-only in JavaScript (even in u
    # mode) and the u-mode dot excludes \r and U+2028/U+2029, while Python's
    # versions are Unicode-aware or narrower. Rewritten patterns use classes
    # that mean the same thing in both engines.
    digit = re.compile(_translate_single_pattern("^(?:\\p{Zl}|\\d)$"))
    assert digit.search("5")
    assert not digit.search("\u0661")  # Arabic-Indic digit: JS \d rejects it

    non_digit = re.compile(_translate_single_pattern("^(?:\\p{Zl}|\\D)$"))
    assert non_digit.search("\u0661")
    assert not non_digit.search("5")

    word = re.compile(_translate_single_pattern("^(?:\\p{Zl}|\\w)$"))
    assert word.search("a")
    assert not word.search("\u00e0")

    non_word = re.compile(_translate_single_pattern("^(?:\\p{Zl}|\\W)$"))
    assert non_word.search("\u00e0")
    assert not non_word.search("a")

    dot = re.compile(_translate_single_pattern("^(?:\\p{Zl}|.)$"))
    assert dot.search("a")
    assert not dot.search("\r")
    assert not dot.search("\u2029")
    assert dot.search("\u0661")

    in_class_digit = re.compile(_translate_single_pattern("^[\\p{Zl}\\d]$"))
    assert in_class_digit.search("5")
    assert not in_class_digit.search("\u0661")

    in_class_word = re.compile(_translate_single_pattern("^[\\p{Zl}\\w]$"))
    assert in_class_word.search("a")
    assert not in_class_word.search("\u00e0")

    # A dot inside a character class is a literal dot in both engines and
    # must not be rewritten.
    literal_dot = _translate_single_pattern("^[.\\p{Zl}]$")
    assert "[." in literal_dot
    assert re.compile(literal_dot).search(".")


@pytest.mark.parametrize(
    "pattern",
    [
        "\\p{Zl}\\s",  # whitespace sets differ between the engines
        "\\p{Zl}\\S",
        "[\\p{Zl}\\s]",
        "[\\p{Zl}\\D]",  # negated shorthands have no in-class equivalent
        "[\\p{Zl}\\W]",
        "[\\p{Zl}\\S]",
        "\\p{Zl}\\b",  # \b is ASCII-based in JS, Unicode-aware in Python
        "\\p{Zl}\\B",
        "\\p{Zl}\\uD83D\\uDE00",  # surrogate escapes are UTF-16 code units in JS
        "[\\p{Zl}\\uD83D]",
        "\\p{Zl}[]",  # JS: never-matching empty class; Python: literal ]
        "\\p{Zl}[^]",  # JS: any character; Python: unterminated set
    ],
)
def test_cross_engine_divergent_constructs_raise_translation_error(
    pattern: str,
) -> None:
    with pytest.raises(TranslationError, match="diverg"):
        _translate_single_pattern(pattern)


def test_patterns_without_property_escapes_keep_divergent_constructs() -> None:
    # A pattern the rewrite never touches keeps its original text, even for
    # constructs that would be rejected in a rewritten pattern.
    for pattern in [
        "^\\d\\s.$",
        "a[]b",
        "^\\uD83D\\uDE00$",
        "^\\w\\b$",
        "^(a)\\1$",
        "(?s:.)",
        "(?i)x",
    ]:
        assert _translate_single_pattern(pattern) == pattern


def test_pattern_properties_key_collisions_after_translation_fail_fast() -> None:
    # Two different patternProperties keys can map onto the same translated
    # key, which would silently drop one subschema. The translated key
    # includes the strict anchor rewrite, so the colliding key must too.
    def schema_with(keys: list) -> dict:
        return {"type": "object", "patternProperties": {key: True for key in keys}}

    solo = translate_claude_request_to_codex(
        {
            "messages": [],
            "tools": [
                {"name": "t", "input_schema": schema_with(["^\\p{Zl}$"])}
            ],
        },
        codex_model="gpt-5.5",
        codex_regex_compat=True,
    )
    (translated_key,) = solo["tools"][0]["parameters"]["patternProperties"]
    assert translated_key != "^\\p{Zl}$"
    assert translated_key.endswith("$(?![\\s\\S])")

    for first, second in [
        ("^\\p{Zl}$", translated_key),
        (translated_key, "^\\p{Zl}$"),
    ]:
        with pytest.raises(TranslationError, match="patternProperties"):
            translate_claude_request_to_codex(
                {
                    "messages": [],
                    "tools": [
                        {"name": "t", "input_schema": schema_with([first, second])}
                    ],
                },
                codex_model="gpt-5.5",
                codex_regex_compat=True,
            )


def test_pattern_properties_renames_with_local_refs_fail_fast() -> None:
    # A renamed patternProperties key invalidates any local $ref JSON
    # Pointer that addresses the old key, so the translation must stop
    # instead of shipping a schema whose reference now dangles.
    schema = {
        "type": "object",
        "$defs": {
            "holder": {
                "patternProperties": {r"\p{Zl}": {"type": "string"}},
            }
        },
        "properties": {
            "data": {"$ref": "#/$defs/holder/patternProperties/%5Cp%7BZl%7D"}
        },
    }
    # The pointer resolves to the key in the original schema (the key and
    # the pointer token must be the same text).
    assert _reference_resolves(
        schema, "#/$defs/holder/patternProperties/%5Cp%7BZl%7D"
    )
    with pytest.raises(TranslationError, match="\\$ref"):
        translate_claude_request_to_codex(
            {"messages": [], "tools": [{"name": "t", "input_schema": schema}]},
            codex_model="gpt-5.5",
            codex_regex_compat=True,
        )

    # Without a $ref in the schema, renaming keys stays safe.
    no_ref = {
        "type": "object",
        "patternProperties": {r"\p{Zl}": {"type": "string"}},
    }
    payload = translate_claude_request_to_codex(
        {"messages": [], "tools": [{"name": "t", "input_schema": no_ref}]},
        codex_model="gpt-5.5",
        codex_regex_compat=True,
    )
    (renamed,) = payload["tools"][0]["parameters"]["patternProperties"]
    assert renamed != "^\\p{Zl}$"


@pytest.mark.parametrize(
    "pattern",
    [
        # re.compile raises OverflowError at or beyond the compiler's
        # repetition limit (_sre.MAXREPEAT).
        "^\\p{Zl}{4294967296}",
        # A repetition count literal past the interpreter's decimal string
        # conversion limit raises ValueError inside re.compile.
        "\\p{Zl}{" + "1" * 4301 + "}",
        # Deeply nested groups exhaust the parser's recursion and raise
        # RecursionError.
        "(?:" * 600 + "\\p{Zl}" + ")" * 600,
    ],
    ids=["huge-repetition", "digit-limit-repetition", "deep-nesting"],
)
def test_compiler_limits_raise_translation_error(pattern: str) -> None:
    with pytest.raises(TranslationError, match="cannot be translated"):
        _translate_single_pattern(pattern)


def test_no_property_patterns_skip_the_compile_check_entirely() -> None:
    # Patterns without a real property escape are returned before any
    # compile check runs, so even compiler-crashing shapes pass through.
    for pattern in ["(?:" * 600 + "x" + ")" * 600, "x{" + "1" * 4301 + "}"]:
        assert _translate_single_pattern(pattern) == pattern




def test_patterns_with_only_literal_property_text_stay_byte_identical() -> None:
    # A doubled backslash makes the following p/P literal text, so these
    # patterns contain no real property escape. They must stay byte-identical
    # even when they hold constructs the rewrite would reject, because a
    # pattern that is never rewritten is never subject to the bounded
    # grammar.
    for pattern in [
        "\\\\p\\{Zl\\}\\s",
        "\\\\P{Cc}\\S",
        "\\\\p\\{Zl\\}[]",
        "\\\\p\\{Zl\\}\\b",
        "\\\\p\\{Zl\\}\\uD83D",
    ]:
        assert _translate_single_pattern(pattern) == pattern

    # Control: an escaped backslash followed by a REAL escape is still
    # detected escape-aware and translated.
    transformed = _translate_single_pattern("\\\\\\p{Zl}")
    assert transformed != "\\\\\\p{Zl}"
    assert "\\p{Zl}" not in transformed
    re.compile(transformed)


def _pointer_tokens(fragment: str) -> list[str]:
    # Stdlib URI-fragment JSON Pointer tokens, sufficient for the fixtures
    # below: percent-decode each segment, then unescape ~1/~0.
    decoded = urllib.parse.unquote(fragment)
    return [
        token.replace("~1", "/").replace("~0", "~")
        for token in decoded.split("/")
        if token
    ]


def _reference_resolves(schema: dict, ref: str) -> bool:
    # Focused same-resource resolution check for these fixtures: when the
    # schema declares an $id, the ref's base (the ref resolved against it)
    # must be that same resource, and the fragment must walk to a node.
    parsed = urllib.parse.urlsplit(ref)
    if not parsed.fragment:
        return False
    base = schema.get("$id")
    if base is not None:
        if urllib.parse.urljoin(base, ref).split("#", 1)[0] != base:
            return False
    node: Any = schema
    for token in _pointer_tokens(parsed.fragment):
        if not isinstance(node, dict) or token not in node:
            return False
        node = node[token]
    return True


def test_reference_instructions_with_renamed_keys_fail_fast() -> None:
    # A renamed patternProperties key can dangle any reference instruction:
    # same-resource refs arrive not only as '#/...' fragments but as
    # percent-encoded fragments, absolute URIs with a fragment, and relative
    # URIs with a fragment, and $dynamicRef/$recursiveRef address keys the
    # same way. The guard therefore rejects any schema-position reference
    # when a key was renamed, rather than guessing locality from the URI
    # spelling.
    refs = [
        "#/$defs/holder/patternProperties/%5Cp%7BZl%7D",
        "#%2F$defs%2Fholder%2FpatternProperties%2F%5Cp%7BZl%7D",
        "https://example.test/tool.json#/$defs/holder/patternProperties/%5Cp%7BZl%7D",
        "tool.json#/$defs/holder/patternProperties/%5Cp%7BZl%7D",
    ]
    for ref in refs:
        schema = {
            "$id": "https://example.test/tool.json",
            "type": "object",
            "$defs": {
                "holder": {
                    "patternProperties": {r"\p{Zl}": {"type": "string"}}
                }
            },
            "properties": {"x": {"$ref": ref}},
        }
        # The original ref genuinely resolves to the key about to be renamed.
        assert _reference_resolves(schema, ref)
        with pytest.raises(TranslationError, match="\\$ref"):
            translate_claude_request_to_codex(
                {"messages": [], "tools": [{"name": "t", "input_schema": schema}]},
                codex_model="gpt-5.5",
                codex_regex_compat=True,
            )

    for keyword in ("$dynamicRef", "$recursiveRef"):
        ref = "#/$defs/holder/patternProperties/%5Cp%7BZl%7D"
        schema = {
            "type": "object",
            "$defs": {
                "holder": {
                    "patternProperties": {r"\p{Zl}": {"type": "string"}}
                }
            },
            "properties": {"x": {keyword: ref}},
        }
        assert _reference_resolves(schema, ref)
        with pytest.raises(TranslationError, match="reference instruction"):
            translate_claude_request_to_codex(
                {"messages": [], "tools": [{"name": "t", "input_schema": schema}]},
                codex_model="gpt-5.5",
                codex_regex_compat=True,
            )


def test_external_refs_with_renamed_keys_are_rejected_conservatively() -> None:
    # Documented conservative boundary: even a genuinely external ref is
    # rejected in combination with a renamed key. Establishing true
    # externality would need resolved resource identifiers, which the guard
    # deliberately does not attempt.
    schema = {
        "type": "object",
        "patternProperties": {r"\p{Zl}": {"type": "string"}},
        "properties": {"x": {"$ref": "https://elsewhere.test/other.json#/x"}},
    }
    with pytest.raises(TranslationError, match="\\$ref"):
        translate_claude_request_to_codex(
            {"messages": [], "tools": [{"name": "t", "input_schema": schema}]},
            codex_model="gpt-5.5",
            codex_regex_compat=True,
        )


def test_references_without_renames_pass_through_unchanged() -> None:
    # With no patternProperties key renamed, reference instructions keep
    # their exact text: the guard is rename-gated, not reference-gated.
    schema = {
        "$id": "https://example.test/tool.json",
        "type": "object",
        "$defs": {
            "holder": {"patternProperties": {"^[a-z]+$": {"type": "string"}}}
        },
        "properties": {
            "x": {"$ref": "#/$defs/holder/patternProperties/%5E%5Ba-z%5D%2B%24"},
        },
        "default": {"$ref": "#/anything"},
    }
    payload = translate_claude_request_to_codex(
        {"messages": [], "tools": [{"name": "t", "input_schema": schema}]},
        codex_model="gpt-5.5",
        codex_regex_compat=True,
    )
    assert payload["tools"][0]["parameters"] == schema


def test_literal_ref_shaped_data_is_not_a_reference_instruction() -> None:
    # $ref text inside default/examples values is literal data, not a
    # reference the validator follows, so a rename elsewhere must not fail
    # the translation on its behalf.
    schema = {
        "type": "object",
        "patternProperties": {"^\\p{Zl}$": {"type": "string"}},
        "default": {"$ref": "#/patternProperties/%5Cp%7BZl%7D"},
        "examples": [{"$ref": "#%2Fanything"}],
    }
    payload = translate_claude_request_to_codex(
        {"messages": [], "tools": [{"name": "t", "input_schema": schema}]},
        codex_model="gpt-5.5",
        codex_regex_compat=True,
    )
    parameters = payload["tools"][0]["parameters"]
    (renamed_key,) = parameters["patternProperties"]
    assert renamed_key != "^\\p{Zl}$"
    assert parameters["default"] == {"$ref": "#/patternProperties/%5Cp%7BZl%7D"}


@pytest.mark.parametrize(
    "pattern",
    [
        # A backreference to a group that did not participate matches empty
        # in JavaScript but fails in Python re.
        "^(a)?\\1\\p{Zl}$",
        # Scoped inline flags change what the dot or the anchors mean
        # (dotAll includes \n; JS multiline line terminators include
        # \r and U+2028/U+2029 while Python's include only \n).
        "(?s:.\\p{Zl})",
        "(?im:.)\\p{Zl}",
        # Bare inline flags are Python-valid but not JavaScript-valid; a
        # rewritten pattern must not carry them either.
        "(?i)\\p{Zl}$",
        "(?m)\\p{Zl}$",
    ],
)
def test_backreferences_and_inline_flags_raise_translation_error(
    pattern: str,
) -> None:
    with pytest.raises(TranslationError, match="diverg"):
        _translate_single_pattern(pattern)


def test_thinking_budget_to_reasoning_effort() -> None:
    def effort_for(thinking: dict) -> str:
        payload = translate_claude_request_to_codex(
            {"messages": [], "thinking": thinking}, codex_model="gpt-5.5"
        )
        return payload["reasoning"]["effort"]

    assert effort_for({"type": "enabled", "budget_tokens": 400}) == "minimal"
    assert effort_for({"type": "enabled", "budget_tokens": 2048}) == "medium"
    assert effort_for({"type": "enabled", "budget_tokens": 16000}) == "high"
    assert effort_for({"type": "enabled", "budget_tokens": 30000}) == "xhigh"
    assert effort_for({"type": "disabled"}) == "low"
    assert effort_for({"type": "adaptive"}) == "xhigh"
    payload = translate_claude_request_to_codex(
        {
            "messages": [],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
        codex_model="gpt-5.5",
    )
    assert payload["reasoning"]["effort"] == "max"


def test_reasoning_effort_override_wins() -> None:
    payload = translate_claude_request_to_codex(
        {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 400}},
        codex_model="gpt-5.5",
        reasoning_effort_override="xhigh",
    )
    assert payload["reasoning"]["effort"] == "xhigh"


def test_shorten_call_id_is_stable_and_bounded() -> None:
    long_id = "toolu_" + "a" * 100
    first = shorten_call_id(long_id)
    second = shorten_call_id(long_id)
    assert first == second
    assert len(first) <= 64
    assert shorten_call_id("toolu_short") == "toolu_short"


_IMAGE_ANALYSIS_CALL_ID = "call_120ce1bcb52744d6a5034c48"


def _image_analysis_messages(*, merged: bool, result_content: object) -> list[dict]:
    call = {
        "type": "server_tool_use",
        "name": "analyze_image",
        "id": _IMAGE_ANALYSIS_CALL_ID,
        "input": {},
    }
    result = {
        "type": "tool_result",
        "tool_use_id": _IMAGE_ANALYSIS_CALL_ID,
        "content": result_content,
    }
    if merged:
        return [{"role": "assistant", "content": [call, result]}]
    return [
        {"role": "assistant", "content": [call]},
        {"role": "assistant", "content": [result]},
    ]


@pytest.mark.parametrize("merged", [False, True], ids=["separate", "merged"])
@pytest.mark.parametrize(
    "result_content",
    [
        "The image shows a blue triangle.",
        [
            {"type": "text", "text": "The image shows a blue triangle."},
            {"type": "text", "text": "A small caption is visible."},
        ],
    ],
    ids=["string-result", "text-block-result"],
)
def test_image_analysis_history_becomes_text_without_orphan_function_output(
    merged: bool, result_content: object
) -> None:
    request = {
        "messages": [
            {"role": "user", "content": "Describe the image."},
            *_image_analysis_messages(merged=merged, result_content=result_content),
            {"role": "user", "content": "Read the accompanying file."},
            {"role": "assistant", "content": [_tool_use_block("call_client", "read_file")]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_client", "content": "File contents."},
            ]},
        ],
    }
    original = deepcopy(request)

    payload = translate_claude_request_to_codex(request, codex_model="gpt-5.5")

    assert request == original
    assert _find_items(payload, "function_call") == [{
        "type": "function_call", "call_id": "call_client",
        "name": "read_file", "arguments": "{}",
    }]
    assert _find_items(payload, "function_call_output") == [{
        "type": "function_call_output", "call_id": "call_client", "output": "File contents.",
    }]
    assistant_parts = [
        part
        for item in _find_items(payload, "message") if item["role"] == "assistant"
        for part in item["content"]
    ]
    assert all(part["type"] == "output_text" for part in assistant_parts)
    assistant_text = "\n".join(part["text"] for part in assistant_parts)
    assert "analyze_image" in assistant_text
    assert "{}" in assistant_text
    assert "The image shows a blue triangle." in assistant_text
    if isinstance(result_content, list):
        assert "A small caption is visible." in assistant_text


def test_image_analysis_id_reused_by_later_client_tool_keeps_legitimate_result() -> None:
    request = {"messages": [
        {"role": "user", "content": "Describe the image."},
        *_image_analysis_messages(merged=False, result_content="An internal image analysis."),
        {"role": "user", "content": "Read the accompanying file."},
        {"role": "assistant", "content": [_tool_use_block(_IMAGE_ANALYSIS_CALL_ID, "read_file")]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": _IMAGE_ANALYSIS_CALL_ID,
             "content": "Later client tool result."},
        ]},
    ]}
    original = deepcopy(request)

    payload = translate_claude_request_to_codex(request, codex_model="gpt-5.5")

    assert request == original
    assert _find_items(payload, "function_call") == [{
        "type": "function_call", "call_id": _IMAGE_ANALYSIS_CALL_ID,
        "name": "read_file", "arguments": "{}",
    }]
    assert _find_items(payload, "function_call_output") == [{
        "type": "function_call_output", "call_id": _IMAGE_ANALYSIS_CALL_ID,
        "output": "Later client tool result.",
    }]
    assert "An internal image analysis." in str(_find_items(payload, "message"))


def test_image_analysis_repair_does_not_make_native_search_a_client_function_call() -> None:
    request = {
        "messages": [
            {"role": "user", "content": "Describe the image."},
            *_image_analysis_messages(merged=True, result_content="An internal image analysis."),
            {"role": "user", "content": "Search for context and read the file."},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "name": "web_search",
                 "id": "srvtoolu_native_search", "input": {"query": "image context"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_native_search",
                 "content": [{"type": "web_search_result", "title": "Context",
                              "url": "https://example.invalid/context", "page_age": None,
                              "encrypted_content": "native-opaque-content"}]},
                _tool_use_block("call_client", "read_file"),
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_client", "content": "File contents."},
            ]},
        ],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "read_file", "input_schema": {"type": "object", "properties": {}}},
        ],
    }
    original = deepcopy(request)

    payload = translate_claude_request_to_codex(request, codex_model="gpt-5.5")

    assert request == original
    assert _find_items(payload, "function_call") == [{
        "type": "function_call", "call_id": "call_client",
        "name": "read_file", "arguments": "{}",
    }]
    assert _find_items(payload, "function_call_output") == [{
        "type": "function_call_output", "call_id": "call_client", "output": "File contents.",
    }]
    assert {"type": "web_search"} in payload["tools"]
    assert "An internal image analysis." in str(_find_items(payload, "message"))
