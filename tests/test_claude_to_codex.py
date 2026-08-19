"""Tests for the Anthropic Messages -> Codex Responses request translation."""

import pytest

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
