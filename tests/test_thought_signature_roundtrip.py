"""Integration tests for thought-signature replay across translation boundaries."""

import copy
from typing import Any

from claudex.compaction import build_reroute_payload
from claudex.translate.claude_to_codex import translate_claude_request_to_codex
from claudex.translate.codex_to_claude import (
    CodexToClaudeStreamTranslator,
    assemble_claude_message,
)
from claudex.translate.thought_signature import CARRIER_PREFIX, MAX_SIGNATURE_BYTES

_ORIGIN_PROVIDER = "gemini-primary"
_SIGNED_CALL_ID = "call_signed"
_UNSIGNED_CALL_ID = "call_unsigned"


def _response_events(signature: str, response_model: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "response.created",
            "response": {"id": "response-round-trip", "model": response_model},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": _SIGNED_CALL_ID,
                "name": "signed_lookup",
                "extra_content": {"google": {"thought_signature": signature}},
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": '{"query":"signed"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": _SIGNED_CALL_ID,
                "name": "signed_lookup",
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": _UNSIGNED_CALL_ID,
                "name": "unsigned_lookup",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "delta": '{"query":"unsigned"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": _UNSIGNED_CALL_ID,
                "name": "unsigned_lookup",
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "response-round-trip",
                "model": response_model,
                "usage": {},
                "output": [],
            },
        },
    ]


def _assemble_assistant_message(
    signature: str,
    *,
    custom_provider: str = _ORIGIN_PROVIDER,
    response_model: str = "gemini-2.5-pro",
) -> dict[str, Any]:
    translator = CodexToClaudeStreamTranslator(
        {"model": "claude-sonnet-4-5"},
        custom_provider=custom_provider,
    )
    translated_events: list[tuple[str, dict[str, Any]]] = []
    for response_event in _response_events(signature, response_model):
        translated_events.extend(translator.translate_event(response_event))

    assembled_message = assemble_claude_message(translated_events)
    assert assembled_message is not None
    return assembled_message


def _next_turn_request(assistant_message: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-5",
        "messages": [
            {
                "role": "assistant",
                "content": assistant_message["content"],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": _SIGNED_CALL_ID,
                        "content": "signed result",
                    }
                ],
            },
        ],
    }


def _translate_next_turn(
    assistant_message: dict[str, Any],
    *,
    codex_model: str = "gemini-2.5-pro",
    custom_provider: str | None = _ORIGIN_PROVIDER,
) -> dict[str, Any]:
    return translate_claude_request_to_codex(
        _next_turn_request(assistant_message),
        codex_model=codex_model,
        custom_provider=custom_provider,
    )


def _function_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in payload["input"] if item["type"] == "function_call"]


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def _content_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for message in payload["messages"]
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict)
    ]


def test_full_round_trip_reattaches_signature_verbatim() -> None:
    signature = "  verbatim-signature+/=한글\n  "
    assistant_message = _assemble_assistant_message(signature)

    payload = _translate_next_turn(assistant_message)
    function_calls = _function_calls(payload)

    assert [call["call_id"] for call in function_calls] == [
        _SIGNED_CALL_ID,
        _UNSIGNED_CALL_ID,
    ]
    assert sum("extra_content" in call for call in function_calls) == 1
    assert function_calls[0]["extra_content"] == {
        "google": {"thought_signature": signature}
    }
    assert "extra_content" not in function_calls[1]


def test_round_trip_survives_same_provider_model_switch() -> None:
    signature = "same-provider-model-switch-signature"
    assistant_message = _assemble_assistant_message(
        signature,
        response_model="gemini-2.5-pro",
    )

    payload = _translate_next_turn(
        assistant_message,
        codex_model="gemini-2.5-flash",
        custom_provider=_ORIGIN_PROVIDER,
    )

    assert payload["model"] == "gemini-2.5-flash"
    assert _function_calls(payload)[0]["extra_content"] == {
        "google": {"thought_signature": signature}
    }


def test_round_trip_blocked_across_custom_providers() -> None:
    assistant_message = _assemble_assistant_message(
        "cross-provider-signature",
        custom_provider="gemini-primary",
    )

    payload = _translate_next_turn(
        assistant_message,
        custom_provider="gemini-secondary",
    )

    assert len(_function_calls(payload)) == 2
    assert not _contains_key(payload["input"], "extra_content")


def test_round_trip_blocked_on_builtin_route() -> None:
    assistant_message = _assemble_assistant_message("builtin-blocked-signature")

    payload = _translate_next_turn(assistant_message, custom_provider=None)

    assert not _contains_key(payload["input"], "extra_content")
    assert [item["type"] for item in payload["input"]] == [
        "function_call",
        "function_call",
        "function_call_output",
    ]


def test_compaction_payload_strips_carrier_and_keeps_tool_use() -> None:
    assistant_message = _assemble_assistant_message("compaction-carrier-signature")
    request = _next_turn_request(assistant_message)
    request_snapshot = copy.deepcopy(request)
    original_tool_uses = [
        block
        for block in assistant_message["content"]
        if block["type"] == "tool_use"
    ]
    assert any(
        block.get("type") == "thinking"
        and block.get("signature", "").startswith(CARRIER_PREFIX)
        for block in assistant_message["content"]
    )

    payload = build_reroute_payload(request, "claude-haiku-4-5")
    payload_blocks = _content_blocks(payload)

    assert request == request_snapshot
    assert payload["model"] == "claude-haiku-4-5"
    assert all(block.get("type") != "thinking" for block in payload_blocks)
    assert all(
        not block.get("signature", "").startswith(CARRIER_PREFIX)
        for block in payload_blocks
    )
    assert [block for block in payload_blocks if block.get("type") == "tool_use"] == (
        original_tool_uses
    )


def test_round_trip_preserves_unusual_opaque_signatures() -> None:
    signatures = [
        " \x00not-base64!?+/=\\\"'\n\t한글🚀é ",
        "opaque!?+/= 한글🚀\n\t" * 512,
    ]

    for signature in signatures:
        assert len(signature.encode("utf-8")) <= MAX_SIGNATURE_BYTES
        assistant_message = _assemble_assistant_message(signature)
        payload = _translate_next_turn(assistant_message)
        replayed_signature = _function_calls(payload)[0]["extra_content"]["google"][
            "thought_signature"
        ]

        assert replayed_signature.encode("utf-8") == signature.encode("utf-8")
