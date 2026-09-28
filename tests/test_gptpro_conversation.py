"""Tests for pure gptpro conversation URL and turn extraction."""

from __future__ import annotations

from typing import Any

import pytest

from claudex.gptpro import conversation

NONCE = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
NONCE_MARKER = conversation.build_nonce_marker(NONCE)
CONVERSATION_ID = "0fdbc1f1-8f05-4cba-865b-b677490fba9c"


def test_is_conversation_id_accepts_mixed_case_canonical_uuid() -> None:
    assert conversation.is_conversation_id(CONVERSATION_ID.upper())


@pytest.mark.parametrize(
    "value",
    [f"WEB:{CONVERSATION_ID}", "", None, 123],
)
def test_is_conversation_id_rejects_noncanonical_values(value: object) -> None:
    assert not conversation.is_conversation_id(value)


def test_build_conversation_url_normalizes_conversation_id() -> None:
    assert conversation.build_conversation_url(CONVERSATION_ID.upper()) == (
        f"https://chatgpt.com/c/{CONVERSATION_ID}"
    )


def test_build_conversation_url_rejects_invalid_id() -> None:
    with pytest.raises(ValueError, match="conversation_id"):
        conversation.build_conversation_url(f"WEB:{CONVERSATION_ID}")


def _node(
    node_id: str,
    parent: str | None,
    role: str,
    parts: list[object],
    *,
    content_type: str = "text",
    status: str | None = None,
    end_turn: bool | None = None,
    channel: str | None = None,
    recipient: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "author": {"role": role},
        "content": {"content_type": content_type, "parts": parts},
    }
    if status is not None:
        message["status"] = status
    if end_turn is not None:
        message["end_turn"] = end_turn
    if channel is not None:
        message["channel"] = channel
    if recipient is not None:
        message["recipient"] = recipient
    return {"id": node_id, "parent": parent, "message": message}


def _conversation(*nodes: dict[str, Any], current_node: str) -> dict[str, Any]:
    return {
        "mapping": {node["id"]: node for node in nodes},
        "current_node": current_node,
    }


def test_extracts_nonce_anchored_text_parts_and_finished_state() -> None:
    fixture = _conversation(
        {"id": "root", "parent": None, "message": None},
        _node("user", "root", "user", ["review", NONCE_MARKER]),
        _node(
            "answer-1",
            "user",
            "assistant",
            ["Part one.", {"ignored": True}, "Still part one."],
            status="finished_successfully",
        ),
        _node(
            "thought",
            "answer-1",
            "assistant",
            ["private reasoning"],
            content_type="thoughts",
            status="finished_successfully",
        ),
        _node(
            "answer-2",
            "thought",
            "assistant",
            ["Part two."],
            end_turn=True,
        ),
        current_node="answer-2",
    )

    turn = conversation.extract_assistant_turn(fixture, NONCE_MARKER)

    assert turn == conversation.AssistantTurn(
        text="Part one.\nStill part one.\n\nPart two.", finished=True
    )


@pytest.mark.parametrize(
    ("status", "end_turn", "expected"),
    [
        ("finished_successfully", False, False),
        ("finished_successfully", None, False),
        ("finished_successfully", True, True),
        ("in_progress", True, True),
        ("in_progress", False, False),
        (None, None, False),
    ],
)
def test_finished_requires_end_turn_regardless_of_status(
    status: str | None, end_turn: bool | None, expected: bool
) -> None:
    fixture = _conversation(
        _node("user", None, "user", [NONCE_MARKER]),
        _node(
            "answer",
            "user",
            "assistant",
            ["answer"],
            status=status,
            end_turn=end_turn,
        ),
        current_node="answer",
    )

    turn = conversation.extract_assistant_turn(fixture, NONCE_MARKER)

    assert turn is not None
    assert turn.finished is expected


def test_nonce_marked_unanswered_user_returns_empty_unfinished_turn() -> None:
    fixture = _conversation(
        _node("user", None, "user", [NONCE_MARKER]), current_node="user"
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="", finished=False)


def test_nonce_anchored_turn_stops_at_the_next_user_message() -> None:
    fixture = _conversation(
        _node("original-user", None, "user", [NONCE_MARKER]),
        _node(
            "original-answer",
            "original-user",
            "assistant",
            ["Original answer."],
            end_turn=True,
        ),
        _node("next-user", "original-answer", "user", ["Unrelated question"]),
        _node(
            "next-answer", "next-user", "assistant", ["Unrelated answer."], end_turn=True
        ),
        current_node="next-answer",
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="Original answer.", finished=True)


def test_nonce_anchored_unanswered_turn_ignores_later_answer() -> None:
    fixture = _conversation(
        _node("original-user", None, "user", [NONCE_MARKER]),
        _node("next-user", "original-user", "user", ["Unrelated question"]),
        _node(
            "next-answer", "next-user", "assistant", ["Unrelated answer."], end_turn=True
        ),
        current_node="next-answer",
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="", finished=False)


def test_missing_nonce_returns_none_instead_of_stale_answer() -> None:
    fixture = _conversation(
        _node("user", None, "user", ["ordinary prompt"]),
        _node("answer", "user", "assistant", ["stale"], end_turn=True),
        current_node="answer",
    )

    assert conversation.extract_assistant_turn(fixture, NONCE_MARKER) is None


@pytest.mark.parametrize(
    "fixture",
    [
        {},
        {"mapping": {}, "current_node": None},
        {"mapping": [], "current_node": "answer"},
        {"mapping": {"answer": "broken"}, "current_node": "answer"},
    ],
)
def test_malformed_conversation_returns_none(fixture: dict[str, Any]) -> None:
    assert conversation.extract_assistant_turn(fixture, NONCE_MARKER) is None


def test_parent_cycle_is_bounded_and_can_still_resolve_the_turn() -> None:
    fixture = _conversation(
        _node("user", "answer", "user", ["question", NONCE_MARKER]),
        _node("answer", "user", "assistant", ["answer"], end_turn=True),
        current_node="answer",
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="answer", finished=True)
    assert conversation.extract_assistant_turn(fixture, "missing") is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            f"https://chatgpt.com/backend-api/conversation/{CONVERSATION_ID}",
            CONVERSATION_ID,
        ),
        (
            "https://chatgpt.com/backend-api/conversation/gen_title/"
            f"{CONVERSATION_ID.upper()}?source=title",
            CONVERSATION_ID,
        ),
        (
            f"https://chatgpt.com/backend-api/f/conversation/{CONVERSATION_ID}/",
            CONVERSATION_ID,
        ),
        (
            f"https://chatgpt.com/backend-api/conversations/{CONVERSATION_ID}",
            CONVERSATION_ID,
        ),
        (
            f"https://chatgpt.com/backend-api/conversations/{CONVERSATION_ID}/",
            CONVERSATION_ID,
        ),
        ("https://chatgpt.com/backend-api/conversations/batch", None),
        ("https://chatgpt.com/backend-api/conversations", None),
        (f"https://chatgpt.com/backend-api/conversation/WEB:{CONVERSATION_ID}", None),
        (f"https://chatgpt.com/backend-api/lat/r?conversation_id={CONVERSATION_ID}", None),
        (f"https://chatgpt.com/conversation/{CONVERSATION_ID}", None),
        ("not a url", None),
    ],
)
def test_extract_conversation_id_from_backend_url(
    url: str, expected: str | None
) -> None:
    assert conversation.extract_conversation_id_from_url(url) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (f'{{"conversation_id":"{CONVERSATION_ID}"}}', CONVERSATION_ID),
        (
            f'{{"conversation_id":"{CONVERSATION_ID.upper()}","model":"pro"}}',
            CONVERSATION_ID,
        ),
        (f'{{"conversation_id":"WEB:{CONVERSATION_ID}"}}', None),
        ('{"model":"pro"}', None),
        (f'garbage "conversation_id": "{CONVERSATION_ID}" trailing', None),
        ("", None),
        ("[]", None),
    ],
)
def test_extract_conversation_id_only_from_valid_json_body(
    body: str, expected: str | None
) -> None:
    assert conversation.extract_conversation_id_from_body(body) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://chatgpt.com/backend-api/conversation", True),
        ("https://chatgpt.com/backend-api/f/conversation?model=pro", True),
        ("https://chatgpt.com:443/backend-api/conversation", True),
        (
            f"https://chatgpt.com/backend-api/conversation/{CONVERSATION_ID}",
            False,
        ),
        ("https://chatgpt.com/backend-api/conversations", False),
        ("https://evil.example/backend-api/conversation", False),
        ("https://chatgpt.com.evil.example/backend-api/conversation", False),
        ("http://chatgpt.com/backend-api/conversation", False),
        ("not a url", False),
    ],
)
def test_conversation_stream_url_requires_exact_path_and_trusted_origin(
    url: str, expected: bool
) -> None:
    assert conversation.is_conversation_stream_url(url) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://chatgpt.com/backend-api/lat/r", True),
        ("https://chatgpt.com/prefix/backend-api/lat/report?x=1", True),
        ("https://chatgpt.com/backend-api/lat", False),
        ("https://evil.example/backend-api/lat/r", False),
        ("http://chatgpt.com/backend-api/lat/r", False),
        ("not a url", False),
    ],
)
def test_completion_report_url_requires_path_fragment_and_trusted_origin(
    url: str, expected: bool
) -> None:
    assert conversation.is_completion_report_url(url) is expected


def test_build_nonce_marker_uses_frozen_transport_label() -> None:
    assert NONCE_MARKER == f"[gptpro-transport-nonce:{NONCE}]"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://chatgpt.com/", True),
        ("https://chatgpt.com:443/backend-api/conversation", True),
        ("https://evil.example/backend-api/conversation", False),
        ("https://chatgpt.com.evil.example/", False),
        ("http://chatgpt.com/", False),
        ("not a url", False),
    ],
)
def test_trusted_origin_url(url: str, expected: bool) -> None:
    assert conversation.is_trusted_origin_url(url) is expected


def test_finished_state_ignores_later_non_text_assistant_nodes() -> None:
    fixture = _conversation(
        _node("user", None, "user", [NONCE_MARKER]),
        _node(
            "answer",
            "user",
            "assistant",
            ["partial answer"],
            status="in_progress",
        ),
        _node(
            "thought",
            "answer",
            "assistant",
            ["finished internal state"],
            content_type="thoughts",
            status="finished_successfully",
        ),
        current_node="thought",
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="partial answer", finished=False)


def _reasoning_turn_nodes() -> list[dict[str, Any]]:
    """Synthetic reasoning-model turn: commentary, tool calls, then a final answer."""
    done = "finished_successfully"
    return [
        _node("user", None, "user", [NONCE_MARKER]),
        _node(
            "search-call", "user", "assistant", [""],
            recipient="web.run", status=done, end_turn=False,
        ),
        _node(
            "preamble", "search-call", "assistant", ["Checking sources."],
            channel="commentary", recipient="all", status=done, end_turn=False,
        ),
        _node("search-result", "preamble", "tool", [""], status=done),
        _node(
            "second-call", "search-result", "assistant", [""],
            recipient="web.run", status=done, end_turn=False,
        ),
        _node(
            "thought", "second-call", "assistant", ["thinking"],
            content_type="thoughts", status=done, end_turn=False,
        ),
        _node(
            "code", "thought", "assistant", ["print(1)"],
            content_type="code", recipient="container.exec",
            status=done, end_turn=False,
        ),
        _node(
            "exec-output", "code", "tool", ["1"],
            content_type="execution_output", status=done,
        ),
        _node(
            "progress", "exec-output", "assistant", ["Verifying the result."],
            channel="commentary", recipient="all", status=done, end_turn=False,
        ),
        _node(
            "recap", "progress", "assistant", ["recap"],
            content_type="reasoning_recap", status=done, end_turn=True,
        ),
        _node(
            "final", "recap", "assistant", ["The final answer."],
            channel="final", recipient="all", status=done, end_turn=True,
        ),
    ]


def test_reasoning_turn_returns_only_the_final_channel_answer() -> None:
    nodes = _reasoning_turn_nodes()
    fixture = _conversation(*nodes, current_node="final")

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="The final answer.", finished=True)


@pytest.mark.parametrize(
    "current_node",
    [
        node["id"]
        for node in _reasoning_turn_nodes()
        if node["id"] != "final"
    ],
)
def test_every_snapshot_before_the_final_answer_is_unfinished(
    current_node: str,
) -> None:
    fixture = _conversation(*_reasoning_turn_nodes(), current_node=current_node)

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="", finished=False)


def test_in_progress_final_answer_is_partial_and_unfinished() -> None:
    nodes = _reasoning_turn_nodes()
    nodes[-1] = _node(
        "final", "recap", "assistant", ["The final"],
        channel="final", recipient="all", status="in_progress",
    )
    fixture = _conversation(*nodes, current_node="final")

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="The final", finished=False)


def test_final_channel_answer_excludes_unchannelled_preamble() -> None:
    fixture = _conversation(
        _node("user", None, "user", [NONCE_MARKER]),
        _node(
            "preamble", "user", "assistant", ["Implicit preamble."],
            recipient="all", end_turn=False,
        ),
        _node(
            "final", "preamble", "assistant", ["Explicit final."],
            channel="final", end_turn=True,
        ),
        current_node="final",
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="Explicit final.", finished=True)


def test_unknown_channel_text_is_not_part_of_the_answer() -> None:
    fixture = _conversation(
        _node("user", None, "user", [NONCE_MARKER]),
        _node(
            "analysis", "user", "assistant", ["hidden analysis"],
            channel="analysis", end_turn=True,
        ),
        current_node="analysis",
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="", finished=False)


def test_unchannelled_answer_excludes_text_addressed_to_tools() -> None:
    fixture = _conversation(
        _node("user", None, "user", [NONCE_MARKER]),
        _node(
            "tool-call", "user", "assistant", ['{"query": "x"}'],
            recipient="web.run", end_turn=False,
        ),
        _node("tool-result", "tool-call", "tool", ["result"]),
        _node(
            "answer", "tool-result", "assistant", ["Legacy answer."],
            recipient="all", end_turn=True,
        ),
        current_node="answer",
    )

    assert conversation.extract_assistant_turn(
        fixture, NONCE_MARKER
    ) == conversation.AssistantTurn(text="Legacy answer.", finished=True)
