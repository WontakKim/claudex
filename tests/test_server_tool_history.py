"""Tests for preserving provider server-tool history across backends."""

from copy import deepcopy

import pytest

from claudex.translate.server_tool_history import normalize_server_tool_history


def _search_blocks(tool_id: str, *, signed: bool = False) -> list[dict]:
    result = {
        "type": "web_search_result", "title": "Python release",
        "url": "https://python.org/news", "page_age": None,
    }
    if signed:
        result["encrypted_content"] = "native-opaque-data"
    return [
        {"type": "server_tool_use", "id": tool_id, "name": "web_search",
         "input": {"query": "python release date"}},
        {"type": "web_search_tool_result", "tool_use_id": tool_id, "content": [result]},
    ]


@pytest.mark.parametrize("tool_id", ["ws_1", "web_search_0", "srvtoolu_gateway_test", "srvtoolu_unsigned"])
def test_normalize_preserves_search_data_and_does_not_mutate_input(tool_id: str) -> None:
    blocks = _search_blocks(tool_id)
    blocks[1]["cache_control"] = {"type": "ephemeral"}
    body = {"model": "claude-fable-5", "messages": [
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": "Continue"},
    ]}
    original = deepcopy(body)
    normalized = normalize_server_tool_history(body)
    assert body == original
    assert normalized is not body
    assert normalized["messages"][1] is body["messages"][1]
    content = normalized["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text", "text"]
    assert "python release date" in content[0]["text"]
    assert "Python release" in content[1]["text"]
    assert "https://python.org/news" in content[1]["text"]
    assert content[1]["cache_control"] == {"type": "ephemeral"}
    assert normalize_server_tool_history(normalized) is normalized


def test_normalize_preserves_native_search_and_unrelated_blocks_in_mixed_history() -> None:
    native = _search_blocks("srvtoolu_native", signed=True)
    other_blocks = [
        {"type": "thinking", "thinking": "opaque", "signature": "native-signature"},
        {"type": "text", "text": "Native citation", "citations": [{
            "type": "web_search_result_location", "encrypted_index": "opaque-index",
            "url": "https://python.org/news", "title": "Python release", "cited_text": "Release",
        }]},
        {"type": "tool_use", "name": "Bash", "id": "call_1", "input": {}},
        {"type": "server_tool_use", "name": "web_fetch", "id": "srvtoolu_native_fetch", "input": {}},
    ]
    body = {"messages": [{"role": "assistant", "content": [
        *native, *_search_blocks("ws_1"), *other_blocks,
    ]}]}
    content = normalize_server_tool_history(body)["messages"][0]["content"]
    assert content[:2] == native
    assert content[4:] == other_blocks
    assert [block["type"] for block in content[2:4]] == ["text", "text"]


@pytest.mark.parametrize("result_content", [[], {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}])
@pytest.mark.parametrize("tool_id", ["srvtoolu_native", "ws_1", "srvtoolu_gateway_test"])
def test_empty_and_error_results_follow_the_call_id(tool_id: str, result_content: object) -> None:
    blocks = _search_blocks(tool_id)
    blocks[1]["content"] = result_content
    body = {"messages": [{"role": "assistant", "content": blocks}]}
    normalized = normalize_server_tool_history(body)
    if tool_id == "srvtoolu_native":
        assert normalized is body
    else:
        assert [block["type"] for block in normalized["messages"][0]["content"]] == ["text", "text"]


def test_orphan_synthetic_result_is_converted_without_inventing_a_call() -> None:
    body = {"messages": [{"role": "assistant", "content": [_search_blocks("ws_1")[1]]}]}
    content = normalize_server_tool_history(body)["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"


@pytest.mark.parametrize("body", [
    None, [], {}, {"messages": None}, {"messages": "invalid"},
    {"messages": [None, "invalid", {"role": "assistant", "content": "plain text"}]},
    {"messages": [{"role": "assistant", "content": [None, "invalid", {}]}]},
    {"messages": [{"role": "assistant", "content": _search_blocks("srvtoolu_native", signed=True)}]},
    {"messages": [{"role": "user", "content": _search_blocks("ws_1")}]},
])
def test_unaffected_history_preserves_object_identity(body: object) -> None:
    assert normalize_server_tool_history(body) is body


@pytest.mark.parametrize("tool_name", ["analyze_image", "provider_ocr"])
@pytest.mark.parametrize("tool_id", ["call_image", "srvtoolu_provider_image"])
@pytest.mark.parametrize("split_messages", [False, True])
@pytest.mark.parametrize("result_content", [
    "The image contains a red square.",
    [{"type": "text", "text": "The image contains a red square."}],
])
def test_provider_server_call_and_assistant_result_are_replayed_together(
    tool_name: str, tool_id: str, split_messages: bool, result_content: object
) -> None:
    call = {"type": "server_tool_use", "id": tool_id, "name": tool_name,
            "input": {"image_source": "test-image"}}
    result = {"type": "tool_result", "tool_use_id": tool_id, "content": result_content,
              "cache_control": {"type": "ephemeral"}}
    if split_messages:
        messages = [
            {"role": "assistant", "content": [call]},
            {"role": "assistant", "content": "Image analysis completed."},
            {"role": "assistant", "content": [result]},
        ]
    else:
        messages = [{"role": "assistant", "content": [call, result]}]
    body = {"messages": messages}
    original = deepcopy(body)
    normalized = normalize_server_tool_history(body)
    assert body == original
    assert normalized is not body
    content = [block for message in normalized["messages"]
               if isinstance(message["content"], list) for block in message["content"]]
    assert [block["type"] for block in content] == ["text", "text"]
    assert tool_name in content[0]["text"]
    assert "test-image" in content[0]["text"]
    assert "The image contains a red square." in content[1]["text"]
    assert content[1]["cache_control"] == {"type": "ephemeral"}
    assert normalize_server_tool_history(normalized) is normalized


def test_internal_error_result_is_preserved_as_an_error() -> None:
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "call_image", "name": "analyze_image", "input": {}},
        {"type": "tool_result", "tool_use_id": "call_image", "content": "Unreadable image",
         "is_error": True},
    ]}]}
    content = normalize_server_tool_history(body)["messages"][0]["content"]
    assert "error" in content[1]["text"].lower()
    assert "Unreadable image" in content[1]["text"]


def test_server_tool_id_does_not_capture_a_client_result_in_a_later_turn() -> None:
    client_call = {"type": "tool_use", "id": "call_reused", "name": "Read", "input": {}}
    client_result = {"type": "tool_result", "tool_use_id": "call_reused", "content": "Client output"}
    body = {"messages": [
        {"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "call_reused", "name": "analyze_image", "input": {}},
            {"type": "tool_result", "tool_use_id": "call_reused", "content": "Server output"},
        ]},
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": [client_call]},
        {"role": "user", "content": [client_result]},
    ]}
    normalized = normalize_server_tool_history(body)
    assert [block["type"] for block in normalized["messages"][0]["content"]] == ["text", "text"]
    assert normalized["messages"][2] is body["messages"][2]
    assert normalized["messages"][3] is body["messages"][3]


@pytest.mark.parametrize("tool_name,result_type", [
    ("web_fetch", "web_fetch_tool_result"),
    ("bash_code_execution", "bash_code_execution_tool_result"),
    ("text_editor_code_execution", "text_editor_code_execution_tool_result"),
])
def test_native_nonsearch_server_tools_are_unchanged(tool_name: str, result_type: str) -> None:
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "srvtoolu_native", "name": tool_name, "input": {}},
        {"type": result_type, "tool_use_id": "srvtoolu_native", "content": {"type": "result"}},
        {"type": "mcp_tool_use", "id": "mcptoolu_native", "name": "remote_read", "input": {}},
        {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_native", "content": []},
    ]}]}
    assert normalize_server_tool_history(body) is body


def test_invalid_nonsearch_server_tool_and_typed_result_are_replayed_together() -> None:
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "call_fetch", "name": "web_fetch", "input": {}},
        {"type": "web_fetch_tool_result", "tool_use_id": "call_fetch", "content": {"title": "Result"}},
    ]}]}
    content = normalize_server_tool_history(body)["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text", "text"]
    assert "Result" in content[1]["text"]
