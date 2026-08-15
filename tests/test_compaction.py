"""Tests for the Signal A Claude Code compaction detector and its reroute.

The positive fixtures below are test-local literals copied verbatim from
`.docs/research/claude-code/compact-detection.md` (the fixed header and the
fixed `REMINDER: ...` suffix shared by all three compaction variants). They
are deliberately NOT built by interpolating or concatenating
`compaction.SIGNAL_A_PREFIX` / `compaction.SIGNAL_A_MARKER`: if the
production constants and this test module both drifted from the real
Claude Code contract in the same way, fixtures built from the constants
would still pass and the mistake would go undetected. Duplicating the
literal text here means a drift in either file breaks the tests.
"""

import copy

from claudex import compaction

FULL_COMPACT_PROMPT = (
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n"
    "\n"
    "- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.\n"
    "- You already have all the context you need in the conversation above.\n"
    "- Tool calls will be REJECTED and will waste your only turn — you will fail the task.\n"
    "- Your entire response must be plain text: an <analysis> block followed by a <summary> block.\n"
    "\n"
    "Your task is to create a detailed summary of the conversation so far, paying "
    "close attention to the user's explicit requests and your previous actions.\n"
    "\n"
    "REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis>\n"
    "block followed by a <summary> block. Tool calls will be rejected and you will\n"
    "fail the task."
)

FROM_COMPACT_PROMPT = (
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n"
    "\n"
    "- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.\n"
    "- You already have all the context you need in the conversation above.\n"
    "- Tool calls will be REJECTED and will waste your only turn — you will fail the task.\n"
    "- Your entire response must be plain text: an <analysis> block followed by a <summary> block.\n"
    "\n"
    "Your task is to create a detailed summary of the conversation from the message "
    "index given below onward, paying close attention to the user's explicit requests "
    "and your previous actions.\n"
    "\n"
    "REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis>\n"
    "block followed by a <summary> block. Tool calls will be rejected and you will\n"
    "fail the task."
)

UP_TO_COMPACT_PROMPT = (
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n"
    "\n"
    "- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.\n"
    "- You already have all the context you need in the conversation above.\n"
    "- Tool calls will be REJECTED and will waste your only turn — you will fail the task.\n"
    "- Your entire response must be plain text: an <analysis> block followed by a <summary> block.\n"
    "\n"
    "Your task is to create a detailed summary of the conversation up to the message "
    "index given below, paying close attention to the user's explicit requests and "
    "your previous actions.\n"
    "\n"
    "REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis>\n"
    "block followed by a <summary> block. Tool calls will be rejected and you will\n"
    "fail the task."
)


def _user_message(content: object) -> dict[str, object]:
    return {"role": "user", "content": content}


# --- Positive cases: string content -----------------------------------------


def test_detects_full_compact_prompt_as_string_content() -> None:
    body = {"messages": [_user_message(FULL_COMPACT_PROMPT)]}
    assert compaction.is_compaction_request(body) is True


def test_detects_from_compact_prompt_as_string_content() -> None:
    body = {"messages": [_user_message(FROM_COMPACT_PROMPT)]}
    assert compaction.is_compaction_request(body) is True


def test_detects_up_to_compact_prompt_as_string_content() -> None:
    body = {"messages": [_user_message(UP_TO_COMPACT_PROMPT)]}
    assert compaction.is_compaction_request(body) is True


# --- Positive cases: text-block-list content ---------------------------------


def test_detects_full_compact_prompt_with_prefix_fragmented_across_blocks() -> None:
    # Split inside "CRITICAL: Respond with TEXT ONLY" itself, so neither
    # block alone contains the full prefix -- only their concatenation does.
    split_at = len("CRITICAL: Respond with TEXT ON")
    body = {
        "messages": [
            _user_message(
                [
                    {"type": "text", "text": FULL_COMPACT_PROMPT[:split_at]},
                    {"type": "text", "text": FULL_COMPACT_PROMPT[split_at:]},
                ]
            )
        ]
    }
    assert compaction.is_compaction_request(body) is True


def test_detects_from_compact_prompt_with_marker_fragmented_across_blocks() -> None:
    # Split inside "Your task is to create a detailed summary" itself, so
    # neither block alone contains the full marker.
    marker_index = FROM_COMPACT_PROMPT.index("Your task is to create a detailed summary")
    split_at = marker_index + len("Your task is to create a detailed su")
    body = {
        "messages": [
            _user_message(
                [
                    {"type": "text", "text": FROM_COMPACT_PROMPT[:split_at]},
                    {"type": "text", "text": FROM_COMPACT_PROMPT[split_at:]},
                ]
            )
        ]
    }
    assert compaction.is_compaction_request(body) is True


def test_detects_up_to_compact_prompt_with_non_text_block_interleaved() -> None:
    split_at = len(UP_TO_COMPACT_PROMPT) // 2
    body = {
        "messages": [
            _user_message(
                [
                    {"type": "text", "text": UP_TO_COMPACT_PROMPT[:split_at]},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "Zm9vYmFy",
                        },
                    },
                    {"type": "text", "text": UP_TO_COMPACT_PROMPT[split_at:]},
                ]
            )
        ]
    }
    assert compaction.is_compaction_request(body) is True


# --- Negative cases -----------------------------------------------------------


def test_prefix_without_marker_returns_false() -> None:
    text = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\nPlease just say hi."
    body = {"messages": [_user_message(text)]}
    assert compaction.is_compaction_request(body) is False


def test_marker_without_prefix_returns_false() -> None:
    text = "Hey! Your task is to create a detailed summary of last night's dinner party."
    body = {"messages": [_user_message(text)]}
    assert compaction.is_compaction_request(body) is False


def test_matching_text_on_a_non_final_message_returns_false() -> None:
    body = {
        "messages": [
            _user_message(FULL_COMPACT_PROMPT),
            {"role": "assistant", "content": "Sure, here is the summary."},
        ]
    }
    assert compaction.is_compaction_request(body) is False


def test_assistant_role_returns_false() -> None:
    body = {"messages": [{"role": "assistant", "content": FULL_COMPACT_PROMPT}]}
    assert compaction.is_compaction_request(body) is False


def test_empty_messages_returns_false() -> None:
    body = {"messages": []}
    assert compaction.is_compaction_request(body) is False


def test_non_text_only_blocks_return_false() -> None:
    body = {
        "messages": [
            _user_message(
                [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "Zm9v",
                        },
                    }
                ]
            )
        ]
    }
    assert compaction.is_compaction_request(body) is False


def test_tool_result_only_content_returns_false() -> None:
    body = {
        "messages": [
            _user_message(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "some tool output",
                    }
                ]
            )
        ]
    }
    assert compaction.is_compaction_request(body) is False


def test_ordinary_chat_returns_false() -> None:
    body = {"messages": [_user_message("Hey, can you help me refactor this function?")]}
    assert compaction.is_compaction_request(body) is False


def test_non_dict_body_returns_false() -> None:
    assert compaction.is_compaction_request(["not", "a", "dict"]) is False
    assert compaction.is_compaction_request("also not a dict") is False
    assert compaction.is_compaction_request(None) is False


def test_non_list_messages_returns_false() -> None:
    body = {"messages": "not-a-list"}
    assert compaction.is_compaction_request(body) is False


def test_non_dict_final_message_returns_false() -> None:
    body = {"messages": [_user_message("hi"), "not-a-dict"]}
    assert compaction.is_compaction_request(body) is False


# --- Contract-drift guard ------------------------------------------------------


def test_signal_a_constants_match_the_locked_claude_code_2_1_223_contract() -> None:
    failure_hint = (
        "the Claude Code 2.1.223 binary must be re-verified before updating "
        "this constant"
    )
    assert compaction.SIGNAL_A_PREFIX == "CRITICAL: Respond with TEXT ONLY", failure_hint
    assert compaction.SIGNAL_A_MARKER == "Your task is to create a detailed summary", failure_hint
    assert compaction.SIGNAL_A_CONTRACT_VERSION == "2.1.223", failure_hint


# --- build_reroute_payload ------------------------------------------------


def _tool_use_block() -> dict[str, object]:
    return {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "read_file",
        "input": {"path": "/tmp/example.txt"},
    }


def _tool_result_block() -> dict[str, object]:
    return {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": "file contents",
    }


def test_build_reroute_payload_replaces_model_with_target_model_id() -> None:
    body = {"model": "claude-opus-4-6", "messages": [_user_message("hi")]}
    result = compaction.build_reroute_payload(body, "claude-haiku-4-6")
    assert result["model"] == "claude-haiku-4-6"


def test_build_reroute_payload_strips_thinking_blocks() -> None:
    body = {
        "model": "claude-opus-4-6",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "internal reasoning", "signature": "sig"},
                    {"type": "text", "text": "final answer"},
                ],
            }
        ],
    }
    result = compaction.build_reroute_payload(body, "claude-haiku-4-6")
    assert result["messages"][0]["content"] == [{"type": "text", "text": "final answer"}]


def test_build_reroute_payload_strips_redacted_thinking_blocks() -> None:
    body = {
        "model": "claude-opus-4-6",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "final answer"},
                ],
            }
        ],
    }
    result = compaction.build_reroute_payload(body, "claude-haiku-4-6")
    assert result["messages"][0]["content"] == [{"type": "text", "text": "final answer"}]


def test_build_reroute_payload_drops_thinking_only_messages() -> None:
    body = {
        "model": "claude-opus-4-6",
        "messages": [
            _user_message("hi"),
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "internal", "signature": "sig"}],
            },
            _user_message("second turn"),
        ],
    }
    result = compaction.build_reroute_payload(body, "claude-haiku-4-6")
    assert [message["role"] for message in result["messages"]] == ["user", "user"]


def test_build_reroute_payload_preserves_string_and_non_thinking_blocks() -> None:
    body = {
        "model": "claude-opus-4-6",
        "messages": [
            _user_message("plain string content"),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "reply"}, _tool_use_block()],
            },
            _user_message([_tool_result_block()]),
        ],
    }
    result = compaction.build_reroute_payload(body, "claude-haiku-4-6")
    assert result["messages"][0]["content"] == "plain string content"
    assert result["messages"][1]["content"] == [
        {"type": "text", "text": "reply"},
        _tool_use_block(),
    ]
    assert result["messages"][2]["content"] == [_tool_result_block()]


def test_build_reroute_payload_leaves_original_body_unchanged_immediately() -> None:
    body = {
        "model": "claude-opus-4-6",
        "metadata": {"user_id": "abc"},
        "messages": [_user_message("hi")],
    }
    pre_call_copy = copy.deepcopy(body)
    compaction.build_reroute_payload(body, "claude-haiku-4-6")
    assert body == pre_call_copy


def test_build_reroute_payload_return_value_does_not_alias_original_body() -> None:
    body = {
        "model": "claude-opus-4-6",
        "metadata": {"user_id": "abc"},
        "messages": [
            {
                "role": "assistant",
                "content": [_tool_use_block()],
            }
        ],
    }
    pre_call_copy = copy.deepcopy(body)
    result = compaction.build_reroute_payload(body, "claude-haiku-4-6")

    # Mutate nested structures reachable from the returned payload: tool
    # input, metadata, and content.
    result["messages"][0]["content"][0]["input"]["path"] = "/mutated/path"
    result["metadata"]["user_id"] = "mutated"
    result["messages"][0]["content"].append({"type": "text", "text": "injected"})

    # The original body must be completely unaffected by those mutations.
    assert body == pre_call_copy


# --- build_reroute_headers -------------------------------------------------


def test_build_reroute_headers_x_api_key_only_is_copied() -> None:
    result = compaction.build_reroute_headers({"x-api-key": "sk-ant-real-key"}, "local-secret")
    assert result == {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": "sk-ant-real-key",
    }


def test_build_reroute_headers_non_local_bearer_only_is_copied() -> None:
    result = compaction.build_reroute_headers(
        {"authorization": "Bearer sk-ant-real-oauth-token"}, "local-secret"
    )
    assert result == {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "authorization": "Bearer sk-ant-real-oauth-token",
    }


def test_build_reroute_headers_both_credential_forms_are_copied() -> None:
    result = compaction.build_reroute_headers(
        {
            "x-api-key": "sk-ant-real-key",
            "authorization": "Bearer sk-ant-real-oauth-token",
        },
        "local-secret",
    )
    assert result["x-api-key"] == "sk-ant-real-key"
    assert result["authorization"] == "Bearer sk-ant-real-oauth-token"


def test_build_reroute_headers_mixed_case_input_names_are_recognized() -> None:
    result = compaction.build_reroute_headers(
        {
            "X-Api-Key": "sk-ant-real-key",
            "Anthropic-Version": "2024-10-01",
            "Anthropic-Beta": "some-beta-flag",
            "Accept": "application/json",
        },
        None,
    )
    assert result == {
        "content-type": "application/json",
        "accept": "application/json",
        "anthropic-version": "2024-10-01",
        "anthropic-beta": "some-beta-flag",
        "x-api-key": "sk-ant-real-key",
    }


def test_build_reroute_headers_local_bearer_token_is_excluded() -> None:
    result = compaction.build_reroute_headers(
        {"authorization": "Bearer local-secret"}, "local-secret"
    )
    assert result is None


def test_build_reroute_headers_local_bearer_excluded_but_x_api_key_kept() -> None:
    result = compaction.build_reroute_headers(
        {"authorization": "Bearer local-secret", "x-api-key": "sk-ant-real-key"},
        "local-secret",
    )
    assert result is not None
    assert "authorization" not in result
    assert result["x-api-key"] == "sk-ant-real-key"


def test_build_reroute_headers_no_credentials_returns_none() -> None:
    assert compaction.build_reroute_headers({}, "local-secret") is None
    assert (
        compaction.build_reroute_headers({"accept": "application/json"}, "local-secret") is None
    )


def test_build_reroute_headers_blank_x_api_key_returns_none() -> None:
    assert compaction.build_reroute_headers({"x-api-key": "   "}, None) is None


def test_build_reroute_headers_bearer_with_no_token_returns_none() -> None:
    assert compaction.build_reroute_headers({"authorization": "Bearer"}, None) is None
    assert compaction.build_reroute_headers({"authorization": "Bearer "}, None) is None


def test_build_reroute_headers_basic_authorization_returns_none() -> None:
    assert compaction.build_reroute_headers({"authorization": "Basic dXNlcjpwYXNz"}, None) is None


def test_build_reroute_headers_authorization_without_separator_returns_none() -> None:
    result = compaction.build_reroute_headers({"authorization": "sk-ant-token-no-scheme"}, None)
    assert result is None


def test_build_reroute_headers_never_discloses_the_local_token_value() -> None:
    local_token = "super-secret-local-value"
    result = compaction.build_reroute_headers(
        {
            "authorization": f"Bearer {local_token}",
            "x-api-key": "sk-ant-real-key",
        },
        local_token,
    )
    assert result is not None
    assert local_token not in result.values()
    assert "authorization" not in result


def test_build_reroute_headers_never_forwards_transport_or_arbitrary_headers() -> None:
    headers = {
        "x-api-key": "sk-ant-real-key",
        "host": "attacker.example.com",
        "content-length": "1234",
        "cookie": "session=abc123",
        "x-forwarded-for": "10.0.0.1",
    }
    result = compaction.build_reroute_headers(headers, None)
    assert set(result) == {"content-type", "anthropic-version", "x-api-key"}
