"""Tests for the Signal A Claude Code compaction detector.

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

from claudex_gateway import compaction

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
