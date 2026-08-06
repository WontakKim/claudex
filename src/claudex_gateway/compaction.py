"""Detect Claude Code compaction requests ("Signal A").

Claude Code has no metadata field, model name, or `querySource` marker that
survives onto the wire for a compaction request; the only reliable signal is
the literal text of the final `user` message the client sends when it asks
the model to summarize the conversation. This single detector covers all
three compaction variants (full compact, partial `from`, partial `up_to`)
because they share the same header and marker text. See
`.docs/research/claude-code/compact-detection.md` for how these literals
were extracted.
"""

from __future__ import annotations

from typing import Any

# These three literals were extracted from the Claude Code 2.1.223 client
# binary and form a versioned contract with that exact release. They must
# only be changed after re-verifying the new literal text against the
# updated Claude Code client binary -- never edit them speculatively.
SIGNAL_A_PREFIX = "CRITICAL: Respond with TEXT ONLY"
SIGNAL_A_MARKER = "Your task is to create a detailed summary"
SIGNAL_A_CONTRACT_VERSION = "2.1.223"


def _extract_message_text(content: Any) -> str | None:
    """Return the concatenated text of a message `content` field.

    `content` may be a plain string or a list of Anthropic content blocks.
    Only dict blocks with `type == "text"` and a string `text` field
    contribute; every other block (image, tool_result, thinking, ...) is
    ignored rather than raising. Any other `content` shape returns `None`.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return None


def is_compaction_request(body: object) -> bool:
    """Return True when `body` is a Claude Code compaction request.

    A request is a compaction request when `body` is a dict, `messages` is a
    non-empty list, the final message is a dict with `role == "user"`, and
    its extracted text starts with `SIGNAL_A_PREFIX` and contains
    `SIGNAL_A_MARKER`. Any other shape returns `False` without raising,
    since request bodies come from an untrusted client.
    """
    if not isinstance(body, dict):
        return False
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    last_message = messages[-1]
    if not isinstance(last_message, dict) or last_message.get("role") != "user":
        return False
    text = _extract_message_text(last_message.get("content"))
    if text is None:
        return False
    return text.startswith(SIGNAL_A_PREFIX) and SIGNAL_A_MARKER in text
