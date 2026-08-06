"""Detect Claude Code compaction requests ("Signal A") and build the reroute.

Claude Code has no metadata field, model name, or `querySource` marker that
survives onto the wire for a compaction request; the only reliable signal is
the literal text of the final `user` message the client sends when it asks
the model to summarize the conversation. This single detector covers all
three compaction variants (full compact, partial `from`, partial `up_to`)
because they share the same header and marker text. See
`.docs/research/claude-code/compact-detection.md` for how these literals
were extracted.

Once a compaction request is detected, `build_reroute_payload` and
`build_reroute_headers` prepare an independent Anthropic-bound call: the
original request is kept untouched for mapped fallback, and the reroute
never carries the gateway-local bearer token or a translated (and therefore
invalid) thinking signature.
"""

from __future__ import annotations

import copy
import hmac
from collections.abc import Mapping
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


_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def build_reroute_payload(body: dict, target_model_id: str) -> dict:
    """Build an isolated Anthropic payload for the compaction reroute call.

    Returns a deep copy of `body` with `model` replaced by `target_model_id`
    and every `thinking`/`redacted_thinking` content block stripped from
    block-list message content -- a thinking signature produced by
    translating a non-Anthropic backend's response is never a valid
    Anthropic signature, so it cannot be replayed to Anthropic. A message
    left with an empty content list after stripping is dropped entirely.
    String content, non-thinking blocks, and every other field are left
    untouched. No nested mutable object in the return value aliases `body`,
    so the caller can still send the original, unmodified request for
    mapped fallback.
    """
    payload = copy.deepcopy(body)
    payload["model"] = target_model_id
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload

    kept_messages: list[Any] = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            stripped_content = [
                block
                for block in message["content"]
                if not (
                    isinstance(block, dict) and block.get("type") in _THINKING_BLOCK_TYPES
                )
            ]
            if not stripped_content:
                continue
            message["content"] = stripped_content
        kept_messages.append(message)
    payload["messages"] = kept_messages
    return payload


_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


def build_reroute_headers(
    headers: Mapping[str, str], local_token: str | None
) -> dict[str, str] | None:
    """Build a fresh allowlisted header set for the compaction reroute call.

    This is a fresh construction, not a filtered copy of `headers`, so it
    never reuses the broad `_PASSTHROUGH_SKIP_REQUEST_HEADERS` denylist
    pattern: `host`, `content-length`, cookies, forwarding headers, and
    every other incoming header are simply never considered. Header names
    are matched case-insensitively.

    Always sets `content-type: application/json`. Copies `accept`,
    `anthropic-version`, and `anthropic-beta` when present, defaulting
    `anthropic-version` to `2023-06-01`. Copies `x-api-key` only when it is
    non-blank. Copies `authorization` only when it has the same syntactic
    Bearer shape `_require_local_token` checks (a "Bearer <token>" value
    with a non-empty token) and either no `local_token` is configured or
    the parsed token differs from `local_token` -- compared with
    `hmac.compare_digest` rather than `==` so the gateway-local token can
    never leak through a timing side channel, and so the gateway-local
    bearer token itself is never forwarded to Anthropic. Returns `None`
    when neither credential form survives, since no reroute is attempted
    without a genuine Anthropic credential.
    """
    lowered = {key.lower(): value for key, value in headers.items()}
    result: dict[str, str] = {"content-type": "application/json"}
    if "accept" in lowered:
        result["accept"] = lowered["accept"]
    result["anthropic-version"] = lowered.get("anthropic-version", _DEFAULT_ANTHROPIC_VERSION)
    if "anthropic-beta" in lowered:
        result["anthropic-beta"] = lowered["anthropic-beta"]

    has_credential = False

    x_api_key = lowered.get("x-api-key")
    if x_api_key is not None and x_api_key.strip():
        result["x-api-key"] = x_api_key
        has_credential = True

    authorization = lowered.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and token:
        if local_token is None or not hmac.compare_digest(token, local_token):
            result["authorization"] = authorization
            has_credential = True

    return result if has_credential else None
