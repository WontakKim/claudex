"""Pure URL and turn extraction for ChatGPT conversation traffic."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, urlsplit

TRUSTED_ORIGIN = "https://chatgpt.com"
CHATGPT_URL = f"{TRUSTED_ORIGIN}/"
TRANSPORT_NONCE_LABEL = "gptpro-transport-nonce"
COMPLETION_REPORT_PATH_FRAGMENT = "/backend-api/lat/"

_UUID_SOURCE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_UUID_EXACT_PATTERN = re.compile(rf"{_UUID_SOURCE}", re.IGNORECASE)
_CONVERSATION_ID_PATH_PATTERN = re.compile(
    rf"/backend-api/[^?#]*conversation/(?:gen_title/)?({_UUID_SOURCE})(?:/|$)",
    re.IGNORECASE,
)
_CONVERSATION_STREAM_PATHS = {
    "/backend-api/conversation",
    "/backend-api/f/conversation",
}


@dataclass(frozen=True)
class AssistantTurn:
    """Raw markdown and completion state for one assistant turn."""

    text: str
    finished: bool


def _parse_url(url: str) -> SplitResult | None:
    if not isinstance(url, str):
        return None
    try:
        return urlsplit(url)
    except ValueError:
        return None


def _has_trusted_origin(parsed: SplitResult) -> bool:
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    normalized_origin = f"{parsed.scheme.lower()}://{hostname or ''}"
    return normalized_origin == TRUSTED_ORIGIN and port in (None, 443)


def is_trusted_origin_url(url: str) -> bool:
    """Return whether a URL uses the canonical trusted ChatGPT origin."""
    parsed = _parse_url(url)
    return parsed is not None and _has_trusted_origin(parsed)


def is_conversation_stream_url(url: str) -> bool:
    """Return whether a URL is an exact ChatGPT conversation stream endpoint."""
    parsed = _parse_url(url)
    return bool(
        parsed is not None
        and _has_trusted_origin(parsed)
        and parsed.path in _CONVERSATION_STREAM_PATHS
    )


def is_completion_report_url(url: str) -> bool:
    """Return whether a trusted ChatGPT URL contains the completion-report path."""
    parsed = _parse_url(url)
    return bool(
        parsed is not None
        and _has_trusted_origin(parsed)
        and COMPLETION_REPORT_PATH_FRAGMENT in parsed.path
    )


def extract_conversation_id_from_url(url: str) -> str | None:
    """Extract a canonical conversation UUID from a backend-api request path."""
    parsed = _parse_url(url)
    if parsed is None:
        return None
    match = _CONVERSATION_ID_PATH_PATTERN.search(parsed.path)
    return match.group(1).lower() if match is not None else None


def extract_conversation_id_from_body(body: str) -> str | None:
    """Extract a canonical conversation UUID from a JSON request body."""
    if not isinstance(body, str) or not body:
        return None
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(parsed, dict):
        return None
    conversation_id = parsed.get("conversation_id")
    if not isinstance(conversation_id, str):
        return None
    if _UUID_EXACT_PATTERN.fullmatch(conversation_id) is None:
        return None
    return conversation_id.lower()


def _message_text(message: object) -> str:
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if not isinstance(content, Mapping) or content.get("content_type") != "text":
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    return "\n".join(part for part in parts if isinstance(part, str))


def _message_role(message: object) -> str | None:
    if not isinstance(message, Mapping):
        return None
    author = message.get("author")
    if not isinstance(author, Mapping):
        return None
    role = author.get("role")
    return role if isinstance(role, str) else None


def extract_assistant_turn(
    conversation: Mapping[str, Any], nonce_marker: str
) -> AssistantTurn | None:
    """Extract the nonce-anchored assistant turn from the active branch."""
    if not isinstance(conversation, Mapping):
        return None
    if not isinstance(nonce_marker, str) or not nonce_marker:
        return None

    mapping = conversation.get("mapping")
    current_node = conversation.get("current_node")
    if not isinstance(mapping, Mapping) or not isinstance(current_node, str):
        return None

    chain: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    cursor: object = current_node
    while isinstance(cursor, str) and cursor not in seen:
        seen.add(cursor)
        node = mapping.get(cursor)
        if not isinstance(node, Mapping):
            break
        chain.append(node)
        cursor = node.get("parent")

    anchor_index: int | None = None
    for index, node in enumerate(chain):
        message = node.get("message")
        if _message_role(message) != "user":
            continue
        if nonce_marker in _message_text(message):
            anchor_index = index
            break
    if anchor_index is None:
        return None

    texts: list[str] = []
    last_text_assistant: Mapping[str, Any] | None = None
    for node in reversed(chain[:anchor_index]):
        message = node.get("message")
        if _message_role(message) != "assistant" or not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, Mapping) or content.get("content_type") != "text":
            continue
        last_text_assistant = message
        text = _message_text(message)
        if text:
            texts.append(text)

    if last_text_assistant is None:
        return AssistantTurn(text="", finished=False)

    finished = (
        last_text_assistant.get("status") == "finished_successfully"
        or last_text_assistant.get("end_turn") is True
    )
    return AssistantTurn(text="\n\n".join(texts), finished=finished)


def build_nonce_marker(nonce: str) -> str:
    """Build the marker embedded in a submitted prompt to identify its turn."""
    return f"[{TRANSPORT_NONCE_LABEL}:{nonce}]"
