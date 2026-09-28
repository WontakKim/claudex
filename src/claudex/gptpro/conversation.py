"""Pure URL and turn extraction for ChatGPT conversation traffic."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, unquote, urlsplit

TRUSTED_ORIGIN = "https://chatgpt.com"
CHATGPT_URL = f"{TRUSTED_ORIGIN}/"
TRANSPORT_NONCE_LABEL = "gptpro-transport-nonce"
COMPLETION_REPORT_PATH_FRAGMENT = "/backend-api/lat/"
_SANDBOX_SCHEME = "sandbox:"

_UUID_SOURCE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_UUID_EXACT_PATTERN = re.compile(rf"{_UUID_SOURCE}", re.IGNORECASE)
_CONVERSATION_ID_PATH_PATTERN = re.compile(
    rf"/backend-api/[^?#]*conversations?/(?:gen_title/)?({_UUID_SOURCE})(?:/|$)",
    re.IGNORECASE,
)
_CONVERSATION_STREAM_PATHS = {
    "/backend-api/conversation",
    "/backend-api/f/conversation",
}
# A fenced code block runs from an opening ``` or ~~~ line to a line starting
# with the same fence, or to the end of the text when it is never closed.
_FENCED_CODE_PATTERN = re.compile(
    r"^ {0,3}(`{3,}|~{3,}).*?(?:^ {0,3}\1|\Z)", re.MULTILINE | re.DOTALL
)
# What may follow a link destination: an optional title in double quotes,
# single quotes, or parentheses after whitespace, then the closing ")".
_LINK_TAIL_PATTERN = re.compile(
    r"""(?:\s+(?:"[^"]*"|'[^']*'|\([^()]*\)))?\s*\)"""
)
# An inline code span is delimited by equal-length backtick runs.
_INLINE_CODE_PATTERN = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)", re.DOTALL)


@dataclass(frozen=True)
class SandboxFileReference:
    """A ``sandbox:`` file link in one answer message.

    ``sandbox_path`` is the percent-decoded link path exactly as linked; it is
    not validated here, so callers must validate it before any use.
    ``message_id`` is the linking message's ``id``, when it has one.
    """

    message_id: str | None
    sandbox_path: str


@dataclass(frozen=True)
class AssistantTurn:
    """Raw markdown, completion state, and linked files for one assistant turn.

    ``file_references`` come only from the messages whose text forms ``text``.
    """

    text: str
    finished: bool
    file_references: tuple[SandboxFileReference, ...] = ()


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


def is_conversation_id(value: object) -> bool:
    """Return whether a value is a canonical ChatGPT conversation UUID."""
    return (
        isinstance(value, str)
        and _UUID_EXACT_PATTERN.fullmatch(value) is not None
    )


def build_conversation_url(conversation_id: str) -> str:
    """Build the canonical ChatGPT URL for an existing conversation."""
    if not is_conversation_id(conversation_id):
        raise ValueError("conversation_id must be a canonical UUID")
    return f"{TRUSTED_ORIGIN}/c/{conversation_id.lower()}"


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


def _sandbox_link_targets(text: str) -> list[str]:
    """Return the ``sandbox:`` destinations of inline Markdown links in order.

    Links inside code blocks and code spans are examples, not links. A
    destination is either ``<...>`` or runs until the ``)`` that closes the
    link, allowing balanced parentheses inside it as Markdown does. An
    optional link title may follow; a destination that the link's ``)`` does
    not close is ignored.
    """
    text = _INLINE_CODE_PATTERN.sub(" ", _FENCED_CODE_PATTERN.sub(" ", text))
    targets: list[str] = []
    cursor = text.find("](")
    while cursor != -1:
        start = cursor + 2
        end = start
        if text.startswith("<" + _SANDBOX_SCHEME, start):
            closing = text.find(">", start)
            if (
                closing != -1
                and "\n" not in text[start:closing]
                and _LINK_TAIL_PATTERN.match(text, closing + 1)
            ):
                targets.append(text[start + 1:closing])
                end = closing
        elif text.startswith(_SANDBOX_SCHEME, start):
            depth = 0
            while end < len(text) and not text[end].isspace():
                if text[end] == "(":
                    depth += 1
                elif text[end] == ")":
                    if depth == 0:
                        break
                    depth -= 1
                end += 1
            if _LINK_TAIL_PATTERN.match(text, end):
                targets.append(text[start:end])
        cursor = text.find("](", end)
    return targets


def _file_references(
    messages: list[Mapping[str, Any]],
) -> tuple[SandboxFileReference, ...]:
    """Collect sandbox links from answer messages, first occurrence per path."""
    references: dict[str, SandboxFileReference] = {}
    for message in messages:
        message_id = message.get("id")
        for target in _sandbox_link_targets(_message_text(message)):
            sandbox_path = unquote(target.removeprefix(_SANDBOX_SCHEME))
            references.setdefault(
                sandbox_path,
                SandboxFileReference(
                    message_id=message_id if isinstance(message_id, str) else None,
                    sandbox_path=sandbox_path,
                ),
            )
    return tuple(references.values())


def _is_user_facing_text(message: Mapping[str, Any]) -> bool:
    """Return whether an assistant message is answer text shown to the user.

    Tool calls are addressed to a tool recipient, and reasoning-model progress
    notes use the ``commentary`` channel; only unaddressed text on no channel
    or the ``final`` channel is part of the answer.
    """
    content = message.get("content")
    if not isinstance(content, Mapping) or content.get("content_type") != "text":
        return False
    return (
        message.get("recipient") in (None, "all")
        and message.get("channel") in (None, "final")
    )


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

    final_channel: list[Mapping[str, Any]] = []
    unchannelled: list[Mapping[str, Any]] = []
    for node in reversed(chain[:anchor_index]):
        message = node.get("message")
        role = _message_role(message)
        if role == "user":
            break
        if role != "assistant" or not isinstance(message, Mapping):
            continue
        if not _is_user_facing_text(message):
            continue
        if message.get("channel") == "final":
            final_channel.append(message)
        else:
            unchannelled.append(message)

    # An explicit final-channel answer supersedes unchannelled text, which
    # only forms the answer for models that do not tag channels.
    answer = final_channel or unchannelled
    if not answer:
        return AssistantTurn(text="", finished=False)

    texts = [text for text in map(_message_text, answer) if text]
    finished = answer[-1].get("end_turn") is True
    return AssistantTurn(
        text="\n\n".join(texts),
        finished=finished,
        file_references=_file_references(answer),
    )


def build_nonce_marker(nonce: str) -> str:
    """Build the marker embedded in a submitted prompt to identify its turn."""
    return f"[{TRANSPORT_NONCE_LABEL}:{nonce}]"
