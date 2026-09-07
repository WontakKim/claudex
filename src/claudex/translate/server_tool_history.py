"""Preserve provider-internal tool history without replaying incompatible blocks."""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import re
from typing import Any

GATEWAY_SERVER_TOOL_ID_PREFIX = "srvtoolu_gateway_"
_SERVER_TOOL_ID_PATTERN = re.compile(r"srvtoolu_[a-zA-Z0-9_]+")

logger = logging.getLogger("claudex.server")


def _server_tool_block_id(block: dict[str, Any]) -> str | None:
    block_type = block.get("type")
    if block_type == "server_tool_use":
        value = block.get("id")
    elif isinstance(block_type, str) and (
        block_type == "tool_result" or block_type.endswith("_tool_result")
    ):
        value = block.get("tool_use_id")
    else:
        return None
    return value if isinstance(value, str) else None


def _assistant_block_groups(
    messages: list[Any],
) -> Iterator[list[tuple[int, int, dict[str, Any]]]]:
    # Streaming clients may persist one assistant response as several messages.
    # A user turn ends the group so reused IDs cannot capture later client calls.
    group: list[tuple[int, int, dict[str, Any]]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            if group:
                yield group
                group = []
            continue
        content = message.get("content")
        if isinstance(content, list):
            group.extend(
                (message_index, block_index, block)
                for block_index, block in enumerate(content)
                if isinstance(block, dict)
            )
    if group:
        yield group


def normalize_server_tool_history(body: Any) -> Any:
    """Copy repaired blocks, preserving native tools and unchanged request identity.

    Provider-internal calls may use non-Anthropic IDs or return tool_result
    inside the assistant response. Replay those pairs as text rather than as
    client-executed calls. Unsigned Responses search results also need text
    replay because they cannot supply Anthropic's native encrypted_content.
    """
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return body

    messages = list(body["messages"])
    converted_count = 0
    for group in _assistant_block_groups(body["messages"]):
        server_calls = {
            block["id"]: block
            for _, _, block in group
            if block.get("type") == "server_tool_use"
            and isinstance(block.get("id"), str)
        }
        text_ids = {
            tool_id for tool_id in server_calls
            if tool_id.startswith(GATEWAY_SERVER_TOOL_ID_PREFIX)
            or not _SERVER_TOOL_ID_PATTERN.fullmatch(tool_id)
        }
        for _, _, block in group:
            tool_id = _server_tool_block_id(block)
            if tool_id is None:
                continue
            if block.get("type") == "tool_result" and tool_id in server_calls:
                text_ids.add(tool_id)
            if block.get("type") != "web_search_tool_result":
                continue
            if (
                tool_id.startswith(GATEWAY_SERVER_TOOL_ID_PREFIX)
                or not _SERVER_TOOL_ID_PATTERN.fullmatch(tool_id)
            ):
                text_ids.add(tool_id)
            results = block.get("content")
            if isinstance(results, list) and any(
                isinstance(result, dict)
                and result.get("type") == "web_search_result"
                and (
                    not isinstance(result.get("encrypted_content"), str)
                    or not result["encrypted_content"]
                )
                for result in results
            ):
                text_ids.add(tool_id)
        if not text_ids:
            continue

        for message_index, block_index, block in group:
            tool_id = _server_tool_block_id(block)
            if tool_id not in text_ids:
                continue
            call = server_calls.get(tool_id, {})
            name = call.get("name", "web_search")
            label = "Web search" if name == "web_search" else f"Server tool {name}"
            if block["type"] == "server_tool_use":
                label, value = f"{label} input", block.get("input", {})
            else:
                label, value = f"{label} results", block.get("content", [])
                if block.get("is_error"):
                    label += " (error)"
            text_block: dict[str, Any] = {
                "type": "text",
                "text": f"{label}: {json.dumps(value, ensure_ascii=False)}",
            }
            if "cache_control" in block:
                text_block["cache_control"] = block["cache_control"]
            if messages[message_index] is body["messages"][message_index]:
                messages[message_index] = {
                    **messages[message_index],
                    "content": list(messages[message_index]["content"]),
                }
            messages[message_index]["content"][block_index] = text_block
            converted_count += 1

    if not converted_count:
        return body
    logger.info("replaying %d incompatible server-tool blocks as text", converted_count)
    return {**body, "messages": messages}
