"""Translate Anthropic Messages API requests into Codex Responses API payloads.

Ported (and trimmed) from CLIProxyAPI's codex/claude request translator:
- system prompt and mid-conversation system messages -> "developer" role input messages
- text/image/document/tool_use/tool_result/thinking content blocks -> Responses input items
- Claude tools -> function tools with 64-char-safe names and normalized JSON schemas
- Anthropic's server-side web_search tool -> the Codex web_search tool
- thinking.budget_tokens -> reasoning.effort

Content the private Codex backend cannot receive raises TranslationError
instead of being dropped, so nothing a client sent silently disappears.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

# OpenAI Responses API limit for tool names and call ids.
_NAME_LIMIT = 64

# Codex reasoning signatures are Fernet tokens; anything else (e.g. a genuine
# Anthropic thinking signature replayed by the client) must not be sent upstream.
_GPT_REASONING_SIGNATURE_RE = re.compile(r"^gAAAA[A-Za-z0-9_=-]+$")

# Claude Code prepends billing attribution system text that has no value upstream.
_CLAUDE_ATTRIBUTION_PREFIX = "x-anthropic-billing-header:"

_CLAUDE_WEB_SEARCH_TOOL_TYPES = frozenset({"web_search_20250305", "web_search_20260209"})

# thinking.budget_tokens upper bounds per effort level, mirroring CLIProxyAPI.
_EFFORT_BUDGET_THRESHOLDS = (
    (512, "minimal"),
    (1024, "low"),
    (8192, "medium"),
    (24576, "high"),
)

_SESSION_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_OID, "claudex-gateway:session")

# The only document form the private Codex backend has been verified to accept.
_SUPPORTED_DOCUMENT_MEDIA_TYPE = "application/pdf"


class TranslationError(Exception):
    """Raised when a mapped Claude request contains content that cannot be
    represented on the Codex backend; surfaced as invalid_request_error."""


def is_gpt_reasoning_signature(signature: str) -> bool:
    return bool(_GPT_REASONING_SIGNATURE_RE.match(signature))


def shorten_tool_name(name: str) -> str:
    """Shorten a tool name to the Responses API limit, keeping MCP names readable."""
    if len(name) <= _NAME_LIMIT:
        return name
    if name.startswith("mcp__"):
        separator_index = name.rfind("__")
        if separator_index > 0:
            candidate = "mcp__" + name[separator_index + 2 :]
            return candidate[:_NAME_LIMIT]
    return name[:_NAME_LIMIT]


def build_tool_name_shortening_map(claude_request: dict[str, Any]) -> dict[str, str]:
    """Map original tool names to unique shortened names for this request."""
    names = [
        tool["name"]
        for tool in claude_request.get("tools") or []
        if isinstance(tool, dict) and isinstance(tool.get("name"), str) and tool["name"]
    ]

    used: set[str] = set()
    mapping: dict[str, str] = {}
    for name in names:
        candidate = shorten_tool_name(name)
        if candidate in used:
            suffix_counter = 1
            while True:
                suffix = f"_{suffix_counter}"
                truncated = candidate[: max(_NAME_LIMIT - len(suffix), 0)]
                unique = truncated + suffix
                if unique not in used:
                    candidate = unique
                    break
                suffix_counter += 1
        used.add(candidate)
        mapping[name] = candidate
    return mapping


def shorten_call_id(call_id: str) -> str:
    """Keep tool call ids within the Responses API limit with a stable hash suffix."""
    if len(call_id) <= _NAME_LIMIT:
        return call_id
    digest = hashlib.sha256(call_id.encode()).hexdigest()[:16]
    suffix = "_" + digest
    return call_id[: _NAME_LIMIT - len(suffix)] + suffix


def derive_session_id(claude_request: dict[str, Any]) -> str:
    """Derive a stable prompt-cache/session id from Claude Code request metadata."""
    metadata = claude_request.get("metadata")
    user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
    if isinstance(user_id, str) and user_id:
        return str(uuid.uuid5(_SESSION_ID_NAMESPACE, user_id))
    return str(uuid.uuid4())


def _normalize_tool_parameters(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    normalized = {key: value for key, value in schema.items() if key != "$schema"}
    normalized.setdefault("type", "object")
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized["properties"] = {}
    return normalized


def _image_data_url(source: dict[str, Any]) -> str | None:
    data = source.get("data") or source.get("base64")
    if not data:
        return None
    media_type = source.get("media_type") or source.get("mime_type") or "application/octet-stream"
    return f"data:{media_type};base64,{data}"


def _translate_system_prompt(system: Any) -> dict[str, Any] | None:
    texts: list[str] = []

    def append_text(text: str) -> None:
        if text and not text.lstrip().startswith(_CLAUDE_ATTRIBUTION_PREFIX):
            texts.append(text)

    if isinstance(system, str):
        append_text(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                append_text(block.get("text", ""))

    if not texts:
        return None
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text} for text in texts],
    }


def _translate_tool_result_output(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    items: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            items.append({"type": "input_text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            data_url = _image_data_url(block.get("source") or {})
            if data_url:
                items.append({"type": "input_image", "image_url": data_url})
        elif block.get("type") == "document":
            raise TranslationError(
                "document blocks inside tool_result content are not supported "
                "on mapped Codex models"
            )
    return items if items else ""


def _translate_document_block(block: dict[str, Any], role: str) -> dict[str, Any]:
    """Translate a base64 PDF document block into a Responses input_file.

    Only the verified-accepted form is translated; everything else is
    rejected so no document silently disappears from the conversation."""
    if role != "user":
        raise TranslationError(
            f"document blocks are only supported in user messages, not {role!r} messages"
        )
    citations = block.get("citations")
    if isinstance(citations, dict) and citations.get("enabled"):
        raise TranslationError("document citations are not supported on mapped Codex models")
    source = block.get("source")
    source_type = source.get("type") if isinstance(source, dict) else None
    media_type = source.get("media_type") if isinstance(source, dict) else None
    if source_type != "base64" or media_type != _SUPPORTED_DOCUMENT_MEDIA_TYPE:
        raise TranslationError(
            "mapped Codex models only support base64 application/pdf documents; "
            f"got source type {source_type!r} with media type {media_type!r}"
        )
    data = source.get("data")
    if not isinstance(data, str) or not data:
        raise TranslationError("document source has no base64 data")
    title = block.get("title")
    filename = title if isinstance(title, str) and title else "document.pdf"
    return {
        "type": "input_file",
        "filename": filename,
        "file_data": f"data:{_SUPPORTED_DOCUMENT_MEDIA_TYPE};base64,{data}",
    }


def _translate_messages(
    claude_request: dict[str, Any], name_map: dict[str, str]
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []

    system_message = _translate_system_prompt(claude_request.get("system"))
    if system_message is not None:
        input_items.append(system_message)

    for message in claude_request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "")
        content = message.get("content")

        if role == "system":
            # Mid-conversation system messages keep their position but carry
            # operator authority, which Responses expresses as "developer".
            text = content if isinstance(content, str) else _collect_text_blocks(content)
            if text:
                input_items.append(
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
            continue

        content_parts: list[dict[str, Any]] = []

        def flush(parts: list[dict[str, Any]] = content_parts, message_role: str = role) -> None:
            if parts:
                input_items.append(
                    {"type": "message", "role": message_role, "content": list(parts)}
                )
                parts.clear()

        text_part_type = "output_text" if role == "assistant" else "input_text"

        if isinstance(content, str):
            content_parts.append({"type": text_part_type, "text": content})
            flush()
            continue
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                content_parts.append({"type": text_part_type, "text": block.get("text", "")})
            elif block_type == "document":
                content_parts.append(_translate_document_block(block, role))
            elif block_type == "image":
                data_url = _image_data_url(block.get("source") or {})
                if data_url:
                    content_parts.append({"type": "input_image", "image_url": data_url})
            elif block_type == "thinking":
                if role != "assistant":
                    continue
                signature = block.get("signature") or ""
                if not is_gpt_reasoning_signature(signature):
                    continue
                flush()
                input_items.append(
                    {
                        "type": "reasoning",
                        "summary": [],
                        "content": None,
                        "encrypted_content": signature,
                    }
                )
            elif block_type == "tool_use":
                flush()
                name = block.get("name", "")
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": shorten_call_id(block.get("id", "")),
                        "name": name_map.get(name, shorten_tool_name(name)),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    }
                )
            elif block_type == "tool_result":
                flush()
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": shorten_call_id(block.get("tool_use_id", "")),
                        "output": _translate_tool_result_output(block.get("content")),
                    }
                )
        flush()

    return input_items


def _collect_text_blocks(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def web_search_tool_names(claude_request: dict[str, Any]) -> set[str]:
    return {
        tool["name"]
        for tool in claude_request.get("tools") or []
        if isinstance(tool, dict)
        and tool.get("type") in _CLAUDE_WEB_SEARCH_TOOL_TYPES
        and tool.get("name")
    }


def _translate_web_search_tool(tool: dict[str, Any]) -> dict[str, Any]:
    web_search_tool: dict[str, Any] = {"type": "web_search"}
    allowed_domains = tool.get("allowed_domains")
    if isinstance(allowed_domains, list) and allowed_domains:
        web_search_tool["filters"] = {"allowed_domains": allowed_domains}
    user_location = tool.get("user_location")
    if isinstance(user_location, dict) and user_location:
        web_search_tool["user_location"] = user_location
    return web_search_tool


def _translate_tools(
    claude_request: dict[str, Any], name_map: dict[str, str]
) -> list[dict[str, Any]] | None:
    tools = claude_request.get("tools")
    if not isinstance(tools, list):
        return None

    translated: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") in _CLAUDE_WEB_SEARCH_TOOL_TYPES:
            translated.append(_translate_web_search_tool(tool))
            continue
        name = tool.get("name", "")
        function_tool: dict[str, Any] = {
            "type": "function",
            "name": name_map.get(name, shorten_tool_name(name)),
            "parameters": _normalize_tool_parameters(tool.get("input_schema")),
            "strict": False,
        }
        if tool.get("description"):
            function_tool["description"] = tool["description"]
        translated.append(function_tool)
    return translated


def _translate_tool_choice(
    tool_choice: Any, name_map: dict[str, str], web_search_names: set[str]
) -> Any:
    if not isinstance(tool_choice, dict):
        return "auto"
    choice_type = tool_choice.get("type", "")
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool":
        name = tool_choice.get("name", "")
        if not name:
            return "auto"
        if name in web_search_names:
            return {"type": "web_search"}
        return {"type": "function", "name": name_map.get(name, shorten_tool_name(name))}
    return "auto"


def _budget_to_effort(budget: int) -> str:
    for threshold, effort in _EFFORT_BUDGET_THRESHOLDS:
        if budget <= threshold:
            return effort
    return "xhigh"


def _derive_reasoning_effort(claude_request: dict[str, Any]) -> str:
    thinking = claude_request.get("thinking")
    if not isinstance(thinking, dict):
        return "medium"

    thinking_type = thinking.get("type")
    if thinking_type == "enabled":
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int) and budget > 0:
            return _budget_to_effort(budget)
        return "medium"
    if thinking_type in ("adaptive", "auto"):
        output_config = claude_request.get("output_config")
        effort = output_config.get("effort") if isinstance(output_config, dict) else None
        if isinstance(effort, str) and effort.strip().lower() in (
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            return effort.strip().lower()
        return "xhigh"
    if thinking_type == "disabled":
        return "low"
    return "medium"


def translate_claude_request_to_codex(
    claude_request: dict[str, Any],
    codex_model: str,
    reasoning_effort_override: str | None = None,
    *,
    service_tier: str | None = None,
) -> dict[str, Any]:
    """Build a Codex Responses API payload, optionally with a service tier."""
    name_map = build_tool_name_shortening_map(claude_request)

    payload: dict[str, Any] = {
        "model": codex_model,
        "instructions": "",
        "input": _translate_messages(claude_request, name_map),
        "reasoning": {
            "effort": reasoning_effort_override or _derive_reasoning_effort(claude_request),
            "summary": "auto",
        },
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": derive_session_id(claude_request),
    }
    if service_tier is not None:
        payload["service_tier"] = service_tier

    tools = _translate_tools(claude_request, name_map)
    # An empty tools list means "no tools": neither field may go out, since
    # Grok 400s on tool_choice without tools and Codex rejects
    # parallel_tool_calls without tools.
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = _translate_tool_choice(
            claude_request.get("tool_choice"), name_map, web_search_tool_names(claude_request)
        )

    # The Responses API rejects parallel_tool_calls when no tools are present.
    if tools:
        tool_choice = claude_request.get("tool_choice")
        disable_parallel = isinstance(tool_choice, dict) and bool(
            tool_choice.get("disable_parallel_tool_use")
        )
        payload["parallel_tool_calls"] = not disable_parallel

    return payload
