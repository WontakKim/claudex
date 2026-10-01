"""Compile a self-contained Anthropic tool history for Responses consumers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

_GENERATED_NAME_PREFIX = "mcp__gw_"
_VALID_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_CONTROL_TYPES = frozenset({"tool_addition", "tool_removal"})
_TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01"
_INLINE_TOOLS_BETA = "inline-tools-2026-09-15"
_WEB_SEARCH_TYPES = frozenset({"web_search_20250305", "web_search_20260209"})
_CLIENT_TOOL_TYPE = re.compile(r"(?:bash|text_editor|computer|memory)_[0-9]{8}")
_PLACEHOLDER_DESCRIPTION = (
    "Reserved placeholder that keeps deferred tool loading active; never call this tool."
)


class ToolProtocolError(ValueError):
    """A tool declaration or history cannot be represented safely upstream."""


@dataclass
class ToolState:
    tools: list[dict]
    name_map: dict[str, str]
    active_name_map: dict[str, str]
    markers: dict[tuple[int, int], str]
    references: dict[tuple[int, int, int], str]


def stable_tool_name(name: str) -> str:
    """Keep valid names, reserving a deterministic 224-bit digest namespace."""
    if not isinstance(name, str):
        raise ToolProtocolError("tool name must be a string")
    if _VALID_NAME.fullmatch(name) and not name.startswith(_GENERATED_NAME_PREFIX):
        return name
    try:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    except UnicodeEncodeError as error:
        raise ToolProtocolError("tool name must contain valid Unicode") from error
    return _GENERATED_NAME_PREFIX + digest[: 64 - len(_GENERATED_NAME_PREFIX)]


def _name(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ToolProtocolError(f"{location}: tool name must be a nonempty string")
    return value


def _definition(value: Any, location: str, *, inline: bool) -> dict:
    if not isinstance(value, dict):
        raise ToolProtocolError(f"{location}: tool definition must be an object")
    _name(value.get("name"), f"{location}.name")
    tool_type = value.get("type", "custom")
    if not isinstance(tool_type, str) or not tool_type:
        raise ToolProtocolError(f"{location}.type: tool type must be a nonempty string")
    if tool_type.startswith("tool_search_"):
        raise ToolProtocolError(
            f"{location}: Anthropic hosted tool_search types are unsupported "
            "by the Responses normalizer; use the original native route"
        )
    if tool_type in {"tool_reference", "tool_definition"}:
        raise ToolProtocolError(f"{location}: expected a definition, not {tool_type}")
    if (
        tool_type != "custom"
        and tool_type not in _WEB_SEARCH_TYPES
        and not _CLIENT_TOOL_TYPE.fullmatch(tool_type)
    ):
        raise ToolProtocolError(
            f"{location}: unsupported hosted or unknown tool type {tool_type!r}; "
            "use the original native route"
        )
    if "defer_loading" in value and not isinstance(value["defer_loading"], bool):
        raise ToolProtocolError(f"{location}.defer_loading: must be a boolean")
    if "description" in value and not isinstance(value["description"], str):
        raise ToolProtocolError(f"{location}.description: must be a string")
    if inline:
        schema = value.get("input_schema")
        if tool_type == "custom" or "input_schema" in value:
            if not isinstance(schema, dict):
                raise ToolProtocolError(f"{location}.input_schema: must be an object")
            if schema.get("type") != "object":
                raise ToolProtocolError(f"{location}.input_schema.type: must be object")
            if "properties" in schema and not isinstance(schema["properties"], dict):
                raise ToolProtocolError(
                    f"{location}.input_schema.properties: must be an object"
                )
    return value


def _is_placeholder(tool: dict) -> bool:
    schema = tool.get("input_schema")
    return (
        tool.get("name") == "DeferredToolPlaceholder"
        and tool.get("description") == _PLACEHOLDER_DESCRIPTION
        and tool.get("defer_loading") is True
        and tool.get("type", "custom") == "custom"
        and isinstance(schema, dict)
        and schema.get("type") == "object"
        and schema.get("properties") == {}
        and set(schema) <= {"type", "properties", "required", "additionalProperties"}
        and schema.get("required", []) == []
        and schema.get("additionalProperties", False) is False
    )


def _reference(value: Any, location: str, *, result: bool) -> str:
    name_key, forbidden_key = ("tool_name", "name") if result else ("name", "tool_name")
    if not isinstance(value, dict) or value.get("type") != "tool_reference":
        raise ToolProtocolError(f"{location}: expected a tool_reference object")
    if forbidden_key in value:
        raise ToolProtocolError(f"{location}: use {name_key}, not {forbidden_key}")
    return _name(value.get(name_key), f"{location}.{name_key}")


def _marker(value: dict, location: str) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ToolProtocolError(f"{location}: tool protocol block must be JSON") from error


def _message_blocks(messages: list) -> list[list[dict]]:
    blocks = []
    for message_index, message in enumerate(messages):
        location = f"messages[{message_index}]"
        if not isinstance(message, dict):
            raise ToolProtocolError(f"{location}: message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or role not in {"user", "assistant", "system"}:
            raise ToolProtocolError(f"{location}.role: expected user, assistant, or system")
        content = message.get("content", [])
        if isinstance(content, str):
            blocks.append([])
            continue
        if not isinstance(content, list):
            raise ToolProtocolError(f"{location}.content: expected a string or array")
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ToolProtocolError(
                    f"{location}.content[{block_index}]: block must be an object"
                )
            if not isinstance(block.get("type"), str) or not block["type"]:
                raise ToolProtocolError(
                    f"{location}.content[{block_index}].type: must be a nonempty string"
                )
        blocks.append(content)
    return blocks


def _control_sections(messages: list, blocks: list[list[dict]]) -> set[int]:
    control_messages = set()
    for message_index, content in enumerate(blocks):
        for block_index, block in enumerate(content):
            if block.get("type") not in _CONTROL_TYPES:
                continue
            location = f"messages[{message_index}].content[{block_index}]"
            if messages[message_index]["role"] != "system":
                raise ToolProtocolError(f"{location}: tool controls require a system message")
            control_messages.add(message_index)
    message_index = 0
    while message_index < len(messages):
        if messages[message_index]["role"] != "system":
            message_index += 1
            continue
        first = message_index
        while message_index < len(messages) and messages[message_index]["role"] == "system":
            message_index += 1
        section_controls = [index for index in range(first, message_index) if index in control_messages]
        if not section_controls:
            continue
        location = f"messages[{section_controls[0]}]"
        previous = messages[first - 1] if first else None
        previous_blocks = blocks[first - 1] if first else []
        is_server_result = (
            previous is not None
            and previous["role"] == "assistant"
            and bool(previous_blocks)
            and isinstance(previous_blocks[-1].get("type"), str)
            and previous_blocks[-1]["type"].endswith("_tool_result")
        )
        if is_server_result and previous.get("stop_reason") == "pause_turn":
            raise ToolProtocolError(f"{location}: tool controls cannot follow a paused server result")
        if previous is None or (previous["role"] != "user" and not is_server_result):
            raise ToolProtocolError(
                f"{location}: tool control section must follow a user message "
                "or completed assistant server-tool result"
            )
        if message_index < len(messages) and messages[message_index]["role"] != "assistant":
            raise ToolProtocolError(
                f"{location}: tool control section must precede assistant or end"
            )
    return control_messages


def compile_tool_state(request: dict, beta_header: str | None = None) -> ToolState:
    """Replay definitions and controls without modifying or inferring a catalog.

    Root schemas are a frozen baseline when controls exist. Search references
    load known definitions but cannot undo a removal; only explicit additions
    do that. Marker JSON records each event, never the final state of its tool.
    """
    if not isinstance(request, dict):
        raise ToolProtocolError("request: must be an object")
    if beta_header is not None and not isinstance(beta_header, str):
        raise ToolProtocolError("beta_header: must be a string or None")
    betas = set(beta_header.split(",")) if beta_header is not None else None
    if betas is not None:
        betas = {beta.strip() for beta in betas}
    messages = request.get("messages", [])
    if not isinstance(messages, list):
        raise ToolProtocolError("messages: must be an array")
    blocks = _message_blocks(messages)
    control_messages = _control_sections(messages, blocks)
    system = request.get("system")
    if isinstance(system, list):
        for index, block in enumerate(system):
            if (
                isinstance(block, dict)
                and isinstance(block.get("type"), str)
                and block["type"] in _CONTROL_TYPES
            ):
                raise ToolProtocolError(
                    f"system[{index}]: tool controls require a system message in messages[]"
                )

    name_map: dict[str, str] = {}
    reverse_names: dict[str, str] = {}

    def remember_name(name: str, location: str) -> None:
        if name in name_map:
            return
        try:
            mapped = stable_tool_name(name)
        except ToolProtocolError as error:
            raise ToolProtocolError(f"{location}: {error}") from error
        previous = reverse_names.get(mapped)
        if previous is not None and previous != name:
            raise ToolProtocolError(
                f"{location}: tool name collision between {previous!r} and {name!r}"
            )
        name_map[name] = mapped
        reverse_names[mapped] = name

    registry: dict[str, dict] = {}
    root_definitions: dict[str, dict] = {}
    active: dict[str, dict] = {}
    root_tools = request.get("tools")
    if root_tools is None:
        root_tools = []
    if not isinstance(root_tools, list):
        raise ToolProtocolError("tools: must be an array")
    for index, value in enumerate(root_tools):
        location = f"tools[{index}]"
        tool = _definition(value, location, inline=False)
        name = tool["name"]
        remember_name(name, f"{location}.name")
        if name in root_definitions:
            if _marker(root_definitions[name], location) != _marker(tool, location):
                raise ToolProtocolError(f"{location}: conflicting root definitions for {name!r}")
            continue
        root_definitions[name] = tool
        if _is_placeholder(tool):
            continue
        registry[name] = deepcopy(tool)
        if not control_messages or tool.get("defer_loading") is not True:
            active[name] = registry[name]

    choice = request.get("tool_choice")
    forced_name = None
    if isinstance(choice, dict) and choice.get("type") == "tool":
        forced_name = _name(choice.get("name"), "tool_choice.name")
        remember_name(forced_name, "tool_choice.name")

    markers: dict[tuple[int, int], str] = {}
    references: dict[tuple[int, int, int], str] = {}
    tombstones: set[str] = set()
    pending_calls: dict[str, tuple[str | None, int]] = {}
    unresolved: dict[str, str] = {}
    last_assistant = max(
        (index for index, message in enumerate(messages) if message["role"] == "assistant"),
        default=-1,
    )
    for message_index, content in enumerate(blocks):
        role = messages[message_index]["role"]
        if message_index in control_messages and pending_calls:
            raise ToolProtocolError(
                f"messages[{message_index}]: tool controls cannot split a tool_use/tool_result pair"
            )
        for block_index, block in enumerate(content):
            location = f"messages[{message_index}].content[{block_index}]"
            block_type = block.get("type")
            if block_type in _CONTROL_TYPES:
                tool = block.get("tool")
                is_definition = isinstance(tool, dict) and tool.get("type") == "tool_definition"
                if betas is not None:
                    required = {_INLINE_TOOLS_BETA} if is_definition else {
                        _TOOL_CHANGES_BETA, _INLINE_TOOLS_BETA
                    }
                    if not betas.intersection(required):
                        raise ToolProtocolError(
                            f"{location}: requires beta {' or '.join(sorted(required))}"
                        )
                if is_definition:
                    if block_type != "tool_addition":
                        raise ToolProtocolError(f"{location}.tool: removal requires a tool_reference")
                    definition = _definition(tool.get("definition"), f"{location}.tool.definition", inline=True)
                    name = definition["name"]
                    remember_name(name, f"{location}.tool.definition.name")
                    if _is_placeholder(definition):
                        raise ToolProtocolError(f"{location}: client placeholder is not callable")
                    previous = registry.get(name)
                    if previous is not None and previous.get("type", "custom") != definition.get("type", "custom"):
                        raise ToolProtocolError(f"{location}: cannot change tool type for {name!r}")
                    registry[name] = deepcopy(definition)
                else:
                    name = _reference(tool, f"{location}.tool", result=False)
                    remember_name(name, f"{location}.tool.name")
                if block_type == "tool_addition":
                    if name not in registry:
                        raise ToolProtocolError(f"{location}: no known definition for {name!r}")
                    tombstones.discard(name)
                    active[name] = registry[name]
                else:
                    tombstones.add(name)
                    active.pop(name, None)
                unresolved.pop(name, None)
                markers[message_index, block_index] = _marker(
                    {**block, "call_name": name_map[name]}, location
                )
            elif block_type in {"tool_use", "server_tool_use"}:
                name = _name(block.get("name"), f"{location}.name")
                remember_name(name, f"{location}.name")
                if block_type == "tool_use":
                    if role != "assistant":
                        raise ToolProtocolError(f"{location}: tool_use requires an assistant message")
                    call_id = _name(block.get("id"), f"{location}.id")
                    # Ambiguous historical IDs retain visible history but cannot
                    # establish a successful ToolSearch discovery association.
                    previous_call = pending_calls.get(call_id)
                    pending_calls[call_id] = (None, previous_call[1] + 1) if previous_call else (name, 1)
            elif block_type == "tool_result":
                nested = block.get("content")
                call_name = None
                if role == "user":
                    call_id = block.get("tool_use_id")
                    if isinstance(call_id, str) and call_id in pending_calls:
                        call_name, outstanding_count = pending_calls[call_id]
                        if outstanding_count == 1:
                            del pending_calls[call_id]
                        else:
                            pending_calls[call_id] = (call_name, outstanding_count - 1)
                if not isinstance(nested, list):
                    continue
                if "is_error" in block and not isinstance(block["is_error"], bool):
                    raise ToolProtocolError(f"{location}.is_error: must be a boolean")
                discovery = call_name == "ToolSearch" and not block.get("is_error", False)
                for nested_index, reference in enumerate(nested):
                    if not isinstance(reference, dict) or reference.get("type") != "tool_reference":
                        continue
                    reference_location = f"{location}.content[{nested_index}]"
                    _name(block.get("tool_use_id"), f"{location}.tool_use_id")
                    name = _reference(reference, reference_location, result=True)
                    remember_name(name, f"{reference_location}.tool_name")
                    known = name in registry
                    if discovery and name not in tombstones:
                        if known:
                            active[name] = registry[name]
                        elif message_index > last_assistant:
                            unresolved[name] = reference_location
                    references[message_index, block_index, nested_index] = _marker(
                        {**reference, "call_name": name_map[name], "discovery": discovery},
                        reference_location,
                    )
            elif block_type in {"tool_reference", "tool_definition"}:
                raise ToolProtocolError(f"{location}: {block_type} is not a standalone content block")

    for name, location in unresolved.items():
        raise ToolProtocolError(f"{location}: current ToolSearch reference has no known definition for {name!r}")
    if forced_name is not None and forced_name not in active:
        raise ToolProtocolError(f"tool_choice.name: forced tool {forced_name!r} is not active")
    return ToolState(
        tools=[deepcopy(tool) for tool in active.values()],
        name_map=name_map,
        active_name_map={name: name_map[name] for name in active},
        markers=markers,
        references=references,
    )
