"""Tool-protocol replay and deterministic name mapping invariants."""

from copy import deepcopy
import hashlib
import json
import re

import pytest

import claudex.translate.tool_protocol as tool_protocol
from claudex.translate.tool_protocol import (
    ToolProtocolError,
    ToolState,
    compile_tool_state,
    stable_tool_name,
)


TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01"
INLINE_TOOLS_BETA = "inline-tools-2026-09-15"


def _tool(name: str, *, deferred: bool = False, revision: bool = False) -> dict:
    properties = {"query": {"type": "string"}}
    if revision:
        properties["revision"] = {"type": "integer"}
    definition = {
        "name": name,
        "description": "New revision" if revision else "Original definition",
        "input_schema": {"type": "object", "properties": properties},
    }
    if deferred:
        definition["defer_loading"] = True
    return definition


def _user(content: str | list = "question") -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str | list = "answer") -> dict:
    return {"role": "assistant", "content": content}


def _system(*blocks: dict) -> dict:
    return {"role": "system", "content": list(blocks)}


def _control(name: str, *, remove: bool = False) -> dict:
    return {
        "type": "tool_removal" if remove else "tool_addition",
        "tool": {"type": "tool_reference", "name": name},
    }


def _addition(definition: dict) -> dict:
    return {
        "type": "tool_addition",
        "tool": {"type": "tool_definition", "definition": definition},
    }


def _call(name: str = "ToolSearch", call_id: str = "search_1") -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": {}}


def _reference(name: str) -> dict:
    return {"type": "tool_reference", "tool_name": name}


def _result(*names: str, call_id: str = "search_1", is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": [_reference(name) for name in names],
        "is_error": is_error,
    }


def _search_history(*names: str, is_error: bool = False) -> list:
    return [_assistant([_call()]), _user([_result(*names, is_error=is_error)])]


def _controlled_request(*names: str) -> dict:
    return {
        "tools": [_tool("ToolSearch"), *[_tool(name, deferred=True) for name in names]],
        "messages": [_user(), _system(_control("ToolSearch"))],
    }


@pytest.mark.parametrize("name", ["a", "lookup", "A_0-z", "x" * 64])
def test_short_valid_names_are_unchanged(name: str) -> None:
    assert stable_tool_name(name) == name


@pytest.mark.parametrize("name", ["", "x" * 65, "mcp__gw_reserved", "has.dot", "has space", "검색", "a\n"])
def test_other_names_use_exact_sha256_prefix(name: str) -> None:
    expected = "mcp__gw_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:56]
    assert stable_tool_name(name) == expected
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", expected)


def test_unicode_identity_is_not_normalized() -> None:
    assert stable_tool_name("é") != stable_tool_name("é")


def test_generated_namespace_cannot_be_shadowed_by_a_short_original() -> None:
    original = "invalid.name"
    generated = stable_tool_name(original)
    state = compile_tool_state({"tools": [_tool(original), _tool(generated)]})
    assert state.name_map[original] == generated
    assert state.name_map[generated] != generated
    assert len(set(state.name_map.values())) == 2


def test_mapping_is_independent_of_order_subset_and_schema() -> None:
    names = ["mcp__" + "server_" * 12 + "__lookup", "other.lookup", "short"]
    baseline = compile_tool_state({"tools": [_tool(name) for name in names]})
    reordered = compile_tool_state({"tools": [_tool(name, revision=True) for name in reversed(names)]})
    assert baseline.name_map == reordered.name_map
    for name in names:
        subset = compile_tool_state({"tools": [_tool(name)]})
        assert subset.name_map[name] == baseline.name_map[name]


def test_actual_collisions_fail_instead_of_suffixing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_protocol, "stable_tool_name", lambda name: "collision")
    with pytest.raises(ToolProtocolError, match=r"tools\[1\].name: tool name collision"):
        compile_tool_state({"tools": [_tool("one"), _tool("two")]})


def test_all_historical_names_are_mapped_without_harvesting_data() -> None:
    request = {
        "tools": [_tool("root")],
        "tool_choice": {"type": "tool", "name": "root"},
        "messages": [
            _user(),
            _system(_addition(_tool("inline.invalid"))),
            _assistant([_call("unknown.historical", "old")]),
            _user([{"type": "tool_result", "tool_use_id": "old", "content": [_reference("reference.only"),
                    {"type": "text", "text": '{"name":"not_a_tool"}'}]}]),
            _system(_control("inline.invalid", remove=True)),
            _assistant([{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"name": "argument_name"}}]),
        ],
    }
    request["messages"][2]["content"][0]["input"] = {"name": "argument_name", "tool_name": "input_reference"}
    state = compile_tool_state(request)
    assert set(state.name_map) == {"root", "inline.invalid", "unknown.historical", "reference.only", "web_search"}
    assert state.active_name_map == {"root": "root"}


def test_identical_root_definitions_are_deduplicated() -> None:
    definition = _tool("lookup")
    state = compile_tool_state({"tools": [definition, deepcopy(definition)]})
    assert state.tools == [definition]
    assert state.name_map == {"lookup": "lookup"}


def test_conflicting_root_definitions_fail() -> None:
    with pytest.raises(ToolProtocolError, match=r"tools\[1\]: conflicting root"):
        compile_tool_state({"tools": [_tool("lookup"), _tool("lookup", revision=True)]})


def _placeholder() -> dict:
    return {
        "name": "DeferredToolPlaceholder",
        "description": "Reserved placeholder that keeps deferred tool loading active; never call this tool.",
        "input_schema": {"type": "object", "properties": {}},
        "defer_loading": True,
    }


def test_only_exact_client_placeholder_is_removed() -> None:
    state = compile_tool_state({"tools": [_tool("ToolSearch"), _placeholder()]})
    assert [tool["name"] for tool in state.tools] == ["ToolSearch"]
    assert "DeferredToolPlaceholder" in state.name_map
    assert "DeferredToolPlaceholder" not in state.active_name_map


@pytest.mark.parametrize(
    "update",
    [
        {"description": "A real user tool"},
        {"defer_loading": False},
        {"input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
        {"input_schema": {"type": "object"}},
        {"type": "web_search_20260209"},
    ],
)
def test_same_name_genuine_tools_are_not_removed(update: dict) -> None:
    definition = {**_placeholder(), **update}
    assert compile_tool_state({"tools": [definition]}).tools == [definition]


def test_forcing_stripped_placeholder_is_rejected() -> None:
    with pytest.raises(ToolProtocolError, match="forced tool.*is not active"):
        compile_tool_state({"tools": [_placeholder()], "tool_choice": {"type": "tool", "name": "DeferredToolPlaceholder"}})


def test_classic_root_definitions_remain_callable_even_if_deferred() -> None:
    definitions = [_tool("ToolSearch"), _tool("loaded", deferred=True)]
    state = compile_tool_state({"tools": definitions, "messages": [_user()]})
    assert state.tools == definitions
    assert set(state.active_name_map) == {"ToolSearch", "loaded"}


def test_classic_references_do_not_invent_omitted_catalog() -> None:
    request = {"tools": [_tool("ToolSearch")], "messages": _search_history("not_in_catalog") + [_assistant()]}
    state = compile_tool_state(request)
    assert [tool["name"] for tool in state.tools] == ["ToolSearch"]
    marker = json.loads(state.references[1, 0, 0])
    assert marker == {"type": "tool_reference", "tool_name": "not_in_catalog", "call_name": "not_in_catalog", "discovery": True}


def test_with_controls_only_nondeferred_root_definitions_start_active() -> None:
    state = compile_tool_state(_controlled_request("hidden"))
    assert [tool["name"] for tool in state.tools] == ["ToolSearch"]
    assert set(state.name_map) == {"ToolSearch", "hidden"}


def test_explicit_reference_addition_activates_deferred_definition() -> None:
    request = _controlled_request("hidden")
    request["messages"][1] = _system(_control("hidden"))
    assert set(compile_tool_state(request).active_name_map) == {"ToolSearch", "hidden"}


def test_successful_corresponding_toolsearch_result_activates_known_tool() -> None:
    request = _controlled_request("hidden")
    request["messages"] += _search_history("hidden")
    state = compile_tool_state(request)
    assert set(state.active_name_map) == {"ToolSearch", "hidden"}
    assert json.loads(state.references[3, 0, 0])["discovery"] is True


@pytest.mark.parametrize("case", ["error", "different_call", "different_id", "orphan", "assistant_result"])
def test_references_outside_successful_actual_toolsearch_are_inert(case: str) -> None:
    request = _controlled_request("hidden")
    call = _call()
    result = _result("hidden")
    result_message = _user([result])
    if case == "error":
        result["is_error"] = True
    elif case == "different_call":
        call["name"] = "ordinary"
    elif case == "different_id":
        result["tool_use_id"] = "other"
    elif case == "assistant_result":
        result_message = _assistant([result])
    request["messages"].append(_assistant() if case == "orphan" else _assistant([call]))
    request["messages"].append(result_message)
    state = compile_tool_state(request)
    assert "hidden" not in state.active_name_map
    assert all(json.loads(marker)["discovery"] is False for marker in state.references.values())


def test_search_text_and_input_are_not_interpreted_as_discovery() -> None:
    request = _controlled_request("hidden")
    request["messages"] += [_assistant([{"type": "text", "text": "ToolSearch"}, _call("ordinary")]),
                            _user([_result("hidden")])]
    request["messages"][2]["content"][1]["input"] = {"name": "ToolSearch", "tool_name": "hidden"}
    assert "hidden" not in compile_tool_state(request).active_name_map


def test_reused_call_ids_bind_to_the_corresponding_call_not_old_search() -> None:
    request = _controlled_request("hidden", "other")
    request["messages"] += _search_history("hidden")
    request["messages"] += [_assistant([_call("ordinary")]), _user([_result("other")])]
    state = compile_tool_state(request)
    assert "hidden" in state.active_name_map
    assert "other" not in state.active_name_map
    assert json.loads(state.references[5, 0, 0])["discovery"] is False


def test_search_never_resurrects_a_tombstoned_tool() -> None:
    request = _controlled_request("hidden")
    request["messages"][1] = _system(_control("hidden", remove=True))
    request["messages"] += _search_history("hidden")
    state = compile_tool_state(request)
    assert "hidden" not in state.active_name_map
    marker = json.loads(state.references[3, 0, 0])
    assert marker["discovery"] is True
    assert "active" not in marker


@pytest.mark.parametrize("use_definition", [False, True])
def test_explicit_readdition_clears_tombstone_even_with_identical_definition(use_definition: bool) -> None:
    request = _controlled_request("hidden")
    request["messages"][1] = _system(_control("hidden", remove=True))
    request["messages"] += _search_history("hidden")
    addition = _addition(_tool("hidden", deferred=True)) if use_definition else _control("hidden")
    request["messages"].append(_system(addition))
    assert "hidden" in compile_tool_state(request).active_name_map


def test_historical_marker_does_not_retcon_discovery_after_removal() -> None:
    request = _controlled_request("hidden")
    request["messages"] += _search_history("hidden")
    initial = compile_tool_state(request)
    request["messages"].append(_system(_control("hidden", remove=True)))
    final = compile_tool_state(request)
    assert final.references == initial.references
    assert json.loads(final.references[3, 0, 0])["discovery"] is True
    assert "hidden" not in final.active_name_map


@pytest.mark.parametrize("controls", [False, True])
def test_unknown_current_search_reference_fails_before_transport(controls: bool) -> None:
    request = _controlled_request() if controls else {"tools": [_tool("ToolSearch")], "messages": []}
    request["messages"] += _search_history("unknown")
    with pytest.raises(ToolProtocolError, match=r"content\[0\].content\[0\]: current ToolSearch reference has no known definition"):
        compile_tool_state(request)


def test_newer_assistant_message_makes_unknown_search_reference_historical() -> None:
    request = _controlled_request()
    request["messages"] += _search_history("old_missing") + [_assistant(), _user("next")]
    state = compile_tool_state(request)
    assert "old_missing" in state.name_map
    assert "old_missing" not in state.active_name_map
    assert "known" not in json.loads(state.references[3, 0, 0])


def test_only_new_search_missing_definition_is_rejected() -> None:
    request = _controlled_request()
    request["messages"] += _search_history("old_missing") + _search_history("new_missing")
    with pytest.raises(ToolProtocolError, match="new_missing") as error:
        compile_tool_state(request)
    assert "old_missing" not in str(error.value)


@pytest.mark.parametrize("known", [False, True])
def test_search_immediately_followed_by_removal_is_allowed_and_inactive(known: bool) -> None:
    request = _controlled_request("found") if known else _controlled_request()
    request["messages"] += _search_history("found") + [_system(_control("found", remove=True))]
    assert "found" not in compile_tool_state(request).active_name_map


def test_unknown_search_then_inline_definition_is_resolved_explicitly() -> None:
    request = _controlled_request()
    request["messages"] += _search_history("new") + [_system(_addition(_tool("new")))]
    state = compile_tool_state(request)
    assert "new" in state.active_name_map
    assert "known" not in json.loads(state.references[3, 0, 0])


def test_captured_frozen_root_timeline_redefinition_remove_and_readd() -> None:
    name = "mcp__documents__" + "late_lookup_" * 6
    definition_a = _tool(name)
    definition_b = _tool(name, revision=True)
    request = {
        "tools": [_tool("ToolSearch"), _tool("root_hidden", deferred=True)],
        "messages": [
            _user("make the late tool available"),
            _system(_addition(definition_a)),
            _assistant([_call(name, "call_a")]),
            _user([{"type": "tool_result", "tool_use_id": "call_a", "content": "original"}]),
            _system(_addition(definition_b)),
            _assistant([_call(name, "call_b")]),
            _user([{"type": "tool_result", "tool_use_id": "call_b", "content": "revised"}]),
            _system(_control(name, remove=True)),
            _assistant("tool removed"),
            _user("add the revised tool again"),
            _system(_addition(definition_b)),
        ],
    }
    request["messages"][5]["content"][0]["input"] = {"query": "updated", "revision": 2}
    original = deepcopy(request)
    state = compile_tool_state(request, f"{TOOL_CHANGES_BETA},{INLINE_TOOLS_BETA}")
    assert request == original
    assert state.tools == [_tool("ToolSearch"), definition_b]
    assert set(state.name_map) == {"ToolSearch", "root_hidden", name}
    assert state.active_name_map[name] == stable_tool_name(name)
    assert json.loads(state.markers[1, 0])["tool"]["definition"] == definition_a
    assert json.loads(state.markers[4, 0])["tool"]["definition"] == definition_b
    assert json.loads(state.markers[7, 0])["type"] == "tool_removal"
    assert json.loads(state.markers[10, 0])["tool"]["definition"] == definition_b
    assert len(state.markers) == 4
    removed = compile_tool_state({**request, "messages": request["messages"][:8]})
    assert name not in removed.active_name_map
    assert state.markers[1, 0] == removed.markers[1, 0]
    state.tools[1]["input_schema"]["properties"]["query"]["type"] = "number"
    assert request == original


def test_identical_inline_additions_keep_both_events_but_one_callable_schema() -> None:
    definition = _tool("same")
    state = compile_tool_state({"messages": [_user(), _system(_addition(definition), _addition(deepcopy(definition)))]})
    assert state.tools == [definition]
    assert len(state.markers) == 2


@pytest.mark.parametrize("initial_type,new_type", [(None, "web_search_20260209"), ("web_search_20250305", "web_search_20260209"), ("bash_20250124", None)])
def test_same_name_type_changes_are_rejected_contextually(initial_type: str | None, new_type: str | None) -> None:
    initial = _tool("same")
    replacement = _tool("same", revision=True)
    if initial_type is not None:
        initial["type"] = initial_type
    if new_type is not None:
        replacement["type"] = new_type
    with pytest.raises(ToolProtocolError, match=r"messages\[1\].content\[0\]: cannot change tool type"):
        compile_tool_state({"tools": [initial], "messages": [_user(), _system(_addition(replacement))]})


def test_explicit_custom_type_and_implicit_custom_type_are_equivalent() -> None:
    replacement = {**_tool("same", revision=True), "type": "custom"}
    assert compile_tool_state({"tools": [_tool("same")], "messages": [_user(), _system(_addition(replacement))]}).tools == [replacement]


@pytest.mark.parametrize("tool_type", ["web_search_20250305", "web_search_20260209"])
def test_supported_web_search_tools_preserve_configuration(tool_type: str) -> None:
    definition = {"type": tool_type, "name": "web_search", "allowed_domains": ["example.com"], "max_uses": 2}
    state = compile_tool_state({"messages": [_user(), _system(_addition(definition))]})
    assert state.tools == [definition]


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("tool_type", ["tool_search_tool_regex_20251119", "tool_search_tool_bm25_20251119"])
def test_anthropic_hosted_search_is_explicitly_unsupported(inline: bool, tool_type: str) -> None:
    definition = {"type": tool_type, "name": "search"}
    request = {"messages": [_user(), _system(_addition(definition))]} if inline else {"tools": [definition]}
    with pytest.raises(ToolProtocolError, match="Anthropic hosted tool_search.*original native route"):
        compile_tool_state(request)


def test_root_builtin_missing_schema_keeps_legacy_normalization_compatibility() -> None:
    definitions = [{"type": "bash_20250124", "name": "bash"}, {"name": "old_custom"}]
    assert compile_tool_state({"tools": definitions}).tools == definitions


def test_controls_allow_consecutive_system_sections_after_user_before_assistant() -> None:
    request = {"tools": [_tool("root")], "messages": [_user(), _system({"type": "text", "text": "operator"}),
               _system(_addition(_tool("new"))), _system(_control("root", remove=True)), _assistant()]}
    state = compile_tool_state(request)
    assert state.tools == [_tool("new")]


@pytest.mark.parametrize("messages", [
    [_system(_addition(_tool("new")))],
    [_assistant(), _system(_addition(_tool("new")))],
    [_user(), _system(_addition(_tool("new"))), _user()],
    [_user([_addition(_tool("new"))])],
    [_assistant([_addition(_tool("new"))])],
])
def test_illegal_control_locations_fail(messages: list) -> None:
    with pytest.raises(ToolProtocolError, match=r"messages\[\d+\].*tool control"):
        compile_tool_state({"messages": messages})


def test_controls_in_root_system_are_not_protocol_events() -> None:
    with pytest.raises(ToolProtocolError, match=r"system\[0\]"):
        compile_tool_state({"system": [_addition(_tool("new"))]})


def test_partial_parallel_results_cannot_be_split_by_controls() -> None:
    request = {"tools": [_tool("root")], "messages": [_assistant([_call("root", "one"), _call("root", "two")]),
               _user([_result(call_id="one")]), _system(_control("root", remove=True)),
               _assistant(), _user([_result(call_id="two")])]}
    with pytest.raises(ToolProtocolError, match="cannot split a tool_use/tool_result pair"):
        compile_tool_state(request)


def test_all_parallel_results_then_controls_are_legal() -> None:
    request = {"tools": [_tool("root")], "messages": [_assistant([_call("root", "one"), _call("root", "two")]),
               _user([_result(call_id="one"), _result(call_id="two")]), _system(_control("root", remove=True))]}
    assert compile_tool_state(request).tools == []


@pytest.mark.parametrize("header", [None, INLINE_TOOLS_BETA, f"unrelated, {INLINE_TOOLS_BETA}"])
def test_inline_definitions_accept_optional_or_exact_beta(header: str | None) -> None:
    state = compile_tool_state({"messages": [_user(), _system(_addition(_tool("new")))]}, header)
    assert state.tools == [_tool("new")]


@pytest.mark.parametrize("header", ["", TOOL_CHANGES_BETA, "inline-tools-2026-09-15-extra"])
def test_inline_definitions_require_inline_beta_if_header_supplied(header: str) -> None:
    with pytest.raises(ToolProtocolError, match=INLINE_TOOLS_BETA):
        compile_tool_state({"messages": [_user(), _system(_addition(_tool("new")))]}, header)


@pytest.mark.parametrize("header", [None, TOOL_CHANGES_BETA, INLINE_TOOLS_BETA])
def test_reference_controls_accept_relevant_beta(header: str | None) -> None:
    assert compile_tool_state(_controlled_request(), header).tools == [_tool("ToolSearch")]


def test_reference_controls_reject_unrelated_beta() -> None:
    with pytest.raises(ToolProtocolError, match="requires beta"):
        compile_tool_state(_controlled_request(), "unrelated")


@pytest.mark.parametrize("tool", [None, {}, {"type": "tool_reference"}, {"type": "tool_reference", "tool_name": "root"},
    {"type": "tool_reference", "name": "root", "tool_name": "root"}, {"type": "tool_reference", "name": 1},
    {"type": "tool_reference", "name": ""}, {"type": "tool_definition"}])
def test_malformed_control_reference_or_definition_fails_at_location(tool: object) -> None:
    with pytest.raises(ToolProtocolError, match=r"messages\[1\].content\[0\].tool"):
        compile_tool_state({"tools": [_tool("root")], "messages": [_user(), _system({"type": "tool_addition", "tool": tool})]})


@pytest.mark.parametrize("reference", [
    {"type": "tool_reference"}, {"type": "tool_reference", "name": "hidden"},
    {"type": "tool_reference", "tool_name": "hidden", "name": "hidden"},
    {"type": "tool_reference", "tool_name": None}, {"type": "tool_reference", "tool_name": ""},
])
def test_malformed_result_reference_fails_at_nested_location(reference: dict) -> None:
    request = {"messages": [_user([{"type": "tool_result", "tool_use_id": "ordinary", "content": [reference]}])]}
    with pytest.raises(ToolProtocolError, match=r"messages\[0\].content\[0\].content\[0\]"):
        compile_tool_state(request)


def test_inline_custom_definition_requires_valid_required_fields() -> None:
    for definition in [{"name": "new"}, {"name": "new", "input_schema": []},
                       {"name": "new", "input_schema": {"type": "array"}},
                       {"name": "new", "input_schema": {"type": "object", "properties": []}},
                       {**_tool("new"), "description": None}, {**_tool("new"), "defer_loading": "true"}]:
        with pytest.raises(ToolProtocolError, match=r"messages\[1\].content\[0\].tool.definition"):
            compile_tool_state({"messages": [_user(), _system(_addition(definition))]})


def test_removal_cannot_contain_a_definition() -> None:
    with pytest.raises(ToolProtocolError, match="removal requires a tool_reference"):
        compile_tool_state({"messages": [_user(), _system({"type": "tool_removal", "tool": _addition(_tool("new"))["tool"]})]})


def test_addition_reference_requires_a_known_definition() -> None:
    with pytest.raises(ToolProtocolError, match="no known definition"):
        compile_tool_state({"messages": [_user(), _system(_control("unknown"))]})


def test_mixed_results_preserve_reference_positions_and_do_not_parse_nested_controls() -> None:
    result = _result("one")
    result["content"] = [{"type": "text", "text": "before"}, _reference("one"),
                         {"type": "image", "source": {"type": "url", "url": "https://example.com/image"}},
                         _addition(_tool("not_an_event")), _reference("two"),
                         {"type": "text", "text": '{"type":"tool_removal","tool":{"name":"one"}}'}]
    result["is_error"] = True
    request = {"messages": [_assistant([_call()]), _user([result])]}
    original = deepcopy(request)
    state = compile_tool_state(request)
    assert set(state.references) == {(1, 0, 1), (1, 0, 4)}
    assert state.markers == {}
    assert state.tools == []
    assert set(state.name_map) == {"ToolSearch", "one", "two"}
    assert request == original
    assert request["messages"][1]["content"][0]["is_error"] is True


@pytest.mark.parametrize("choice", [{"type": "tool", "name": "unknown"}, {"type": "tool", "name": "hidden"}])
def test_forced_choice_must_refer_to_final_active_set(choice: dict) -> None:
    request = _controlled_request("hidden")
    request["tool_choice"] = choice
    with pytest.raises(ToolProtocolError, match="tool_choice.name: forced tool.*is not active"):
        compile_tool_state(request)


def test_forced_choice_uses_final_redefined_tool() -> None:
    definition = _tool("new.invalid", revision=True)
    request = {"tool_choice": {"type": "tool", "name": "new.invalid"}, "messages": [_user(), _system(_addition(definition))]}
    state = compile_tool_state(request)
    assert state.tools == [definition]
    assert state.active_name_map == {"new.invalid": stable_tool_name("new.invalid")}


@pytest.mark.parametrize("choice", [{"type": "auto"}, {"type": "any"}, {"type": "none"}, None])
def test_nonforced_choices_do_not_need_active_tools(choice: dict | None) -> None:
    assert compile_tool_state({"tool_choice": choice}).tools == []


@pytest.mark.parametrize("body,location", [
    ([], "request"), ({"messages": None}, "messages"), ({"tools": {}}, "tools"),
    ({"tools": [None]}, r"tools\[0\]"), ({"tools": [{}]}, r"tools\[0\].name"),
    ({"messages": [None]}, r"messages\[0\]"), ({"messages": [_user(None)]}, r"messages\[0\].content"),
    ({"messages": [_user([None])]}, r"messages\[0\].content\[0\]"),
    ({"tool_choice": {"type": "tool"}}, "tool_choice.name"),
    ({"messages": [_assistant([{**_call(), "name": None}])]}, r"messages\[0\].content\[0\].name"),
    ({"messages": [_assistant([{**_call(), "id": None}])]}, r"messages\[0\].content\[0\].id"),
    ({"messages": [_user([{"type": "tool_result", "content": [_reference("name")]}])]}, "tool_use_id"),
])
def test_malformed_required_structures_have_contextual_errors(body: object, location: str) -> None:
    with pytest.raises(ToolProtocolError, match=location):
        compile_tool_state(body)


def test_empty_request_compiles_without_session_state() -> None:
    assert compile_tool_state({}) == ToolState([], {}, {}, {}, {})


def test_unpaired_unicode_names_fail_contextually() -> None:
    with pytest.raises(ToolProtocolError, match=r"tools\[0\].name: tool name must contain valid Unicode"):
        compile_tool_state({"tools": [_tool("\ud800")]})


def test_reference_marker_is_not_retconned_when_classic_catalog_grows() -> None:
    request = {"tools": [_tool("ToolSearch")], "messages": _search_history("later.invalid") + [_assistant()]}
    original = compile_tool_state(request)
    request["tools"].append(_tool("later.invalid", deferred=True))
    expanded = compile_tool_state(request)
    assert original.references == expanded.references
    assert "later.invalid" not in original.active_name_map
    assert "later.invalid" in expanded.active_name_map
    marker = json.loads(original.references[1, 0, 0])
    assert marker["call_name"] == stable_tool_name("later.invalid")
    assert set(marker) == {"type", "tool_name", "call_name", "discovery"}


def test_control_marker_uses_call_name_and_preserves_full_original_event() -> None:
    definition = _tool("inline.invalid")
    definition["input_schema"]["properties"]["query"]["examples"] = ["ToolSearch", "other_tool"]
    block = _addition(definition)
    state = compile_tool_state({"messages": [_user(), _system(block)]})
    assert json.loads(state.markers[1, 0]) == {**block, "call_name": stable_tool_name("inline.invalid")}
    assert set(state.name_map) == {"inline.invalid"}


@pytest.mark.parametrize("is_error", [False, True])
def test_missing_current_target_is_rejected_only_for_successful_search(is_error: bool) -> None:
    request = {"messages": _search_history("missing", is_error=is_error)}
    if not is_error:
        with pytest.raises(ToolProtocolError, match="current ToolSearch reference"):
            compile_tool_state(request)
        return
    state = compile_tool_state(request)
    assert state.tools == []
    assert state.name_map == {"ToolSearch": "ToolSearch", "missing": "missing"}
    assert json.loads(state.references[1, 0, 0])["discovery"] is False


def test_omitted_is_error_is_a_successful_result() -> None:
    request = {"tools": [_tool("found")], "messages": _search_history("found")}
    del request["messages"][1]["content"][0]["is_error"]
    assert json.loads(compile_tool_state(request).references[1, 0, 0])["discovery"] is True


@pytest.mark.parametrize("tool_type", ["web_fetch_20260209", "code_execution_20260521", "advisor_20260301", "mcp_toolset", "unknown_type"])
@pytest.mark.parametrize("inline", [False, True])
def test_other_unsupported_hosted_types_do_not_become_empty_functions(tool_type: str, inline: bool) -> None:
    definition = {"type": tool_type, "name": "hosted"}
    request = {"messages": [_user(), _system(_addition(definition))]} if inline else {"tools": [definition]}
    with pytest.raises(ToolProtocolError, match="unsupported hosted or unknown tool type.*original native route"):
        compile_tool_state(request)


@pytest.mark.parametrize("tool_type", ["bash_20250124", "text_editor_20250728", "computer_20251124", "memory_20250818"])
def test_known_client_tool_types_keep_schema_less_root_compatibility(tool_type: str) -> None:
    definition = {"type": tool_type, "name": "client"}
    assert compile_tool_state({"tools": [definition]}).tools == [definition]


def test_placeholder_and_conflicting_real_root_definition_still_conflict() -> None:
    with pytest.raises(ToolProtocolError, match="conflicting root definitions"):
        compile_tool_state({"tools": [_placeholder(), _tool("DeferredToolPlaceholder")]})


def test_ambiguous_outstanding_call_ids_preserve_history_without_discovery() -> None:
    request = {"messages": [_assistant([_call("ordinary"), _call("ToolSearch")]), _user([_result("missing")])]}
    original = deepcopy(request)
    state = compile_tool_state(request)
    assert state.tools == []
    assert json.loads(state.references[1, 0, 0])["discovery"] is False
    assert request == original


def test_malformed_search_error_flag_is_rejected_contextually() -> None:
    request = {"messages": _search_history("missing")}
    request["messages"][1]["content"][0]["is_error"] = "false"
    with pytest.raises(ToolProtocolError, match=r"messages\[1\].content\[0\].is_error"):
        compile_tool_state(request)


def test_type_change_stays_unsupported_after_removal() -> None:
    request = {"tools": [_tool("same")], "messages": [_user(), _system(_control("same", remove=True)),
               _assistant(), _user(), _system(_addition({"type": "web_search_20260209", "name": "same"}))]}
    with pytest.raises(ToolProtocolError, match="cannot change tool type"):
        compile_tool_state(request)


@pytest.mark.parametrize("result_type", ["web_search_tool_result", "bash_code_execution_tool_result"])
def test_controls_after_completed_assistant_server_result_are_legal(result_type: str) -> None:
    server_response = _assistant([
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
        {"type": result_type, "tool_use_id": "srvtoolu_1", "content": []},
    ])
    server_response["stop_reason"] = "end_turn"
    request = {"messages": [server_response, _system({"type": "text", "text": "operator"}),
                           _system(_addition(_tool("new"))), _assistant()]}
    assert compile_tool_state(request).tools == [_tool("new")]
    del server_response["stop_reason"]
    assert compile_tool_state(request).tools == [_tool("new")]


def test_controls_cannot_follow_paused_server_result() -> None:
    server_response = _assistant([{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}])
    server_response["stop_reason"] = "pause_turn"
    with pytest.raises(ToolProtocolError, match="cannot follow a paused server result"):
        compile_tool_state({"messages": [server_response, _system(_addition(_tool("new")))]})


def test_server_result_does_not_allow_splitting_an_outstanding_client_pair() -> None:
    response = _assistant([_call("client", "client_id"),
                           {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}])
    with pytest.raises(ToolProtocolError, match="cannot split a tool_use/tool_result pair"):
        compile_tool_state({"messages": [response, _system(_addition(_tool("new")))]})


@pytest.mark.parametrize("message", [{"role": [], "content": "text"}, _user([{"type": []}]), _user([{}])])
def test_nonstring_roles_and_block_types_fail_as_protocol_errors(message: dict) -> None:
    with pytest.raises(ToolProtocolError, match=r"messages\[0\]"):
        compile_tool_state({"messages": [message]})


def test_nonjson_inline_schema_values_fail_contextually() -> None:
    definition = _tool("new")
    definition["input_schema"]["properties"]["query"]["default"] = float("nan")
    with pytest.raises(ToolProtocolError, match="tool protocol block must be JSON"):
        compile_tool_state({"messages": [_user(), _system(_addition(definition))]})


def test_ambiguous_search_call_ids_do_not_activate_known_deferred_tools() -> None:
    request = _controlled_request("hidden")
    request["messages"] += [_assistant([_call(), _call()]), _user([_result("hidden")])]
    assert "hidden" not in compile_tool_state(request).active_name_map


def test_ambiguous_outstanding_calls_still_cannot_be_split_by_controls() -> None:
    request = {"messages": [_assistant([_call(), _call()]), _user("partial"),
                            _system(_addition(_tool("new")))]}
    with pytest.raises(ToolProtocolError, match="cannot split a tool_use/tool_result pair"):
        compile_tool_state(request)


@pytest.mark.parametrize("call_count, result_count", [(2, 1), (3, 1), (3, 2)])
def test_partial_duplicate_call_results_cannot_be_split_by_controls(call_count: int, result_count: int) -> None:
    request = {"messages": [
        _assistant([_call("ToolSearch", "dup"), *[_call("ordinary", "dup") for _ in range(call_count - 1)]]),
        _user([_result(call_id="dup") for _ in range(result_count)]),
        _system(_addition(_tool("new"))),
    ]}
    with pytest.raises(ToolProtocolError, match=r"messages\[2\]: tool controls cannot split a tool_use/tool_result pair"):
        compile_tool_state(request)


@pytest.mark.parametrize("call_count", [2, 3])
@pytest.mark.parametrize("separate_messages", [False, True])
def test_all_duplicate_call_results_allow_controls_without_discovery(call_count: int, separate_messages: bool) -> None:
    results = [_result("hidden", "missing", call_id="dup") for _ in range(call_count)]
    result_messages = [_user([result]) for result in results] if separate_messages else [_user(results)]
    request = {"tools": [_tool("ToolSearch"), _tool("hidden", deferred=True)], "messages": [
        _assistant([_call("ToolSearch", "dup"), *[_call("ordinary", "dup") for _ in range(call_count - 1)]]),
        *result_messages,
        _system(_addition(_tool("new"))),
    ]}
    original = deepcopy(request)
    state = compile_tool_state(request)
    assert state.tools == [_tool("ToolSearch"), _tool("new")]
    assert len(state.references) == call_count * 2
    assert all(json.loads(reference)["discovery"] is False for reference in state.references.values())
    assert request == original


@pytest.mark.parametrize("call_count", [1, 2])
def test_unknown_result_ids_do_not_resolve_outstanding_calls(call_count: int) -> None:
    request = {"messages": [
        _assistant([_call("ToolSearch", "dup") for _ in range(call_count)]),
        _user([_result(call_id="unknown")]),
        _system(_addition(_tool("new"))),
    ]}
    with pytest.raises(ToolProtocolError, match="cannot split a tool_use/tool_result pair"):
        compile_tool_state(request)


@pytest.mark.parametrize("call_count", [1, 2])
def test_extra_results_and_unknown_ids_do_not_discover_tools(call_count: int) -> None:
    request = {"tools": [_tool("ToolSearch"), _tool("hidden", deferred=True)], "messages": [
        _assistant([_call("ToolSearch", "dup") for _ in range(call_count)]),
        _user([*[_result(call_id="dup") for _ in range(call_count)],
               _result("hidden", "missing", call_id="dup"), _result("hidden", "missing", call_id="unknown")]),
        _system(_addition(_tool("new"))),
    ]}
    state = compile_tool_state(request)
    assert state.tools == [_tool("ToolSearch"), _tool("new")]
    assert len(state.references) == 4
    assert all(json.loads(reference)["discovery"] is False for reference in state.references.values())


@pytest.mark.parametrize("result_type", ["web_search_tool_result", "bash_code_execution_tool_result"])
def test_server_results_do_not_resolve_remaining_duplicate_client_calls(result_type: str) -> None:
    request = {"messages": [
        _assistant([_call("ToolSearch", "dup"), _call("ordinary", "dup")]),
        _user([_result(call_id="dup")]),
        _assistant([{"type": result_type, "tool_use_id": "dup", "content": []}]),
        _system(_addition(_tool("new"))),
    ]}
    with pytest.raises(ToolProtocolError, match=r"messages\[3\]: tool controls cannot split a tool_use/tool_result pair"):
        compile_tool_state(request)


def test_root_definition_conflict_comparison_preserves_json_boolean_number_types() -> None:
    first = _tool('typed')
    second = deepcopy(first)
    first['input_schema']['properties'] = {'value': {'enum': [True]}}
    second['input_schema']['properties'] = {'value': {'enum': [1]}}
    with pytest.raises(ToolProtocolError, match='conflicting root definitions'):
        compile_tool_state({'tools': [first, second]})


def test_malformed_root_toolsearch_description_is_contextual_protocol_error() -> None:
    with pytest.raises(ToolProtocolError, match=r'tools\[0\].description: must be a string'):
        compile_tool_state({'tools': [{'name': 'ToolSearch', 'description': 1, 'input_schema': {'type': 'object'}}]})


@pytest.mark.parametrize('malformed_type', [None, True, 1, []])
def test_malformed_preceding_server_result_has_contextual_control_error(malformed_type: object) -> None:
    request = {'messages': [
        {'role': 'assistant', 'content': [{'type': malformed_type}]},
        _system(_addition(_tool('lookup'))),
    ]}
    with pytest.raises(ToolProtocolError, match=r'messages\[0\].content\[0\].type: must be a nonempty string'):
        compile_tool_state(request)
