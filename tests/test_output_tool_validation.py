"""Executable mapped calls are withheld until their current schema validates."""

import json
import socket
import urllib.request

import pytest

from claudex.translate.claude_to_codex import TranslationError
from claudex.translate.codex_to_claude import CodexToClaudeStreamTranslator, assemble_claude_message
from claudex.translate.thought_signature import decode_call_signature_carrier
from claudex.translate.tool_validation import compile_function_validators, validate_function_arguments


_SCHEMA = {
    "type": "object",
    "properties": {"mode": {"type": "string", "enum": ["B"]}},
    "required": ["mode"],
    "additionalProperties": False,
}


def _translator(**kwargs):
    return CodexToClaudeStreamTranslator(
        {"model": "mapped", "tools": [{"name": "lookup", "input_schema": _SCHEMA}]},
        tool_name_map={"lookup": "lookup_alias"},
        function_schemas={"lookup_alias": _SCHEMA},
        **kwargs,
    )


def _item(event_type="added", *, output_index=0, **fields):
    return {
        "type": f"response.output_item.{event_type}",
        "output_index": output_index,
        "item": {"type": "function_call", "id": "item_1", "call_id": "call_1",
                 "name": "lookup_alias", **fields},
    }


def _arguments(event_type="delta", *, output_index=0, **fields):
    return {"type": f"response.function_call_arguments.{event_type}",
            "output_index": output_index, **fields}


def _terminal(*items, event_type="completed"):
    return {"type": f"response.{event_type}", "response": {"output": list(items)}}


def _tool_blocks(events):
    return [payload["content_block"] for event_name, payload in events
            if event_name == "content_block_start"
            and payload["content_block"]["type"] == "tool_use"]


def _carriers(events):
    return [decode_call_signature_carrier(payload["delta"]["signature"])
            for event_name, payload in events
            if event_name == "content_block_delta"
            and payload["delta"]["type"] == "signature_delta"]


def test_valid_call_is_buffered_until_item_done():
    translator = _translator()
    for event in [_item(), _arguments(delta='{"mo'), _arguments(delta='de":"B"}'),
                  _arguments("done", arguments='{"mode":"B"}')]:
        assert translator.translate_event(event) == []
    events = translator.translate_event(_item("done"))
    assert _tool_blocks(events) == [{"type": "tool_use", "id": "call_1",
                                    "name": "lookup", "input": {}}]
    assert [name for name, _ in events] == ["content_block_start", "content_block_delta",
                                          "content_block_stop"]
    assert events[1][1]["delta"]["partial_json"] == '{"mode":"B"}'
    assert not _tool_blocks(translator.translate_event(_terminal()))


def test_latest_declared_enum_blocks_stale_arguments_before_executable_start():
    translator = _translator()
    emitted = translator.translate_event(_item())
    emitted += translator.translate_event(_arguments(delta='{"mode":"A"}'))
    with pytest.raises(TranslationError, match="declared schema"):
        translator.translate_event(_item("done"))
    assert emitted == []


def test_arguments_done_without_delta_waits_for_item_completion():
    translator = _translator()
    assert translator.translate_event(_item()) == []
    assert translator.translate_event(_arguments("done", arguments='{"mode":"B"}')) == []
    events = translator.translate_event(_item("done"))
    assert len(_tool_blocks(events)) == 1
    assert events[1][1]["delta"]["partial_json"] == '{"mode":"B"}'


def test_nonstream_aggregation_cannot_turn_bad_call_into_executable_empty_input():
    translator = _translator()
    events = translator.translate_event({"type": "response.created", "response": {"id": "r1"}})
    events.extend(translator.translate_event(_item()))
    events.extend(translator.translate_event(_arguments(delta='{"mode":"A"}')))
    with pytest.raises(TranslationError):
        events.extend(translator.translate_event(_item("done")))
    message = assemble_claude_message(events)
    assert message is not None
    assert message["content"] == []


@pytest.mark.parametrize("arguments", [
    '{"mode":', '{"mode":"B"} trailing', '{"mode":42}', '{}',
    '{"mode":"B","extra":1}', '{"mode":"A"}', '["B"]', 'null',
    '{"mode":NaN}', '{"mode":Infinity}', '', None,
])
def test_invalid_arguments_are_rejected(arguments):
    validators = compile_function_validators({"lookup_alias": _SCHEMA})
    with pytest.raises(TranslationError):
        validate_function_arguments(validators, "lookup_alias", arguments)


@pytest.mark.parametrize("schema,arguments", [
    ({"type": "object", "properties": {"value": {"type": "number"}}}, '{"value":true}'),
    ({"type": "object", "properties": {"value": {"type": "boolean"}}}, '{"value":1}'),
    ({"type": "object", "properties": {"value": {"enum": [1]}}}, '{"value":true}'),
    ({"type": "object", "properties": {"value": {"enum": [False]}}}, '{"value":0}'),
])
def test_booleans_and_numbers_are_distinct(schema, arguments):
    with pytest.raises(TranslationError):
        validate_function_arguments(compile_function_validators({"f": schema}), "f", arguments)


def test_internal_defs_refs_and_composition_validate():
    schema = {"type": "object", "$defs": {"current": {"enum": ["B"]}},
              "properties": {"mode": {"allOf": [{"$ref": "#/$defs/current"}]}},
              "required": ["mode"]}
    validators = compile_function_validators({"f": schema})
    validate_function_arguments(validators, "f", '{"mode":"B"}')
    with pytest.raises(TranslationError):
        validate_function_arguments(validators, "f", '{"mode":"A"}')


@pytest.mark.parametrize("schema", [
    {"$ref": "https://invalid.example/schema"},
    {"$defs": {"unused": {"$ref": "https://invalid.example/schema"}}},
    {"anyOf": [{"type": "object"}, {"$ref": "https://invalid.example/schema"}]},
    {"$ref": "file:///private/secret.json"},
    {"$ref": "#/$defs/missing"},
])
def test_unresolved_references_fail_preflight_without_network(monkeypatch, schema):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("schema validation attempted I/O")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    with pytest.raises(TranslationError, match="reference"):
        compile_function_validators({"f": schema})
    assert calls == []


def test_annotations_containing_ref_or_pattern_are_data_not_schemas():
    schema = {"type": "object", "default": {"$ref": "https://invalid.example/default"},
              "examples": [{"$ref": "https://invalid.example/example", "pattern": "["}],
              "properties": {"mode": {"type": "string", "format": "email"}}}
    validate_function_arguments(compile_function_validators({"f": schema}), "f", '{"mode":"not-email"}')


@pytest.mark.parametrize("schema", [
    {"type": "not-a-type"}, {"required": "mode"},
    {"$schema": "https://invalid.example/custom-dialect"},
])
def test_unusable_schemas_fail_before_streaming(schema):
    with pytest.raises(TranslationError):
        CodexToClaudeStreamTranslator({}, function_schemas={"f": schema})


@pytest.mark.parametrize("name", ["lookup", "removed_alias", "Lookup_Alias"])
def test_names_require_exact_prepared_alias(name):
    translator = _translator()
    assert translator.translate_event(_item(name=name)) == []
    with pytest.raises(TranslationError, match="undeclared or removed"):
        translator.translate_event(_item("done", name=name, arguments='{"mode":"B"}'))


def test_empty_current_schema_map_rejects_all_calls():
    translator = CodexToClaudeStreamTranslator({}, function_schemas={})
    assert translator.translate_event(_item()) == []
    with pytest.raises(TranslationError, match="undeclared or removed"):
        translator.translate_event(_item("done", arguments='{"mode":"B"}'))


def test_active_name_map_guard_is_still_enforced():
    translator = CodexToClaudeStreamTranslator({}, tool_name_map={},
                                              function_schemas={"lookup_alias": _SCHEMA})
    with pytest.raises(TranslationError, match="undeclared or removed"):
        translator.translate_event(_item("done", arguments='{"mode":"B"}'))


def test_none_schema_map_keeps_legacy_start_timing():
    translator = CodexToClaudeStreamTranslator({})
    assert len(_tool_blocks(translator.translate_event(_item()))) == 1


def test_argument_errors_do_not_echo_sensitive_values():
    validators = compile_function_validators({"f": _SCHEMA})
    with pytest.raises(TranslationError) as raised:
        validate_function_arguments(validators, "f", '{"mode":"private-credential"}')
    assert "private-credential" not in str(raised.value)
    assert "$.mode" in str(raised.value)


def test_parallel_interleaving_keeps_native_order_and_call_identity():
    translator = _translator()
    assert translator.translate_event(_item(output_index=2, id="item_a", call_id="call_a")) == []
    assert translator.translate_event(_item(output_index=3, id="item_b", call_id="call_b")) == []
    assert translator.translate_event({"type": "response.function_call_arguments.delta",
                                       "call_id": "call_b", "delta": '{"mo'}) == []
    assert translator.translate_event({"type": "response.function_call_arguments.delta",
                                       "item_id": "item_a", "delta": '{"mode":"B"}'}) == []
    assert translator.translate_event({"type": "response.function_call_arguments.delta",
                                       "item_id": "item_b", "delta": 'de":"B"}'}) == []
    assert translator.translate_event(_item("done", output_index=3, id="item_b", call_id="call_b")) == []
    events = translator.translate_event(_item("done", output_index=2, id="item_a", call_id="call_a"))
    assert [block["id"] for block in _tool_blocks(events)] == ["call_a", "call_b"]
    assert [payload["delta"]["partial_json"] for name, payload in events
            if name == "content_block_delta"] == ['{"mode":"B"}', '{"mode":"B"}']
    assert [payload["index"] for name, payload in events if name == "content_block_start"] == [0, 1]
    assert not _tool_blocks(translator.translate_event(_terminal()))


def test_text_and_reasoning_stream_while_function_arguments_are_held():
    translator = _translator()
    thinking = translator.translate_event({"type": "response.reasoning_summary_text.delta",
                                           "delta": "thinking"})
    assert thinking[-1][1]["delta"]["thinking"] == "thinking"
    assert translator.translate_event(_item()) == []
    text = translator.translate_event({"type": "response.output_text.delta", "delta": "text"})
    assert text[-1][1]["delta"]["text"] == "text"
    assert not _tool_blocks(thinking + text)
    events = translator.translate_event(_item("done", arguments='{"mode":"B"}'))
    assert events[0][0] == "content_block_stop"
    assert _tool_blocks(events)[0]["id"] == "call_1"


@pytest.mark.parametrize("completion", ["done", "terminal"])
def test_nameless_call_hydrates_real_identity_arguments_and_signature(completion):
    translator = _translator(custom_provider="gemprov")
    assert translator.translate_event(_item(output_index=7, name="", id="", call_id="")) == []
    assert translator.translate_event(_arguments(output_index=7, delta='{"mode":"B"}')) == []
    complete = _item("done", output_index=7, call_id="call.real/id", id="late_item",
                     extra_content={"google": {"thought_signature": "opaque-signature"}})
    if completion == "terminal":
        complete = _terminal({**complete["item"], "output_index": 7})
    events = translator.translate_event(complete)
    assert _tool_blocks(events)[0]["id"] == "call_real_id"
    assert _carriers(events)[0].call_id == "call_real_id"
    assert _carriers(events)[0].signature == "opaque-signature"


def test_terminal_only_full_output_validates_and_assembles_complete_tool_input():
    translator = _translator()
    events = translator.translate_event({"type": "response.created", "response": {"id": "r1"}})
    events += translator.translate_event(_terminal(_item("done", arguments='{"mode":"B"}')["item"]))
    message = assemble_claude_message(events)
    assert message["content"] == [{"type": "tool_use", "id": "call_1", "name": "lookup",
                                  "input": {"mode": "B"}}]
    assert message["stop_reason"] == "tool_use"


def test_named_added_call_hydrates_arguments_only_at_terminal():
    translator = _translator()
    assert translator.translate_event(_item(arguments="")) == []
    events = translator.translate_event(_terminal(_item("done", arguments='{"mode":"B"}')["item"]))
    assert len(_tool_blocks(events)) == 1
    assert events[1][1]["delta"]["partial_json"] == '{"mode":"B"}'


@pytest.mark.parametrize("completion", ["done", "terminal"])
def test_latest_signature_on_completion_replaces_added_snapshot(completion):
    translator = _translator(custom_provider="gemprov")
    assert translator.translate_event(_item(extra_content={"google": {"thought_signature": "added"}})) == []
    complete = _item("done", arguments='{"mode":"B"}',
                     extra_content={"google": {"thought_signature": "latest"}})
    if completion == "terminal":
        complete = _terminal(complete["item"])
    events = translator.translate_event(complete)
    assert [carrier.signature for carrier in _carriers(events)] == ["latest"]
    assert _carriers(events)[0].call_id == _tool_blocks(events)[0]["id"]


@pytest.mark.parametrize("identity", ["call_id", "item_id", "output_index"])
def test_terminal_signature_backfill_uses_positive_identity_without_replaying_call(identity):
    translator = _translator(custom_provider="gemprov")
    initial = {"type": "function_call", "name": "lookup_alias", "arguments": '{"mode":"B"}'}
    if identity == "call_id":
        initial["call_id"] = "call_opaque"
    elif identity == "item_id":
        initial["id"] = "item_opaque"
    added = {"type": "response.output_item.added", "item": initial, "output_index": 4}
    done = {**added, "type": "response.output_item.done"}
    assert translator.translate_event(added) == []
    events = translator.translate_event(done)
    tool_id = _tool_blocks(events)[0]["id"]
    terminal_item = {**initial, "extra_content": {"google": {"thought_signature": "terminal"}}}
    if identity == "output_index":
        terminal_item["output_index"] = 4
    terminal_events = translator.translate_event(_terminal(terminal_item))
    assert not _tool_blocks(terminal_events)
    assert [carrier.signature for carrier in _carriers(terminal_events)] == ["terminal"]
    assert _carriers(terminal_events)[0].call_id == tool_id
    assert terminal_events[-1][0] == "message_stop"


def test_parallel_carrier_signatures_do_not_cross_calls():
    translator = _translator(custom_provider="gemprov")
    first = _item(output_index=0, id="item_a", call_id="call_a",
                  extra_content={"google": {"thought_signature": "a"}})
    second = _item(output_index=1, id="item_b", call_id="call_b",
                   extra_content={"google": {"thought_signature": "b"}})
    assert translator.translate_event(first) == []
    assert translator.translate_event(second) == []
    assert translator.translate_event({**second, "type": "response.output_item.done",
                                       "item": {**second["item"], "arguments": '{"mode":"B"}'}}) == []
    events = translator.translate_event({**first, "type": "response.output_item.done",
                                        "item": {**first["item"], "arguments": '{"mode":"B"}'}})
    assert [(carrier.call_id, carrier.signature) for carrier in _carriers(events)] == [
        ("call_a", "a"), ("call_b", "b")]


@pytest.mark.parametrize("terminal", [
    _terminal(),
    _terminal(event_type="incomplete"),
    _terminal(_item("done", arguments='{"mode":')["item"]),
    _terminal(_item("done", arguments='{"mode":"B"}')["item"], event_type="incomplete"),
    _item("done", arguments='{"mode":"B"}', status="incomplete"),
])
def test_partial_call_cannot_emit_a_tool_use_or_stop(terminal):
    translator = _translator()
    emitted = translator.translate_event(_item())
    emitted += translator.translate_event(_arguments(delta='{"mode":'))
    with pytest.raises(TranslationError):
        translator.translate_event(terminal)
    assert emitted == []


def test_argument_done_alone_does_not_make_call_executable_at_empty_terminal():
    translator = _translator()
    assert translator.translate_event(_item()) == []
    assert translator.translate_event(_arguments("done", arguments='{"mode":"B"}')) == []
    with pytest.raises(TranslationError, match="incomplete"):
        translator.translate_event(_terminal())


def test_incomplete_terminal_does_not_discard_already_completed_valid_call():
    translator = _translator()
    events = translator.translate_event(_item("done", arguments='{"mode":"B"}'))
    assert len(_tool_blocks(events)) == 1
    completion = translator.translate_event(_terminal(event_type="incomplete"))
    assert not _tool_blocks(completion)
    assert completion[-1][0] == "message_stop"


def test_later_invalid_parallel_call_does_not_replay_prior_valid_call():
    translator = _translator()
    valid_events = translator.translate_event(_item("done", arguments='{"mode":"B"}'))
    assert len(_tool_blocks(valid_events)) == 1
    assert translator.translate_event(_item(output_index=1, id="item_2", call_id="call_2")) == []
    with pytest.raises(TranslationError):
        translator.translate_event(_item("done", output_index=1, id="item_2", call_id="call_2",
                                         arguments='{"mode":"A"}'))
    assert len(_tool_blocks(valid_events)) == 1


@pytest.mark.parametrize("delta,final_arguments,is_valid", [
    ('{"mode":"A"}', '{"mode":"B"}', True),
    ('{"mode":"B"}', '{"mode":"A"}', False),
    ('{"mode":"B"}', '', False),
])
def test_complete_snapshot_is_validated_not_speculative_deltas(delta, final_arguments, is_valid):
    translator = _translator()
    assert translator.translate_event(_item()) == []
    assert translator.translate_event(_arguments(delta=delta)) == []
    if not is_valid:
        with pytest.raises(TranslationError):
            translator.translate_event(_item("done", arguments=final_arguments))
    else:
        events = translator.translate_event(_item("done", arguments=final_arguments))
        assert events[1][1]["delta"]["partial_json"] == final_arguments


def test_embedded_resources_resolve_without_fetching_their_absolute_ids():
    schema = {"$id": "https://local.example/root", "type": "object",
              "$defs": {"mode": {"$id": "mode", "enum": ["B"]}},
              "properties": {"mode": {"$ref": "https://local.example/mode"}},
              "required": ["mode"]}
    validators = compile_function_validators({"f": schema})
    validate_function_arguments(validators, "f", '{"mode":"B"}')
    with pytest.raises(TranslationError):
        validate_function_arguments(validators, "f", '{"mode":"A"}')


def test_recursive_internal_schema_references_validate():
    schema = {"$defs": {"node": {"type": "object", "properties": {
        "mode": {"enum": ["B"]}, "child": {"$ref": "#/$defs/node"}}, "required": ["mode"]}},
        "$ref": "#/$defs/node"}
    validators = compile_function_validators({"f": schema})
    validate_function_arguments(validators, "f", '{"mode":"B","child":{"mode":"B"}}')
    with pytest.raises(TranslationError):
        validate_function_arguments(validators, "f", '{"mode":"B","child":{"mode":"A"}}')


def test_validator_snapshot_does_not_follow_later_mutations():
    schema = json.loads(json.dumps(_SCHEMA))
    validators = compile_function_validators({"f": schema})
    schema["properties"]["mode"]["enum"] = ["A"]
    validate_function_arguments(validators, "f", '{"mode":"B"}')
    with pytest.raises(TranslationError):
        validate_function_arguments(validators, "f", '{"mode":"A"}')


@pytest.mark.parametrize("source", ["delta", "done"])
def test_late_named_added_snapshot_does_not_erase_received_arguments(source):
    translator = _translator()
    assert translator.translate_event(_item(name="", arguments="")) == []
    argument_field = {"delta": '{"mode":"B"}'} if source == "delta" else {"arguments": '{"mode":"B"}'}
    assert translator.translate_event(_arguments(source, **argument_field)) == []
    assert translator.translate_event(_item(arguments="")) == []
    events = translator.translate_event(_item("done"))
    assert events[1][1]["delta"]["partial_json"] == '{"mode":"B"}'


def test_argument_delta_before_added_keeps_item_identity_and_json():
    translator = _translator()
    assert translator.translate_event({"type": "response.function_call_arguments.delta",
                                       "item_id": "item_1", "delta": '{"mode":"B"}'}) == []
    assert translator.translate_event(_item(arguments="")) == []
    events = translator.translate_event(_item("done"))
    assert _tool_blocks(events)[0]["id"] == "call_1"
    assert events[1][1]["delta"]["partial_json"] == '{"mode":"B"}'


def test_nested_schema_dialect_external_dynamic_reference_is_not_ignored():
    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
              "properties": {"mode": {
                  "$schema": "https://json-schema.org/draft/2020-12/schema",
                  "$dynamicRef": "https://invalid.example/remote"}}}
    with pytest.raises(TranslationError, match="reference"):
        compile_function_validators({"f": schema})


def test_reference_targeting_annotation_data_checks_the_referenced_schema():
    schema = {"default": {"$ref": "https://invalid.example/referenced-default"},
              "$ref": "#/default"}
    with pytest.raises(TranslationError, match="reference"):
        compile_function_validators({"f": schema})


def test_internal_ref_cycle_preflight_finishes_and_validation_fails_safely():
    validators = compile_function_validators({"f": {"$ref": "#"}})
    with pytest.raises(TranslationError, match="cannot validate"):
        validate_function_arguments(validators, "f", '{}')


@pytest.mark.parametrize("schema", [
    {"properties": {"mode": {"pattern": r"^\p{Lu}+$"}}},
    {"patternProperties": {r"\p{Zl}": {"type": "string"}}},
])
def test_unrepresentable_js_regex_preserves_preflight_but_fails_closed_at_call(schema):
    translator = CodexToClaudeStreamTranslator({}, function_schemas={"lookup_alias": schema})
    assert translator.translate_event(_item()) == []
    with pytest.raises(TranslationError, match="unsupported schema regex"):
        translator.translate_event(_item("done", arguments='{"mode":"B"}'))


@pytest.mark.parametrize('arguments', [
    '{"amount":1e999}', '{"amount":-1e999}',
    '{"nested":{"values":[0,1e999]}}', '{"nested":[{"amount":-1e999}]}',
])
def test_overflow_numbers_are_rejected_before_executable_tool_use(arguments):
    translator = CodexToClaudeStreamTranslator(
        {'model': 'mapped'}, tool_name_map={'lookup': 'lookup_alias'},
        function_schemas={'lookup_alias': {'type': 'object'}},
    )
    assert not _tool_blocks(translator.translate_event(_item()))
    assert not _tool_blocks(translator.translate_event(_arguments(delta=arguments)))
    with pytest.raises(TranslationError, match='invalid JSON arguments'):
        translator.translate_event(_item('done', arguments=arguments))


@pytest.mark.parametrize('arguments', ['{"amount":1e999}', '{"nested":[NaN]}', '{"amount":-Infinity}'])
def test_nonstream_assembly_uses_strict_function_argument_parse_policy(arguments):
    events = [
        ('message_start', {'message': {'id': 'msg', 'model': 'mapped'}}),
        ('content_block_start', {'index': 0, 'content_block': {
            'type': 'tool_use', 'id': 'call', 'name': 'lookup', 'input': {}}}),
        ('content_block_delta', {'index': 0, 'delta': {
            'type': 'input_json_delta', 'partial_json': arguments}}),
    ]
    with pytest.raises(TranslationError, match='invalid JSON arguments'):
        assemble_claude_message(events)


@pytest.mark.parametrize("regex_dialect", ["python", "ecmascript"])
def test_regex_timeout_prevents_tool_use_at_execution_boundary(regex_dialect):
    import os
    import signal
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import json, sys, time
        from claudex.translate.claude_to_codex import TranslationError
        from claudex.translate.codex_to_claude import CodexToClaudeStreamTranslator
        schema = {"properties": {"value": {"pattern": "^(a+)+$"}}}
        translator = CodexToClaudeStreamTranslator({}, tool_name_map={"lookup": "lookup"},
            function_schemas={"lookup": schema}, regex_dialect=sys.argv[1])
        event = {"type": "response.output_item.done", "output_index": 0,
                 "item": {"type": "function_call", "name": "lookup", "call_id": "call_1",
                          "arguments": json.dumps({"value": "a" * 35 + "!"})}}
        started = time.monotonic()
        try:
            translator.translate_event(event)
        except TranslationError:
            pass
        else:
            raise AssertionError("regex timeout was not raised before emission")
        assert time.monotonic() - started < 5
        assert not translator._has_emitted_tool_use
        assert not any(call.start_emitted for call in translator._buffered_function_calls)
    """)
    process = subprocess.Popen([sys.executable, "-c", program, regex_dialect],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=6)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("execution-boundary validation exceeded the hard 6-second test guard")
    assert process.returncode == 0, stdout + stderr


@pytest.mark.parametrize("completion", ["done", "completed", "incomplete"])
def test_fully_anonymous_call_binds_late_completion_identity(completion):
    translator = _translator()
    assert translator.translate_event({"type": "response.output_item.added",
                                       "item": {"type": "function_call"}}) == []
    assert translator.translate_event({"type": "response.function_call_arguments.delta",
                                       "delta": '{"mode":"B"}'}) == []
    item = {"type": "function_call", "call_id": "late_call", "name": "lookup_alias",
            "status": "completed"}
    event = ({"type": "response.output_item.done", "item": item} if completion == "done"
             else _terminal(item, event_type=completion))
    emitted = translator.translate_event(event)
    assert [block["id"] for block in _tool_blocks(emitted)] == ["late_call"]
    assert emitted[1][1]["delta"]["partial_json"] == '{"mode":"B"}'
    assert len(translator._buffered_function_calls) == (1 if completion == "done" else 0)
    if completion == "done":
        assert not _tool_blocks(translator.translate_event(_terminal(item)))


def test_ambiguous_late_call_identity_is_rejected_without_execution():
    translator = _translator()
    for index in (3, 4):
        assert translator.translate_event({"type": "response.output_item.added", "output_index": index,
                                           "item": {"type": "function_call"}}) == []
    with pytest.raises(TranslationError, match="ambiguous"):
        translator.translate_event(_terminal({"type": "function_call", "call_id": "late_call",
                                              "name": "lookup_alias", "arguments": '{"mode":"B"}'}))
    assert not translator._has_emitted_tool_use


def test_terminal_position_does_not_duplicate_index_only_emitted_call():
    translator = _translator(custom_provider="gemprov")
    item = {"type": "function_call", "name": "lookup_alias", "arguments": '{"mode":"B"}'}
    emitted = translator.translate_event({"type": "response.output_item.done", "output_index": 3,
                                          "item": item})
    public_id = _tool_blocks(emitted)[0]["id"]
    retained_call = translator._buffered_function_calls[0]
    terminal = translator.translate_event(_terminal({**item, "call_id": "late_call",
        "extra_content": {"google": {"thought_signature": "late_signature"}}}))
    assert not _tool_blocks(terminal)
    assert [carrier.call_id for carrier in _carriers(terminal)] == [public_id]
    assert not translator._buffered_function_calls
    assert retained_call.call_id == "late_call"
    assert retained_call.claude_tool_id == public_id


def test_ambiguous_terminal_position_does_not_duplicate_emitted_calls():
    translator = _translator()
    item = {"type": "function_call", "name": "lookup_alias", "arguments": '{"mode":"B"}'}
    for index in (3, 4):
        assert len(_tool_blocks(translator.translate_event({"type": "response.output_item.done",
            "output_index": index, "item": item}))) == 1
    with pytest.raises(TranslationError, match="ambiguous"):
        translator.translate_event(_terminal({**item, "call_id": "late_call"}))
    assert len(translator._buffered_function_calls) == 2


@pytest.mark.parametrize("validated", [False, True])
def test_native_call_id_sanitization_collision_is_rejected(validated):
    translator = _translator() if validated else CodexToClaudeStreamTranslator({})
    first = _item("done" if validated else "added", call_id="call.real/id", arguments='{"mode":"B"}')
    emitted = translator.translate_event(first)
    assert _tool_blocks(emitted)[0]["id"] == "call_real_id"
    with pytest.raises(TranslationError, match="tool.*id.*collision"):
        translator.translate_event(_item("done" if validated else "added", output_index=1,
            id="item_2", call_id="call_real_id", arguments='{"mode":"B"}'))


@pytest.mark.parametrize("completion", ["done", "arguments_done", "terminal"])
def test_emitted_arguments_cannot_change_in_later_valid_snapshot(completion):
    translator = CodexToClaudeStreamTranslator({}, function_schemas={"lookup_alias": {"type": "object"}},
                                                custom_provider="gemprov")
    emitted = translator.translate_event(_item("done", arguments='{"mode":"A"}'))
    assert len(_tool_blocks(emitted)) == 1
    changed = _item("done", arguments='{"mode":"B"}',
                    extra_content={"google": {"thought_signature": "different_arguments"}})
    if completion == "terminal":
        changed = _terminal(changed["item"])
    elif completion == "arguments_done":
        changed = _arguments("done", arguments='{"mode":"B"}')
    with pytest.raises(TranslationError, match="changed.*arguments"):
        translator.translate_event(changed)
    assert translator._buffered_function_calls[0].arguments == '{"mode":"A"}'
    assert not translator._emitted_carrier_tool_ids


def test_emitted_arguments_allow_semantically_identical_terminal_snapshot():
    translator = CodexToClaudeStreamTranslator({}, function_schemas={"lookup_alias": {"type": "object"}})
    emitted = translator.translate_event(_item("done", arguments='{"mode":"A","count":1}'))
    assert len(_tool_blocks(emitted)) == 1
    completed = translator.translate_event(_terminal(_item("done",
        arguments=' {"count": 1.0, "mode": "\\u0041"} ')["item"]))
    assert not _tool_blocks(completed)
    assert completed[-1][0] == "message_stop"


def test_terminal_only_anonymous_parallel_calls_keep_distinct_positions():
    translator = _translator()
    item = {"type": "function_call", "name": "lookup_alias", "arguments": '{"mode":"B"}'}
    emitted = translator.translate_event(_terminal(item, item))
    blocks = _tool_blocks(emitted)
    assert len(blocks) == 2
    assert blocks[0]["id"] != blocks[1]["id"]


def test_emitted_arguments_distinguish_nested_json_booleans_from_numbers():
    translator = CodexToClaudeStreamTranslator({}, function_schemas={"lookup_alias": {"type": "object"}})
    translator.translate_event(_item("done", arguments='{"nested":[{"value":true}]}'))
    with pytest.raises(TranslationError, match="changed.*arguments"):
        translator.translate_event(_terminal(_item("done", arguments='{"nested":[{"value":1}]}')["item"]))


def test_late_identity_with_explicit_index_selects_one_anonymous_slot():
    translator = _translator()
    for index in (3, 4):
        translator.translate_event({"type": "response.output_item.added", "output_index": index,
                                   "item": {"type": "function_call"}})
        translator.translate_event(_arguments(output_index=index, delta='{"mode":"B"}'))
    emitted = translator.translate_event(_terminal(
        {"type": "function_call", "output_index": 3, "call_id": "call_3", "name": "lookup_alias"},
        {"type": "function_call", "output_index": 4, "call_id": "call_4", "name": "lookup_alias"}))
    assert [block["id"] for block in _tool_blocks(emitted)] == ["call_3", "call_4"]


@pytest.mark.parametrize("changed", [False, True])
def test_deep_completed_snapshot_preserves_emitted_arguments_and_signature(changed):
    translator = CodexToClaudeStreamTranslator({}, function_schemas={"lookup_alias": {"type": "object"}},
                                                custom_provider="gemprov")
    depth = 600
    arguments = '{"x":' * depth + '0' + '}' * depth
    emitted = translator.translate_event(_item("done", arguments=arguments))
    assert len(_tool_blocks(emitted)) == 1
    retained_call = translator._buffered_function_calls[0]
    terminal_arguments = ' ' + '{"x":' * depth + ('1' if changed else '0') + '}' * depth + ' '
    terminal = _terminal(_item("done", arguments=terminal_arguments,
        extra_content={"google": {"thought_signature": "terminal_signature"}})["item"])
    if changed:
        with pytest.raises(TranslationError, match="changed.*arguments"):
            translator.translate_event(terminal)
        assert retained_call.signature == ""
        assert not translator._emitted_carrier_tool_ids
    else:
        completed = translator.translate_event(terminal)
        assert not _tool_blocks(completed)
        assert completed[-2][0] == "message_delta"
        assert completed[-1][0] == "message_stop"
        assert [carrier.signature for carrier in _carriers(completed)] == ["terminal_signature"]
    assert retained_call.arguments == arguments


@pytest.mark.parametrize("arguments", ['{"x":' * 2000 + '0' + '}' * 1999, '{"x":', '{"x":1e999}'],
                         ids=["deep_malformed", "truncated", "nonfinite"])
def test_unparseable_post_emission_snapshot_raises_translation_error(arguments):
    translator = CodexToClaudeStreamTranslator({}, function_schemas={"lookup_alias": {"type": "object"}},
                                                custom_provider="gemprov")
    translator.translate_event(_item("done", arguments='{"x":0}'))
    with pytest.raises(TranslationError, match="invalid JSON arguments"):
        translator.translate_event(_terminal(_item("done", arguments=arguments,
            extra_content={"google": {"thought_signature": "invalid_signature"}})["item"]))
    assert translator._buffered_function_calls[0].signature == ""
    assert not translator._emitted_carrier_tool_ids


@pytest.mark.parametrize("arguments, terminal_arguments, is_equal", [
    ('{"values":[1,2]}', '{"values":[1.0,2.0]}', True),
    ('{"values":[1,2]}', '{"values":[2,1]}', False),
    ('{"values":[true]}', '{"values":[1]}', False),
    ('{"values":[]}', '{"values":{}}', False),
    ('{"values":{}}', '{"values":[]}', False),
    ('{"values":' + '[' * 600 + '0' + ']' * 600 + '}',
     ' {"values":' + '[' * 600 + '0.0' + ']' * 600 + '} ', True),
    ('{"values":' + '[' * 600 + '0' + ']' * 600 + '}',
     ' {"values":' + '[' * 600 + '1' + ']' * 600 + '} ', False),
], ids=["numeric_equality", "array_order", "bool_number", "array_object", "object_array",
        "deep_equal_array", "deep_changed_array"])
def test_emitted_argument_snapshot_preserves_array_comparison_semantics(arguments, terminal_arguments, is_equal):
    translator = CodexToClaudeStreamTranslator({}, function_schemas={"lookup_alias": {"type": "object"}})
    emitted = translator.translate_event(_item("done", arguments=arguments))
    assert len(_tool_blocks(emitted)) == 1
    terminal = _terminal(_item("done", arguments=terminal_arguments)["item"])
    if is_equal:
        completed = translator.translate_event(terminal)
        assert not _tool_blocks(completed)
        assert completed[-1][0] == "message_stop"
    else:
        with pytest.raises(TranslationError, match="changed.*arguments"):
            translator.translate_event(terminal)
