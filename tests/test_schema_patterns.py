"""Tests for JavaScript regex semantics at the executable-call boundary."""

import pytest

from claudex.translate.claude_to_codex import TranslationError
from claudex.translate.schema_patterns import ecmascript_validator_factory


def _validator(schema):
    from jsonschema import Draft202012Validator
    factory, check = ecmascript_validator_factory()
    check(schema)
    return factory(Draft202012Validator)(schema)


@pytest.mark.parametrize('value,valid', [('ABC', True), ('É', True), ('abc', False), ('ABC1', False)])
def test_unicode_letter_pattern_uses_actual_ecmascript(value, valid):
    assert _validator({'type': 'string', 'pattern': r'^\p{Lu}+$'}).is_valid(value) is valid


def test_ecmascript_word_class_does_not_adopt_python_unicode_word_semantics():
    validator = _validator({'type': 'string', 'pattern': r'^\w+$'})
    assert validator.is_valid('ABC_123')
    assert not validator.is_valid('é')


def test_pattern_properties_and_additional_properties_agree_on_matches():
    validator = _validator({'type': 'object', 'patternProperties': {r'^\p{Lu}+$': {'type': 'integer'}}, 'additionalProperties': False})
    assert validator.is_valid({'É': 3})
    assert not validator.is_valid({'é': 3})
    assert not validator.is_valid({'É': 'wrong'})


def test_unevaluated_properties_preserves_regex_evaluation_across_all_of():
    validator = _validator({'type': 'object', 'allOf': [{'patternProperties': {r'^\p{Lu}+$': {'type': 'integer'}}}], 'unevaluatedProperties': False})
    assert validator.is_valid({'É': 3})
    assert not validator.is_valid({'é': 3})
    assert not validator.is_valid({'É': 'wrong'})


def test_nested_explicit_dialect_does_not_drop_pattern_adapter():
    validator = _validator({'type': 'object', 'properties': {'upper': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'string', 'pattern': r'^\p{Lu}+$',
    }}})
    assert validator.is_valid({'upper': 'É'})
    assert not validator.is_valid({'upper': 'wrong'})


def test_invalid_pattern_is_a_contextual_failure_not_an_ignored_constraint():
    with pytest.raises(TranslationError, match='ECMAScript schema pattern'):
        _validator({'pattern': '['})


def test_internal_reference_preserves_ecmascript_pattern_evaluation():
    validator = _validator({'type': 'object', '$defs': {'upper': {'type': 'string', 'pattern': r'^\p{Lu}+$'}},
                            'properties': {'value': {'$ref': '#/$defs/upper'}}})
    assert validator.is_valid({'value': 'É'})
    assert not validator.is_valid({'value': 'lower'})


def test_conditional_pattern_properties_and_unevaluated_properties_agree():
    validator = _validator({'type': 'object', 'if': {'required': ['kind']},
                            'then': {'properties': {'kind': {'type': 'string'}},
                                     'patternProperties': {r'^\p{Lu}+$': {'type': 'integer'}}},
                            'unevaluatedProperties': False})
    assert validator.is_valid({'kind': 'record', 'É': 1})
    assert not validator.is_valid({'kind': 'record', 'é': 1})


def test_older_dialect_does_not_gain_unevaluated_properties_semantics():
    from jsonschema import Draft7Validator
    factory, _ = ecmascript_validator_factory()
    validator = factory(Draft7Validator)({'type': 'object', 'unevaluatedProperties': False})
    assert validator.is_valid({'unrestricted': 1})


@pytest.mark.parametrize('regex_dialect', ['python', 'ecmascript'])
@pytest.mark.parametrize('branch', ['allOf', 'anyOf', 'oneOf', 'dependentSchemas', 'if', 'then', 'else', '$ref', '$dynamicRef'])
def test_unevaluated_properties_uses_child_resource_base(regex_dialect, branch):
    from claudex.translate.tool_validation import compile_function_validators

    child = {'$id': 'child', '$defs': {'fields': {'properties': {'allowed': {}}}},
             '$ref': '#/$defs/fields'}
    schema = {'$id': 'https://local.example/root', 'type': 'object',
              '$defs': {'fields': {'properties': {'forbidden': {}}}},
              'unevaluatedProperties': False}
    instance = {'allowed': 1}
    if branch in ('allOf', 'anyOf', 'oneOf'):
        schema[branch] = [child]
    elif branch == 'dependentSchemas':
        schema['dependentSchemas'] = {'allowed': child, 'forbidden': child}
    elif branch == 'if':
        schema['if'] = child
    elif branch == 'then':
        schema.update({'if': True, 'then': child})
    elif branch == 'else':
        schema.update({'if': False, 'else': child})
    else:
        schema['$defs']['child'] = child
        schema[branch] = 'child'
    validator = compile_function_validators({'lookup': schema}, regex_dialect=regex_dialect)['lookup']
    assert validator.is_valid(instance)
    assert not validator.is_valid({'forbidden': 1})


@pytest.mark.parametrize('regex_dialect', ['python', 'ecmascript'])
@pytest.mark.parametrize('child_resource', [False, True])
def test_recursive_ref_contributes_evaluated_properties_in_draft_2019(regex_dialect, child_resource):
    from claudex.translate.tool_validation import compile_function_validators

    schema = {'$schema': 'https://json-schema.org/draft/2019-09/schema',
              '$id': 'https://local.example/tree', '$recursiveAnchor': True,
              'type': 'object', 'properties': {'allowed': {}, 'next': {
                  '$recursiveRef': '#', 'unevaluatedProperties': False}},
              'unevaluatedProperties': False}
    if child_resource:
        schema = {'$schema': schema['$schema'], '$id': 'https://local.example/root',
                  '$defs': {'fields': {'properties': {'forbidden': {}}}},
                  'allOf': [{'$id': 'child', '$recursiveAnchor': True,
                             '$defs': {'fields': {'properties': {'allowed': {}}}},
                             '$ref': '#/$defs/fields', 'properties': {
                                 'next': {'$recursiveRef': '#', 'unevaluatedProperties': False}}}],
                  'unevaluatedProperties': False}
    validator = compile_function_validators({'lookup': schema}, regex_dialect=regex_dialect)['lookup']
    assert validator.is_valid({'allowed': 1, 'next': {'allowed': 2}})
    assert not validator.is_valid({'next': {'forbidden': 1}})


@pytest.mark.parametrize('regex_dialect', ['python', 'ecmascript'])
def test_only_successful_branches_contribute_evaluated_properties(regex_dialect):
    from claudex.translate.tool_validation import compile_function_validators

    schema = {'type': 'object', 'anyOf': [
        {'properties': {'allowed': {'const': 1}}, 'required': ['allowed']},
        {'properties': {'forbidden': {'const': 2}}, 'required': ['forbidden']},
    ], 'unevaluatedProperties': False}
    validator = compile_function_validators({'lookup': schema}, regex_dialect=regex_dialect)['lookup']
    assert validator.is_valid({'allowed': 1})
    assert not validator.is_valid({'allowed': 1, 'forbidden': 1})


@pytest.mark.parametrize('regex_dialect', ['python', 'ecmascript'])
def test_child_resource_conditional_validity_and_annotations_agree(regex_dialect):
    from claudex.translate.tool_validation import compile_function_validators

    schema = {'$id': 'https://local.example/root', 'type': 'object',
              '$defs': {'fields': {'required': ['forbidden']}},
              'if': {'$id': 'child', '$defs': {'fields': {'required': ['allowed'],
                       'properties': {'allowed': {'const': 1}}}}, '$ref': '#/$defs/fields'},
              'then': True, 'else': False, 'unevaluatedProperties': False}
    validator = compile_function_validators({'lookup': schema}, regex_dialect=regex_dialect)['lookup']
    assert validator.is_valid({'allowed': 1})
    assert not validator.is_valid({'allowed': 2})
    assert not validator.is_valid({'forbidden': 1})


@pytest.mark.parametrize('regex_dialect', ['python', 'ecmascript'])
def test_nested_draft7_reference_siblings_do_not_contribute_annotations(regex_dialect):
    from claudex.translate.tool_validation import compile_function_validators

    schema = {'$id': 'https://local.example/root', 'type': 'object',
              '$defs': {'fields': {'properties': {'allowed': {}}}},
              'allOf': [{'$schema': 'http://json-schema.org/draft-07/schema#',
                         '$ref': '#/$defs/fields', 'properties': {'forbidden': {}}}],
              'unevaluatedProperties': False}
    validator = compile_function_validators({'lookup': schema}, regex_dialect=regex_dialect)['lookup']
    assert validator.is_valid({'allowed': 1})
    assert not validator.is_valid({'forbidden': 1})


@pytest.mark.parametrize("regex_dialect", ["python", "ecmascript"])
@pytest.mark.parametrize("keyword", ["pattern", "patternProperties", "additionalProperties", "unevaluatedProperties"])
def test_catastrophic_regex_is_bounded_and_validation_recovers(regex_dialect, keyword):
    import os
    import signal
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import json, sys, time
        from claudex.translate.claude_to_codex import TranslationError
        from claudex.translate.tool_validation import compile_function_validators, validate_function_arguments
        import claudex.translate.schema_patterns as patterns
        dialect, keyword = sys.argv[1:]
        expression = "^(a+)+$"
        value = "a" * 35 + "!"
        if keyword == "pattern":
            schema = {"properties": {"value": {"pattern": expression}}}
            bad, good = {"value": value}, {"value": "a"}
        else:
            schema = {"patternProperties": {expression: {}}, "additionalProperties": False}
            if keyword == "unevaluatedProperties":
                schema = {"allOf": [{"patternProperties": {expression: {}}}], "unevaluatedProperties": False}
            bad, good = {value: 1}, {"a": 1}
        validators = compile_function_validators({"lookup": schema}, regex_dialect=dialect)
        pool = patterns._python_worker_pool if dialect == "python" else patterns._ecmascript_worker_pool
        pool.matches("a", ["a"])
        worker = pool._idle[-1]
        old_process = worker._process
        started = time.monotonic()
        try:
            validate_function_arguments(validators, "lookup", json.dumps(bad))
        except TranslationError:
            pass
        else:
            raise AssertionError("catastrophic regex did not raise TranslationError")
        assert time.monotonic() - started < 5
        assert old_process.poll() is not None, "timed out worker was not terminated"
        assert worker._process is None
        validate_function_arguments(validators, "lookup", json.dumps(good))
        assert pool._idle[-1]._process.pid != old_process.pid
    """)
    process = subprocess.Popen([sys.executable, "-c", program, regex_dialect, keyword],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=6)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("regex validation exceeded the hard 6-second test guard")
    assert process.returncode == 0, stdout + stderr


@pytest.mark.parametrize("regex_dialect", ["python", "ecmascript"])
def test_validation_budget_is_shared_by_all_regex_keywords(regex_dialect, monkeypatch):
    import time
    import claudex.translate.schema_patterns as patterns
    from claudex.translate.tool_validation import compile_function_validators, validate_function_arguments

    schema = {"properties": {"value": {"allOf": [{"pattern": "a"}, {"pattern": "b"}, {"pattern": "c"}]}}}
    validators = compile_function_validators({"lookup": schema}, regex_dialect=regex_dialect)
    if regex_dialect == "python":
        matcher_class = patterns._PythonPatternMatcher
        program_name = "_PYTHON_PATTERN_PROGRAM"
        program = patterns._PYTHON_PATTERN_PROGRAM.replace("import json, re, sys", "import json, re, sys, time")
        program = program.replace("request = json.loads(line)", "request = json.loads(line); time.sleep(0.1)")
    else:
        matcher_class = patterns._PatternMatcher
        program_name = "_PATTERN_PROGRAM"
        program = patterns._PATTERN_PROGRAM.replace('try{', 'try{Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,100);')
    worker = patterns._RegexWorker(regex_dialect)
    monkeypatch.setattr(matcher_class, "_pool", worker)
    monkeypatch.setattr(patterns, program_name, program)
    monkeypatch.setattr(patterns, "PATTERN_VALIDATION_TIMEOUT_SECONDS", 0.25)
    started = time.monotonic()
    try:
        with pytest.raises(TranslationError, match="timed out"):
            validate_function_arguments(validators, "lookup", '{"value":"abc"}')
        assert time.monotonic() - started < 0.6
        assert worker._process is None
    finally:
        worker.close()


def test_python_adapter_preserves_native_regex_semantics_and_errors():
    from jsonschema import Draft202012Validator
    from claudex.translate.schema_patterns import python_validator_factory

    factory = python_validator_factory()
    for schema, instance in [
        ({"pattern": r"(?i)^é\w+$"}, "Éclair"),
        ({"pattern": r"(?i)^é\w+$"}, "wrong"),
        ({"patternProperties": {r"^a": {"type": "integer"}}, "additionalProperties": False}, {"bad": 1}),
        ({"additionalProperties": False}, {"bad": 1}),
        ({"patternProperties": {r"(?i)^a": {}, r"(?i)^b": {}}, "additionalProperties": False}, {"a": 1, "b": 1}),
        ({"patternProperties": {r"(?i)^a": {}, r"(?i)^b": {}}, "properties": {"allowed": {}}, "additionalProperties": False}, {"allowed": 1}),
    ]:
        import re
        try:
            expected = [error.message for error in Draft202012Validator(schema).iter_errors(instance)]
        except re.error as expected_error:
            with pytest.raises(re.error) as actual_error:
                list(factory(Draft202012Validator)(schema).iter_errors(instance))
            assert str(actual_error.value) == str(expected_error)
        else:
            actual = [error.message for error in factory(Draft202012Validator)(schema).iter_errors(instance)]
            assert actual == expected



def test_persistent_regex_workers_are_reaped_at_interpreter_exit():
    import os
    import subprocess
    import sys

    program = """
from claudex.translate.schema_patterns import _python_worker_pool, _ecmascript_worker_pool
for pool in (_python_worker_pool, _ecmascript_worker_pool):
    workers = [pool.checkout() for _ in range(3)]
    for worker in workers:
        assert worker.matches("a", ["a"]) == [True]
        print(worker._process.pid, flush=True)
    pool.release(workers[0])
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            timeout=5, check=True)
    workers = [int(pid) for pid in result.stdout.splitlines()]
    assert len(workers) == 6
    for pid in workers:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)



@pytest.mark.parametrize("regex_dialect", ["python", "ecmascript"])
def test_concurrent_benign_validation_does_not_wait_for_catastrophic_request(regex_dialect):
    import os
    import signal
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import json, sys, threading, time
        from concurrent.futures import ThreadPoolExecutor
        from claudex.translate.claude_to_codex import TranslationError
        from claudex.translate.tool_validation import compile_function_validators, validate_function_arguments
        from claudex.translate.schema_patterns import _RegexWorker
        dialect = sys.argv[1]
        bad = compile_function_validators({"lookup": {"properties": {"value": {"pattern": "^(a+)+$"}}}}, regex_dialect=dialect)
        good = compile_function_validators({"lookup": {"properties": {"value": {"pattern": "^a$"}}}}, regex_dialect=dialect)
        evaluating = threading.Event()
        original_exchange = _RegexWorker._exchange
        def signal_exchange(self, request):
            if json.loads(request)["pattern"] == "^(a+)+$":
                evaluating.set()
            return original_exchange(self, request)
        _RegexWorker._exchange = signal_exchange
        with ThreadPoolExecutor(max_workers=2) as executor:
            catastrophic = executor.submit(validate_function_arguments, bad, "lookup", json.dumps({"value": "a" * 35 + "!"}))
            assert evaluating.wait(1), "catastrophic evaluation did not start"
            started = time.monotonic()
            benign = executor.submit(validate_function_arguments, good, "lookup", '{"value":"a"}')
            try:
                benign.result(timeout=4)
            except TranslationError as error:
                raise AssertionError("unrelated benign validation was rejected") from error
            elapsed = time.monotonic() - started
            try:
                catastrophic.result(timeout=4)
            except TranslationError:
                pass
            else:
                raise AssertionError("catastrophic validation did not time out")
        assert elapsed < 1, f"benign validation waited {elapsed:.3f}s for another request"
    """)
    process = subprocess.Popen([sys.executable, "-c", program, regex_dialect],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=7)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("concurrent validation exceeded the hard 7-second test guard")
    assert process.returncode == 0, stdout + stderr


def test_catalog_compilation_does_not_share_one_pattern_check_deadline(monkeypatch):
    import time
    import claudex.translate.schema_patterns as patterns
    from claudex.translate.tool_validation import compile_function_validators

    program = patterns._PATTERN_PROGRAM.replace('try{', 'try{Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,250);')
    worker = patterns._RegexWorker("ecmascript")
    monkeypatch.setattr(patterns, "_PATTERN_PROGRAM", program)
    monkeypatch.setattr(patterns._PatternMatcher, "_pool", worker)
    schemas = {f"lookup_{index}": {"properties": {"value": {"pattern": f"^value_{index}$"}}} for index in range(10)}
    started = time.monotonic()
    try:
        validators = compile_function_validators(schemas, regex_dialect="ecmascript")
        assert len(validators) == len(schemas)
        assert time.monotonic() - started > patterns.PATTERN_VALIDATION_TIMEOUT_SECONDS
    finally:
        worker.close()


def test_single_stuck_compile_time_pattern_check_is_terminated(monkeypatch):
    import time
    import claudex.translate.schema_patterns as patterns
    from claudex.translate.tool_validation import compile_function_validators

    program = patterns._PATTERN_PROGRAM.replace('try{', 'try{Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,10000);')
    worker = patterns._RegexWorker("ecmascript")
    monkeypatch.setattr(patterns, "_PATTERN_PROGRAM", program)
    monkeypatch.setattr(patterns._PatternMatcher, "_pool", worker)
    monkeypatch.setattr(patterns, "PATTERN_VALIDATION_TIMEOUT_SECONDS", 0.25)
    started = time.monotonic()
    try:
        with pytest.raises(TranslationError, match="timed out"):
            compile_function_validators({"lookup": {"pattern": "slow"}}, regex_dialect="ecmascript")
        assert time.monotonic() - started < 0.6
        assert worker._process is None
    finally:
        worker.close()


@pytest.mark.parametrize("regex_dialect", ["python", "ecmascript"])
def test_worker_pool_bounds_idle_workers_and_closes_all_checkouts(regex_dialect):
    from claudex.translate.schema_patterns import _RegexWorkerPool, MAX_IDLE_PATTERN_WORKERS_PER_DIALECT

    pool = _RegexWorkerPool(regex_dialect)
    workers = [pool.checkout() for _ in range(MAX_IDLE_PATTERN_WORKERS_PER_DIALECT + 2)]
    processes = []
    try:
        for worker in workers:
            assert worker.matches("a", ["a"]) == [True]
            processes.append(worker._process)
        for worker in workers[:-1]:
            pool.release(worker)
        assert len(pool._idle) == MAX_IDLE_PATTERN_WORKERS_PER_DIALECT
        assert processes[-2].poll() is not None, "excess idle worker was not reaped"
        assert processes[-1].poll() is None, "checked-out worker was closed prematurely"
    finally:
        pool.close()
    assert not pool._workers
    assert not pool._idle
    assert all(process.poll() is not None for process in processes)
