"""Resource-aware schema annotations and dialect-specific regex keywords."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from contextvars import ContextVar
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import threading
import time
from typing import Any

from .claude_to_codex import TranslationError

PATTERN_VALIDATION_TIMEOUT_SECONDS = 2.0
MAX_IDLE_PATTERN_WORKERS_PER_DIALECT = 2
_validation_workers: ContextVar[dict[_RegexWorkerPool, _RegexWorker] | None] = ContextVar("schema_validation_workers", default=None)
_validation_deadline: ContextVar[float | None] = ContextVar("schema_validation_deadline", default=None)
_PATTERN_PROGRAM = (
    'const readline=require("readline");'
    'readline.createInterface({input:process.stdin}).on("line",line=>{'
    'try{const x=JSON.parse(line);const r=new RegExp(x.pattern,"u");'
    'process.stdout.write(JSON.stringify({matches:x.values.map(v=>r.test(v))})+"\\n");}'
    'catch(e){process.stdout.write(JSON.stringify({error:String(e)})+"\\n");}});'
)
_PYTHON_PATTERN_PROGRAM = """
import json, re, sys
for line in sys.stdin:
    try:
        request = json.loads(line)
        reply = {"matches": [re.search(request["pattern"], value) is not None for value in request["values"]]}
    except re.error as error:
        reply = {"error": str(error)}
    print(json.dumps(reply), flush=True)
"""


@contextmanager
def pattern_validation_budget():
    # All regex keywords and annotation traversals share one execution deadline.
    if _validation_deadline.get() is not None:
        yield
        return
    token = _validation_deadline.set(time.monotonic() + PATTERN_VALIDATION_TIMEOUT_SECONDS)
    workers: dict[_RegexWorkerPool, _RegexWorker] = {}
    workers_token = _validation_workers.set(workers)
    is_successful = False
    try:
        yield
        is_successful = True
    finally:
        try:
            for pool, worker in workers.items():
                if not is_successful:
                    worker.close()
                pool.release(worker)
        finally:
            _validation_workers.reset(workers_token)
            _validation_deadline.reset(token)


def _remaining_budget() -> float:
    deadline = _validation_deadline.get()
    assert deadline is not None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TranslationError("schema pattern validation timed out")
    return remaining


class _RegexWorker:
    def __init__(self, dialect: str) -> None:
        self.dialect = dialect
        self._process: subprocess.Popen | None = None

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()

    def _start(self) -> None:
        if self.dialect == "python":
            command = [sys.executable, "-I", "-u", "-c", _PYTHON_PATTERN_PROGRAM]
            environment = {"LANG": "C"}
        else:
            spec = importlib.util.find_spec("playwright")
            if spec is None or spec.origin is None:
                raise TranslationError("ECMAScript schema validation requires the bundled JavaScript runtime")
            directory = Path(spec.origin).parent / "driver"
            node = next((path for path in (directory / "node", directory / "node.exe") if path.is_file()), None)
            if node is None:
                raise TranslationError("the bundled JavaScript runtime is unavailable")
            command = [str(node), "-e", _PATTERN_PROGRAM]
            environment = {"LANG": "C", "NODE_NO_WARNINGS": "1"}
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=environment,
        )
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)

    def matches(self, pattern: str, values: list[str]) -> list[bool]:
        with pattern_validation_budget():
            try:
                _remaining_budget()
                if self._process is None or self._process.poll() is not None:
                    self.close()
                    self._start()
                request = json.dumps({"pattern": pattern, "values": values}, ensure_ascii=True).encode() + b"\n"
                reply = self._exchange(request)
                if "error" in reply:
                    if self.dialect == "python":
                        raise re.error(reply["error"])
                    raise TranslationError("cannot validate ECMAScript schema pattern")
                matched = reply.get("matches")
                if not isinstance(matched, list) or len(matched) != len(values) or any(type(item) is not bool for item in matched):
                    raise TranslationError(f"invalid {self.dialect_name} schema validation result")
                return matched
            except (TranslationError, re.error):
                self.close()
                raise
            except (OSError, ValueError) as exc:
                self.close()
                label = "ECMAScript" if self.dialect == "ecmascript" else "Python"
                raise TranslationError(f"cannot validate {label} schema pattern") from exc

    @property
    def dialect_name(self) -> str:
        return "JavaScript" if self.dialect == "ecmascript" else "Python"

    def _exchange(self, request: bytes) -> dict:
        process = self._process
        assert process is not None
        response = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdin, selectors.EVENT_WRITE)
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                for key, _ in selector.select(_remaining_budget()):
                    if key.fileobj is process.stdin:
                        written = os.write(process.stdin.fileno(), request)
                        request = request[written:]
                        if not request:
                            selector.unregister(process.stdin)
                    else:
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            raise ValueError("regex worker exited without a result")
                        response.extend(chunk)
                        if response.endswith(b"\n"):
                            _remaining_budget()
                            return json.loads(response)


class _RegexWorkerPool:
    def __init__(self, dialect: str) -> None:
        self.dialect = dialect
        self._idle: list[_RegexWorker] = []
        self._workers: set[_RegexWorker] = set()
        self._lock = threading.Lock()

    def checkout(self) -> _RegexWorker:
        # A busy validation never queues another request behind its regex worker.
        with self._lock:
            worker = self._idle.pop() if self._idle else _RegexWorker(self.dialect)
            self._workers.add(worker)
            return worker

    def release(self, worker: _RegexWorker) -> None:
        with self._lock:
            if worker._process is not None and len(self._idle) < MAX_IDLE_PATTERN_WORKERS_PER_DIALECT:
                self._idle.append(worker)
                return
            self._workers.discard(worker)
        worker.close()

    def matches(self, pattern: str, values: list[str]) -> list[bool]:
        with pattern_validation_budget():
            workers = _validation_workers.get()
            assert workers is not None
            if self not in workers:
                workers[self] = self.checkout()
            return workers[self].matches(pattern, values)

    def close(self) -> None:
        with self._lock:
            workers = list(self._workers)
            self._workers.clear()
            self._idle.clear()
        for worker in workers:
            worker.close()


_python_worker_pool = _RegexWorkerPool("python")
_ecmascript_worker_pool = _RegexWorkerPool("ecmascript")
atexit.register(_python_worker_pool.close)
atexit.register(_ecmascript_worker_pool.close)


class _PatternMatcher:
    _pool = _ecmascript_worker_pool

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[str, ...]], list[bool]] = {}

    def matches(self, pattern: str, values: list[str]) -> list[bool]:
        key = (pattern, tuple(values))
        if _validation_deadline.get() is not None:
            _remaining_budget()
        if key in self._cache:
            return self._cache[key]
        matched = self._pool.matches(pattern, values)
        if len(self._cache) >= 64:
            self._cache.clear()
        self._cache[key] = matched
        return matched


class _PythonPatternMatcher(_PatternMatcher):
    _pool = _python_worker_pool


def python_validator_factory() -> Any:
    factory, _ = _validator_factory(_PythonPatternMatcher(), adapt_patterns=False)
    return factory


def ecmascript_validator_factory() -> tuple[Any, Any]:
    """Extend regex-sensitive keywords without changing the upstream schema."""
    return _validator_factory(_PatternMatcher(), adapt_patterns=True)


def _validator_factory(matcher: Any, *, adapt_patterns: bool) -> tuple[Any, Any]:
    from jsonschema import validators
    from jsonschema.exceptions import ValidationError

    classes: dict[Any, Any] = {}

    def pattern(validator: Any, expression: str, instance: Any, schema: dict) -> Any:
        if validator.is_type(instance, "string") and not matcher.matches(expression, [instance])[0]:
            yield ValidationError("string does not match the declared pattern" if adapt_patterns
                                  else f"{instance!r} does not match {expression!r}")

    def pattern_properties(validator: Any, patterns: dict, instance: Any, schema: dict) -> Any:
        if not validator.is_type(instance, "object"):
            return
        names = list(instance)
        for expression, subschema in patterns.items():
            matches = (
                matcher.matches(expression, names) if adapt_patterns
                else (matcher.matches(expression, [name])[0] for name in names)
            )
            for name, matched in zip(names, matches):
                if matched:
                    yield from validator.descend(instance[name], subschema, path=name, schema_path=expression)

    def additional_properties(validator: Any, additional: Any, instance: Any, schema: dict) -> Any:
        if not validator.is_type(instance, "object"):
            return
        names = list(instance)
        covered = set(schema.get("properties", {})).intersection(names)
        patterns = schema.get("patternProperties", {})
        if adapt_patterns:
            for expression in patterns:
                covered.update(name for name, matched in zip(names, matcher.matches(expression, names)) if matched)
        elif patterns:
            # jsonschema's Python additionalProperties joins patterns before searching.
            candidates = [name for name in names if name not in covered]
            if candidates:
                covered.update(name for name, matched in zip(candidates, matcher.matches("|".join(patterns), candidates)) if matched)
        extras = [name for name in names if name not in covered]
        if not adapt_patterns:
            extras = set(extras)
        if validator.is_type(additional, "object"):
            for name in extras:
                yield from validator.descend(instance[name], additional, path=name)
        elif (additional is False if adapt_patterns else not additional) and extras:
            if adapt_patterns:
                yield ValidationError("additional properties are not allowed")
            elif "patternProperties" in schema:
                verb = "does" if len(extras) == 1 else "do"
                joined = ", ".join(repr(name) for name in sorted(extras))
                expressions = ", ".join(repr(expression) for expression in sorted(patterns))
                yield ValidationError(f"{joined} {verb} not match any of the regexes: {expressions}")
            else:
                from jsonschema._utils import extras_msg
                yield ValidationError("Additional properties are not allowed (%s %s unexpected)" % extras_msg(sorted(extras, key=str)))

    def child_validator(validator: Any, schema: Any) -> Any:
        if validator._ref_resolver is not None:
            return validator.evolve(schema=schema)
        from referencing import Resource
        from referencing.jsonschema import specification_with

        dialect = validator.META_SCHEMA.get("$id") or validator.META_SCHEMA["id"]
        resource = Resource.from_contents(schema, default_specification=specification_with(dialect))
        # evolve alone does not enter a subschema resource; descend does.
        resolver = validator._resolver.in_subresource(resource)
        return validator.evolve(schema=schema, _resolver=resolver)

    def conditional(validator: Any, condition: Any, instance: Any, schema: dict) -> Any:
        child = child_validator(validator, condition)
        branch = "then" if child.is_valid(instance) else "else"
        if branch in schema:
            yield from validator.descend(instance, schema[branch], schema_path=branch)

    def evaluated_keys(validator: Any, instance: dict, schema: Any) -> set[str]:
        if isinstance(schema, bool):
            return set()
        # Older drafts ignore every sibling of $ref, including annotations.
        schema = dict(type(validator)._APPLICABLE_VALIDATORS(schema))
        covered: set[str] = set(schema.get("properties", {})).intersection(instance)
        for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
            if keyword not in schema or keyword not in validator.VALIDATORS:
                continue
            if keyword == "$recursiveRef":
                from referencing.jsonschema import lookup_recursive_ref
                resolved = lookup_recursive_ref(validator._resolver)
            else:
                resolved = validator._resolver.lookup(schema[keyword])
            child = validator.evolve(schema=resolved.contents, _resolver=resolved.resolver)
            covered.update(evaluated_keys(child, instance, resolved.contents))
        for keyword in ("additionalProperties", "unevaluatedProperties"):
            if keyword in schema and keyword in validator.VALIDATORS:
                child = child_validator(validator, schema[keyword])
                covered.update(name for name, value in instance.items() if child.is_valid(value))
        names = list(instance)
        for expression in schema.get("patternProperties", {}):
            covered.update(name for name, matched in zip(names, matcher.matches(expression, names)) if matched)
        for name, subschema in schema.get("dependentSchemas", {}).items():
            if name in instance and "dependentSchemas" in validator.VALIDATORS:
                child = child_validator(validator, subschema)
                covered.update(evaluated_keys(child, instance, subschema))
        for keyword in ("allOf", "oneOf", "anyOf"):
            for subschema in schema.get(keyword, []):
                child = child_validator(validator, subschema)
                if child.is_valid(instance):
                    covered.update(evaluated_keys(child, instance, subschema))
        if "if" in schema and "if" in validator.VALIDATORS:
            child = child_validator(validator, schema["if"])
            if child.is_valid(instance):
                covered.update(evaluated_keys(child, instance, schema["if"]))
                branch = schema.get("then", True)
            else:
                branch = schema.get("else", True)
            child = child_validator(validator, branch)
            covered.update(evaluated_keys(child, instance, branch))
        return covered

    def unevaluated_properties(validator: Any, unevaluated: Any, instance: Any, schema: dict) -> Any:
        if not validator.is_type(instance, "object"):
            return
        covered = evaluated_keys(validator, instance, schema)
        for name, value in instance.items():
            if name not in covered:
                yield from validator.descend(value, unevaluated, path=name, schema_path=name)

    def factory(base: Any) -> Any:
        if base in classes:
            return classes[base]
        keywords = {"unevaluatedProperties": unevaluated_properties, "if": conditional}
        keywords.update({
            "pattern": pattern,
            "patternProperties": pattern_properties,
            "additionalProperties": additional_properties,
        })
        extended = validators.extend(base, {
            key: handler for key, handler in keywords.items() if key in base.VALIDATORS
        })
        original_evolve = extended.evolve

        def evolve(self: Any, **changes: Any) -> Any:
            child = original_evolve(self, **changes)
            if child.__class__ in classes.values():
                return child
            child_class = factory(child.__class__)
            return child_class(
                child.schema, resolver=child._ref_resolver,
                format_checker=child.format_checker,
                registry=child._registry, _resolver=child._resolver,
            )

        extended.evolve = evolve
        classes[base] = extended
        return extended

    def check_patterns(schema: Any) -> None:
        if not isinstance(schema, dict):
            return
        if "pattern" in schema:
            matcher.matches(schema["pattern"], [])
        for expression in schema.get("patternProperties", {}):
            matcher.matches(expression, [])

    return factory, check_patterns
