"""Validate executable calls against the current prepared function schemas."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from typing import Any

from .claude_to_codex import TranslationError
from .schema_patterns import pattern_validation_budget


def _refuse_resource_retrieval(uri: str) -> Any:
    from referencing.exceptions import NoSuchResource

    # Schemas may resolve embedded resources, never network or filesystem URLs.
    raise NoSuchResource(ref=uri)


def _check_schema_resources(
    resource: Any, resolver: Any, validator_class: Any, checked: set[int], pattern_check: Any = None
) -> None:
    from jsonschema.validators import validator_for
    from referencing import Resource
    from referencing.jsonschema import specification_with

    schema = resource.contents
    if isinstance(schema, bool) or id(schema) in checked:
        return
    checked.add(id(schema))
    if pattern_check is not None:
        pattern_check(schema)
    validator_class = validator_for(schema, default=validator_class)
    for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
        if keyword in schema and keyword in validator_class.VALIDATORS:
            resolved = resolver.lookup(schema[keyword])
            validator_class.check_schema(resolved.contents, format_checker=None)
            specification = specification_with(
                validator_class.META_SCHEMA.get("$id") or validator_class.META_SCHEMA["id"]
            )
            target = Resource.from_contents(resolved.contents, default_specification=specification)
            _check_schema_resources(target, resolved.resolver, validator_class, checked, pattern_check)
    for child in resource.subresources():
        _check_schema_resources(child, resolver.in_subresource(child), validator_class, checked, pattern_check)


def compile_function_validators(
    schemas: dict[str, dict], *, regex_dialect: str = "python"
) -> dict[str, Any]:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for
    from referencing import Registry, Resource
    from referencing.exceptions import CannotDetermineSpecification, Unresolvable
    from referencing.jsonschema import DRAFT202012

    from .schema_patterns import python_validator_factory

    factory = python_validator_factory()
    pattern_check = None
    if regex_dialect == "ecmascript":
        from .schema_patterns import ecmascript_validator_factory
        factory, pattern_check = ecmascript_validator_factory()
    elif regex_dialect != "python":
        raise TranslationError("unsupported function schema regex dialect")
    validators: dict[str, Any] = {}
    for name, declared_schema in schemas.items():
        try:
            schema = deepcopy(declared_schema)
            if not isinstance(schema, dict):
                raise TranslationError(f"function {name} has an invalid parameter schema")
            validator_class = validator_for(schema, default=Draft202012Validator)
            if "$schema" in schema and validator_for(schema, default=None) is None:
                raise TranslationError(f"function {name} declares an unsupported JSON Schema dialect")
            validator_class.check_schema(schema, format_checker=None)
            resource = Resource.from_contents(schema, default_specification=DRAFT202012)
            registry = Registry(retrieve=_refuse_resource_retrieval).with_resource(
                resource.id() or "", resource
            ).crawl()
            _check_schema_resources(resource, registry.resolver_with_root(resource), validator_class, set(), pattern_check)
            # No format checker: JSON Schema formats are annotations by default.
            validator_class = factory(validator_class)
            validators[name] = validator_class(schema, registry=registry)
        except Unresolvable as exc:
            raise TranslationError(
                f"function {name} has an unresolved or external schema reference; "
                "resource retrieval is disabled"
            ) from exc
        except (SchemaError, CannotDetermineSpecification, re.error, TypeError, ValueError,
                OverflowError, RecursionError) as exc:
            raise TranslationError(f"cannot validate parameter schema for function {name}") from exc
    return validators


def _reject_json_constant(value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _parse_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def parse_function_arguments(arguments: str, name: str) -> dict:
    try:
        instance = json.loads(
            arguments, parse_constant=_reject_json_constant, parse_float=_parse_finite_float
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise TranslationError(f"upstream function {name} returned invalid JSON arguments") from exc
    if not isinstance(instance, dict):
        raise TranslationError(f"upstream function {name} arguments must be a JSON object")
    return instance


@pattern_validation_budget()
def validate_function_arguments(validators: dict[str, Any], name: str, arguments: str) -> None:
    from jsonschema.exceptions import ValidationError
    from referencing.exceptions import Unresolvable

    validator = validators.get(name)
    if validator is None:
        raise TranslationError(f"upstream returned an undeclared or removed function: {name}")
    instance = parse_function_arguments(arguments, name)
    try:
        validator.validate(instance)
    except ValidationError as exc:
        raise TranslationError(
            f"upstream function {name} arguments violate the declared schema "
            f"at {exc.json_path} ({exc.validator})"
        ) from exc
    except re.error as exc:
        raise TranslationError(
            f"cannot validate upstream arguments for function {name}: unsupported schema regex"
        ) from exc
    except (Unresolvable, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise TranslationError(f"cannot validate upstream arguments for function {name}") from exc
