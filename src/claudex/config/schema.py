"""Configuration schema, parsing, and validation."""

from __future__ import annotations

import ipaddress
import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urlsplit

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

# Built-in providers a model_map value may target. Custom providers extend this
# namespace at config load time.
BUILTIN_ROUTE_PROVIDERS = ("codex", "kimi", "grok")
RESERVED_PROVIDER_NAMES = ("codex", "kimi", "grok", "claude", "anthropic")

VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
VALID_CODEX_SERVICE_TIERS = ("fast",)

VALID_LOG_LEVELS = ("debug", "info", "warning", "error")

# Settings-file key for every supported variable: the environment variable
# name minus its CLAUDEX_ prefix, lowercased (the CLI-home variables —
# CODEX_HOME, GROK_HOME, KIMI_CODE_HOME — having no prefix, keep their full
# names). New configuration must be registered here to be loadable from
# either source; read_settings_file rejects unknown keys so typos fail at
# boot instead of being silently ignored.
SETTINGS_KEYS: dict[str, str] = {
    "host": "CLAUDEX_HOST",
    "port": "CLAUDEX_PORT",
    "model_map": "CLAUDEX_MODEL_MAP",
    "context_window_map": "CLAUDEX_CONTEXT_WINDOW_MAP",
    "custom_providers": "CLAUDEX_CUSTOM_PROVIDERS",
    "reasoning_effort": "CLAUDEX_REASONING_EFFORT",
    # Opt-in: which Codex service tier mapped requests use. Absence of the key
    # (or a resolved empty string) means the feature is disabled — there is no
    # separate "enabled" flag.
    "codex.service_tier": "CLAUDEX_CODEX_SERVICE_TIER",
    "codex_home": "CODEX_HOME",
    "grok_home": "GROK_HOME",
    "kimi_code_home": "KIMI_CODE_HOME",
    "local_token": "CLAUDEX_LOCAL_TOKEN",
    "log_level": "CLAUDEX_LOG_LEVEL",
    # Opt-in: which Claude model oversized compaction requests are rerouted
    # to. Absence of the key (or a resolved empty string) means the feature
    # is disabled — there is no separate "enabled" flag.
    "compaction.model": "CLAUDEX_COMPACTION_MODEL",
    # Opt-in: which registered Claude account serves Anthropic passthrough
    # traffic. Absence of the key (or a resolved empty string) means
    # passthrough forwards the client's own credentials untouched.
    "claude_account.id": "CLAUDEX_CLAUDE_ACCOUNT_ID",
    # Multi-account routing policy for the managed Anthropic relay, as a
    # JSON document like {"mode": "fallback"}. Absence of the key (or a
    # resolved empty env string) means "disabled" — single-account serving
    # with rate limits relayed verbatim. "fallback" fails over across the
    # registered pool in order; "balanced" spreads sessions across the pool
    # by weighted HRW (design v2) via ClaudeBalancedRuntime.
    "claude_account.routing": "CLAUDEX_CLAUDE_ACCOUNT_ROUTING",
}


class ConfigError(Exception):
    """Raised when the gateway configuration is invalid."""


@dataclass(frozen=True)
class RouteTarget:
    """A parsed model_map value: which provider serves which upstream model."""

    provider: str
    model: str


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """A static OpenAI Responses-compatible upstream configuration."""

    wire_api: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class AnthropicCompatibleProvider:
    """A static Anthropic Messages-compatible upstream configuration."""

    base_url: str
    api_key: str


_CustomProvider: TypeAlias = OpenAICompatibleProvider | AnthropicCompatibleProvider


def parse_route_target(value: str, known_providers: Sequence[str]) -> RouteTarget:
    """Parse a provider-prefixed model_map value against registered providers.

    The provider prefix is mandatory: a bare model name is rejected rather
    than defaulted to Codex, so every map entry says which backend serves it.
    Only the first colon separates the provider prefix, so upstream model
    names containing colons remain expressible. Unknown prefixes are rejected
    rather than treated as literal model names — a typo like "kim:k2.5" must
    fail at boot, not surface as a baffling upstream 404.
    """
    if ":" not in value:
        raise ConfigError(
            f"model target {value!r} has no provider prefix; "
            f"prefix the serving provider, e.g. 'codex:{value}' "
            f"(known providers: {', '.join(known_providers)})"
        )
    prefix, _, model = value.partition(":")
    prefix = prefix.strip()
    model = model.strip()
    if prefix not in known_providers:
        raise ConfigError(
            f"unknown provider prefix {prefix!r} in model target {value!r}; "
            f"known providers: {', '.join(known_providers)}"
        )
    if not model:
        raise ConfigError(f"model target {value!r} names no model after the provider prefix")
    return RouteTarget(provider=prefix, model=model)


def parse_custom_providers(value: object) -> dict[str, _CustomProvider]:
    """Parse custom provider families from a settings object or env JSON."""
    example = (
        '{"openai_compatible": {"wrtn": {"wire_api": "responses", '
        '"base_url": "https://model.example/api/v1", "api_key": "secret"}}}'
    )
    family_fields = {
        "openai_compatible": {"wire_api", "base_url", "api_key"},
        "anthropic_compatible": {"base_url", "api_key"},
    }
    valid_families = tuple(family_fields)
    if isinstance(value, str):
        if not value:
            return {}
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                "custom providers contains invalid JSON; valid value is an object "
                f"grouped by family; example: {example}: {exc}"
            ) from exc
    if not isinstance(value, dict):
        raise ConfigError(
            "custom providers must be a JSON object grouped by family, got "
            f"{type(value).__name__}; valid families: {', '.join(valid_families)}; "
            f"example: {example}"
        )

    unknown_families = sorted(
        str(family) for family in value if family not in valid_families
    )
    if unknown_families:
        raise ConfigError(
            f"custom providers has unknown families: {', '.join(unknown_families)}; "
            f"valid families: {', '.join(valid_families)}; example: {example}"
        )

    providers: dict[str, _CustomProvider] = {}
    provider_families: dict[str, str] = {}
    for family, required_fields in family_fields.items():
        entries = value.get(family, {})
        if not isinstance(entries, dict):
            raise ConfigError(
                f"custom providers family {family!r} must be a JSON object "
                f"mapping provider names, got {type(entries).__name__}; "
                f"example: {example}"
            )

        for name, entry in entries.items():
            if not isinstance(name, str) or re.fullmatch(
                r"[a-z][a-z0-9-]{0,31}", name
            ) is None:
                raise ConfigError(
                    f"custom provider name {name!r} is invalid; valid names match "
                    "^[a-z][a-z0-9-]{0,31}$; example: 'wrtn'"
                )
            if name in RESERVED_PROVIDER_NAMES:
                raise ConfigError(
                    f"custom provider name {name!r} is reserved; reserved names: "
                    f"{', '.join(RESERVED_PROVIDER_NAMES)}; example: 'wrtn'"
                )
            if name in providers:
                raise ConfigError(
                    f"custom provider name {name!r} is configured in both "
                    f"{provider_families[name]!r} and {family!r}; provider names "
                    "must be unique across families"
                )
            if not isinstance(entry, dict):
                raise ConfigError(
                    f"custom provider {name!r} must be a JSON object, got "
                    f"{type(entry).__name__}; example entry: {example}"
                )

            unknown_fields = sorted(
                str(field) for field in entry if field not in required_fields
            )
            if unknown_fields:
                raise ConfigError(
                    f"custom provider {name!r} has unknown keys: "
                    f"{', '.join(unknown_fields)}; valid keys: "
                    f"{', '.join(sorted(required_fields))}; example: {example}"
                )
            missing_fields = sorted(required_fields - set(entry))
            if missing_fields:
                raise ConfigError(
                    f"custom provider {name!r} is missing required keys: "
                    f"{', '.join(missing_fields)}; required keys: "
                    f"{', '.join(sorted(required_fields))}; example: {example}"
                )

            if family == "openai_compatible":
                wire_api = entry["wire_api"]
                if wire_api == "chat":
                    raise ConfigError(
                        f"custom provider {name!r} uses wire_api 'chat'; chat "
                        "completions upstreams are not supported and only "
                        f"'responses' is valid; example: {example}"
                    )
                if wire_api != "responses":
                    raise ConfigError(
                        f"custom provider {name!r} wire_api must be exactly "
                        f"'responses', got {wire_api!r}; example: {example}"
                    )

            raw_base_url = entry["base_url"]
            if not isinstance(raw_base_url, str) or not raw_base_url.strip():
                found = (
                    "empty string"
                    if isinstance(raw_base_url, str)
                    else type(raw_base_url).__name__
                )
                raise ConfigError(
                    f"custom provider {name!r} base_url must be a non-empty URL "
                    f"string, got {found}; example: 'https://model.example/api/v1'"
                )
            base_url = raw_base_url.strip().rstrip("/")
            try:
                parsed_url = urlsplit(base_url)
                base_url_host = parsed_url.hostname
            except ValueError:
                parsed_url = None
                base_url_host = None
            if parsed_url is None or not parsed_url.scheme or base_url_host is None:
                raise ConfigError(
                    f"custom provider {name!r} base_url is not a valid absolute URL; "
                    "valid URLs use https, e.g. 'https://model.example/api/v1'"
                )
            is_secure = parsed_url.scheme == "https"
            is_loopback_http = parsed_url.scheme == "http" and is_loopback_host(
                base_url_host
            )
            if not is_secure and not is_loopback_http:
                raise ConfigError(
                    f"custom provider {name!r} base_url uses scheme "
                    f"{parsed_url.scheme!r} for host {base_url_host!r}; https is "
                    "required except for http loopback URLs, e.g. "
                    "'http://127.0.0.1:8080/v1'"
                )

            api_key = entry["api_key"]
            if not isinstance(api_key, str) or not api_key.strip():
                found = (
                    "empty string"
                    if isinstance(api_key, str)
                    else type(api_key).__name__
                )
                raise ConfigError(
                    f"custom provider {name!r} api_key must be a non-empty string, "
                    f"got {found}; example: 'secret'"
                )

            if family == "openai_compatible":
                providers[name] = OpenAICompatibleProvider(
                    wire_api=wire_api,
                    base_url=base_url,
                    api_key=api_key,
                )
            else:
                providers[name] = AnthropicCompatibleProvider(
                    base_url=base_url,
                    api_key=api_key,
                )
            provider_families[name] = family
    return providers


def parse_compaction_model(value: str) -> str:
    """Parse a compaction.model value like "claude:claude-opus-5".

    Only the "claude:" prefix is accepted. This is deliberately independent
    of parse_route_target/BUILTIN_ROUTE_PROVIDERS: compaction requests are
    rerouted to a Claude model, not to a model_map upstream, and "claude"
    must never become a valid model_map route provider. Returns the
    canonical model id with the prefix stripped.
    """
    prefix, sep, model_id = value.partition(":")
    if not sep or prefix != "claude":
        raise ConfigError(
            f"compaction model {value!r} must be prefixed with 'claude:', "
            f"e.g. 'claude:claude-opus-5'"
        )
    if not model_id or any(char.isspace() for char in model_id):
        raise ConfigError(
            f"compaction model {value!r} names no valid model after the "
            f"'claude:' prefix"
        )
    return model_id


def parse_claude_account_id(value: str) -> str:
    """Validate a claude_account.id value: a canonical account UUID.

    Only the id form is accepted here — email resolution happens in the CLI,
    where the registry is at hand. Registry membership is deliberately not
    checked at boot: accounts can be added or removed by the CLI while the
    daemon runs, so the serving path re-resolves the id on every request.
    """
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        parsed = None
    if parsed is None or str(parsed) != value:
        raise ConfigError(
            f"claude account id {value!r} is not a canonical account UUID; "
            "use `claudex-gateway account list` to see registered ids"
        )
    return value


VALID_CLAUDE_ACCOUNT_ROUTING_MODES = ("disabled", "fallback", "balanced")


@dataclass(frozen=True)
class ClaudeAccountRoutingPolicy:
    """A parsed claude_account.routing policy document."""

    mode: str
    include_local_login: bool = True


def parse_claude_account_routing(value: object) -> ClaudeAccountRoutingPolicy:
    """Parse a claude_account.routing value into its routing policy.

    The setting is a policy document — {"mode": "fallback"} — so a mode can
    carry its own config without renaming the key. The settings file holds the
    document itself; the environment variable holds it JSON-encoded, with an
    empty string meaning disabled like the other opt-in settings. "balanced"
    is the weighted-HRW pool-wide routing mode (design v2), constructed and
    persisted the same way as "fallback".
    """
    if isinstance(value, str):
        if not value:
            return ClaudeAccountRoutingPolicy(mode="disabled")
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"claude account routing {value!r} is not valid JSON: {exc}"
            ) from exc
    if not isinstance(value, dict):
        raise ConfigError(
            "claude account routing must be a JSON object like "
            f'{{"mode": "fallback"}}, got {value!r}'
        )
    mode = value.get("mode")
    include_local_login = value.get("include_local_login", True)
    unknown = sorted(set(value) - {"mode", "include_local_login"})
    if unknown:
        raise ConfigError(
            f"claude account routing has unknown keys: {', '.join(map(str, unknown))}; "
            "valid keys: mode, include_local_login"
        )
    if mode not in VALID_CLAUDE_ACCOUNT_ROUTING_MODES:
        raise ConfigError(
            "claude account routing mode must be one of "
            f"{', '.join(VALID_CLAUDE_ACCOUNT_ROUTING_MODES)}, got {mode!r}"
        )
    if not isinstance(include_local_login, bool):
        raise ConfigError(
            "claude account routing include_local_login must be a JSON boolean, "
            f"got {include_local_login!r}"
        )
    return ClaudeAccountRoutingPolicy(
        mode=mode, include_local_login=include_local_login
    )


def validate_model_map(
    variable: str,
    parsed: dict[object, object],
    known_providers: Sequence[str],
) -> dict[str, str]:
    if not all(
        isinstance(key, str)
        and bool(key.strip())
        and isinstance(value, str)
        and bool(value.strip())
        for key, value in parsed.items()
    ):
        raise ConfigError(f"{variable} must be a JSON object of non-empty strings")
    model_map = {key.strip(): value.strip() for key, value in parsed.items()}
    for value in model_map.values():
        try:
            parse_route_target(value, known_providers)
        except ConfigError as exc:
            raise ConfigError(f"{variable}: {exc}") from exc
    return model_map


def validate_context_window_map(
    variable: str,
    parsed: dict[object, object],
    known_providers: Sequence[str],
) -> dict[str, int]:
    if not all(
        isinstance(key, str)
        and bool(key.strip())
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        for key, value in parsed.items()
    ):
        raise ConfigError(
            f"{variable} must be a JSON object mapping non-empty model targets "
            "to positive integers"
        )
    context_window_map = {key.strip(): value for key, value in parsed.items()}
    for key in context_window_map:
        try:
            parse_route_target(key, known_providers)
        except ConfigError as exc:
            raise ConfigError(f"{variable}: {exc}") from exc
    return context_window_map


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
