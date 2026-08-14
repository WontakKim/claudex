"""Gateway configuration loaded from settings.json and environment variables."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from claudex_gateway import paths

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
# either source; _read_settings_file rejects unknown keys so typos fail at
# boot instead of being silently ignored.
SETTINGS_KEYS: dict[str, str] = {
    "host": "CLAUDEX_HOST",
    "port": "CLAUDEX_PORT",
    "model_map": "CLAUDEX_MODEL_MAP",
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


def parse_custom_providers(value: object) -> dict[str, OpenAICompatibleProvider]:
    """Parse custom provider families from a settings object or env JSON."""
    example = (
        '{"openai_compatible": {"wrtn": {"wire_api": "responses", '
        '"base_url": "https://model.example/api/v1", "api_key": "secret"}}}'
    )
    valid_families = ("openai_compatible",)
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

    entries = value.get("openai_compatible", {})
    if not isinstance(entries, dict):
        raise ConfigError(
            "custom providers family 'openai_compatible' must be a JSON object "
            f"mapping provider names, got {type(entries).__name__}; example: {example}"
        )

    required_fields = {"wire_api", "base_url", "api_key"}
    providers: dict[str, OpenAICompatibleProvider] = {}
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

        wire_api = entry["wire_api"]
        if wire_api == "chat":
            raise ConfigError(
                f"custom provider {name!r} uses wire_api 'chat'; chat completions "
                "upstreams are not supported and only 'responses' is valid; "
                f"example: {example}"
            )
        if wire_api != "responses":
            raise ConfigError(
                f"custom provider {name!r} wire_api must be exactly 'responses', "
                f"got {wire_api!r}; example: {example}"
            )

        raw_base_url = entry["base_url"]
        if not isinstance(raw_base_url, str) or not raw_base_url.strip():
            found = (
                "empty string"
                if isinstance(raw_base_url, str)
                else type(raw_base_url).__name__
            )
            raise ConfigError(
                f"custom provider {name!r} base_url must be a non-empty URL string, "
                f"got {found}; example: 'https://model.example/api/v1'"
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
        is_loopback_http = parsed_url.scheme == "http" and _is_loopback_host(
            base_url_host
        )
        if not is_secure and not is_loopback_http:
            raise ConfigError(
                f"custom provider {name!r} base_url uses scheme "
                f"{parsed_url.scheme!r} for host {base_url_host!r}; https is required "
                "except for http loopback URLs, e.g. 'http://127.0.0.1:8080/v1'"
            )

        api_key = entry["api_key"]
        if not isinstance(api_key, str) or not api_key.strip():
            found = (
                "empty string"
                if isinstance(api_key, str)
                else type(api_key).__name__
            )
            raise ConfigError(
                f"custom provider {name!r} api_key must be a non-empty string, got "
                f"{found}; example: 'secret'"
            )

        providers[name] = OpenAICompatibleProvider(
            wire_api=wire_api,
            base_url=base_url,
            api_key=api_key,
        )
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


def _default_kimi_code_home() -> Path:
    return Path.home() / ".kimi-code"


@dataclass(frozen=True)
class GatewayConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # Maps a Claude model name (exact or substring, e.g. "haiku") to a target
    # model prefixed with its provider ("codex:gpt-5.6-luna", "kimi:k2.5").
    # Unmapped models are relayed verbatim to Anthropic, so there is no
    # default target — the map alone decides what runs where.
    model_map: dict[str, str] = field(default_factory=dict)
    custom_providers: dict[str, OpenAICompatibleProvider] = field(default_factory=dict)
    # When set, overrides the reasoning effort derived from the Claude request.
    reasoning_effort_override: str | None = None
    # When set to "fast", opts supported Codex models into the Fast tier.
    codex_service_tier: str | None = None
    codex_home: Path = field(default_factory=lambda: Path.home() / ".codex")
    # Where the Grok CLI login (`grok login`) lives; mirrors the CLI's own
    # GROK_HOME. auth.json sits directly inside.
    grok_home: Path = field(default_factory=lambda: Path.home() / ".grok")
    # Where the Kimi Code CLI login (`kimi login`) lives; mirrors the CLI's
    # own KIMI_CODE_HOME. credentials/kimi-code.json sits inside.
    kimi_code_home: Path = field(default_factory=_default_kimi_code_home)
    local_token: str | None = None
    log_level: str = "info"
    # Raw "claude:<canonical-model-id>" value naming the Claude model to
    # reroute oversized compaction requests to. None means the feature is
    # disabled — there is no separate "enabled" flag.
    compaction_model: str | None = None
    # Canonical id of the registered Claude account that serves Anthropic
    # passthrough traffic. None means passthrough forwards the client's own
    # credentials untouched — there is no separate "enabled" flag.
    claude_account_id: str | None = None
    # Multi-account routing mode for the managed relay. "disabled" (the
    # default) serves with the single configured account and relays rate
    # limits verbatim; "fallback" fails over across registered accounts in
    # order; "balanced" spreads sessions across the registered pool by
    # weighted HRW, served through a ClaudeBalancedRuntime.
    claude_account_routing_mode: str = "disabled"
    # Whether balanced routing includes the local Claude login in its pool.
    claude_account_include_local_login: bool = True
    # Where settings are read from and where runtime changes are persisted.
    settings_file: Path = field(default_factory=paths.settings_file)

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Build from environment variables only, ignoring any settings file."""
        return cls._from_sources({}, paths.settings_file())

    @classmethod
    def load(cls, settings_file: Path | None = None) -> "GatewayConfig":
        """Build from the settings file overlaid by environment variables.

        A variable set in the environment (even to an empty string) wins over
        its settings key; a variable present in neither source uses the
        default. The file defaults to ~/.claudex/settings.json and is
        optional — a missing file behaves like an empty one.
        """
        if settings_file is None:
            settings_file = paths.settings_file()
        return cls._from_sources(_read_settings_file(settings_file), settings_file)

    @classmethod
    def _from_sources(
        cls, settings: dict[str, object], settings_file: Path
    ) -> "GatewayConfig":
        value, label = _resolve("port", settings)
        if value is None:
            port = DEFAULT_PORT
        elif isinstance(value, str):
            try:
                port = int(value)
            except ValueError as exc:
                raise ConfigError(f"{label} must be an integer, got {value!r}") from exc
        elif isinstance(value, int) and not isinstance(value, bool):
            port = value
        else:
            raise ConfigError(f"{label} must be an integer, got {value!r}")
        if not 1 <= port <= 65535:
            raise ConfigError(f"{label} must be between 1 and 65535, got {port}")

        value, label = _resolve("custom_providers", settings)
        if value is None:
            custom_providers = {}
        else:
            try:
                custom_providers = parse_custom_providers(value)
            except ConfigError as exc:
                raise ConfigError(f"{label}: {exc}") from exc
        route_providers = (*BUILTIN_ROUTE_PROVIDERS, *custom_providers)
        model_map = _map_setting(
            "model_map",
            settings,
            '{"haiku": "codex:gpt-5.6-luna"}',
            route_providers,
        )

        value, label = _resolve("reasoning_effort", settings)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{label} must be a string, got {value!r}")
        effort = value or None
        if effort is not None and effort not in VALID_REASONING_EFFORTS:
            raise ConfigError(
                f"{label} must be one of {', '.join(VALID_REASONING_EFFORTS)}, "
                f"got {effort!r}"
            )

        value, label = _resolve("codex.service_tier", settings)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{label} must be a string, got {value!r}")
        codex_service_tier = value or None
        if (
            codex_service_tier is not None
            and codex_service_tier not in VALID_CODEX_SERVICE_TIERS
        ):
            raise ConfigError(
                f"{label} must be one of {', '.join(VALID_CODEX_SERVICE_TIERS)}, "
                f"got {codex_service_tier!r}"
            )

        codex_home = _path_setting("codex_home", settings, Path.home() / ".codex")
        grok_home = _path_setting("grok_home", settings, Path.home() / ".grok")
        kimi_code_home = _path_setting(
            "kimi_code_home", settings, _default_kimi_code_home()
        )

        value, label = _resolve("host", settings)
        if value is None:
            host = DEFAULT_HOST
        elif isinstance(value, str):
            host = value.strip()
        else:
            raise ConfigError(f"{label} must be a string, got {value!r}")
        if not host:
            raise ConfigError(f"{label} must not be empty")

        value, label = _resolve("local_token", settings)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{label} must be a string, got {value!r}")
        local_token = value or None
        if not _is_loopback_host(host) and local_token is None:
            raise ConfigError(
                "CLAUDEX_LOCAL_TOKEN is required when CLAUDEX_HOST is not loopback"
            )

        value, label = _resolve("log_level", settings)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{label} must be a string, got {value!r}")
        log_level = (value or "info").strip().lower() or "info"
        if log_level not in VALID_LOG_LEVELS:
            raise ConfigError(
                f"{label} must be one of {', '.join(VALID_LOG_LEVELS)}, got {log_level!r}"
            )

        value, label = _resolve("compaction.model", settings)
        env_name = SETTINGS_KEYS["compaction.model"]
        if value is None:
            if label == env_name:
                compaction_model = None
            else:
                # The key is present in the settings file (e.g. JSON null) —
                # that must not be conflated with "not configured".
                raise ConfigError(f"{label} must be a string, got {value!r}")
        elif not isinstance(value, str):
            raise ConfigError(f"{label} must be a string, got {value!r}")
        else:
            compaction_model = value or None
            if compaction_model is not None:
                try:
                    parse_compaction_model(compaction_model)
                except ConfigError as exc:
                    raise ConfigError(f"{label}: {exc}") from exc

        value, label = _resolve("claude_account.id", settings)
        env_name = SETTINGS_KEYS["claude_account.id"]
        if value is None:
            if label == env_name:
                claude_account_id = None
            else:
                # The key is present in the settings file (e.g. JSON null) —
                # that must not be conflated with "not configured".
                raise ConfigError(f"{label} must be a string, got {value!r}")
        elif not isinstance(value, str):
            raise ConfigError(f"{label} must be a string, got {value!r}")
        else:
            claude_account_id = value or None
            if claude_account_id is not None:
                try:
                    parse_claude_account_id(claude_account_id)
                except ConfigError as exc:
                    raise ConfigError(f"{label}: {exc}") from exc

        value, label = _resolve("claude_account.routing", settings)
        env_name = SETTINGS_KEYS["claude_account.routing"]
        if value is None:
            if label == env_name:
                claude_account_routing_mode = "disabled"
                claude_account_include_local_login = True
            else:
                # The key is present in the settings file (e.g. JSON null) —
                # that must not be conflated with "not configured".
                raise ConfigError(f"{label} must be a JSON object, got {value!r}")
        else:
            try:
                routing_policy = parse_claude_account_routing(value)
            except ConfigError as exc:
                raise ConfigError(f"{label}: {exc}") from exc
            claude_account_routing_mode = routing_policy.mode
            claude_account_include_local_login = routing_policy.include_local_login

        return cls(
            host=host,
            port=port,
            model_map=model_map,
            custom_providers=custom_providers,
            reasoning_effort_override=effort,
            codex_service_tier=codex_service_tier,
            codex_home=codex_home,
            grok_home=grok_home,
            kimi_code_home=kimi_code_home,
            local_token=local_token,
            log_level=log_level,
            compaction_model=compaction_model,
            claude_account_id=claude_account_id,
            claude_account_routing_mode=claude_account_routing_mode,
            claude_account_include_local_login=claude_account_include_local_login,
            settings_file=settings_file,
        )

    @property
    def route_providers(self) -> tuple[str, ...]:
        """Return every provider prefix available to model_map routes."""
        return (*BUILTIN_ROUTE_PROVIDERS, *self.custom_providers)

    def mapped_route(self, claude_model: str | None) -> RouteTarget | None:
        """Return the parsed route for a Claude model, or None when unmapped."""
        target = _mapped_model(claude_model, self.model_map)
        return (
            parse_route_target(target, self.route_providers)
            if target is not None
            else None
        )

    def maps_to_provider(self, provider: str) -> bool:
        """Report whether any model_map value routes to the given provider."""
        return any(
            parse_route_target(value, self.route_providers).provider == provider
            for value in self.model_map.values()
        )



def _read_settings_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read settings file {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"settings file {path} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"settings file {path} must contain a JSON object")
    group_names = {
        key.split(".", 1)[0] for key in SETTINGS_KEYS if "." in key
    }
    unknown = sorted(set(parsed) - set(SETTINGS_KEYS) - group_names)
    if unknown:
        raise ConfigError(
            f"settings file {path} has unknown keys: {', '.join(unknown)}; "
            f"valid keys: {', '.join(SETTINGS_KEYS)}"
        )

    settings = {
        key: value for key, value in parsed.items() if key in SETTINGS_KEYS
    }
    for group_name, group_value in parsed.items():
        if group_name in SETTINGS_KEYS:
            continue
        valid_group_keys = sorted(
            key for key in SETTINGS_KEYS if key.startswith(f"{group_name}.")
        )
        if not isinstance(group_value, dict):
            raise ConfigError(
                f'settings file {path} key "{group_name}" must contain a JSON object; '
                f"valid keys: {', '.join(valid_group_keys)}"
            )
        flattened = {
            f"{group_name}.{subkey}": value
            for subkey, value in group_value.items()
        }
        unknown_group_keys = sorted(set(flattened) - set(valid_group_keys))
        if unknown_group_keys:
            raise ConfigError(
                f"settings file {path} has unknown keys: "
                f"{', '.join(unknown_group_keys)}; "
                f"valid keys: {', '.join(valid_group_keys)}"
            )
        duplicated = sorted(set(flattened) & set(settings))
        if duplicated:
            raise ConfigError(
                f"settings file {path} has duplicate settings keys: "
                f"{', '.join(duplicated)}"
            )
        settings.update(flattened)
    return settings


def update_settings_file(
    path: Path,
    updates: dict[str, object],
    deletions: tuple[str, ...] = (),
) -> None:
    """Merge updates into the settings file, creating it if missing.

    The existing file is validated first so a corrupt or unknown-key file
    fails loudly instead of being silently overwritten, and the write is
    atomic so a crash cannot leave a half-written file behind.

    `deletions` removes keys from the settings dict before it is written, so
    a disabled feature is represented by the key's absence rather than a
    persisted JSON `null`. Deleting a key that is not present in the file is
    a no-op, not an error.
    """
    unknown = sorted(set(updates) - set(SETTINGS_KEYS))
    if unknown:
        raise ConfigError(f"cannot persist unknown settings keys: {', '.join(unknown)}")
    unknown_deletions = sorted(set(deletions) - set(SETTINGS_KEYS))
    if unknown_deletions:
        raise ConfigError(
            f"cannot delete unknown settings keys: {', '.join(unknown_deletions)}"
        )
    settings = _read_settings_file(path)
    settings.update(updates)
    for key in deletions:
        settings.pop(key, None)

    rendered: dict[str, object] = {}
    for key, value in settings.items():
        group_name, separator, subkey = key.partition(".")
        if not separator:
            rendered[key] = value
            continue
        group = rendered.setdefault(group_name, {})
        if not isinstance(group, dict):
            raise ConfigError(
                f"cannot persist settings group {group_name!r} because it conflicts "
                "with a top-level settings key"
            )
        group[subkey] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    staging_file = path.with_name(path.name + ".tmp")
    staging_file.write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(staging_file, path)


def _resolve(key: str, settings: dict[str, object]) -> tuple[object | None, str]:
    """Resolve one variable to (value, source label for error messages).

    The environment wins whenever the variable is set — even to an empty
    string, preserving env-only semantics such as CLAUDEX_MODEL_MAP='' meaning
    "no mapping". Returns (None, env name) when neither source has the key.
    """
    env_name = SETTINGS_KEYS[key]
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value, env_name
    if key in settings:
        return settings[key], f'settings.json key "{key}"'
    return None, env_name


def _map_setting(
    key: str,
    settings: dict[str, object],
    example: str,
    known_providers: Sequence[str],
) -> dict[str, str]:
    value, label = _resolve(key, settings)
    if value is None:
        return {}
    if isinstance(value, str):
        return _parse_model_map(label, value, example, known_providers)
    if not isinstance(value, dict):
        raise ConfigError(
            f"{label} must be a JSON object mapping model names, e.g. {example}"
        )
    return validate_model_map(label, value, known_providers)


def _path_setting(key: str, settings: dict[str, object], default: Path) -> Path:
    value, label = _resolve(key, settings)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path string, got {value!r}")
    return Path(value).expanduser()


def _parse_model_map(
    variable: str,
    raw: str,
    example: str,
    known_providers: Sequence[str],
) -> dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{variable} must be a JSON object mapping model names, e.g. {example}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{variable} must be a JSON object of non-empty strings")
    return validate_model_map(variable, parsed, known_providers)


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


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _mapped_model(requested: str | None, model_map: dict[str, str]) -> str | None:
    if not requested:
        return None
    if requested in model_map:
        return model_map[requested]
    # Among substring matches the longest (most specific) key wins, so
    # "claude-haiku" beats a catch-all "claude" regardless of map order.
    best_key = max(
        (key for key in model_map if key in requested), key=len, default=None
    )
    return model_map[best_key] if best_key is not None else None
