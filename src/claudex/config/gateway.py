"""Runtime gateway configuration and source overlays."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from claudex import paths

from .schema import (
    BUILTIN_ROUTE_PROVIDERS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    SETTINGS_KEYS,
    VALID_CODEX_SERVICE_TIERS,
    VALID_LOG_LEVELS,
    VALID_REASONING_EFFORTS,
    ConfigError,
    OpenAICompatibleProvider,
    RouteTarget,
    is_loopback_host,
    parse_claude_account_id,
    parse_claude_account_routing,
    parse_compaction_model,
    parse_custom_providers,
    parse_route_target,
    validate_model_map,
)
from .settings_io import read_settings_file


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
        return cls._from_sources(read_settings_file(settings_file), settings_file)

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
        if not is_loopback_host(host) and local_token is None:
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
