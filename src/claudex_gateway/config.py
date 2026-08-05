"""Gateway configuration loaded from settings.json and environment variables."""

from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from claudex_gateway import paths

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

# Providers a model_map value may target. Every value must name one via a
# "provider:" prefix — bare model names are rejected so a map entry always
# says which backend serves it.
KNOWN_ROUTE_PROVIDERS = ("codex", "kimi", "grok")

VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

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
    "reasoning_effort": "CLAUDEX_REASONING_EFFORT",
    "codex_home": "CODEX_HOME",
    "grok_home": "GROK_HOME",
    "kimi_code_home": "KIMI_CODE_HOME",
    "local_token": "CLAUDEX_LOCAL_TOKEN",
    "log_level": "CLAUDEX_LOG_LEVEL",
}


class ConfigError(Exception):
    """Raised when the gateway configuration is invalid."""


@dataclass(frozen=True)
class RouteTarget:
    """A parsed model_map value: which provider serves which upstream model."""

    provider: str  # one of KNOWN_ROUTE_PROVIDERS
    model: str


def parse_route_target(value: str) -> RouteTarget:
    """Parse a model_map value like "codex:gpt-5.6-luna" or "kimi:k2.5".

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
            f"(known providers: {', '.join(KNOWN_ROUTE_PROVIDERS)})"
        )
    prefix, _, model = value.partition(":")
    prefix = prefix.strip()
    model = model.strip()
    if prefix not in KNOWN_ROUTE_PROVIDERS:
        raise ConfigError(
            f"unknown provider prefix {prefix!r} in model target {value!r}; "
            f"known providers: {', '.join(KNOWN_ROUTE_PROVIDERS)}"
        )
    if not model:
        raise ConfigError(f"model target {value!r} names no model after the provider prefix")
    return RouteTarget(provider=prefix, model=model)


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
    # When set, overrides the reasoning effort derived from the Claude request.
    reasoning_effort_override: str | None = None
    codex_home: Path = field(default_factory=lambda: Path.home() / ".codex")
    # Where the Grok CLI login (`grok login`) lives; mirrors the CLI's own
    # GROK_HOME. auth.json sits directly inside.
    grok_home: Path = field(default_factory=lambda: Path.home() / ".grok")
    # Where the Kimi Code CLI login (`kimi login`) lives; mirrors the CLI's
    # own KIMI_CODE_HOME. credentials/kimi-code.json sits inside.
    kimi_code_home: Path = field(default_factory=_default_kimi_code_home)
    local_token: str | None = None
    log_level: str = "info"
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

        model_map = _map_setting("model_map", settings, '{"haiku": "codex:gpt-5.6-luna"}')

        value, label = _resolve("reasoning_effort", settings)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{label} must be a string, got {value!r}")
        effort = value or None
        if effort is not None and effort not in VALID_REASONING_EFFORTS:
            raise ConfigError(
                f"{label} must be one of {', '.join(VALID_REASONING_EFFORTS)}, "
                f"got {effort!r}"
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

        return cls(
            host=host,
            port=port,
            model_map=model_map,
            reasoning_effort_override=effort,
            codex_home=codex_home,
            grok_home=grok_home,
            kimi_code_home=kimi_code_home,
            local_token=local_token,
            log_level=log_level,
            settings_file=settings_file,
        )

    def mapped_route(self, claude_model: str | None) -> RouteTarget | None:
        """Return the parsed route for a Claude model, or None when unmapped."""
        target = _mapped_model(claude_model, self.model_map)
        return parse_route_target(target) if target is not None else None

    def maps_to_provider(self, provider: str) -> bool:
        """Report whether any model_map value routes to the given provider."""
        return any(
            parse_route_target(value).provider == provider
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
    unknown = sorted(set(parsed) - set(SETTINGS_KEYS))
    if unknown:
        raise ConfigError(
            f"settings file {path} has unknown keys: {', '.join(unknown)}; "
            f"valid keys: {', '.join(SETTINGS_KEYS)}"
        )
    return parsed


def update_settings_file(path: Path, updates: dict[str, object]) -> None:
    """Merge updates into the settings file, creating it if missing.

    The existing file is validated first so a corrupt or unknown-key file
    fails loudly instead of being silently overwritten, and the write is
    atomic so a crash cannot leave a half-written file behind.
    """
    unknown = sorted(set(updates) - set(SETTINGS_KEYS))
    if unknown:
        raise ConfigError(f"cannot persist unknown settings keys: {', '.join(unknown)}")
    settings = _read_settings_file(path)
    settings.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_file = path.with_name(path.name + ".tmp")
    staging_file.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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


def _map_setting(key: str, settings: dict[str, object], example: str) -> dict[str, str]:
    value, label = _resolve(key, settings)
    if value is None:
        return {}
    if isinstance(value, str):
        return _parse_model_map(label, value, example)
    if not isinstance(value, dict):
        raise ConfigError(
            f"{label} must be a JSON object mapping model names, e.g. {example}"
        )
    return validate_model_map(label, value)


def _path_setting(key: str, settings: dict[str, object], default: Path) -> Path:
    value, label = _resolve(key, settings)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path string, got {value!r}")
    return Path(value).expanduser()


def _parse_model_map(variable: str, raw: str, example: str) -> dict[str, str]:
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
    return validate_model_map(variable, parsed)


def validate_model_map(variable: str, parsed: dict[object, object]) -> dict[str, str]:
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
            parse_route_target(value)
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
