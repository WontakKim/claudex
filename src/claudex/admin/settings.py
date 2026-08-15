"""Gateway and Claude serving settings admin handlers."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from claudex import paths, server_support
from claudex.admin.common import (
    _admin_guard,
    _read_json_object,
    _require_json_content_type,
)
from claudex.balanced.runtime import BalancedPrepareError, ClaudeBalancedRuntime
from claudex.claude.accounts import AccountRegistryError, list_accounts, load_registry
from claudex.config import (
    SETTINGS_KEYS,
    VALID_CLAUDE_ACCOUNT_ROUTING_MODES,
    VALID_CODEX_SERVICE_TIERS,
    VALID_LOG_LEVELS,
    ConfigError,
    GatewayConfig,
    parse_claude_account_id,
    parse_compaction_model,
    update_settings_file,
    validate_model_map,
)

logger = logging.getLogger("claudex.server")


_ADMIN_MAP_KEYS = ("model_map",)


def _mapping_payload(config: GatewayConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_map": config.model_map,
        # The dashboard renders the board view-only while the corresponding
        # environment variable overrides the settings file.
        "env_locked": {
            key: SETTINGS_KEYS[key] if os.environ.get(SETTINGS_KEYS[key]) is not None else None
            for key in _ADMIN_MAP_KEYS
        },
        "codex_home": str(config.codex_home),
        "grok_home": str(config.grok_home),
        "kimi_code_home": str(config.kimi_code_home),
    }
    if config.custom_providers:
        payload["custom_providers"] = [
            {
                "name": name,
                "wire_api": provider.wire_api,
                "base_url": provider.base_url,
            }
            for name, provider in config.custom_providers.items()
        ]
    return payload


async def _handle_admin_mapping_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_mapping_payload(request.app.state.config))


async def _handle_admin_mapping_put(request: Request) -> JSONResponse:
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None:
        return error
    unknown = sorted(set(body) - set(_ADMIN_MAP_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_ADMIN_MAP_KEYS)}",
            ),
            status_code=400,
        )
    if not body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"provide at least one of: {', '.join(_ADMIN_MAP_KEYS)}",
            ),
            status_code=400,
        )

    updates: dict[str, dict[str, str]] = {}
    for key in _ADMIN_MAP_KEYS:
        if key not in body:
            continue
        value = body[key]
        if not isinstance(value, dict):
            return JSONResponse(
                server_support._openai_error_body(
                    "invalid_request_error",
                    f"{key} must be a JSON object mapping model names",
                ),
                status_code=400,
            )
        try:
            updates[key] = validate_model_map(
                key,
                value,
                known_providers=request.app.state.config.route_providers,
            )
        except ConfigError as exc:
            return JSONResponse(
                server_support._openai_error_body("invalid_request_error", str(exc)),
                status_code=400,
            )
        # An environment variable outranks settings.json at every boot, so a
        # persisted change would silently vanish on restart — refuse instead.
        env_name = SETTINGS_KEYS[key]
        if os.environ.get(env_name) is not None:
            return JSONResponse(
                server_support._openai_error_body(
                    "invalid_request_error",
                    f"{env_name} is set in the gateway's environment and overrides "
                    f"{key}; unset it to manage the mapping at runtime",
                ),
                status_code=409,
            )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(config.settings_file, dict(updates))
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        # Swap only after the file write succeeded, atomically and only for
        # runtime-safe fields; in-flight requests keep their config snapshot.
        new_config = replace(config, **updates)
        request.app.state.config = new_config
    return JSONResponse(_mapping_payload(new_config))


_LOG_LEVEL_LOGGER_NAMES = ("", "uvicorn", "uvicorn.access", "uvicorn.error")


def _apply_log_level(log_level: str) -> None:
    level = getattr(logging, log_level.upper())
    for name in _LOG_LEVEL_LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)


def _log_level_payload(config: GatewayConfig) -> dict[str, Any]:
    env_name = SETTINGS_KEYS["log_level"]
    return {
        "log_level": config.log_level,
        "choices": list(VALID_LOG_LEVELS),
        "env_locked": env_name if os.environ.get(env_name) is not None else None,
    }


async def _handle_admin_log_level_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_log_level_payload(request.app.state.config))


async def _handle_admin_log_level_put(request: Request) -> JSONResponse:
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    value = body.get("log_level")
    if not isinstance(value, str) or value.strip().lower() not in VALID_LOG_LEVELS:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"log_level must be one of: {', '.join(VALID_LOG_LEVELS)}",
            ),
            status_code=400,
        )
    value = value.strip().lower()
    env_name = SETTINGS_KEYS["log_level"]
    if os.environ.get(env_name) is not None:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"{env_name} is set in the gateway's environment and overrides "
                f"log_level; unset it to manage the level at runtime",
            ),
            status_code=409,
        )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(config.settings_file, {"log_level": value})
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                server_support._openai_error_body("server_error", f"could not persist settings: {exc}"),
                status_code=500,
            )
        new_config = replace(config, log_level=value)
        request.app.state.config = new_config
    _apply_log_level(value)
    logger.info("log level set to %s", value)
    return JSONResponse(_log_level_payload(new_config))


_COMPACTION_KEYS = ("model",)


def _compaction_payload(config: GatewayConfig, app_state: Any) -> dict[str, Any]:
    """Pinned {model, env_locked, last_reroute} envelope for /admin/compaction.

    `model` is the raw "claude:<id>" value (or None), so a GET/PUT round-trip
    is loss-free. `env_locked` is a plain boolean — true whenever
    CLAUDEX_COMPACTION_MODEL is present in the environment, including an
    empty value, mirroring _resolve's own "present even if empty" env
    precedence. `last_reroute` is exactly the pinned seven-key diagnostics
    record from _assign_compaction_reroute (or None before any reroute has
    been attempted); its internal sequence counter is never part of the
    record and so never serialized here.
    """
    env_name = SETTINGS_KEYS["compaction.model"]
    return {
        "model": config.compaction_model,
        "env_locked": os.environ.get(env_name) is not None,
        "last_reroute": app_state.compaction_last_reroute,
    }


async def _handle_admin_compaction_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_compaction_payload(request.app.state.config, request.app.state))


async def _handle_admin_compaction_put(request: Request) -> JSONResponse:
    """Set or clear the compaction reroute target.

    Only a successful PUT returns the state envelope; the 409 (env-locked)
    path uses the existing admin error envelope, so a caller needing current
    state issues a GET afterward.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_COMPACTION_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_COMPACTION_KEYS)}",
            ),
            status_code=400,
        )
    if "model" not in body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error", "provide 'model' (a string or null)"
            ),
            status_code=400,
        )

    value = body["model"]
    if value is not None:
        if not isinstance(value, str):
            return JSONResponse(
                server_support._openai_error_body(
                    "invalid_request_error", "model must be a string or null"
                ),
                status_code=400,
            )
        try:
            parse_compaction_model(value)
        except ConfigError as exc:
            return JSONResponse(
                server_support._openai_error_body("invalid_request_error", str(exc)),
                status_code=400,
            )

    # An environment variable outranks settings.json at every boot (even set
    # to an empty string), so a persisted change would silently vanish on
    # restart — refuse before the lock or any file/config read.
    env_name = SETTINGS_KEYS["compaction.model"]
    if os.environ.get(env_name) is not None:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"{env_name} is set in the gateway's environment and overrides "
                f"compaction.model; unset it to manage the setting at runtime",
            ),
            status_code=409,
        )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            if value is None:
                # A disabled setting is represented by the key's absence, so
                # a JSON null is never persisted.
                update_settings_file(
                    config.settings_file, {}, deletions=("compaction.model",)
                )
            else:
                update_settings_file(config.settings_file, {"compaction.model": value})
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        # Swap only after the file write succeeded, atomically; in-flight
        # requests keep their config snapshot.
        new_config = replace(config, compaction_model=value)
        request.app.state.config = new_config
    return JSONResponse(_compaction_payload(new_config, request.app.state))


_CODEX_KEYS = ("service_tier",)


def _codex_payload(config: GatewayConfig) -> dict[str, Any]:
    """Pinned {service_tier, env_locked} envelope for /admin/settings/codex.

    `service_tier` is the raw supported tier (or None), so a GET/PUT
    round-trip is loss-free. `env_locked` mirrors the compaction envelope:
    true whenever CLAUDEX_CODEX_SERVICE_TIER is present in the environment,
    including an empty value.
    """
    env_name = SETTINGS_KEYS["codex.service_tier"]
    return {
        "service_tier": config.codex_service_tier,
        "env_locked": os.environ.get(env_name) is not None,
    }


async def _handle_admin_codex_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_codex_payload(request.app.state.config))


async def _handle_admin_codex_put(request: Request) -> JSONResponse:
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CODEX_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_CODEX_KEYS)}",
            ),
            status_code=400,
        )
    if "service_tier" not in body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error", "provide 'service_tier' ('fast' or null)"
            ),
            status_code=400,
        )

    value = body["service_tier"]
    if value is not None and (
        not isinstance(value, str) or value not in VALID_CODEX_SERVICE_TIERS
    ):
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "service_tier must be null or one of: "
                f"{', '.join(VALID_CODEX_SERVICE_TIERS)}",
            ),
            status_code=400,
        )

    env_name = SETTINGS_KEYS["codex.service_tier"]
    if os.environ.get(env_name) is not None:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"{env_name} is set in the gateway's environment and overrides "
                f"codex.service_tier; unset it to manage the setting at runtime",
            ),
            status_code=409,
        )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            if value is None:
                update_settings_file(
                    config.settings_file, {}, deletions=("codex.service_tier",)
                )
            else:
                update_settings_file(config.settings_file, {"codex.service_tier": value})
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        new_config = replace(config, codex_service_tier=value)
        request.app.state.config = new_config
    return JSONResponse(_codex_payload(new_config))


_CLAUDE_ACCOUNT_KEYS = ("account_id",)


def _claude_account_payload(config: GatewayConfig) -> dict[str, Any]:
    """Pinned {account_id, env_locked} envelope for claude pool/serving.

    `account_id` is the raw canonical uuid (or None), so a GET/PUT
    round-trip is loss-free. `env_locked` mirrors the compaction envelope:
    true whenever CLAUDEX_CLAUDE_ACCOUNT_ID is present in the environment,
    including an empty value.
    """
    env_name = SETTINGS_KEYS["claude_account.id"]
    return {
        "account_id": config.claude_account_id,
        "env_locked": os.environ.get(env_name) is not None,
    }


async def _handle_admin_claude_serving_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_claude_account_payload(request.app.state.config))


def _claude_account_env_locked() -> JSONResponse | None:
    """409 when CLAUDEX_CLAUDE_ACCOUNT_ID overrides runtime writes.

    An environment variable outranks settings.json at every boot (even set
    to an empty string), so a persisted change would silently vanish on
    restart — refuse before the lock or any file/config read.
    """
    env_name = SETTINGS_KEYS["claude_account.id"]
    if os.environ.get(env_name) is None:
        return None
    return JSONResponse(
        server_support._openai_error_body(
            "invalid_request_error",
            f"{env_name} is set in the gateway's environment and overrides "
            f"claude_account.id; unset it to manage the setting at runtime",
        ),
        status_code=409,
    )


async def _handle_admin_claude_serving_put(request: Request) -> JSONResponse:
    """Pin the registered account serving Anthropic passthrough.

    The pin is cleared with DELETE, never with a null PUT — the two writes
    stay distinct so a partial payload can't silently disable serving.
    Only a successful write returns the state envelope; the 409
    (env-locked) path uses the existing admin error envelope, so a caller
    needing current state issues a GET afterward.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_ACCOUNT_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_CLAUDE_ACCOUNT_KEYS)}",
            ),
            status_code=400,
        )
    value = body.get("account_id")
    if not isinstance(value, str):
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "provide 'account_id' as a string; to clear the serving "
                "account, DELETE this endpoint instead",
            ),
            status_code=400,
        )
    try:
        parse_claude_account_id(value)
    except ConfigError as exc:
        return JSONResponse(
            server_support._openai_error_body("invalid_request_error", str(exc)),
            status_code=400,
        )
    # The env lock dooms every write, so it is checked before any registry
    # I/O — a registry hiccup must not turn the required 409 into a 500.
    denied = _claude_account_env_locked()
    if denied is not None:
        return denied

    async with request.app.state.admin_lock:
        # Selecting an unregistered account would turn every passthrough
        # request into a 503, so refuse it here where the mistake is cheap.
        # The membership check runs under the same lock that serializes the
        # accounts/{id} DELETE, so pinning and removal are linearizable
        # within this daemon (a concurrent CLI `account remove` can still
        # race from another process; the serve path re-resolves per request
        # and degrades to a loud 503 by design).
        try:
            records = load_registry()
        except AccountRegistryError as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"cannot read the claude account registry: {exc}"
                ),
                status_code=500,
            )
        if not any(record.id == value for record in records):
            return JSONResponse(
                server_support._openai_error_body(
                    "invalid_request_error",
                    f"no account registered with id {value}; "
                    "see `claudex-gateway account list`",
                ),
                status_code=400,
            )
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(config.settings_file, {"claude_account.id": value})
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        # Swap only after the file write succeeded, atomically; in-flight
        # requests keep their config snapshot.
        new_config = replace(config, claude_account_id=value)
        request.app.state.config = new_config
    return JSONResponse(_claude_account_payload(new_config))


async def _handle_admin_claude_serving_delete(request: Request) -> JSONResponse:
    """Clear the serving pin: passthrough forwards client credentials again.

    A disabled setting is represented by the key's absence, so a JSON null
    is never persisted. Clearing an already-clear pin is a no-op 200.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    denied = _claude_account_env_locked()
    if denied is not None:
        return denied

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(
                config.settings_file, {}, deletions=("claude_account.id",)
            )
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        new_config = replace(config, claude_account_id=None)
        request.app.state.config = new_config
    return JSONResponse(_claude_account_payload(new_config))


_CLAUDE_ROUTING_KEYS = ("mode",)


def _claude_routing_payload(config: GatewayConfig) -> dict[str, Any]:
    """Pinned {mode, env_locked} envelope for claude pool/routing."""
    env_name = SETTINGS_KEYS["claude_account.routing"]
    return {
        "mode": config.claude_account_routing_mode,
        "env_locked": os.environ.get(env_name) is not None,
    }


async def _handle_admin_claude_routing_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_claude_routing_payload(request.app.state.config))


def _persist_claude_routing_mode(config: GatewayConfig, mode: str) -> None:
    """Write `claude_account.routing` to `mode`'s on-disk representation.

    "disabled" is represented by the settings key's absence; every other
    mode ("fallback", "balanced") persists the policy document
    ({"mode": mode}), whose object form leaves room for a mode to carry its
    own config block without renaming the key.
    """
    if mode == "disabled":
        update_settings_file(config.settings_file, {}, deletions=("claude_account.routing",))
    else:
        update_settings_file(config.settings_file, {"claude_account.routing": {"mode": mode}})


async def _handle_admin_claude_routing_put(request: Request) -> JSONResponse:
    """Select the pool routing policy: "disabled", "fallback", or "balanced".

    Enabling ("disabled"/"fallback" -> "balanced") and intentionally exiting
    ("balanced" -> "disabled"/"fallback") balanced routing are both
    transactional under `admin_lock`. Enabling prepares the complete
    `ClaudeBalancedRuntime` — opening the runtime store, restoring its state,
    constructing the router, and verifying every ready account's
    `profile_fingerprint` — while the old mode keeps serving. Settings are
    persisted only once every check passes, immediately before publishing the
    prepared runtime; a failure at any point tears preparation down and leaves
    the old mode untouched. Exiting drains in-flight balanced dispatch and
    durably persists the target mode first. A persistence failure aborts the
    exit, leaving the runtime active with its epoch and pins untouched and this
    handler returning 500 with the mode unchanged. After persistence succeeds,
    the runtime rotates the epoch and publishes the target mode in memory before
    waking requests that arrived mid-transition. Switching between "disabled"
    and "fallback" uses the settings-file swap directly.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    mode = body.get("mode")
    unknown = sorted(set(body) - set(_CLAUDE_ROUTING_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_CLAUDE_ROUTING_KEYS)}",
            ),
            status_code=400,
        )
    if mode not in VALID_CLAUDE_ACCOUNT_ROUTING_MODES:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "provide 'mode' as one of "
                f"{', '.join(VALID_CLAUDE_ACCOUNT_ROUTING_MODES)}",
            ),
            status_code=400,
        )
    env_name = SETTINGS_KEYS["claude_account.routing"]
    if os.environ.get(env_name) is not None:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"{env_name} is set in the gateway's environment and overrides "
                f"claude_account.routing; unset it to manage the setting at runtime",
            ),
            status_code=409,
        )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        current_mode = config.claude_account_routing_mode
        runtime: ClaudeBalancedRuntime = request.app.state.claude_balanced_runtime

        if mode == "balanced" and current_mode != "balanced":
            lease = getattr(request.app.state, "claude_pool_lease", None)
            if lease is None:
                return JSONResponse(
                    server_support._openai_error_body(
                        "server_error",
                        "the claude account pool lease is not held; balanced "
                        "routing cannot be enabled",
                    ),
                    status_code=500,
                )
            try:
                accounts = list_accounts()
            except AccountRegistryError as exc:
                return JSONResponse(
                    server_support._openai_error_body(
                        "server_error", f"cannot read the claude account registry: {exc}"
                    ),
                    status_code=500,
                )

            def _persist_balanced() -> None:
                _persist_claude_routing_mode(config, "balanced")
                request.app.state.config = replace(config, claude_account_routing_mode="balanced")

            try:
                await runtime.prepare_and_publish(
                    accounts=accounts,
                    accounts_root=paths.accounts_dir("claude"),
                    runtime_db_path=paths.claude_account_pool_runtime_db(),
                    persist=_persist_balanced,
                    entry="admin_enable",
                    usage_cache=request.app.state.claude_account_usage_cache,
                )
            except BalancedPrepareError as exc:
                return JSONResponse(
                    server_support._openai_error_body("invalid_request_error", str(exc)), status_code=400
                )
            except Exception as exc:
                return JSONResponse(
                    server_support._openai_error_body(
                        "server_error", f"could not enable balanced routing: {exc}"
                    ),
                    status_code=500,
                )
            return JSONResponse(_claude_routing_payload(request.app.state.config))

        if mode != "balanced" and current_mode == "balanced":

            def _persist_target() -> None:
                _persist_claude_routing_mode(config, mode)

            def _publish_target() -> None:
                request.app.state.config = replace(config, claude_account_routing_mode=mode)

            try:
                await runtime.exit_mode(mode, persist=_persist_target, publish=_publish_target)
            except (ConfigError, OSError) as exc:
                return JSONResponse(
                    server_support._openai_error_body(
                        "server_error", f"could not persist settings: {exc}"
                    ),
                    status_code=500,
                )
            return JSONResponse(_claude_routing_payload(request.app.state.config))

        try:
            _persist_claude_routing_mode(config, mode)
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        new_config = replace(config, claude_account_routing_mode=mode)
        request.app.state.config = new_config
    return JSONResponse(_claude_routing_payload(new_config))
