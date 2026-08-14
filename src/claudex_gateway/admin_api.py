"""Dashboard, health, identity, and admin API handlers."""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

import claudex_gateway
from claudex_gateway import paths, server_support
from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache
from claudex_gateway.claude_account_pool import AccountCooldownTracker
from claudex_gateway.claude_accounts import (
    AccountNotFoundError,
    AccountRegistryError,
    list_accounts,
    load_registry,
    remove_account,
)
from claudex_gateway.claude_balanced_router import BalancedPrepareError, ClaudeBalancedRuntime
from claudex_gateway.claude_login_session import (
    ClaudeLoginSession,
    LoginSessionStateError,
    capture_lock_path,
)
from claudex_gateway.codex_auth import CodexAuthError, CodexAuthManager
from claudex_gateway.codex_client import CodexClient, CodexUpstreamError
from claudex_gateway.config import (
    SETTINGS_KEYS,
    VALID_CLAUDE_ACCOUNT_ROUTING_MODES,
    VALID_CODEX_SERVICE_TIERS,
    VALID_LOG_LEVELS,
    ConfigError,
    GatewayConfig,
    parse_claude_account_id,
    parse_compaction_model,
    parse_route_target,
    update_settings_file,
    validate_model_map,
)
from claudex_gateway.grok_auth import GrokAuthError, GrokAuthManager
from claudex_gateway.grok_client import GrokClient, GrokUpstreamError, sanitize_grok_payload
from claudex_gateway.kimi_auth import KimiAuthError, KimiAuthManager
from claudex_gateway.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.locking import try_file_lock
from claudex_gateway.openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleUpstreamError,
)
from claudex_gateway.translate import translate_claude_request_to_codex
from claudex_gateway.upstream_errors import UpstreamAuthError, UpstreamError
from claudex_gateway.usage import (
    _provider_result,
    consume_codex_reset_credit,
    fetch_claude_usage,
    fetch_codex_usage,
    fetch_grok_usage,
    fetch_kimi_usage,
)

logger = logging.getLogger("claudex_gateway.server")

_STATUS_TO_OPENAI_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "server_error",
    503: "server_error",
    529: "server_error",
}


async def _read_json_object(
    request: Request, error_factory: Any
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JSONResponse(
            error_factory("invalid_request_error", "request body is not valid JSON"),
            status_code=400,
        )
    if not isinstance(body, dict):
        return None, JSONResponse(
            error_factory("invalid_request_error", "request body must be a JSON object"),
            status_code=400,
        )
    return body, None


async def _handle_hello(request: Request) -> JSONResponse:
    # The launcher compares the version against its own to detect a stale
    # daemon left running across a package update, and matches pid/nonce
    # against its daemon record before signaling the process. The dashboard
    # reads local_auth_required to know whether to ask for the local token —
    # only the boolean is exposed, never the token itself.
    config: GatewayConfig = request.app.state.config
    return JSONResponse(
        {
            "hello": "claudex-gateway",
            "version": claudex_gateway.__version__,
            "pid": os.getpid(),
            "nonce": request.app.state.daemon_nonce,
            "local_auth_required": config.local_token is not None,
        }
    )


async def _handle_health(request: Request) -> JSONResponse:
    config: GatewayConfig = request.app.state.config
    codex_auth_manager: CodexAuthManager = request.app.state.codex_auth_manager
    kimi_auth_manager: KimiAuthManager = request.app.state.kimi_auth_manager
    grok_auth_manager: GrokAuthManager = request.app.state.grok_auth_manager
    providers: dict[str, dict[str, Any]] = {}

    try:
        credentials = await codex_auth_manager.get_credentials()
        providers["codex"] = {
            "status": "ok",
            "auth_mode": "api_key" if credentials.is_api_key else "chatgpt",
            "account": credentials.account_id,
            "email": credentials.email,
        }
    except CodexAuthError as exc:
        providers["codex"] = {"status": "error", "detail": str(exc)}

    # A missing OAuth login only degrades readiness when the map routes to
    # that provider, so setups not using it keep reporting healthy. The flag
    # is exposed so the dashboard can render an unused login failure as
    # neutral, not error.
    kimi_required = config.maps_to_provider("kimi")
    try:
        kimi_credentials = await kimi_auth_manager.get_credentials()
        providers["kimi"] = {
            "status": "ok",
            "required": kimi_required,
            "account": kimi_credentials.account,
        }
    except KimiAuthError as exc:
        providers["kimi"] = {"status": "error", "detail": str(exc), "required": kimi_required}

    grok_required = config.maps_to_provider("grok")
    try:
        grok_credentials = await grok_auth_manager.get_credentials()
        providers["grok"] = {
            "status": "ok",
            "required": grok_required,
            "auth_mode": "api_key" if grok_credentials.is_api_key else "oauth",
            "account": grok_credentials.email,
        }
    except GrokAuthError as exc:
        providers["grok"] = {"status": "error", "detail": str(exc), "required": grok_required}

    custom_providers_ready = True
    for name, custom_client in request.app.state.custom_provider_clients.items():
        required = config.maps_to_provider(name)
        try:
            await custom_client.list_models()
            providers[name] = {"status": "ok", "required": required}
        except (UpstreamError, httpx.HTTPError) as exc:
            providers[name] = {"status": "error", "detail": str(exc), "required": required}
            if required:
                custom_providers_ready = False

    is_ready = (
        providers["codex"]["status"] == "ok"
        and (providers["kimi"]["status"] == "ok" or not kimi_required)
        and (providers["grok"]["status"] == "ok" or not grok_required)
        and custom_providers_ready
    )
    return JSONResponse(
        {"status": "ok" if is_ready else "error", "providers": providers},
        status_code=200 if is_ready else 503,
    )


# The single runtime-editable map. Everything else in GatewayConfig is either
# fixed at startup (bind address, auth directories) or out of the admin API's
# mapping-only scope.
_ADMIN_MAP_KEYS = ("model_map",)


def _admin_guard(request: Request) -> JSONResponse | None:
    """Reject admin requests that could originate from another origin.

    Browsers can fire requests at localhost from any web page (drive-by
    requests, DNS rebinding), so beyond the optional bearer token the admin
    surface only answers when the Host header names the gateway itself.
    """
    denied = server_support._require_local_token(request)
    if denied is not None:
        return denied
    config: GatewayConfig = request.app.state.config
    hostname = (request.url.hostname or "").lower()
    if hostname not in {"localhost", "127.0.0.1", "::1", config.host.lower()}:
        return JSONResponse(
            server_support._openai_error_body(
                "permission_error",
                f"admin API refuses Host {hostname!r} (DNS-rebinding guard)",
            ),
            status_code=403,
        )
    return None


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


def _require_json_content_type(request: Request) -> JSONResponse | None:
    """Reject admin writes that are not application/json.

    Requiring application/json forces cross-origin browser requests into a
    CORS preflight, which this server never approves.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.partition(";")[0].strip().lower() != "application/json":
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "admin API requires Content-Type: application/json",
            ),
            status_code=415,
        )
    return None


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


async def _handle_admin_logs(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    log_buffer = request.app.state.log_buffer
    return JSONResponse({"logs": list(log_buffer.records)})


async def _handle_admin_usage(request: Request) -> JSONResponse:
    """Probe provider subscription usage for the dashboard's usage cards.

    Each provider answers from its own usage endpoint with the local CLI
    credentials (see usage.py); a failure on one side never masks the other.
    ?provider=claude|codex|kimi|grok refreshes a single card; without it all run.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    provider = request.query_params.get("provider")
    if provider not in (None, "claude", "codex", "kimi", "grok"):
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error", "provider must be one of: claude, codex, kimi, grok"
            ),
            status_code=400,
        )
    probes: dict[str, Any] = {}
    if provider in (None, "claude"):
        probes["claude"] = fetch_claude_usage(request.app.state.http_client)
    if provider in (None, "codex"):
        probes["codex"] = fetch_codex_usage(
            request.app.state.http_client, request.app.state.codex_auth_manager
        )
    if provider in (None, "kimi"):
        probes["kimi"] = fetch_kimi_usage(
            request.app.state.http_client, request.app.state.kimi_auth_manager
        )
    if provider in (None, "grok"):
        probes["grok"] = fetch_grok_usage(
            request.app.state.http_client, request.app.state.grok_auth_manager
        )
    results = await asyncio.gather(*probes.values())
    payload = dict(zip(probes, results))
    payload["fetched_at"] = time.time()
    return JSONResponse(payload)


async def _handle_admin_codex_reset_credit(request: Request) -> JSONResponse:
    """Spend one Codex reset credit — irreversible, so never call it implicitly.

    The admin lock serializes attempts so two clicks cannot both reach the
    backend, and the redeem key is held until an attempt settles: a request
    that timed out may or may not have burned the credit, so the retry reuses
    the key and lets the backend deduplicate instead of spending a second one.
    The key lives in memory only, so a gateway restart forfeits that guard.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    async with request.app.state.admin_lock:
        redeem_request_id = request.app.state.codex_reset_key or str(uuid.uuid4())
        request.app.state.codex_reset_key = redeem_request_id
        result = await consume_codex_reset_credit(
            request.app.state.http_client,
            request.app.state.codex_auth_manager,
            redeem_request_id,
        )
        if result["status"] == "ok":
            request.app.state.codex_reset_key = None
    return JSONResponse(result)


# Root covers the gateway's own loggers; the uvicorn loggers do not
# propagate to root, so they are adjusted explicitly.
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


# The single runtime-editable field on the compaction admin surface.
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


# The single runtime-editable field on the claude-account admin surface.
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
    transactional under `admin_lock` (T-10 Step 5): enabling prepares the
    complete `ClaudeBalancedRuntime` — opening the runtime store, restoring
    its state, constructing the router, and verifying every ready account's
    T-3 profile_fingerprint — while the OLD mode keeps serving, and persists
    settings only once every check passes, immediately before publishing the
    prepared runtime; a failure at any point tears preparation down and
    leaves the old mode untouched. Exiting drains in-flight balanced
    dispatch, durably PERSISTS the target mode first (T-20, fix for gap
    G-3 — a failure here aborts the exit, leaving the runtime "active" with
    its epoch and pins untouched and this handler returning 500 with the
    mode unchanged), THEN rotates (invalidates) the epoch, THEN publishes
    the target mode in memory before waking any request that arrived
    mid-transition. Switching between "disabled" and "fallback" is
    unaffected — the pre-existing settings-file swap.
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


# --------------------------------------------------------------------------
# Balanced-mode usage isolation (T-13): while balanced routing is the
# currently PUBLISHED and ACTIVE mode, usage reads are cache-only (never
# `ClaudeAccountUsageCache.get`/upstream) and manual refresh only ever
# enqueues on the coordinator -- fallback/disabled mode is entirely
# untouched by any of this and keeps the pre-existing fetch path/envelope.
# --------------------------------------------------------------------------

_USAGE_WINDOW_FRESH_MAX_AGE_SECONDS = 5 * 60
_USAGE_WINDOW_AGING_MAX_AGE_SECONDS = 30 * 60


def _active_balanced_runtime(request: Request) -> ClaudeBalancedRuntime | None:
    """The live runtime iff "balanced" is the currently PUBLISHED routing mode
    AND the runtime itself is active -- the exact isolation boundary Steps
    4/5/6 draw between balanced-only usage behavior and every other mode. A
    non-balanced request must never see this as non-`None` (Step 5's "never
    queued for a coordinator that is not running").
    """
    config: GatewayConfig = request.app.state.config
    if config.claude_account_routing_mode != "balanced":
        return None
    runtime: ClaudeBalancedRuntime = request.app.state.claude_balanced_runtime
    return runtime if runtime.status == "active" else None


def _usage_window_state(age_seconds: float) -> str:
    """Per-window freshness label for the balanced-mode usage read (Step 4)."""
    if age_seconds <= _USAGE_WINDOW_FRESH_MAX_AGE_SECONDS:
        return "fresh"
    if age_seconds <= _USAGE_WINDOW_AGING_MAX_AGE_SECONDS:
        return "aging"
    return "stale"



# The account-level binding-window pair required for `fresh` (§2.1):
# envelope names for `five_hour`/`seven_day`. `fable_weekly` is a scoped,
# Fable-only extra window and must never be required for -- or substitute
# for a missing member of -- this pair.
_USAGE_FRESHNESS_BINDING_WINDOWS = ("session", "weekly")


def _compute_usage_freshness(
    ready_ids: list[str], cache: ClaudeAccountUsageCache, *, persistence_degraded: bool
) -> tuple[str, dict[str, Any]]:
    """Step 6's aggregate `usage_freshness` plus per-account diagnostics.

    `"fresh"`: every ready account has BOTH binding windows
    (`_USAGE_FRESHNESS_BINDING_WINDOWS`) present, each at most 5 minutes
    old -- a ready account missing either window does not count as fresh,
    even if every window it does have is recent. `"degraded"`: persistence
    is degraded, or no window across the whole ready set is at most 30
    minutes old. Otherwise `"partial"`.
    """
    per_account: dict[str, Any] = {}
    all_fresh = True
    any_within_degraded_window = False
    for account_id in ready_ids:
        peeked = cache.peek_with_metadata(account_id)
        if peeked is None or not peeked[1]:
            all_fresh = False
            per_account[account_id] = {"oldest_age_seconds": None, "window_count": 0}
            continue
        _, metadata = peeked
        ages = [window["age_seconds"] for window in metadata.values()]
        per_account[account_id] = {
            "oldest_age_seconds": max(ages),
            "window_count": len(ages),
        }
        binding_ages = [
            metadata[window_name]["age_seconds"]
            for window_name in _USAGE_FRESHNESS_BINDING_WINDOWS
            if window_name in metadata
        ]
        if (
            len(binding_ages) < len(_USAGE_FRESHNESS_BINDING_WINDOWS)
            or max(binding_ages) > _USAGE_WINDOW_FRESH_MAX_AGE_SECONDS
        ):
            all_fresh = False
        if min(ages) <= _USAGE_WINDOW_AGING_MAX_AGE_SECONDS:
            any_within_degraded_window = True

    if persistence_degraded or (ready_ids and not any_within_degraded_window):
        return "degraded", per_account
    if all_fresh:
        return "fresh", per_account
    return "partial", per_account


async def _handle_admin_claude_pool_status(request: Request) -> JSONResponse:
    """Per-account routing state: what the serving chain would see right now.

    This is telemetry over the registry plus the daemon-memory cooldown
    tracker — never the configured pin, which lives at pool/serving. While
    balanced routing is active, this also carries the balanced `usage_freshness`
    diagnostic (Step 6); in every other mode both fields are `None`.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    try:
        records = list_accounts()
    except AccountRegistryError as exc:
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", f"cannot read the claude account registry: {exc}"
            ),
            status_code=500,
        )
    tracker: AccountCooldownTracker = request.app.state.claude_account_cooldowns
    members: list[dict[str, Any]] = []
    for record in records:
        if record.state != "ready":
            members.append(
                {
                    "account_id": record.id,
                    "routing_state": "unavailable",
                    "reason": record.state,
                }
            )
            continue
        cooldown_until = _cooling_down_until_millis(tracker, record.id)
        if cooldown_until is not None:
            members.append(
                {
                    "account_id": record.id,
                    "routing_state": "cooldown",
                    "cooldown_until": cooldown_until,
                }
            )
        else:
            members.append({"account_id": record.id, "routing_state": "ready"})

    usage_freshness: str | None = None
    usage_diagnostics: dict[str, Any] | None = None
    runtime = _active_balanced_runtime(request)
    if runtime is not None:
        ready_ids = [record.id for record in records if record.state == "ready"]
        cache: ClaudeAccountUsageCache = request.app.state.claude_account_usage_cache
        persistence_degraded = runtime.router.persistence_degraded if runtime.router is not None else True
        usage_freshness, per_account = _compute_usage_freshness(
            ready_ids, cache, persistence_degraded=persistence_degraded
        )
        coordinator = runtime.usage_poll_coordinator
        usage_diagnostics = {
            "persistence_degraded": persistence_degraded,
            "accounts": per_account,
            "coordinator": vars(coordinator.diagnostics()) if coordinator is not None else None,
        }
    return JSONResponse(
        {
            "members": members,
            "usage_freshness": usage_freshness,
            "usage_diagnostics": usage_diagnostics,
        }
    )


# --------------------------------------------------------------------------
# Admin claude-accounts surface (dashboard account management)
# --------------------------------------------------------------------------

_CLAUDE_LOGIN_CODE_KEYS = ("code",)
_CLAUDE_LOGIN_REPLACE_KEYS = ("existing_account_id",)
_LOGIN_ATTEMPT_HEADER = "x-login-attempt"
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


def _require_login_attempt(
    request: Request, session: ClaudeLoginSession | None
) -> JSONResponse | None:
    """409 unless X-Login-Attempt names the active session's attempt.

    Runs immediately after the admin guard — before content-type, body,
    or state validation — so a stale caller always learns it is stale
    instead of receiving an incidental 400/415/state error first. With no
    active session there is no attempt any header could name, so every
    attached call is stale by definition.
    """
    if (
        session is not None
        and request.headers.get(_LOGIN_ATTEMPT_HEADER) == session.attempt_id
    ):
        return None
    return JSONResponse(
        server_support._openai_error_body(
            "invalid_request_error",
            "X-Login-Attempt does not name the active login attempt; "
            "re-attach via GET /admin/providers/claude/login",
            "stale_login",
        ),
        status_code=409,
    )


def _local_claude_login_fields() -> dict[str, Any] | None:
    """Identity snapshot of this machine's ambient Claude Code login.

    Read from the CLI's own config file (`~/.claude.json`, or
    `$CLAUDE_CONFIG_DIR/.claude.json` when the override is set): the same
    `oauthAccount` block a capture snapshots — identity and plan metadata,
    never secrets. The dashboard's accounts screen shows it as the "로컬
    CLI 로그인" hero, which is informational only and unrelated to serving.
    Missing or malformed files degrade to None (no local login).
    """
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    path = (
        Path(override).expanduser() / ".claude.json"
        if override
        else Path.home() / ".claude.json"
    )
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    account = parsed.get("oauthAccount")
    if not isinstance(account, dict):
        return None

    def _text_field(key: str) -> str | None:
        value = account.get(key)
        return value if isinstance(value, str) and value else None

    email = _text_field("emailAddress")
    if email is None:
        return None
    return {
        "accountUuid": _text_field("accountUuid"),
        "email": email,
        "organizationName": _text_field("organizationName"),
        "planType": _text_field("organizationType"),
        "rateLimitTier": _text_field("organizationRateLimitTier"),
    }


def _account_plan_fields(account_id: str) -> dict[str, Any]:
    """Plan metadata from the account's captured oauth-account.json.

    `organizationType` (e.g. claude_max) and `organizationRateLimitTier`
    (e.g. default_claude_max_20x) are login-time snapshots — refreshed only
    by a re-login — and the file holds no secrets. Missing or malformed
    files degrade to nulls; the account list never fails over plan info.
    """
    path = paths.accounts_dir("claude") / account_id / "oauth-account.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {"planType": None, "rateLimitTier": None}
    if not isinstance(parsed, dict):
        return {"planType": None, "rateLimitTier": None}
    organization_type = parsed.get("organizationType")
    rate_limit_tier = parsed.get("organizationRateLimitTier")
    return {
        "planType": organization_type if isinstance(organization_type, str) else None,
        "rateLimitTier": rate_limit_tier if isinstance(rate_limit_tier, str) else None,
    }


async def _handle_admin_claude_accounts_get(request: Request) -> JSONResponse:
    """List every registered account (registry metadata only — never secrets).

    Deliberately just the collection: the local-login hero lives at
    `claude/local`, the serving pin at `claude/pool/serving`, and cooldown
    telemetry at `claude/pool/status` — each readable (and cacheable) on
    its own.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    try:
        records = list_accounts()
    except AccountRegistryError as exc:
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", f"cannot read the claude account registry: {exc}"
            ),
            status_code=500,
        )
    return JSONResponse(
        {
            "accounts": [
                {**record.to_row(), **_account_plan_fields(record.id)}
                for record in records
            ]
        }
    )


async def _handle_admin_claude_local_get(request: Request) -> JSONResponse:
    """This machine's ambient Claude Code login (informational hero card)."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse({"local": _local_claude_login_fields()})


async def _handle_admin_claude_account_delete(request: Request) -> Response:
    """Remove a registered account (dashboard analog of `account remove`).

    Refuses while the account is the serving pin — silently unpinning here
    would flip passthrough back to client credentials as a side effect.
    The registry mutation itself is crash-safe under registry.lock
    (tombstone protocol); afterwards the daemon-memory remnants (cached
    auth manager, cooldown) are dropped, and — while balanced routing is
    active (T-12, design v2 §5.7) — the router's own removal matrix runs and
    every durable row (pins, cooldowns, usage observations, capability
    evidence) for the removed incarnation is deleted, awaited before this
    responds.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    account_id = request.path_params["account_id"]
    # The pin check and the removal share the admin lock with the
    # pool/serving writes, so a concurrent serving PUT cannot pin this
    # account between the check and the removal (or vice versa).
    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        if config.claude_account_id == account_id:
            return JSONResponse(
                server_support._openai_error_body(
                    "invalid_request_error",
                    f"account {account_id} is the serving account; clear the pin "
                    "at pool/serving first",
                ),
                status_code=409,
            )
        try:
            records = list_accounts()
        except AccountRegistryError as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"cannot read the claude account registry: {exc}"
                ),
                status_code=500,
            )
        removed_record = next((record for record in records if record.id == account_id), None)
        try:
            await asyncio.to_thread(remove_account, account_id)
        except AccountNotFoundError as exc:
            return JSONResponse(
                server_support._openai_error_body("invalid_request_error", str(exc)), status_code=404
            )
        except AccountRegistryError as exc:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", f"could not remove the account: {exc}"
                ),
                status_code=500,
            )
        request.app.state.claude_account_auth_managers.pop(account_id, None)
        request.app.state.claude_account_cooldowns.clear(account_id)
        runtime: ClaudeBalancedRuntime = request.app.state.claude_balanced_runtime
        if runtime.status == "active" and runtime.router is not None and removed_record is not None:
            runtime.router.remove_account(account_id, removed_record.account_incarnation_id)
            await runtime.router.await_account_removal_durability(removed_record.account_incarnation_id)
    return Response(status_code=204)


def _cooling_down_until_millis(tracker: AccountCooldownTracker, account_id: str) -> int | None:
    """Epoch-ms cooldown deadline for the row overlay (registry-timestamp unit)."""
    remaining = tracker.remaining_seconds(account_id)
    if remaining <= 0.0:
        return None
    return int((time.time() + remaining) * 1000)


async def _handle_admin_claude_accounts_usage(request: Request) -> JSONResponse:
    """Per-account usage.

    Fallback/disabled mode (and balanced published but not yet/no-longer
    active) is served exactly as before, through the TTL cache's fetch path
    — unchanged envelope, TTL/backoff/global-cooldown semantics, no
    force-refresh; needs-reauth rows get a synthesized "unavailable" without
    touching the network.

    Active balanced mode is cache-only (T-13 Step 4): it never calls
    `cache.get`/upstream, reading `peek_with_metadata` instead and reporting
    each window's age/source/reset/state. A `?refresh` request in this mode
    (Step 5) enqueues a coalesced, globally rate-limited manual poll on the
    balanced coordinator and reports it as `queued` in the response — it
    never fetches inline, and cached data is returned immediately either
    way. `?refresh` outside active balanced mode is inert: a non-balanced
    request must never be queued for a coordinator that is not running.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    try:
        records = list_accounts()
    except AccountRegistryError as exc:
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", f"cannot read the claude account registry: {exc}"
            ),
            status_code=500,
        )
    account_param = request.query_params.get("account")
    if account_param is not None:
        records = [record for record in records if record.id == account_param]
        if not records:
            return JSONResponse(
                server_support._openai_error_body(
                    "invalid_request_error",
                    f"no account registered with id {account_param}",
                ),
                status_code=400,
            )
    ready_ids = [record.id for record in records if record.state == "ready"]
    cache: ClaudeAccountUsageCache = request.app.state.claude_account_usage_cache
    runtime = _active_balanced_runtime(request)

    if runtime is not None:
        refresh_requested = request.query_params.get("refresh") is not None
        coordinator = runtime.usage_poll_coordinator
        results: dict[str, Any] = {}
        for account_id in ready_ids:
            if refresh_requested and coordinator is not None:
                coordinator.request_manual_refresh(account_id)
            peeked = cache.peek_with_metadata(account_id)
            if peeked is None:
                envelope = _provider_result(
                    "claude",
                    status="unavailable",
                    error="no usage observation yet; the balanced poll coordinator "
                    "has not polled this account",
                )
                windows: dict[str, Any] = {}
            else:
                envelope = dict(peeked[0])
                windows = {
                    window_name: {
                        "age_seconds": metadata["age_seconds"],
                        "source": metadata["source"],
                        "reset_at": metadata["reset_at"],
                        "state": _usage_window_state(metadata["age_seconds"]),
                    }
                    for window_name, metadata in peeked[1].items()
                }
            envelope["windows"] = windows
            envelope["queued"] = (
                coordinator.is_manual_refresh_pending(account_id) if coordinator is not None else False
            )
            results[account_id] = envelope
        for record in records:
            if record.state != "ready":
                results[record.id] = {
                    **_provider_result(
                        "claude",
                        status="unavailable",
                        error="account needs re-authentication; log in again from the dashboard",
                    ),
                    "windows": {},
                    "queued": False,
                }
        return JSONResponse(
            {
                "accounts": results,
                "fetched_at": time.time(),
                "queued": any(account["queued"] for account in results.values()),
            }
        )

    results = await cache.get(ready_ids)
    for record in records:
        if record.state != "ready":
            results[record.id] = _provider_result(
                "claude",
                status="unavailable",
                error="account needs re-authentication; log in again from the dashboard",
            )
    return JSONResponse({"accounts": results, "fetched_at": time.time()})


async def _handle_admin_claude_login_get(request: Request) -> JSONResponse:
    """Poll the login session. A bare GET is discovery/attach — it returns
    the full status (including attempt_id) so a fresh tab can pin itself;
    a GET that carries X-Login-Attempt is an attached poll and is guarded
    (including against a cleared slot: a dead attempt is a stale one)."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    if _LOGIN_ATTEMPT_HEADER in request.headers:
        denied = _require_login_attempt(request, session)
        if denied is not None:
            return denied
    if session is None:
        return JSONResponse({"status": "idle"})
    return JSONResponse(session.status())


async def _handle_admin_claude_login_post(request: Request) -> JSONResponse:
    """Start a dashboard login session (single concurrent session).

    The slot check and assignment have no await between them, so two
    concurrent POSTs cannot both create a session. The cross-process capture
    lock additionally excludes a CLI `account add` running on this machine.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    if body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unexpected keys: {', '.join(sorted(body))}; POST an empty JSON object",
            ),
            status_code=400,
        )

    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    if session is not None and not session.is_terminal:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "a login session is already active; poll GET /admin/providers/claude/login",
                "login-active",
            ),
            status_code=409,
        )
    lock_handle = try_file_lock(capture_lock_path())
    if lock_handle is None:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "another Claude login is in progress on this machine "
                "(a CLI `account add`?); retry once it finishes",
                "login-locked",
            ),
            status_code=409,
        )
    session = ClaudeLoginSession(lock_handle)
    request.app.state.claude_login_session = session
    session.start()
    # The full envelope (not a minimal status) so the creating tab can pin
    # its attempt_id without a follow-up GET racing another tab's POST.
    return JSONResponse(session.status(), status_code=201)


async def _handle_admin_claude_login_code_post(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    denied = _require_login_attempt(request, session)
    if denied is not None:
        return denied
    denied = _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_LOGIN_CODE_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; supported: code",
            ),
            status_code=400,
        )
    code = body.get("code")
    code = code.strip() if isinstance(code, str) else None
    # A pasted code must be exactly one stdin line for the login child;
    # control characters would smuggle extra lines or terminal noise.
    if not code or _CONTROL_CHARACTER_PATTERN.search(code):
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "provide 'code' as a non-empty single-line string",
            ),
            status_code=400,
        )
    try:
        await session.submit_code(code)
    except LoginSessionStateError as exc:
        return JSONResponse(
            server_support._openai_error_body("invalid_request_error", str(exc)),
            status_code=409,
        )
    return JSONResponse({"status": session.status()["status"]})


async def _handle_admin_claude_login_replace_post(request: Request) -> JSONResponse:
    """Confirm replacing the duplicate registration the session collided with.

    The body names the account being replaced (the `existing_account_id`
    from status()) — a confirmation, not a generation token; the session
    rejects a mismatch. Declining is DELETE (cancel), not a body variant.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    denied = _require_login_attempt(request, session)
    if denied is not None:
        return denied
    denied = _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_LOGIN_REPLACE_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; supported: existing_account_id",
            ),
            status_code=400,
        )
    existing_account_id = body.get("existing_account_id")
    if not isinstance(existing_account_id, str) or not existing_account_id:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "provide 'existing_account_id' as a non-empty string",
            ),
            status_code=400,
        )
    try:
        session.confirm_replace(existing_account_id)
    except LoginSessionStateError as exc:
        return JSONResponse(
            server_support._openai_error_body("invalid_request_error", str(exc)),
            status_code=409,
        )
    return JSONResponse({"status": session.status()["status"]})


async def _handle_admin_claude_login_delete(request: Request) -> JSONResponse:
    """Cancel an active session, or clear a terminal one from the slot.

    Attempt-addressed like every mutating login command: the caller must
    name the attempt it is cancelling, so a stale tab (or a call after
    the slot was cleared) gets 409 stale_login instead of touching a
    session it never attached to.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    denied = _require_login_attempt(request, session)
    if denied is not None:
        return denied
    if session.is_terminal:
        request.app.state.claude_login_session = None
        return JSONResponse({"status": "idle"})
    session.request_cancel()
    return JSONResponse({"status": "cancelling"})


async def _handle_dashboard(request: Request) -> Response:
    """Serve the runtime dashboard, embedded in the package as dashboard.html."""
    try:
        page = (
            importlib.resources.files("claudex_gateway")
            .joinpath("dashboard.html")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError) as exc:
        logger.warning("dashboard asset unavailable: %s", exc)
        return JSONResponse(
            server_support._openai_error_body("server_error", "dashboard.html is missing from the package"),
            status_code=500,
        )
    return Response(page, media_type="text/html; charset=utf-8")


# Inline SVG so the package ships no binary asset; the glyph is a two-way
# relay in the dashboard's accent color.
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<rect width="16" height="16" rx="3" fill="#1d5fbf"/>'
    '<path d="M10.2 3.6 12.6 6l-2.4 2.4M12.2 6H3.4M5.8 12.4 3.4 10l2.4-2.4M3.8 10h8.8"'
    ' fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"'
    ' stroke-linejoin="round"/>'
    "</svg>"
)


async def _handle_favicon(request: Request) -> Response:
    """Answer the browser's automatic /favicon.ico probe for the dashboard.

    Without this route every dashboard visit logs a 404 in the access log.
    """
    return Response(
        _FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


async def _handle_admin_codex_models(request: Request) -> JSONResponse:
    """Proxy the live Codex model catalog for the dashboard's model columns."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    codex_client: CodexClient = request.app.state.codex_client
    try:
        models = await codex_client.list_models()
    except CodexAuthError as exc:
        return JSONResponse(
            server_support._openai_error_body("authentication_error", str(exc)), status_code=401
        )
    except CodexUpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        return JSONResponse(
            server_support._openai_error_body(error_type, server_support._upstream_error_message(exc.body)),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            server_support._openai_error_body("server_error", f"failed to reach the Codex backend: {exc}"),
            status_code=502,
        )
    return JSONResponse({"models": models})


async def _handle_admin_grok_models(request: Request) -> JSONResponse:
    """Relay Grok's live model catalog (IDs only) for model_map authoring.

    Same convenience role as the Codex/Kimi catalog endpoints: the gateway
    never validates map targets against the list, so this only feeds the
    dashboard's add-node suggestions.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    grok_client: GrokClient = request.app.state.grok_client
    try:
        models = await grok_client.list_models()
    except GrokAuthError as exc:
        return JSONResponse(
            server_support._openai_error_body("authentication_error", str(exc)), status_code=401
        )
    except GrokUpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        return JSONResponse(
            server_support._openai_error_body(error_type, server_support._upstream_error_message(exc.body)),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            server_support._openai_error_body("server_error", f"failed to reach the Grok backend: {exc}"),
            status_code=502,
        )
    return JSONResponse({"models": models})


async def _handle_admin_custom_models(request: Request) -> JSONResponse:
    """Relay a configured custom provider's live model catalog."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    name = request.path_params["name"]
    config: GatewayConfig = request.app.state.config
    if name not in config.custom_providers:
        return JSONResponse(
            server_support._openai_error_body(
                "not_found_error", f"custom provider {name!r} is not configured"
            ),
            status_code=404,
        )
    custom_client: OpenAICompatibleClient = request.app.state.custom_provider_clients[name]
    try:
        models = await custom_client.list_models()
    except OpenAICompatibleUpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        return JSONResponse(
            server_support._openai_error_body(error_type, server_support._upstream_error_message(exc.body)),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", f"failed to reach custom provider {name!r}: {exc}"
            ),
            status_code=502,
        )
    return JSONResponse({"models": models})


async def _handle_admin_kimi_models(request: Request) -> JSONResponse:
    """Relay Kimi's live model catalog verbatim for model_map authoring.

    The gateway never validates map targets against a model list — values
    after the kimi: prefix bypass untouched so newly released models work
    without a gateway update. This endpoint only exists as a convenience
    source of valid IDs (and future dashboard presets).
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    kimi_client: KimiClient = request.app.state.kimi_client
    try:
        catalog = await kimi_client.list_models()
    except KimiAuthError as exc:
        return JSONResponse(
            server_support._openai_error_body("authentication_error", str(exc)), status_code=401
        )
    except KimiUpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        return JSONResponse(
            server_support._openai_error_body(error_type, server_support._upstream_error_message(exc.body)),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            server_support._openai_error_body("server_error", f"failed to reach the Kimi backend: {exc}"),
            status_code=502,
        )
    return JSONResponse(catalog)


_CONNECTION_TEST_TIMEOUT = 30.0


async def _probe_codex_route(codex_client: CodexClient, target: str) -> str:
    claude_request = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    payload = translate_claude_request_to_codex(claude_request, target, "low")
    events = codex_client.stream_responses(payload, payload["prompt_cache_key"])
    try:
        first_event = await anext(events, None)
    finally:
        await events.aclose()
    if first_event is None:
        raise CodexUpstreamError(502, "codex stream ended without any events")
    response = first_event.get("response") if isinstance(first_event, dict) else None
    model = response.get("model") if isinstance(response, dict) else None
    return model if isinstance(model, str) else target


async def _probe_grok_route(grok_client: GrokClient, target: str) -> str:
    claude_request = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    payload = translate_claude_request_to_codex(claude_request, target, "low")
    payload = sanitize_grok_payload(payload, target)
    events = grok_client.stream_responses(payload, payload["prompt_cache_key"])
    try:
        first_event = await anext(events, None)
    finally:
        await events.aclose()
    if first_event is None:
        raise GrokUpstreamError(502, "grok stream ended without any events")
    response = first_event.get("response") if isinstance(first_event, dict) else None
    model = response.get("model") if isinstance(response, dict) else None
    return model if isinstance(model, str) else target


async def _probe_custom_route(
    custom_client: OpenAICompatibleClient, provider_name: str, target: str
) -> str:
    claude_request = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    payload = translate_claude_request_to_codex(claude_request, target, "low")
    events = custom_client.stream_responses(payload, payload["prompt_cache_key"])
    try:
        first_event = await anext(events, None)
    finally:
        await events.aclose()
    if first_event is None:
        raise OpenAICompatibleUpstreamError(
            502,
            f"custom provider {provider_name!r} stream ended without any events",
            provider_name,
        )
    response = first_event.get("response") if isinstance(first_event, dict) else None
    model = response.get("model") if isinstance(response, dict) else None
    return model if isinstance(model, str) else target


async def _probe_kimi_route(kimi_client: KimiClient, target_model: str) -> str:
    claude_request = {
        "model": target_model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20",
    }
    response = await kimi_client.send_messages(json.dumps(claude_request).encode(), headers)
    try:
        payload = await response.aread()
    finally:
        await response.aclose()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return target_model
    model = parsed.get("model") if isinstance(parsed, dict) else None
    return model if isinstance(model, str) else target_model


async def _handle_admin_connection_test(request: Request) -> JSONResponse:
    """Send one minimal request through the gateway to verify a target model id.

    The result — success or failure — is always a 200 with the outcome in the
    body; non-200 responses are reserved for invalid test requests themselves.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    target = body.get("target")
    if not isinstance(target, str) or not target.strip():
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error", "target must be a non-empty string"
            ),
            status_code=400,
        )
    target = target.strip()

    started_at = time.monotonic()

    def result(
        ok: bool, status: int | None, detail: str | None = None, response_model: str | None = None
    ) -> JSONResponse:
        return JSONResponse(
            {
                "ok": ok,
                "status": status,
                "latency_ms": int((time.monotonic() - started_at) * 1000),
                "target": target,
                "response_model": response_model,
                "detail": detail,
            }
        )

    # The target carries the same provider-prefix syntax as model_map values,
    # so the dashboard's test box works for every configured route.
    config: GatewayConfig = request.app.state.config
    try:
        route = parse_route_target(target, config.route_providers)
    except ConfigError as exc:
        return JSONResponse(
            server_support._openai_error_body("invalid_request_error", str(exc)), status_code=400
        )

    try:
        if route.provider == "kimi":
            probe = _probe_kimi_route(request.app.state.kimi_client, route.model)
        elif route.provider == "grok":
            probe = _probe_grok_route(request.app.state.grok_client, route.model)
        elif route.provider in config.custom_providers:
            probe = _probe_custom_route(
                request.app.state.custom_provider_clients[route.provider],
                route.provider,
                route.model,
            )
        else:
            probe = _probe_codex_route(request.app.state.codex_client, route.model)
        response_model = await asyncio.wait_for(probe, _CONNECTION_TEST_TIMEOUT)
    except UpstreamError as exc:
        return result(False, exc.status_code, server_support._upstream_error_message(exc.body))
    except UpstreamAuthError as exc:
        return result(False, 401, str(exc))
    except TimeoutError:
        return result(False, None, f"no response within {_CONNECTION_TEST_TIMEOUT:.0f}s")
    except httpx.HTTPError as exc:
        return result(False, None, f"failed to reach the upstream: {exc}")
    return result(True, 200, response_model=response_model)
