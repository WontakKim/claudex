"""Shared request, authorization, response, and identity handlers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

import claudex
from claudex import server_support
from claudex.providers.codex_auth import CodexAuthError, CodexAuthManager
from claudex.config import (
    AnthropicCompatibleProvider,
    ConfigError,
    GatewayConfig,
    OpenAICompatibleProvider,
)
from claudex.providers.backends import AnthropicBackend, ResponsesBackend, RouteBackend
from claudex.providers.grok_auth import GrokAuthError, GrokAuthManager
from claudex.providers.kimi_auth import KimiAuthError, KimiAuthManager
from claudex.upstream_errors import UpstreamError


def _get_custom_provider_binding(
    config: GatewayConfig, route_backends: Mapping[str, RouteBackend], name: str
) -> tuple[OpenAICompatibleProvider | AnthropicCompatibleProvider, RouteBackend]:
    provider = config.custom_providers.get(name)
    if provider is None:
        raise RuntimeError("custom provider configuration is missing")
    if any(
        configured_provider.api_key in name
        or configured_provider.api_key in provider.base_url
        for configured_provider in config.custom_providers.values()
    ):
        raise RuntimeError(
            "custom provider public fields overlap a configured credential"
        )

    backend = route_backends.get(name)
    if backend is None:
        raise RuntimeError("custom provider binding is missing")
    if isinstance(provider, OpenAICompatibleProvider):
        if not isinstance(backend, ResponsesBackend):
            raise RuntimeError(
                "custom provider binding does not match its configured family"
            )
    elif isinstance(provider, AnthropicCompatibleProvider):
        if not isinstance(backend, AnthropicBackend):
            raise RuntimeError(
                "custom provider binding does not match its configured family"
            )
    else:
        raise RuntimeError("custom provider family is unsupported")
    return provider, backend


def _safe_route_target_error_detail(
    exc: ConfigError, config: GatewayConfig
) -> str:
    try:
        detail = str(exc)
    except BaseException:
        return "route target is invalid"
    for provider in config.custom_providers.values():
        redacted_detail = _redact_configured_credential(
            detail, provider.api_key
        )
        if redacted_detail != detail:
            return "route target is invalid"
    return detail


def _redact_configured_credential(value: str, credential: str) -> str:
    replacement = "[REDACTED]"
    if credential in replacement:
        replacement = "*" if credential != "*" else "?"

    bearer_value = f"Bearer {credential}"
    credential_bytes = credential.encode("utf-8", errors="replace")
    bearer_bytes = bearer_value.encode("utf-8", errors="replace")
    forms = {
        credential,
        json.dumps(credential, ensure_ascii=False)[1:-1],
        json.dumps(credential, ensure_ascii=True)[1:-1],
        credential.encode("unicode_escape").decode("ascii"),
        repr(credential),
        repr(credential)[1:-1],
        ascii(credential),
        ascii(credential)[1:-1],
        repr(credential_bytes),
        repr(credential_bytes)[2:-1],
        repr(bearer_value),
        repr(bearer_value)[1:-1],
        repr(bearer_bytes),
        repr(bearer_bytes)[2:-1],
    }
    for form in sorted((form for form in forms if form), key=len, reverse=True):
        value = value.replace(form, replacement)
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _safe_custom_provider_exception_detail(
    exc: BaseException, credential: str
) -> str:
    error_name = type(exc).__name__
    try:
        message = str(exc)
    except BaseException:
        message = "diagnostic unavailable"
    detail = f"{error_name}: {message}" if message else error_name
    return _redact_configured_credential(detail, credential)


def _safe_custom_provider_upstream_detail(body: str, credential: str) -> str:
    redacted_body = _redact_configured_credential(body, credential)
    try:
        detail = server_support._upstream_error_message(redacted_body)
    except (TypeError, ValueError, RecursionError):
        detail = "upstream returned a malformed error response"
    return _redact_configured_credential(detail, credential)


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
            "version": claudex.__version__,
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
    for index, (name, configured_provider) in enumerate(
        config.custom_providers.items(), start=1
    ):
        required = config.maps_to_provider(name)
        response_name = (
            name
            if not any(
                provider.api_key in name
                for provider in config.custom_providers.values()
            )
            else f"custom_provider_{index}"
        )
        try:
            provider, backend = _get_custom_provider_binding(
                config, request.app.state.route_backends, name
            )
        except RuntimeError as exc:
            providers[response_name] = {
                "status": "error",
                "detail": str(exc),
                "required": required,
            }
            if required:
                custom_providers_ready = False
            continue

        catalog_loader = backend.catalog_loader
        if catalog_loader is None:
            providers[response_name] = {"status": "ok", "required": required}
            continue
        try:
            await catalog_loader()
            providers[response_name] = {"status": "ok", "required": required}
        except UpstreamError as exc:
            detail = _safe_custom_provider_upstream_detail(exc.body, provider.api_key)
            providers[response_name] = {
                "status": "error",
                "detail": f"upstream returned HTTP {exc.status_code}: {detail}",
                "required": required,
            }
            if required:
                custom_providers_ready = False
        except httpx.HTTPError as exc:
            providers[response_name] = {
                "status": "error",
                "detail": _safe_custom_provider_exception_detail(
                    exc, provider.api_key
                ),
                "required": required,
            }
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
