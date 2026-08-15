"""Shared request, authorization, response, and identity handlers."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

import claudex
from claudex import server_support
from claudex.providers.codex_auth import CodexAuthError, CodexAuthManager
from claudex.config import GatewayConfig
from claudex.providers.grok_auth import GrokAuthError, GrokAuthManager
from claudex.providers.kimi_auth import KimiAuthError, KimiAuthManager
from claudex.upstream_errors import UpstreamError


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
