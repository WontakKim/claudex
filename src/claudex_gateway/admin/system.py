"""Dashboard, observability, model catalog, and connection admin handlers."""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
import time
import uuid
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from claudex_gateway import server_support
from claudex_gateway.admin.common import (
    _admin_guard,
    _read_json_object,
    _require_json_content_type,
)
from claudex_gateway.providers.codex_auth import CodexAuthError
from claudex_gateway.providers.codex_client import CodexClient, CodexUpstreamError
from claudex_gateway.config import ConfigError, GatewayConfig, parse_route_target
from claudex_gateway.providers.grok_auth import GrokAuthError
from claudex_gateway.providers.grok_client import GrokClient, GrokUpstreamError, sanitize_grok_payload
from claudex_gateway.providers.kimi_auth import KimiAuthError
from claudex_gateway.providers.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.providers.openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleUpstreamError,
)
from claudex_gateway.translate import translate_claude_request_to_codex
from claudex_gateway.upstream_errors import UpstreamAuthError, UpstreamError
from claudex_gateway.claude.usage import fetch_claude_usage
from claudex_gateway.usage.providers import (
    consume_codex_reset_credit,
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


async def _handle_dashboard_css(request: Request) -> Response:
    """Serve the dashboard stylesheet embedded in the package."""
    try:
        stylesheet = (
            importlib.resources.files("claudex_gateway")
            .joinpath("dashboard.css")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError) as exc:
        logger.warning("dashboard asset unavailable: %s", exc)
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", "dashboard.css is missing from the package"
            ),
            status_code=500,
        )
    return Response(stylesheet, media_type="text/css")


async def _handle_dashboard_js(request: Request) -> Response:
    """Serve the dashboard JavaScript embedded in the package."""
    try:
        javascript = (
            importlib.resources.files("claudex_gateway")
            .joinpath("dashboard.js")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError) as exc:
        logger.warning("dashboard asset unavailable: %s", exc)
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", "dashboard.js is missing from the package"
            ),
            status_code=500,
        )
    return Response(javascript, media_type="application/javascript")


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
