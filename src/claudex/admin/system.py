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

from claudex import server_support
from claudex.admin.common import (
    _admin_guard,
    _get_custom_provider_binding,
    _read_json_object,
    _require_json_content_type,
    _safe_custom_provider_exception_detail,
    _safe_custom_provider_upstream_detail,
    _safe_route_target_error_detail,
)
from claudex.providers.backends import (
    AnthropicBackend,
    ResponsesBackend,
    RouteBackend,
)
from claudex.providers.codex_auth import CodexAuthError
from claudex.providers.codex_client import CodexUpstreamError
from claudex.config import ConfigError, GatewayConfig, parse_route_target
from claudex.providers.grok_auth import GrokAuthError
from claudex.providers.grok_client import GrokUpstreamError
from claudex.providers.kimi_auth import KimiAuthError
from claudex.providers.kimi_client import KimiUpstreamError
from claudex.translate import translate_claude_request_to_codex
from claudex.upstream_errors import UpstreamAuthError, UpstreamError
from claudex.claude.usage import fetch_claude_usage
from claudex.usage.providers import (
    consume_codex_reset_credit,
    fetch_codex_usage,
    fetch_grok_usage,
    fetch_kimi_usage,
)

logger = logging.getLogger("claudex.server")


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


def _serve_dashboard_asset(basename: str, media_type: str) -> Response:
    try:
        text = (
            importlib.resources.files("claudex")
            .joinpath("dashboard", basename)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError) as exc:
        logger.warning("dashboard asset unavailable: %s", exc)
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", f"{basename} is missing from the package"
            ),
            status_code=500,
        )
    return Response(text, media_type=media_type)


async def _handle_dashboard(request: Request) -> Response:
    """Serve the runtime dashboard embedded in the package."""
    return _serve_dashboard_asset("dashboard.html", "text/html; charset=utf-8")


async def _handle_dashboard_css(request: Request) -> Response:
    """Serve the dashboard stylesheet embedded in the package."""
    return _serve_dashboard_asset("dashboard.css", "text/css")


async def _handle_dashboard_js(request: Request) -> Response:
    """Serve the dashboard JavaScript embedded in the package."""
    return _serve_dashboard_asset("dashboard.js", "application/javascript")


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


def _get_route_backend(request: Request, provider: str) -> RouteBackend:
    try:
        backend = request.app.state.route_backends[provider]
    except KeyError as exc:
        raise RuntimeError(
            f"route backend registry has no binding for provider {provider!r}"
        ) from exc
    if not isinstance(backend, (ResponsesBackend, AnthropicBackend)):
        raise RuntimeError(
            "route backend registry contains an unsupported binding for "
            f"provider {provider!r}: {type(backend).__name__}"
        )
    return backend


async def _handle_admin_codex_models(request: Request) -> JSONResponse:
    """Proxy the live Codex model catalog for the dashboard's model columns."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    backend = _get_route_backend(request, "codex")
    if not isinstance(backend, ResponsesBackend):
        raise RuntimeError("codex route backend must use the Responses wire")
    try:
        models = await backend.transport.list_models()
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
    backend = _get_route_backend(request, "grok")
    if not isinstance(backend, ResponsesBackend):
        raise RuntimeError("grok route backend must use the Responses wire")
    try:
        models = await backend.transport.list_models()
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
                "not_found_error", "custom provider is not configured"
            ),
            status_code=404,
        )
    try:
        provider, backend = _get_custom_provider_binding(
            config, request.app.state.route_backends, name
        )
    except RuntimeError:
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", "custom provider binding is unavailable"
            ),
            status_code=500,
        )
    if backend.catalog_loader is None:
        return JSONResponse(
            server_support._openai_error_body(
                "not_found_error",
                f"custom provider {name!r} does not provide a model catalog",
            ),
            status_code=404,
        )
    try:
        models = await backend.catalog_loader()
    except UpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        detail = _safe_custom_provider_upstream_detail(exc.body, provider.api_key)
        return JSONResponse(
            server_support._openai_error_body(error_type, detail),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        detail = _safe_custom_provider_exception_detail(exc, provider.api_key)
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", f"failed to reach custom provider: {detail}"
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
    backend = _get_route_backend(request, "kimi")
    if not isinstance(backend, AnthropicBackend):
        raise RuntimeError("kimi route backend must use the Anthropic Messages wire")
    if backend.catalog_loader is None:
        return JSONResponse(
            server_support._openai_error_body(
                "not_found_error", "Kimi does not provide a model catalog"
            ),
            status_code=404,
        )
    try:
        catalog = await backend.catalog_loader()
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


async def _probe_responses_route(
    backend: ResponsesBackend, provider_name: str, target: str
) -> str:
    claude_request = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    translated_payload = translate_claude_request_to_codex(
        claude_request, target, "low"
    )
    payload = backend.adapt_probe_payload(translated_payload, target)
    events = backend.transport.stream_responses(
        payload, payload["prompt_cache_key"]
    )
    try:
        first_event = await anext(events, None)
    finally:
        await events.aclose()
    if first_event is None:
        if backend.signature_namespace is None:
            detail = f"{provider_name} stream ended without any events"
        else:
            detail = (
                f"custom provider {provider_name!r} stream ended without any events"
            )
        raise UpstreamError(502, detail, provider_name)
    response = first_event.get("response") if isinstance(first_event, dict) else None
    model = response.get("model") if isinstance(response, dict) else None
    return model if isinstance(model, str) else target


async def _probe_anthropic_route(
    backend: AnthropicBackend, request: Request, target_model: str
) -> str:
    claude_request = {
        "model": target_model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    headers = backend.header_policy(request)
    response = await backend.transport.send_messages(
        json.dumps(claude_request).encode(), headers
    )
    # httpx auto-closes a fully read stream. Reserve that close for the
    # explicit ownership paths below so each path attempts it exactly once.
    close_response = response.aclose

    async def defer_automatic_close() -> None:
        return None

    response.aclose = defer_automatic_close
    try:
        payload = await response.aread()
    except BaseException:
        response.aclose = close_response
        try:
            await close_response()
        except BaseException:
            # The read failure is the operation's primary error; cleanup must
            # not replace it or disclose an unrelated cleanup diagnostic.
            pass
        raise
    else:
        response.aclose = close_response
        await close_response()
    invalid_response_detail = "upstream returned an invalid Anthropic Messages response"
    try:
        parsed = json.loads(payload)
    except (ValueError, RecursionError):
        raise UpstreamError(502, invalid_response_detail) from None
    if not isinstance(parsed, dict) or parsed.get("type") != "message":
        raise UpstreamError(502, invalid_response_detail)
    model = parsed.get("model")
    if not isinstance(model, str) or not model.strip():
        raise UpstreamError(502, invalid_response_detail)
    return model


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
        detail = _safe_route_target_error_detail(exc, config)
        return JSONResponse(
            server_support._openai_error_body("invalid_request_error", detail),
            status_code=400,
        )

    custom_provider = None
    if route.provider in config.custom_providers:
        try:
            custom_provider, backend = _get_custom_provider_binding(
                config, request.app.state.route_backends, route.provider
            )
        except RuntimeError:
            return JSONResponse(
                server_support._openai_error_body(
                    "server_error", "custom provider binding is unavailable"
                ),
                status_code=500,
            )
    else:
        backend = _get_route_backend(request, route.provider)
    try:
        if isinstance(backend, AnthropicBackend):
            probe = _probe_anthropic_route(backend, request, route.model)
        else:
            probe = _probe_responses_route(
                backend, route.provider, route.model
            )
        response_model = await asyncio.wait_for(probe, _CONNECTION_TEST_TIMEOUT)
    except UpstreamError as exc:
        detail = (
            _safe_custom_provider_upstream_detail(
                exc.body, custom_provider.api_key
            )
            if custom_provider is not None
            else server_support._upstream_error_message(exc.body)
        )
        return result(False, exc.status_code, detail)
    except UpstreamAuthError as exc:
        detail = (
            _safe_custom_provider_exception_detail(
                exc, custom_provider.api_key
            )
            if custom_provider is not None
            else str(exc)
        )
        return result(False, 401, detail)
    except TimeoutError:
        return result(False, None, f"no response within {_CONNECTION_TEST_TIMEOUT:.0f}s")
    except httpx.HTTPError as exc:
        detail = (
            _safe_custom_provider_exception_detail(
                exc, custom_provider.api_key
            )
            if custom_provider is not None
            else str(exc)
        )
        return result(False, None, f"failed to reach the upstream: {detail}")
    return result(True, 200, response_model=response_model)
