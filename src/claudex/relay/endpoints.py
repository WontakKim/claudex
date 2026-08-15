"""HTTP endpoint handlers and top-level relay target dispatch."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from claudex import server_support
from claudex.config import GatewayConfig, RouteTarget
from claudex.relay import balanced as balanced_relay
from claudex.relay.balanced import _passthrough_with_claude_balanced
from claudex.relay.common import (
    _PASSTHROUGH_SKIP_REQUEST_HEADERS,
    _relay_anthropic_response,
    _send_to_anthropic,
)
from claudex.relay.kimi import _count_tokens_via_kimi, _relay_to_kimi
from claudex.relay.openai_backend import _relay_via_responses_backend
from claudex.relay.registered import (
    _passthrough_with_claude_account,
    _passthrough_with_claude_pool,
)

logger = logging.getLogger("claudex.server")


def _route_for_request(config: GatewayConfig, parsed: Any) -> RouteTarget | None:
    """Return the mapped route, or None for verbatim Anthropic passthrough.

    Only requests whose model has a model_map entry are handed to a backend;
    everything else — unmapped models and unparseable bodies alike — passes
    through so the gateway stays transparent for plain Anthropic traffic.
    """
    if not isinstance(parsed, dict):
        return None
    model = parsed.get("model")
    return config.mapped_route(model if isinstance(model, str) else None)


async def _passthrough_to_anthropic(
    request: Request, raw_body: bytes, parsed_body: Any
) -> JSONResponse | StreamingResponse:
    """Forward the request to the real Anthropic API and relay the response verbatim.

    With no claude_account.id configured, the client's own credentials and
    beta headers are forwarded untouched, so passthrough traffic behaves
    exactly as if Claude Code talked to Anthropic directly. With one
    configured, the request is served with the registered account instead —
    and when claude_account.routing selects the "fallback" mode, with the
    whole pool, serving account first (see `_passthrough_with_claude_pool`).
    The "balanced" mode is checked first, ahead of the claude_account.id
    gate: it spreads sessions across the whole registered pool by weighted
    HRW and never falls through to single-account or fallback routing. It
    dispatches only through an active `ClaudeBalancedRuntime`, otherwise it
    returns the reserved fail-closed 503.
    """
    config: GatewayConfig = request.app.state.config
    if config.claude_account_routing_mode == "balanced":
        return await _passthrough_with_claude_balanced(request, raw_body, parsed_body)
    if config.claude_account_id is not None:
        if config.claude_account_routing_mode == "fallback":
            return await _passthrough_with_claude_pool(
                request, raw_body, parsed_body, config.claude_account_id
            )
        return await _passthrough_with_claude_account(
            request, raw_body, parsed_body, config.claude_account_id
        )
    logger.info("anthropic passthrough: verbatim client credentials for %s", request.url.path)
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _PASSTHROUGH_SKIP_REQUEST_HEADERS
    }
    try:
        upstream_response = await _send_to_anthropic(request, headers, raw_body)
    except httpx.HTTPError as exc:
        logger.warning("anthropic passthrough failed: %s", exc)
        return JSONResponse(
            server_support._claude_error_body("api_error", f"failed to reach the Anthropic API: {exc}"),
            status_code=502,
        )
    return _relay_anthropic_response(upstream_response)


async def _handle_messages(request: Request) -> JSONResponse | StreamingResponse:
    unauthorized = server_support._require_local_token(request, claude_error=True)
    if unauthorized is not None:
        return unauthorized

    config: GatewayConfig = request.app.state.config

    raw_body = await request.body()
    try:
        claude_request = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        claude_request = None

    model = claude_request.get("model") if isinstance(claude_request, dict) else None
    route = _route_for_request(config, claude_request)
    if route is None:
        logger.info("%s -> anthropic passthrough", model or "?")
        return await _passthrough_to_anthropic(request, raw_body, claude_request)
    if route.provider == "kimi":
        return await _relay_to_kimi(request, claude_request, route.model)
    return await _relay_via_responses_backend(request, claude_request, route)


async def _handle_count_tokens(request: Request) -> JSONResponse | StreamingResponse:
    unauthorized = server_support._require_local_token(request, claude_error=True)
    if unauthorized is not None:
        return unauthorized

    config: GatewayConfig = request.app.state.config

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None

    route = _route_for_request(config, body)
    if route is None:
        return await _passthrough_to_anthropic(request, raw_body, body)
    if route.provider == "kimi":
        counted = await _count_tokens_via_kimi(request, body, route.model)
        if counted is not None:
            return counted

    # Rough characters/4 estimate: mapped prompts must not be sent to
    # Anthropic just to be counted, and no Codex tokenizer or token-count
    # endpoint is available (Kimi's native counter is preferred above but
    # may be down). Good enough for Claude Code's context-usage display;
    # not exact for billing or hard context-limit decisions.
    estimated = max(len(json.dumps(body, ensure_ascii=False)) // 4, 1)
    return JSONResponse({"input_tokens": estimated})

# Balanced transition dispatch resolves this endpoint-owned callback without a
# balanced-to-endpoints import edge.
balanced_relay._passthrough_to_anthropic = _passthrough_to_anthropic
