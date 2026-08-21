"""Kimi-specific policies for the native Anthropic Messages binding."""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

import httpx
from starlette.requests import Request

from claudex import server_support
from claudex.providers.backends import AnthropicRelayError, AnthropicStreamReadFailure
from claudex.relay.common import (
    _MANAGED_RELAY_SKIP_REQUEST_HEADERS,
    _OAUTH_BETA,
    _STATUS_TO_CLAUDE_ERROR_TYPE,
)
from claudex.upstream_errors import UpstreamAuthError, UpstreamError

logger = logging.getLogger("claudex.server")


def _kimi_request_headers(request: Request) -> dict[str, str]:
    """Forward the client's headers with the gateway's OAuth identity.

    The caller is real Claude Code, so its own fingerprint and beta headers
    are kept; only credentials are replaced (by KimiClient) and the OAuth
    beta is guaranteed to be present.
    """
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _MANAGED_RELAY_SKIP_REQUEST_HEADERS
    }
    headers.setdefault("anthropic-version", "2023-06-01")
    betas = [
        beta.strip()
        for beta in headers.get("anthropic-beta", "").split(",")
        if beta.strip()
    ]
    if _OAUTH_BETA not in betas:
        betas.append(_OAUTH_BETA)
    headers["anthropic-beta"] = ",".join(betas)
    return headers


def _kimi_error_to_claude(exc: AnthropicRelayError) -> tuple[int, dict[str, Any]]:
    """Map Kimi transport, credential, and stream failures to Claude errors."""
    if isinstance(exc, AnthropicStreamReadFailure):
        logger.warning("kimi stream aborted: %r", exc.error)
        return 502, server_support._claude_error_body(
            "api_error", f"kimi stream aborted: {exc.error!r}"
        )
    if isinstance(exc, UpstreamError):
        logger.warning("kimi upstream error %s: %s", exc.status_code, exc.body[:500])
        if exc.status_code == 401:
            # A post-retry 401 means the gateway's credential is bad, not the
            # client's; relaying it verbatim would trigger a Claude Code re-auth.
            return 401, server_support._claude_error_body(
                "authentication_error",
                "Kimi rejected the gateway credentials: "
                f"{server_support._upstream_error_message(exc.body)}; "
                "run `kimi login` again",
            )
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(exc.body)
            if isinstance(parsed, dict) and parsed.get("type") == "error":
                # Kimi speaks the Anthropic error shape natively; relay it.
                return exc.status_code, parsed
        error_type = _STATUS_TO_CLAUDE_ERROR_TYPE.get(exc.status_code, "api_error")
        return exc.status_code, server_support._claude_error_body(
            error_type, server_support._upstream_error_message(exc.body)
        )
    if isinstance(exc, UpstreamAuthError):
        return 401, server_support._claude_error_body(
            "authentication_error", str(exc)
        )
    if isinstance(exc, httpx.HTTPError):
        logger.warning("kimi backend unreachable: %r", exc)
        return 502, server_support._claude_error_body(
            "api_error", f"failed to reach the Kimi backend: {exc!r}"
        )
    raise TypeError(f"unsupported Kimi relay error: {type(exc).__name__}")
