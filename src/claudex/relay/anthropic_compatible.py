"""Policies for static Anthropic Messages-compatible backends."""

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

_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_HOP_BY_HOP_REQUEST_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_STATIC_SKIP_REQUEST_HEADERS = (
    _MANAGED_RELAY_SKIP_REQUEST_HEADERS | _HOP_BY_HOP_REQUEST_HEADERS
)


def _anthropic_compatible_request_headers(request: Request) -> dict[str, str]:
    """Keep safe Anthropic headers while removing credentials and connection state."""
    raw_headers = [
        (key.decode("latin-1").lower(), value.decode("latin-1"))
        for key, value in request.headers.raw
    ]
    connection_tokens = {
        token.strip().lower()
        for key, value in raw_headers
        if key == "connection"
        for token in value.split(",")
        if token.strip()
    }

    headers: dict[str, str] = {}
    beta_values: list[str] = []
    for key, value in raw_headers:
        if key in _STATIC_SKIP_REQUEST_HEADERS or key in connection_tokens:
            continue
        if key == "anthropic-beta":
            beta_values.append(value)
            continue
        headers[key] = value
    headers.setdefault("anthropic-version", _DEFAULT_ANTHROPIC_VERSION)

    betas = [
        beta.strip()
        for value in beta_values
        for beta in value.split(",")
        if beta.strip() and beta.strip().lower() != _OAUTH_BETA
    ]
    if betas:
        headers["anthropic-beta"] = ",".join(betas)
    return headers


def _safe_anthropic_error(body: str) -> dict[str, Any] | None:
    with contextlib.suppress(ValueError, RecursionError):
        parsed = json.loads(body)
        if not isinstance(parsed, dict) or parsed.get("type") != "error":
            return None
        detail = parsed.get("error")
        if not (
            isinstance(detail, dict)
            and isinstance(detail.get("type"), str)
            and isinstance(detail.get("message"), str)
        ):
            return None
        try:
            json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
            return None
        return parsed
    return None


def _safe_upstream_message(body: str) -> str:
    try:
        message = server_support._upstream_error_message(body)
    except (ValueError, RecursionError):
        message = body
    return message.encode("utf-8", errors="replace").decode("utf-8")


def _configured_credential_error() -> dict[str, Any]:
    return server_support._claude_error_body(
        "authentication_error",
        "The Anthropic-compatible backend rejected the configured custom-provider API key",
    )


def _anthropic_compatible_error_to_claude(
    exc: AnthropicRelayError,
) -> tuple[int, dict[str, Any]]:
    """Map static-credential transport failures to Anthropic error responses."""
    if isinstance(exc, AnthropicStreamReadFailure):
        logger.warning(
            "Anthropic-compatible stream aborted: %s", type(exc.error).__name__
        )
        return 502, server_support._claude_error_body(
            "api_error", "The Anthropic-compatible backend stream was interrupted"
        )
    if isinstance(exc, UpstreamError):
        logger.warning("Anthropic-compatible upstream error %s", exc.status_code)
        if exc.status_code == 401:
            return 401, _configured_credential_error()
        preserved = _safe_anthropic_error(exc.body)
        if preserved is not None:
            return exc.status_code, preserved
        error_type = _STATUS_TO_CLAUDE_ERROR_TYPE.get(
            exc.status_code, "api_error"
        )
        return exc.status_code, server_support._claude_error_body(
            error_type, _safe_upstream_message(exc.body)
        )
    if isinstance(exc, UpstreamAuthError):
        return 401, _configured_credential_error()
    if isinstance(exc, httpx.HTTPError):
        logger.warning(
            "Anthropic-compatible backend unreachable: %s", type(exc).__name__
        )
        return 502, server_support._claude_error_body(
            "api_error", "Failed to reach the Anthropic-compatible backend"
        )
    raise TypeError(
        f"unsupported Anthropic-compatible relay error: {type(exc).__name__}"
    )
