"""Streaming HTTP client for the xAI Grok Responses backend.

xAI speaks the same Responses API family as the Codex backend, so the Claude
translation layer is reused wholesale; this module owns only the xAI-side
wire quirks — the chat-proxy endpoint, its identity headers, and the payload
fields xAI rejects. Ported from router-for-me/CLIProxyAPI's xAI executor.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from claudex_gateway.xai_auth import XAIAuthManager, XAICredentials

XAI_RESPONSES_URL = "https://cli-chat-proxy.grok.com/v1/responses"
XAI_MODELS_URL = "https://cli-chat-proxy.grok.com/v1/models"

# Identity headers the Grok CLI chat-proxy expects; mirrors CLIProxyAPI's
# applyXAIChatHeaders for the OAuth (non-official-API) path.
_XAI_TOKEN_AUTH_HEADER = "X-XAI-Token-Auth"
_XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
_XAI_CLIENT_VERSION = "0.2.93"

# Fields CLIProxyAPI strips before forwarding to xAI: accepted by the Codex
# backend but rejected (or silently harmful) on xAI's Responses surface.
_XAI_UNSUPPORTED_FIELDS = (
    "previous_response_id",
    "prompt_cache_retention",
    "safety_identifier",
    "stream_options",
    "stop",
)

# Models whose registry entry carries thinking levels (low/medium/high), per
# CLIProxyAPI's catalog. Anything else gets reasoning stripped entirely —
# sending an effort to a non-thinking model fails upstream, while a newly
# released thinking model merely runs at its default until listed here.
_XAI_THINKING_MODELS = frozenset(
    {
        "grok-4.5",
        "grok-4.3",
        "grok-4.20-multi-agent-0309",
        "grok-3-mini",
        "grok-3-mini-fast",
    }
)

# The Claude-side effort vocabulary is wider than xAI's low/medium/high.
_XAI_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class XAIUpstreamError(Exception):
    """Raised when the xAI backend returns a non-success HTTP response."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"xai upstream returned {status_code}: {body[:2000]}")
        self.status_code = status_code
        self.body = body


def sanitize_xai_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Adapt a Codex-shaped Responses payload to what xAI's backend accepts."""
    sanitized = {key: value for key, value in payload.items() if key not in _XAI_UNSUPPORTED_FIELDS}
    if model in _XAI_THINKING_MODELS:
        reasoning = sanitized.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if isinstance(effort, str):
                reasoning["effort"] = _XAI_EFFORT_MAP.get(effort.strip().lower(), "medium")
    else:
        sanitized.pop("reasoning", None)
    return sanitized


class XAIClient:
    def __init__(self, auth_manager: XAIAuthManager, http_client: httpx.AsyncClient) -> None:
        self._auth_manager = auth_manager
        self._http_client = http_client

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """POST the Responses payload and yield each SSE data event as a dict.

        Retries exactly once with force-refreshed credentials on HTTP 401.
        """
        credentials = await self._auth_manager.get_credentials()
        try:
            async for event in self._stream_once(payload, session_id, credentials):
                yield event
            return
        except XAIUpstreamError as exc:
            if exc.status_code != 401:
                raise

        credentials = await self._auth_manager.get_credentials(force_refresh=True)
        async for event in self._stream_once(payload, session_id, credentials):
            yield event

    async def list_models(self) -> list[str]:
        """Return the model IDs from the live catalog (OpenAI list shape)."""
        credentials = await self._auth_manager.get_credentials()
        headers = self._base_headers(credentials)
        headers["Accept"] = "application/json"
        response = await self._http_client.get(XAI_MODELS_URL, headers=headers)
        if response.status_code != 200:
            raise XAIUpstreamError(response.status_code, response.text)
        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise XAIUpstreamError(502, "xai models response is not valid JSON") from exc
        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list):
            raise XAIUpstreamError(502, "xai models response has no data list")
        return [
            model["id"]
            for model in data
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]

    @staticmethod
    def _base_headers(credentials: XAICredentials) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "Keep-Alive",
            _XAI_TOKEN_AUTH_HEADER: _XAI_TOKEN_AUTH_VALUE,
            "x-grok-client-version": _XAI_CLIENT_VERSION,
            "User-Agent": f"xai-grok-workspace/{_XAI_CLIENT_VERSION}",
        }

    async def _stream_once(
        self, payload: dict[str, Any], session_id: str, credentials: XAICredentials
    ) -> AsyncIterator[dict[str, Any]]:
        headers = self._base_headers(credentials)
        headers["x-grok-conv-id"] = session_id

        async with self._http_client.stream(
            "POST", XAI_RESPONSES_URL, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise XAIUpstreamError(response.status_code, body)

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
