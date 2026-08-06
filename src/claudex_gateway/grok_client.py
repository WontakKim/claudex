"""Streaming HTTP client for the Grok Responses backend.

Grok speaks the same Responses API family as the Codex backend, so the Claude
translation layer is reused wholesale; this module owns only the Grok-side
wire quirks — the chat-proxy endpoint, its identity headers, and the payload
fields Grok rejects. Ported from router-for-me/CLIProxyAPI's Grok executor.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from claudex_gateway.context_window_cache import ContextWindowCache
from claudex_gateway.grok_auth import GrokAuthError, GrokAuthManager, GrokCredentials

GROK_RESPONSES_URL = "https://cli-chat-proxy.grok.com/v1/responses"
GROK_MODELS_URL = "https://cli-chat-proxy.grok.com/v1/models"

# Identity headers the Grok CLI chat-proxy expects; mirrors CLIProxyAPI's
# applyXAIChatHeaders for the OAuth (non-official-API) path.
_XAI_TOKEN_AUTH_HEADER = "X-XAI-Token-Auth"
_XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
_GROK_CLIENT_VERSION = "0.2.93"

# Fields CLIProxyAPI strips before forwarding to Grok: accepted by the Codex
# backend but rejected (or silently harmful) on Grok's Responses surface.
_GROK_UNSUPPORTED_FIELDS = (
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
_GROK_THINKING_MODELS = frozenset(
    {
        "grok-4.5",
        "grok-4.3",
        "grok-4.20-multi-agent-0309",
        "grok-3-mini",
        "grok-3-mini-fast",
    }
)

# The Claude-side effort vocabulary is wider than Grok's low/medium/high.
_GROK_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class GrokUpstreamError(Exception):
    """Raised when the Grok backend returns a non-success HTTP response."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"grok upstream returned {status_code}: {body[:2000]}")
        self.status_code = status_code
        self.body = body


def _coerce_context_window(value: Any) -> int | None:
    """Apply the catalog's context-window type policy.

    Accepts a positive `int` (excluding `bool`, which is technically an
    `int` subclass) or a positive, integral `float` coerced to `int`.
    Everything else — missing, non-numeric, zero, negative, or fractional —
    yields `None`.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if value > 0 else None
    return None


def sanitize_grok_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """Adapt a Codex-shaped Responses payload to what Grok's backend accepts."""
    sanitized = {key: value for key, value in payload.items() if key not in _GROK_UNSUPPORTED_FIELDS}
    if model in _GROK_THINKING_MODELS:
        reasoning = sanitized.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if isinstance(effort, str):
                reasoning["effort"] = _GROK_EFFORT_MAP.get(effort.strip().lower(), "medium")
    else:
        sanitized.pop("reasoning", None)
    return sanitized


class GrokClient:
    def __init__(self, auth_manager: GrokAuthManager, http_client: httpx.AsyncClient) -> None:
        self._auth_manager = auth_manager
        self._http_client = http_client
        self._context_windows = ContextWindowCache(
            self._fetch_context_windows,
            expected_errors=(GrokAuthError, GrokUpstreamError, httpx.HTTPError),
        )

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
        except GrokUpstreamError as exc:
            if exc.status_code != 401:
                raise

        credentials = await self._auth_manager.get_credentials(force_refresh=True)
        async for event in self._stream_once(payload, session_id, credentials):
            yield event

    async def list_models(self) -> list[str]:
        """Return the model IDs from the live catalog (OpenAI list shape)."""
        data = await self._fetch_catalog_entries()
        return [
            model["id"]
            for model in data
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]

    async def context_window(self, model: str) -> int | None:
        """Return the model's context window size from the cached catalog."""
        return await self._context_windows.get(model)

    async def _fetch_context_windows(self) -> dict[str, int]:
        """Fetch the catalog and map each valid entry's id to its context window."""
        data = await self._fetch_catalog_entries()
        windows: dict[str, int] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            window = _coerce_context_window(entry.get("context_window"))
            if window is not None:
                windows[model_id] = window
        return windows

    async def _fetch_catalog_entries(self) -> list[Any]:
        """GET the live model catalog and return its `data` list.

        Raises `GrokUpstreamError` on any structural failure: a non-200
        response, invalid JSON, a non-object JSON root, or a missing/
        non-list `data` field.
        """
        credentials = await self._auth_manager.get_credentials()
        headers = self._base_headers(credentials)
        headers["Accept"] = "application/json"
        response = await self._http_client.get(GROK_MODELS_URL, headers=headers)
        if response.status_code != 200:
            raise GrokUpstreamError(response.status_code, response.text)
        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise GrokUpstreamError(502, "grok models response is not valid JSON") from exc
        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list):
            raise GrokUpstreamError(502, "grok models response has no data list")
        return data

    @staticmethod
    def _base_headers(credentials: GrokCredentials) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "Keep-Alive",
            _XAI_TOKEN_AUTH_HEADER: _XAI_TOKEN_AUTH_VALUE,
            "x-grok-client-version": _GROK_CLIENT_VERSION,
            "User-Agent": f"xai-grok-workspace/{_GROK_CLIENT_VERSION}",
        }

    async def _stream_once(
        self, payload: dict[str, Any], session_id: str, credentials: GrokCredentials
    ) -> AsyncIterator[dict[str, Any]]:
        headers = self._base_headers(credentials)
        headers["x-grok-conv-id"] = session_id

        async with self._http_client.stream(
            "POST", GROK_RESPONSES_URL, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise GrokUpstreamError(response.status_code, body)

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
