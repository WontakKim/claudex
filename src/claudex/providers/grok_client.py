"""Streaming HTTP client for the Grok Responses backend.

Grok speaks the same Responses API family as the Codex backend, so the Claude
translation layer is reused wholesale; this module owns only the Grok-side
wire quirks — the chat-proxy endpoint, its identity headers, and the payload
fields Grok rejects. Ported from router-for-me/CLIProxyAPI's Grok executor.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx

from claudex.providers.client_support import (
    coerce_context_window,
    fetch_models_list,
    stream_sse_events,
    stream_with_one_retry,
)
from claudex.providers.grok_auth import GrokAuthError, GrokAuthManager, GrokCredentials
from claudex.providers.model_catalog_cache import ModelCatalogCache
from claudex.upstream_errors import UpstreamError

logger = logging.getLogger(__name__)

GROK_RESPONSES_URL = "https://cli-chat-proxy.grok.com/v1/responses"
GROK_MODELS_URL = "https://cli-chat-proxy.grok.com/v1/models"

# Identity headers the Grok CLI chat-proxy expects; mirrors CLIProxyAPI's
# applyXAIChatHeaders for the OAuth (non-official-API) path.
_XAI_TOKEN_AUTH_HEADER = "X-XAI-Token-Auth"
_XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
# The verified bundled identity used when the local CLI is unavailable.
_GROK_CLIENT_VERSION = "0.2.93"
_GROK_VERSION_PROBE_TIMEOUT = 2.0
_GROK_VERSION_CLEANUP_TIMEOUT = 1.0
_GROK_VERSION_CACHE_SECONDS = 60.0
_GROK_VERSION_OUTPUT = re.compile(r"^grok (\d+\.\d+\.\d+)\b")

# Fields CLIProxyAPI strips before forwarding to Grok: accepted by the Codex
# backend but rejected (or silently harmful) on Grok's Responses surface.
_GROK_UNSUPPORTED_FIELDS = (
    "previous_response_id",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
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


class GrokVersionDiscovery:
    def __init__(self) -> None:
        self._clock = time.monotonic
        self._lock = asyncio.Lock()
        self._version = _GROK_CLIENT_VERSION
        self._checked_at: float | None = None

    @property
    def current_version(self) -> str:
        return self._version

    async def get_version(self) -> str:
        now = self._clock()
        if self._checked_at is not None and now - self._checked_at < _GROK_VERSION_CACHE_SECONDS:
            return self._version
        async with self._lock:
            now = self._clock()
            if self._checked_at is not None and now - self._checked_at < _GROK_VERSION_CACHE_SECONDS:
                return self._version
            self._version = await self._probe_version()
            self._checked_at = self._clock()
            return self._version

    async def _probe_version(self) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                "grok", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _GROK_CLIENT_VERSION
        except OSError as exc:
            logger.warning("could not run grok --version: %s", exc)
            return _GROK_CLIENT_VERSION

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_GROK_VERSION_PROBE_TIMEOUT
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    process.communicate(), timeout=_GROK_VERSION_CLEANUP_TIMEOUT
                )
            except TimeoutError:
                if isinstance(exc, asyncio.CancelledError):
                    raise exc
                logger.warning("grok --version timed out and could not reap its process")
                return _GROK_CLIENT_VERSION
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("grok --version timed out")
            return _GROK_CLIENT_VERSION

        if process.returncode != 0:
            logger.warning(
                "grok --version exited with status %s: %s",
                process.returncode, stderr.decode("utf-8", errors="replace").strip(),
            )
            return _GROK_CLIENT_VERSION
        first_line = stdout.decode("utf-8", errors="replace").splitlines()
        match = _GROK_VERSION_OUTPUT.match(first_line[0]) if first_line else None
        if match is None:
            logger.warning("grok --version returned an invalid version")
            return _GROK_CLIENT_VERSION
        version = match.group(1)
        numbers = tuple(int(part) for part in version.split("."))
        floor = tuple(int(part) for part in _GROK_CLIENT_VERSION.split("."))
        return version if numbers > floor else _GROK_CLIENT_VERSION


class GrokUpstreamError(UpstreamError):
    """Raised when the Grok backend returns a non-success HTTP response."""

    provider_label = "grok"


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
        self._version_discovery = GrokVersionDiscovery()
        self._context_windows: ModelCatalogCache[int] = ModelCatalogCache(
            self._fetch_context_windows,
            expected_errors=(GrokAuthError, GrokUpstreamError, httpx.HTTPError),
        )

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """POST the Responses payload and yield each SSE data event as a dict.

        Retries exactly once with force-refreshed credentials on HTTP 401.
        """
        async for event in stream_with_one_retry(
            self._auth_manager.get_credentials,
            lambda credentials: self._stream_once(payload, session_id, credentials),
            upstream_error=GrokUpstreamError,
            should_retry=lambda exc, credentials: exc.status_code == 401,
        ):
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
            window = coerce_context_window(entry.get("context_window"))
            if window is not None:
                windows[model_id] = window
        return windows

    async def _fetch_catalog_entries(self) -> list[Any]:
        """GET the live model catalog and return its `data` list.

        Raises `GrokUpstreamError` on any structural failure: a non-200
        response, invalid JSON, a non-object JSON root, or a missing/
        non-list `data` field.
        """
        version = await self._version_discovery.get_version()
        credentials = await self._auth_manager.get_credentials()
        headers = self._base_headers(credentials, version)
        headers["Accept"] = "application/json"
        return await fetch_models_list(
            self._http_client,
            GROK_MODELS_URL,
            headers,
            label="grok",
            make_error=GrokUpstreamError,
        )

    @staticmethod
    def _base_headers(credentials: GrokCredentials, version: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "Keep-Alive",
            _XAI_TOKEN_AUTH_HEADER: _XAI_TOKEN_AUTH_VALUE,
            "x-grok-client-version": version,
            "User-Agent": f"xai-grok-workspace/{version}",
        }

    async def _stream_once(
        self, payload: dict[str, Any], session_id: str, credentials: GrokCredentials
    ) -> AsyncIterator[dict[str, Any]]:
        version = await self._version_discovery.get_version()
        headers = self._base_headers(credentials, version)
        headers["x-grok-conv-id"] = session_id

        async with aclosing(
            stream_sse_events(
                self._http_client,
                GROK_RESPONSES_URL,
                payload,
                headers,
                make_error=GrokUpstreamError,
            )
        ) as events:
            async for event in events:
                yield event
