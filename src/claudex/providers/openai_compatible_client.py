"""Streaming HTTP client for custom OpenAI-compatible Responses backends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx

from claudex.config.schema import OpenAICompatibleProvider
from claudex.providers.client_support import (
    coerce_context_window,
    fetch_models_list,
    stream_sse_events,
)
from claudex.providers.model_catalog_cache import ModelCatalogCache
from claudex.upstream_errors import UpstreamError


class OpenAICompatibleUpstreamError(UpstreamError):
    """Raised when a custom provider returns a non-success HTTP response."""


class OpenAICompatibleClient:
    def __init__(
        self,
        name: str,
        provider: OpenAICompatibleProvider,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._name = name
        self._base_url = provider.base_url
        self._api_key = provider.api_key
        self._http_client = http_client
        self._context_windows: ModelCatalogCache[int] = ModelCatalogCache(
            self._fetch_context_windows,
            expected_errors=(OpenAICompatibleUpstreamError, httpx.HTTPError),
        )

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """POST a Responses payload and yield each SSE data event as a dict."""
        headers = self._base_headers()
        async with aclosing(
            stream_sse_events(
                self._http_client,
                f"{self._base_url}/responses",
                payload,
                headers,
                make_error=lambda code, body: OpenAICompatibleUpstreamError(
                    code, self._redact_api_key(body), self._name
                ),
            )
        ) as events:
            async for event in events:
                yield event

    async def list_models(self) -> list[str]:
        """Return the model IDs from the live OpenAI-shaped catalog."""
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

            if "context_window" in entry:
                raw_window = entry["context_window"]
            else:
                limits = entry.get("limits")
                raw_window = limits.get("contextWindow") if isinstance(limits, dict) else None
            window = coerce_context_window(raw_window)
            if window is not None:
                windows[model_id] = window
        return windows

    async def _fetch_catalog_entries(self) -> list[Any]:
        """GET the live model catalog and return its `data` list."""
        headers = self._base_headers()
        headers["Accept"] = "application/json"
        return await fetch_models_list(
            self._http_client,
            f"{self._base_url}/models",
            headers,
            label=self._name,
            make_error=lambda code, body: OpenAICompatibleUpstreamError(
                code, body, self._name
            ),
            redact=self._redact_api_key,
        )

    def _base_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _redact_api_key(self, text: str) -> str:
        if not self._api_key:
            return text
        return text.replace(self._api_key, "[REDACTED]")
