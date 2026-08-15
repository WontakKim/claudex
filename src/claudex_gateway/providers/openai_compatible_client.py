"""Streaming HTTP client for custom OpenAI-compatible Responses backends."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from claudex_gateway.config import OpenAICompatibleProvider
from claudex_gateway.providers.model_catalog_cache import ModelCatalogCache
from claudex_gateway.upstream_errors import UpstreamError


class OpenAICompatibleUpstreamError(UpstreamError):
    """Raised when a custom provider returns a non-success HTTP response."""

    def __init__(self, status_code: int, body: str, provider_name: str) -> None:
        super().__init__(status_code, body, provider_name)


def _coerce_context_window(value: Any) -> int | None:
    """Apply the catalog's context-window type policy."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if value > 0 else None
    return None


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
        async with self._http_client.stream(
            "POST", f"{self._base_url}/responses", json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise OpenAICompatibleUpstreamError(
                    response.status_code,
                    self._redact_api_key(body),
                    self._name,
                )

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
            window = _coerce_context_window(raw_window)
            if window is not None:
                windows[model_id] = window
        return windows

    async def _fetch_catalog_entries(self) -> list[Any]:
        """GET the live model catalog and return its `data` list."""
        headers = self._base_headers()
        headers["Accept"] = "application/json"
        response = await self._http_client.get(
            f"{self._base_url}/models", headers=headers
        )
        if response.status_code != 200:
            raise OpenAICompatibleUpstreamError(
                response.status_code,
                self._redact_api_key(response.text),
                self._name,
            )
        try:
            parsed = response.json()
        except ValueError as exc:
            raise OpenAICompatibleUpstreamError(
                502,
                f"{self._name} models response is not valid JSON",
                self._name,
            ) from exc
        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list):
            raise OpenAICompatibleUpstreamError(
                502,
                f"{self._name} models response has no data list",
                self._name,
            )
        return data

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
