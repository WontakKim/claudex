"""Streaming HTTP client for the ChatGPT Codex Responses backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

import httpx

from claudex.providers.client_support import (
    coerce_context_window,
    fetch_models_list,
    stream_sse_events,
    stream_with_one_retry,
)
from claudex.providers.codex_auth import CodexAuthError, CodexAuthManager, CodexCredentials
from claudex.providers.model_catalog_cache import ModelCatalogCache
from claudex.upstream_errors import UpstreamError

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
# The UI name is "Fast", but the wire keeps the legacy pre-rename value.
CODEX_FAST_TIER_WIRE_VALUE = "priority"

# Mirrors the header set CLIProxyAPI sends; the backend rejects unknown clients
# and silently downgrades gpt-5.6-luna requests from clients older than 0.144.0.
_CODEX_USER_AGENT = "codex-tui/0.144.0 (Mac OS 26.5.1; arm64) iTerm.app/3.6.11 (codex-tui; 0.144.0)"
_CODEX_ORIGINATOR = "codex-tui"
# The models endpoint 400s without an explicit client_version query parameter.
_CODEX_CLIENT_VERSION = "0.146.0"


class CodexUpstreamError(UpstreamError):
    """Raised when the Codex backend returns a non-success HTTP response."""

    provider_label = "codex"


@dataclass(frozen=True)
class CodexModelEntry:
    context_window: int | None
    supports_fast_tier: bool


class CodexClient:
    def __init__(self, auth_manager: CodexAuthManager, http_client: httpx.AsyncClient) -> None:
        self._auth_manager = auth_manager
        self._http_client = http_client
        self._catalog_entries: ModelCatalogCache[CodexModelEntry] = ModelCatalogCache(
            self._fetch_catalog_entries,
            expected_errors=(CodexAuthError, CodexUpstreamError, httpx.HTTPError),
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
            upstream_error=CodexUpstreamError,
            should_retry=lambda exc, credentials: (
                exc.status_code == 401 and not credentials.is_api_key
            ),
        ):
            yield event

    async def list_models(self) -> list[str]:
        """Return the visible Codex model slugs from the live catalog."""
        models = await self._fetch_model_entries()
        return [
            model["slug"]
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("slug"), str)
            and model.get("visibility") != "hide"
        ]

    async def context_window(self, model: str) -> int | None:
        """Return the cached context-window size for ``model``, or ``None``."""
        entry = await self._catalog_entries.get(model)
        return entry.context_window if entry else None

    async def supports_fast_tier(self, model: str) -> bool:
        """Return whether the live catalog lists the Fast tier for ``model``."""
        entry = await self._catalog_entries.get(model)
        return entry.supports_fast_tier if entry is not None else False

    async def _fetch_catalog_entries(self) -> dict[str, CodexModelEntry]:
        """Resolve slug -> catalog entries from the raw, unfiltered catalog."""
        models = await self._fetch_model_entries()
        entries: dict[str, CodexModelEntry] = {}
        for model in models:
            if not isinstance(model, dict):
                continue
            slug = model.get("slug")
            if not isinstance(slug, str) or not slug:
                continue
            service_tiers = model.get("service_tiers")
            supports_fast_tier = isinstance(service_tiers, list) and any(
                isinstance(tier, dict)
                and tier.get("id") == CODEX_FAST_TIER_WIRE_VALUE
                for tier in service_tiers
            )
            entries[slug] = CodexModelEntry(
                context_window=coerce_context_window(model.get("context_window")),
                supports_fast_tier=supports_fast_tier,
            )
        return entries

    async def _fetch_model_entries(self) -> list[Any]:
        """GET the Codex model catalog and return its raw ``models`` list.

        Raises ``CodexUpstreamError`` on any structural failure: a non-200
        response, a non-JSON body, a non-object JSON root, or a missing/
        non-list ``models`` field.
        """
        credentials = await self._auth_manager.get_credentials()
        headers = self._base_headers(credentials)
        headers["Accept"] = "application/json"
        return await fetch_models_list(
            self._http_client,
            CODEX_MODELS_URL,
            headers,
            label="codex",
            make_error=CodexUpstreamError,
            params={"client_version": _CODEX_CLIENT_VERSION},
            items_key="models",
            require_object_root=True,
        )

    @staticmethod
    def _base_headers(credentials: CodexCredentials) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "User-Agent": _CODEX_USER_AGENT,
        }
        if not credentials.is_api_key:
            headers["Originator"] = _CODEX_ORIGINATOR
            if credentials.account_id:
                headers["Chatgpt-Account-Id"] = credentials.account_id
        return headers

    async def _stream_once(
        self, payload: dict[str, Any], session_id: str, credentials: CodexCredentials
    ) -> AsyncIterator[dict[str, Any]]:
        headers = self._base_headers(credentials)
        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Session_id": session_id,
            }
        )
        service_tier = payload.get("service_tier")
        if service_tier:
            headers["x-codex-routing-hint"] = f"model={payload['model']};tier={service_tier}"

        async with aclosing(
            stream_sse_events(
                self._http_client,
                CODEX_RESPONSES_URL,
                payload,
                headers,
                make_error=CodexUpstreamError,
            )
        ) as events:
            async for event in events:
                yield event
