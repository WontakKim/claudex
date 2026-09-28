"""Streaming HTTP client for the ChatGPT Codex Responses backend."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import time
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

# The models endpoint requires an explicit client_version; this is the last
# verified bundled stable identity, not a substitute for an absent CLI catalog.
_CODEX_CLIENT_VERSION = "0.157.1"
_CODEX_VERSION_PROBE_TIMEOUT = 2.0
_CODEX_VERSION_CLEANUP_TIMEOUT = 1.0
_CODEX_VERSION_CACHE_SECONDS = 60.0
_CODEX_VERSION_OUTPUT = re.compile(r"codex-cli (\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\Z")
_CODEX_PRESET_MODELS = [
    "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol",
    "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
]
_CODEX_ORIGINATOR = "codex-tui"


def _codex_user_agent(version: str) -> str:
    # Mirrors the header set CLIProxyAPI sends; the backend rejects unknown
    # clients and downgrades gpt-5.6-luna for versions older than 0.144.0.
    return (
        f"codex-tui/{version} (Mac OS 26.5.1; arm64) "
        f"iTerm.app/3.6.11 (codex-tui; {version})"
    )


class CodexDiscoveryError(Exception):
    """A present Codex CLI could not provide a usable version."""


class CodexVersionDiscovery:
    def __init__(self) -> None:
        self._clock = time.monotonic
        self._lock = asyncio.Lock()
        self._version: str | None = None
        self._checked_at: float | None = None

    @property
    def current_version(self) -> str:
        return self._version or _CODEX_CLIENT_VERSION

    async def get_version(self) -> str | None:
        now = self._clock()
        if self._checked_at is not None and now - self._checked_at < _CODEX_VERSION_CACHE_SECONDS:
            return self._version
        async with self._lock:
            now = self._clock()
            if self._checked_at is not None and now - self._checked_at < _CODEX_VERSION_CACHE_SECONDS:
                return self._version
            version = await self._probe_version()
            self._version = version
            self._checked_at = self._clock()
            return version

    async def _probe_version(self) -> str | None:
        try:
            process = await asyncio.create_subprocess_exec(
                "codex", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            if shutil.which("codex") is not None:
                raise CodexDiscoveryError(f"could not run codex --version: {exc}") from exc
            return None
        except OSError as exc:
            raise CodexDiscoveryError(f"could not run codex --version: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_CODEX_VERSION_PROBE_TIMEOUT
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            try:
                if os.name == "posix":
                    # The npm launcher can exit while a native child still owns
                    # stdout/stderr, so killing only the launcher cannot close the pipes.
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    process.communicate(), timeout=_CODEX_VERSION_CLEANUP_TIMEOUT
                )
            except TimeoutError:
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=_CODEX_VERSION_CLEANUP_TIMEOUT
                    )
                except TimeoutError as cleanup_error:
                    if isinstance(exc, asyncio.CancelledError):
                        raise exc
                    raise CodexDiscoveryError(
                        "codex --version timed out and could not reap its process"
                    ) from cleanup_error
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise CodexDiscoveryError("codex --version timed out") from exc

        if process.returncode != 0:
            raise CodexDiscoveryError(
                f"codex --version exited with status {process.returncode}: "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
        match = _CODEX_VERSION_OUTPUT.fullmatch(stdout.decode("utf-8", errors="replace").strip())
        if match is None:
            raise CodexDiscoveryError("codex --version returned an invalid version")
        version = match.group(1)
        try:
            numbers = tuple(int(part) for part in version.split("-", 1)[0].split("."))
        except ValueError as exc:
            raise CodexDiscoveryError("codex --version returned an invalid version") from exc
        floor = tuple(int(part) for part in _CODEX_CLIENT_VERSION.split("."))
        return version if numbers > floor else _CODEX_CLIENT_VERSION


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
        self._version_discovery = CodexVersionDiscovery()
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
        """Return live visible slugs when installed, otherwise preset suggestions."""
        version = await self._version_discovery.get_version()
        if version is None:
            return list(_CODEX_PRESET_MODELS)
        models = await self._fetch_model_entries(version)
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
            context_window = coerce_context_window(model.get("context_window"))
            max_context_window = coerce_context_window(model.get("max_context_window"))
            # The catalog's context_window is a conservative client default and
            # max_context_window is the server's actual input ceiling, so the
            # larger valid value is the enforceable window.
            effective_context_window = max(
                (
                    window
                    for window in (context_window, max_context_window)
                    if window is not None
                ),
                default=None,
            )
            entries[slug] = CodexModelEntry(
                context_window=effective_context_window,
                supports_fast_tier=supports_fast_tier,
            )
        return entries

    async def _fetch_model_entries(self, client_version: str | None = None) -> list[Any]:
        """GET the Codex model catalog and return its raw ``models`` list.

        Raises ``CodexUpstreamError`` on any structural failure: a non-200
        response, a non-JSON body, a non-object JSON root, or a missing/
        non-list ``models`` field.
        """
        version = client_version or self._version_discovery.current_version
        credentials = await self._auth_manager.get_credentials()
        headers = self._base_headers(credentials, version)
        headers["Accept"] = "application/json"
        return await fetch_models_list(
            self._http_client,
            CODEX_MODELS_URL,
            headers,
            label="codex",
            make_error=CodexUpstreamError,
            params={"client_version": version},
            items_key="models",
            require_object_root=True,
        )

    @staticmethod
    def _base_headers(credentials: CodexCredentials, version: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "User-Agent": _codex_user_agent(version),
        }
        if not credentials.is_api_key:
            headers["Originator"] = _CODEX_ORIGINATOR
            if credentials.account_id:
                headers["Chatgpt-Account-Id"] = credentials.account_id
        return headers

    async def _stream_once(
        self, payload: dict[str, Any], session_id: str, credentials: CodexCredentials
    ) -> AsyncIterator[dict[str, Any]]:
        headers = self._base_headers(credentials, self._version_discovery.current_version)
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
