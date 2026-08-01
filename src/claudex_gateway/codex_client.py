"""Streaming HTTP client for the ChatGPT Codex Responses backend."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from claudex_gateway.codex_auth import CodexAuthManager, CodexCredentials

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"

# Mirrors the header set CLIProxyAPI sends; the backend rejects unknown clients
# and silently downgrades gpt-5.6-luna requests from clients older than 0.144.0.
_CODEX_USER_AGENT = "codex-tui/0.144.0 (Mac OS 26.5.1; arm64) iTerm.app/3.6.11 (codex-tui; 0.144.0)"
_CODEX_ORIGINATOR = "codex-tui"
# The models endpoint 400s without an explicit client_version query parameter.
_CODEX_CLIENT_VERSION = "0.146.0"


class CodexUpstreamError(Exception):
    """Raised when the Codex backend returns a non-success HTTP response."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"codex upstream returned {status_code}: {body[:2000]}")
        self.status_code = status_code
        self.body = body


class CodexClient:
    def __init__(self, auth_manager: CodexAuthManager, http_client: httpx.AsyncClient) -> None:
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
        except CodexUpstreamError as exc:
            if exc.status_code != 401 or credentials.is_api_key:
                raise

        credentials = await self._auth_manager.get_credentials(force_refresh=True)
        async for event in self._stream_once(payload, session_id, credentials):
            yield event

    async def list_models(self) -> list[str]:
        """Return the visible Codex model slugs from the live catalog."""
        credentials = await self._auth_manager.get_credentials()
        headers = self._base_headers(credentials)
        headers["Accept"] = "application/json"
        response = await self._http_client.get(
            CODEX_MODELS_URL,
            params={"client_version": _CODEX_CLIENT_VERSION},
            headers=headers,
        )
        if response.status_code != 200:
            raise CodexUpstreamError(response.status_code, response.text)
        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise CodexUpstreamError(502, "codex models response is not valid JSON") from exc
        models = parsed.get("models") if isinstance(parsed, dict) else None
        if not isinstance(models, list):
            raise CodexUpstreamError(502, "codex models response has no models list")
        return [
            model["slug"]
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("slug"), str)
            and model.get("visibility") != "hide"
        ]

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

        async with self._http_client.stream(
            "POST", CODEX_RESPONSES_URL, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise CodexUpstreamError(response.status_code, body)

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
