"""Relay HTTP client for the Kimi coding backend.

Kimi's coding endpoint speaks the Anthropic Messages API natively, so unlike
the Codex client this one never parses payloads: it forwards the caller's
bytes with the gateway's Kimi Bearer token and returns the open response for
the server to relay.
"""

from __future__ import annotations

from typing import Any

import httpx

from claudex_gateway.kimi_auth import KimiAuthManager, KimiCredentials

KIMI_MESSAGES_URL = "https://api.kimi.com/coding/v1/messages"
KIMI_COUNT_TOKENS_URL = "https://api.kimi.com/coding/v1/messages/count_tokens"
KIMI_MODELS_URL = "https://api.kimi.com/coding/v1/models"


class KimiUpstreamError(Exception):
    """Raised when the Kimi backend returns a non-success HTTP response."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"kimi upstream returned {status_code}: {body[:2000]}")
        self.status_code = status_code
        self.body = body


class KimiClient:
    def __init__(self, auth_manager: KimiAuthManager, http_client: httpx.AsyncClient) -> None:
        self._auth_manager = auth_manager
        self._http_client = http_client

    async def send_messages(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        """POST a Messages request and return the open streaming response.

        Ownership of the response transfers to the caller, which must aclose()
        it on every path. Retries exactly once with force-refreshed
        credentials on HTTP 401; any remaining non-200 raises.
        """
        return await self._post(KIMI_MESSAGES_URL, body, headers)

    async def count_tokens(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        """POST a count_tokens request; same contract as send_messages."""
        return await self._post(KIMI_COUNT_TOKENS_URL, body, headers)

    async def list_models(self) -> Any:
        """Return Kimi's live model catalog exactly as the backend reports it.

        model_map values after the kimi: prefix are sent unprocessed, so the
        raw catalog is the authority on valid IDs; no reshaping happens here.
        Retries exactly once with force-refreshed credentials on HTTP 401.
        """
        credentials = await self._auth_manager.get_credentials()
        response = await self._get_models(credentials)
        if response.status_code == 401:
            credentials = await self._auth_manager.get_credentials(force_refresh=True)
            response = await self._get_models(credentials)
        if response.status_code != 200:
            raise KimiUpstreamError(response.status_code, response.text)
        try:
            return response.json()
        except ValueError as exc:
            raise KimiUpstreamError(502, "kimi models response is not valid JSON") from exc

    async def _get_models(self, credentials: KimiCredentials) -> httpx.Response:
        return await self._http_client.get(
            KIMI_MODELS_URL,
            headers={
                "Authorization": f"Bearer {credentials.access_token}",
                "Accept": "application/json",
            },
        )

    async def _post(self, url: str, body: bytes, headers: dict[str, str]) -> httpx.Response:
        credentials = await self._auth_manager.get_credentials()
        response = await self._send_once(url, body, headers, credentials)
        if response.status_code == 401:
            await self._discard(response)
            credentials = await self._auth_manager.get_credentials(force_refresh=True)
            response = await self._send_once(url, body, headers, credentials)
        if response.status_code != 200:
            upstream_body = (await response.aread()).decode("utf-8", errors="replace")
            await response.aclose()
            raise KimiUpstreamError(response.status_code, upstream_body)
        return response

    async def _send_once(
        self, url: str, body: bytes, headers: dict[str, str], credentials: KimiCredentials
    ) -> httpx.Response:
        request_headers = dict(headers)
        request_headers["Authorization"] = f"Bearer {credentials.access_token}"
        request = self._http_client.build_request(
            "POST", url, params={"beta": "true"}, content=body, headers=request_headers
        )
        return await self._http_client.send(request, stream=True)

    @staticmethod
    async def _discard(response: httpx.Response) -> None:
        await response.aread()
        await response.aclose()
