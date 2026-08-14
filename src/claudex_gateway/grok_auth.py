"""Grok credential management backed by the Grok CLI's auth.json.

The gateway reuses the login produced by ``grok login``: the store lives at
``$GROK_HOME/auth.json`` (default ``~/.grok/auth.json``) keyed by
``"<issuer>::<client_id>"``, with first-party OAuth entries under the
``https://auth.x.ai`` issuer. Expired OAuth access tokens are refreshed via
OIDC discovery on the entry's issuer and the rotated tokens are persisted
back, exactly like the Grok CLI does. Plain API-key entries (``grok login
--api-key``) are used as-is and never refreshed.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from claudex_gateway.upstream_errors import UpstreamAuthError

XAI_ISSUER = "https://auth.x.ai"

# Refresh the access token this many seconds before its recorded expiry.
_EXPIRY_SKEW_SECONDS = 300


class GrokAuthError(UpstreamAuthError):
    """Raised when Grok credentials are missing or cannot be refreshed."""


@dataclass(frozen=True)
class GrokCredentials:
    access_token: str
    email: str | None
    is_api_key: bool = False
    user_id: str | None = None


def _parse_expiry(value: object) -> float | None:
    """Parse the entry's expires_at (RFC3339) to epoch seconds; None when absent."""
    if not isinstance(value, str) or not value:
        return None
    try:
        expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry.timestamp()


def _rfc3339_in(seconds: float) -> str:
    expiry = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return expiry.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class GrokAuthManager:
    """Loads, refreshes, and persists the Grok CLI's credentials."""

    def __init__(self, auth_file: Path, http_client: httpx.AsyncClient) -> None:
        self._auth_file = auth_file
        self._http_client = http_client
        self._refresh_lock = asyncio.Lock()

    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        store = self._load_auth_file()
        scope, entry = _select_entry(store, self._auth_file)

        if entry.get("auth_mode") == "api_key":
            return GrokCredentials(
                access_token=entry["key"],
                email=entry.get("email"),
                is_api_key=True,
                user_id=entry.get("user_id"),
            )

        expiry = _parse_expiry(entry.get("expires_at"))
        needs_refresh = (
            force_refresh
            or expiry is None  # OAuth entries always carry an expiry; treat absence as stale
            or expiry - time.time() < _EXPIRY_SKEW_SECONDS
        )
        if needs_refresh:
            # Remember which token looked stale: concurrent 401 retries all
            # force-refresh, and only the first one holding the lock should
            # actually rotate — with rotating refresh tokens a second POST
            # for the same generation can invalidate the fresh credentials.
            # The re-read also picks up a rotation the CLI itself just wrote.
            stale_access_token = entry.get("key", "")
            async with self._refresh_lock:
                store = self._load_auth_file()
                scope, entry = _select_entry(store, self._auth_file)
                expiry = _parse_expiry(entry.get("expires_at"))
                if (
                    (force_refresh and entry.get("key", "") == stale_access_token)
                    or expiry is None
                    or expiry - time.time() < _EXPIRY_SKEW_SECONDS
                ):
                    store, entry = await self._refresh_tokens(store, scope, entry)

        return GrokCredentials(
            access_token=entry["key"],
            email=entry.get("email"),
            user_id=entry.get("user_id"),
        )

    def _load_auth_file(self) -> dict[str, Any]:
        try:
            raw = self._auth_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise GrokAuthError(
                f"{self._auth_file} not found; install the Grok CLI and run `grok login`"
            ) from exc
        try:
            store = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GrokAuthError(f"{self._auth_file} is not valid JSON: {exc}") from exc
        if not isinstance(store, dict):
            raise GrokAuthError(f"{self._auth_file} has an unexpected format")
        return store

    async def _refresh_tokens(
        self, store: dict[str, Any], scope: str, entry: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        refresh_token = entry.get("refresh_token")
        if not refresh_token:
            raise GrokAuthError(
                f"access token expired and no refresh token in {self._auth_file}; "
                "run `grok login` again"
            )
        issuer = entry.get("oidc_issuer") or scope.partition("::")[0] or XAI_ISSUER
        client_id = entry.get("oidc_client_id") or scope.partition("::")[2]
        if not client_id:
            raise GrokAuthError(
                f"entry {scope!r} in {self._auth_file} names no OAuth client; "
                "run `grok login` again"
            )

        token_endpoint = await self._discover_token_endpoint(issuer)
        form = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
        # Team principals refresh under their principal identity, mirroring
        # the Grok CLI's refresh form.
        for field in ("principal_type", "principal_id"):
            if isinstance(entry.get(field), str) and entry[field]:
                form[field] = entry[field]
        try:
            response = await self._http_client.post(
                token_endpoint, data=form, headers={"Accept": "application/json"}
            )
        except httpx.HTTPError as exc:
            raise GrokAuthError(f"token refresh request failed: {exc}") from exc
        if response.status_code != 200:
            raise GrokAuthError(
                f"token refresh failed with status {response.status_code}: {response.text}"
            )
        try:
            refreshed = response.json()
        except ValueError as exc:
            raise GrokAuthError("token refresh response is not valid JSON") from exc

        access_token = refreshed.get("access_token")
        if not access_token:
            raise GrokAuthError(f"token refresh returned no access token: {refreshed!r}")
        new_entry = dict(entry)
        new_entry["key"] = access_token
        if refreshed.get("refresh_token"):
            new_entry["refresh_token"] = refreshed["refresh_token"]
        expires_in = refreshed.get("expires_in")
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
            new_entry["expires_at"] = _rfc3339_in(float(expires_in))

        new_store = dict(store)
        new_store[scope] = new_entry
        self._persist_auth_file(new_store)
        return new_store, new_entry

    async def _discover_token_endpoint(self, issuer: str) -> str:
        url = issuer.rstrip("/") + "/.well-known/openid-configuration"
        try:
            response = await self._http_client.get(url, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise GrokAuthError(f"OIDC discovery request failed: {exc}") from exc
        if response.status_code != 200:
            raise GrokAuthError(
                f"OIDC discovery failed with status {response.status_code}: {response.text}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GrokAuthError("OIDC discovery response is not valid JSON") from exc
        token_endpoint = payload.get("token_endpoint")
        if not isinstance(token_endpoint, str) or not token_endpoint:
            raise GrokAuthError(f"OIDC discovery response has no token_endpoint: {payload!r}")
        return token_endpoint

    def _persist_auth_file(self, store: dict[str, Any]) -> None:
        temp_file = self._auth_file.with_name(self._auth_file.name + ".tmp")
        temp_file.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
        os.chmod(temp_file, 0o600)
        os.replace(temp_file, self._auth_file)


def _select_entry(store: dict[str, Any], auth_file: Path) -> tuple[str, dict[str, Any]]:
    """Pick the credential entry: the first-party Grok OAuth entry, then any
    API-key entry. Legacy web-login entries are skipped like the CLI does."""
    fallback: tuple[str, dict[str, Any]] | None = None
    for scope, entry in store.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
            continue
        if entry.get("auth_mode") == "web_login":
            continue
        if scope.startswith(XAI_ISSUER + "::") and entry.get("auth_mode") != "api_key":
            return scope, entry
        if fallback is None:
            fallback = scope, entry
    if fallback is not None:
        return fallback
    raise GrokAuthError(f"no usable Grok credentials in {auth_file}; run `grok login` first")
