"""Kimi credential management backed by the Kimi Code CLI's credential store.

The gateway reuses the login produced by ``kimi login`` (the Kimi Code CLI):
tokens live at ``~/.kimi-code/credentials/kimi-code.json`` with the device
identity beside them at ``~/.kimi-code/device_id``. Expired access tokens are
refreshed via the Kimi OAuth token endpoint — the same endpoint and client ID
the CLI itself uses — and the rotated tokens are persisted back, exactly like
the CLI does.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import platform
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

import claudex_gateway

KIMI_TOKEN_URL = "https://auth.kimi.com/api/oauth/token"
KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"

# Refresh the access token this many seconds before its recorded expiry.
_EXPIRY_SKEW_SECONDS = 300


class KimiAuthError(Exception):
    """Raised when Kimi credentials are missing or cannot be obtained."""


@dataclass(frozen=True)
class KimiCredentials:
    access_token: str
    device_id: str | None
    account: str | None = None


def _account_from_token(access_token: str) -> str | None:
    """Read the user id from the access token JWT (unverified, like the CLI does)."""
    try:
        payload = access_token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (IndexError, ValueError, binascii.Error):
        return None
    account = claims.get("user_id") or claims.get("sub")
    return account if isinstance(account, str) and account else None


def identity_headers(device_id: str | None) -> dict[str, str]:
    """Device identity headers the Kimi OAuth endpoints expect."""
    headers = {
        "X-Msh-Platform": "claudex-gateway",
        "X-Msh-Version": claudex_gateway.__version__,
        "X-Msh-Device-Name": socket.gethostname(),
        "X-Msh-Device-Model": f"{platform.system()} {platform.machine()}",
    }
    if device_id:
        headers["X-Msh-Device-Id"] = device_id
    return headers


class KimiAuthManager:
    """Loads, refreshes, and persists the Kimi Code CLI's credentials."""

    def __init__(self, kimi_code_home: Path, http_client: httpx.AsyncClient) -> None:
        self._credentials_file = kimi_code_home / "credentials" / "kimi-code.json"
        self._device_id_file = kimi_code_home / "device_id"
        self._http_client = http_client
        self._refresh_lock = asyncio.Lock()

    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        auth_data = self._load_credentials_file()
        access_token = auth_data.get("access_token", "")
        if force_refresh or not access_token or _is_expiring(auth_data):
            # Remember which token looked stale: concurrent 401 retries all
            # force-refresh, and only the first one holding the lock should
            # actually rotate — with rotating refresh tokens a second POST
            # for the same generation can invalidate the fresh credentials.
            # The re-read also picks up a rotation the CLI itself just wrote.
            stale_access_token = access_token
            async with self._refresh_lock:
                auth_data = self._load_credentials_file()
                access_token = auth_data.get("access_token", "")
                if (
                    (force_refresh and access_token == stale_access_token)
                    or not access_token
                    or _is_expiring(auth_data)
                ):
                    auth_data = await self._refresh_tokens(auth_data)
                    access_token = auth_data.get("access_token", "")
        return KimiCredentials(
            access_token=access_token,
            device_id=self._load_device_id(),
            account=_account_from_token(access_token),
        )

    def _load_credentials_file(self) -> dict[str, Any]:
        try:
            raw = self._credentials_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise KimiAuthError(
                f"no Kimi Code credentials at {self._credentials_file}; "
                "run `kimi login` first"
            ) from exc
        try:
            auth_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KimiAuthError(f"{self._credentials_file} is not valid JSON: {exc}") from exc
        if not isinstance(auth_data, dict):
            raise KimiAuthError(f"{self._credentials_file} has an unexpected format")
        return auth_data

    def _load_device_id(self) -> str | None:
        # A missing or unreadable device identity never blocks a request; the
        # header is advisory metadata on the OAuth endpoints.
        try:
            device_id = self._device_id_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return device_id or None

    async def _refresh_tokens(self, auth_data: dict[str, Any]) -> dict[str, Any]:
        refresh_token = auth_data.get("refresh_token")
        if not refresh_token:
            raise KimiAuthError(
                f"access token expired and no refresh token in {self._credentials_file}; "
                "run `kimi login` again"
            )

        try:
            response = await self._http_client.post(
                KIMI_TOKEN_URL,
                data={
                    "client_id": KIMI_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                headers={
                    "Accept": "application/json",
                    **identity_headers(self._load_device_id()),
                },
            )
        except httpx.HTTPError as exc:
            raise KimiAuthError(f"token refresh request failed: {exc}") from exc
        if response.status_code != 200:
            raise KimiAuthError(
                f"token refresh failed with status {response.status_code}: {response.text}"
            )
        try:
            refreshed = response.json()
        except ValueError as exc:
            raise KimiAuthError("token refresh response is not valid JSON") from exc

        access_token = refreshed.get("access_token")
        if not access_token:
            raise KimiAuthError(f"token refresh returned no access token: {refreshed!r}")
        new_auth = dict(auth_data)
        new_auth["access_token"] = access_token
        if refreshed.get("refresh_token"):
            new_auth["refresh_token"] = refreshed["refresh_token"]
        if refreshed.get("scope"):
            new_auth["scope"] = refreshed["scope"]
        if refreshed.get("token_type"):
            new_auth["token_type"] = refreshed["token_type"]
        expires_in = refreshed.get("expires_in")
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
            # The CLI's record shape: absolute expiry plus the granted TTL.
            new_auth["expires_in"] = expires_in
            new_auth["expires_at"] = time.time() + float(expires_in)
        else:
            # Without a fresh expiry the old one would claim the new token is
            # already stale; drop it and let the 401 retry path catch decay.
            new_auth.pop("expires_at", None)
        self._persist_credentials_file(new_auth)
        return new_auth

    def _persist_credentials_file(self, auth_data: dict[str, Any]) -> None:
        temp_file = self._credentials_file.with_name(self._credentials_file.name + ".tmp")
        temp_file.write_text(json.dumps(auth_data, indent=2) + "\n", encoding="utf-8")
        os.chmod(temp_file, 0o600)
        os.replace(temp_file, self._credentials_file)


def _is_expiring(auth_data: dict[str, Any]) -> bool:
    """True when expires_at (epoch seconds) is within the refresh skew.

    A missing expires_at counts as valid — refreshing on absence would loop
    forever when the token endpoint omits expires_in, and the 401 retry path
    already recovers from silent expiry. A non-numeric value refreshes once,
    which rewrites the record into a well-formed one.
    """
    expires_at = auth_data.get("expires_at")
    if expires_at is None:
        return False
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        return True
    return float(expires_at) - time.time() < _EXPIRY_SKEW_SECONDS
