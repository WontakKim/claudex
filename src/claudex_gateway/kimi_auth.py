"""Kimi credential management backed by ~/.claudex/kimi-auth.json.

The gateway performs the RFC 8628 OAuth device flow itself (`claudex-gateway
login kimi`), stores the resulting tokens, and refreshes them via the Kimi
OAuth token endpoint before they expire. Ported from the Kimi auth layer of
router-for-me/CLIProxyAPI.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

import claudex_gateway

KIMI_DEVICE_AUTH_URL = "https://auth.kimi.com/api/oauth/device_authorization"
KIMI_TOKEN_URL = "https://auth.kimi.com/api/oauth/token"
KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"

_DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

# Refresh the access token this many seconds before its recorded expiry.
_EXPIRY_SKEW_SECONDS = 300

# RFC 8628 floor for the token-poll interval; the server may ask for more.
_MIN_POLL_INTERVAL_SECONDS = 5.0

# Give up on the device flow after this long even if the server allows more.
_POLL_DEADLINE_SECONDS = 900.0


class KimiAuthError(Exception):
    """Raised when Kimi credentials are missing or cannot be obtained."""


@dataclass(frozen=True)
class KimiCredentials:
    access_token: str
    device_id: str | None


@dataclass(frozen=True)
class DeviceAuthorization:
    """One issued device authorization plus the identity it was requested with."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    interval: float
    expires_in: float
    device_id: str


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


def _utc_now_rfc3339() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def _expires_at_from(expires_in: object) -> str | None:
    if not isinstance(expires_in, (int, float)) or isinstance(expires_in, bool):
        return None
    expiry = datetime.now(timezone.utc) + timedelta(seconds=float(expires_in))
    return expiry.isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def request_device_authorization(http_client: httpx.AsyncClient) -> DeviceAuthorization:
    """Start the device flow: ask Kimi for a user code and verification URL."""
    device_id = str(uuid.uuid4())
    try:
        response = await http_client.post(
            KIMI_DEVICE_AUTH_URL,
            data={"client_id": KIMI_CLIENT_ID},
            headers={"Accept": "application/json", **identity_headers(device_id)},
        )
    except httpx.HTTPError as exc:
        raise KimiAuthError(f"device authorization request failed: {exc}") from exc
    if response.status_code != 200:
        raise KimiAuthError(
            f"device authorization failed with status {response.status_code}: {response.text}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise KimiAuthError("device authorization response is not valid JSON") from exc

    device_code = payload.get("device_code")
    user_code = payload.get("user_code")
    verification_uri = payload.get("verification_uri")
    if not (device_code and user_code and verification_uri):
        raise KimiAuthError(f"device authorization response is incomplete: {payload!r}")

    interval = payload.get("interval")
    expires_in = payload.get("expires_in")
    return DeviceAuthorization(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=payload.get("verification_uri_complete") or None,
        interval=float(interval) if isinstance(interval, (int, float)) else _MIN_POLL_INTERVAL_SECONDS,
        expires_in=float(expires_in) if isinstance(expires_in, (int, float)) else _POLL_DEADLINE_SECONDS,
        device_id=device_id,
    )


async def poll_device_token(
    http_client: httpx.AsyncClient, authorization: DeviceAuthorization
) -> dict[str, Any]:
    """Poll the token endpoint until the user approves; return persist-ready auth data."""
    interval = max(authorization.interval, _MIN_POLL_INTERVAL_SECONDS)
    deadline = time.monotonic() + min(authorization.expires_in, _POLL_DEADLINE_SECONDS)
    while True:
        try:
            response = await http_client.post(
                KIMI_TOKEN_URL,
                data={
                    "client_id": KIMI_CLIENT_ID,
                    "grant_type": _DEVICE_CODE_GRANT,
                    "device_code": authorization.device_code,
                },
                headers={"Accept": "application/json", **identity_headers(authorization.device_id)},
            )
        except httpx.HTTPError as exc:
            raise KimiAuthError(f"device token request failed: {exc}") from exc

        if response.status_code == 200:
            try:
                token_response = response.json()
            except ValueError as exc:
                raise KimiAuthError("device token response is not valid JSON") from exc
            return _auth_data_from_token_response(token_response, authorization.device_id)

        error = None
        try:
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error")
        except ValueError:
            pass
        if error == "authorization_pending":
            pass
        elif error == "slow_down":
            interval += 5.0
        else:
            raise KimiAuthError(
                f"device login failed: {error or f'status {response.status_code}: {response.text}'}"
            )

        if time.monotonic() + interval > deadline:
            raise KimiAuthError(
                "device login timed out before approval; run `claudex-gateway login kimi` again"
            )
        await asyncio.sleep(interval)


def _auth_data_from_token_response(
    token_response: dict[str, Any], device_id: str
) -> dict[str, Any]:
    access_token = token_response.get("access_token")
    if not access_token:
        raise KimiAuthError(f"token response contains no access token: {token_response!r}")
    auth_data: dict[str, Any] = {
        "type": "kimi",
        "access_token": access_token,
        "refresh_token": token_response.get("refresh_token", ""),
        "token_type": token_response.get("token_type", "Bearer"),
        "scope": token_response.get("scope", ""),
        "device_id": device_id,
        "last_refresh": _utc_now_rfc3339(),
    }
    expires_at = _expires_at_from(token_response.get("expires_in"))
    if expires_at is not None:
        auth_data["expires_at"] = expires_at
    return auth_data


def write_auth_file(auth_file: Path, auth_data: dict[str, Any]) -> None:
    """Atomically persist credentials, readable by the owner only."""
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = auth_file.with_name(auth_file.name + ".tmp")
    temp_file.write_text(json.dumps(auth_data, indent=2) + "\n", encoding="utf-8")
    os.chmod(temp_file, 0o600)
    os.replace(temp_file, auth_file)


class KimiAuthManager:
    """Loads, refreshes, and persists the gateway's Kimi credentials."""

    def __init__(self, auth_file: Path, http_client: httpx.AsyncClient) -> None:
        self._auth_file = auth_file
        self._http_client = http_client
        self._refresh_lock = asyncio.Lock()

    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        auth_data = self._load_auth_file()
        access_token = auth_data.get("access_token", "")
        if force_refresh or not access_token or _is_expiring(auth_data):
            # Remember which token looked stale: concurrent 401 retries all
            # force-refresh, and only the first one holding the lock should
            # actually rotate — with rotating refresh tokens a second POST
            # for the same generation can invalidate the fresh credentials.
            stale_access_token = access_token
            async with self._refresh_lock:
                # Another request may have refreshed while we waited for the lock.
                auth_data = self._load_auth_file()
                access_token = auth_data.get("access_token", "")
                if (
                    (force_refresh and access_token == stale_access_token)
                    or not access_token
                    or _is_expiring(auth_data)
                ):
                    auth_data = await self._refresh_tokens(auth_data)
                    access_token = auth_data.get("access_token", "")
        return KimiCredentials(
            access_token=access_token, device_id=auth_data.get("device_id") or None
        )

    def _load_auth_file(self) -> dict[str, Any]:
        try:
            raw = self._auth_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise KimiAuthError(
                f"no Kimi credentials at {self._auth_file}; "
                "run `claudex-gateway login kimi` first"
            ) from exc
        try:
            auth_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KimiAuthError(f"{self._auth_file} is not valid JSON: {exc}") from exc
        if not isinstance(auth_data, dict):
            raise KimiAuthError(f"{self._auth_file} has an unexpected format")
        return auth_data

    async def _refresh_tokens(self, auth_data: dict[str, Any]) -> dict[str, Any]:
        refresh_token = auth_data.get("refresh_token")
        if not refresh_token:
            raise KimiAuthError(
                f"access token expired and no refresh token in {self._auth_file}; "
                "run `claudex-gateway login kimi` again"
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
                    **identity_headers(auth_data.get("device_id")),
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

        new_auth = dict(auth_data)
        access_token = refreshed.get("access_token")
        if not access_token:
            raise KimiAuthError(f"token refresh returned no access token: {refreshed!r}")
        new_auth["access_token"] = access_token
        if refreshed.get("refresh_token"):
            new_auth["refresh_token"] = refreshed["refresh_token"]
        if refreshed.get("token_type"):
            new_auth["token_type"] = refreshed["token_type"]
        if refreshed.get("scope"):
            new_auth["scope"] = refreshed["scope"]
        expires_at = _expires_at_from(refreshed.get("expires_in"))
        if expires_at is not None:
            new_auth["expires_at"] = expires_at
        else:
            # Without a fresh expiry the old one would claim the new token is
            # already stale; drop it and let the 401 retry path catch decay.
            new_auth.pop("expires_at", None)
        new_auth["last_refresh"] = _utc_now_rfc3339()
        write_auth_file(self._auth_file, new_auth)
        return new_auth


def _is_expiring(auth_data: dict[str, Any]) -> bool:
    """True when expires_at is within the refresh skew or unreadable.

    A missing expires_at counts as valid — refreshing on absence would loop
    forever when the token endpoint omits expires_in, and the 401 retry path
    already recovers from silent expiry. A malformed value refreshes once,
    which rewrites the record into a well-formed one.
    """
    expires_at = auth_data.get("expires_at")
    if expires_at is None:
        return False
    if not isinstance(expires_at, str):
        return True
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return (expiry - datetime.now(timezone.utc)).total_seconds() < _EXPIRY_SKEW_SECONDS
