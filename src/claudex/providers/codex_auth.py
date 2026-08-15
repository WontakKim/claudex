"""Codex credential management backed by the Codex CLI's ~/.codex/auth.json.

The gateway reuses the login produced by ``codex login``. Access tokens are
JWTs; when one is expired (or about to expire) the gateway refreshes it via the
OpenAI OAuth token endpoint and persists the rotated tokens back to auth.json,
exactly like the Codex CLI does.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from claudex.providers.auth_support import (
    load_json_credentials,
    refresh_when_still_stale,
    replace_credentials_file,
)
from claudex.upstream_errors import UpstreamAuthError

CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Refresh the access token this many seconds before its JWT exp claim.
_EXPIRY_SKEW_SECONDS = 300


class CodexAuthError(UpstreamAuthError):
    """Raised when Codex credentials are missing or cannot be refreshed."""


@dataclass(frozen=True)
class CodexCredentials:
    access_token: str
    account_id: str | None
    is_api_key: bool = False
    email: str | None = None


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except (IndexError, ValueError, binascii.Error):
        return {}


def _jwt_expiry(token: str) -> float | None:
    exp = _decode_jwt_claims(token).get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def _tokens_stale(auth_data: dict[str, Any]) -> bool:
    tokens = auth_data.get("tokens") or {}
    access_token = tokens.get("access_token", "")
    expiry = _jwt_expiry(access_token)
    return not access_token or (
        expiry is not None and expiry - time.time() < _EXPIRY_SKEW_SECONDS
    )


def _account_id_from_token(token: str) -> str | None:
    auth_claims = _decode_jwt_claims(token).get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _email_from_id_token(id_token: object) -> str | None:
    """Read the account email from the OpenAI id_token, like the Codex CLI does."""
    if not isinstance(id_token, str) or not id_token:
        return None
    claims = _decode_jwt_claims(id_token)
    email = claims.get("email")
    if not isinstance(email, str) or not email:
        profile = claims.get("profile")
        email = profile.get("email") if isinstance(profile, dict) else None
    return email if isinstance(email, str) and email else None


class CodexAuthManager:
    """Loads, refreshes, and persists Codex CLI credentials."""

    def __init__(self, auth_file: Path, http_client: httpx.AsyncClient) -> None:
        self._auth_file = auth_file
        self._http_client = http_client
        self._refresh_lock = asyncio.Lock()

    async def get_credentials(self, force_refresh: bool = False) -> CodexCredentials:
        auth_data = self._load_auth_file()

        api_key = auth_data.get("OPENAI_API_KEY")
        tokens = auth_data.get("tokens")
        if not isinstance(tokens, dict):
            if isinstance(api_key, str) and api_key:
                return CodexCredentials(access_token=api_key, account_id=None, is_api_key=True)
            raise CodexAuthError(
                f"no ChatGPT tokens in {self._auth_file}; run `codex login` first"
            )

        access_token = tokens.get("access_token", "")
        needs_refresh = force_refresh or _tokens_stale(auth_data)
        if needs_refresh:
            stale_access_token = access_token

            async def refresh(state: dict[str, Any]) -> dict[str, Any]:
                return {**state, "tokens": await self._refresh_tokens(state)}

            auth_data = await refresh_when_still_stale(
                self._refresh_lock,
                force_refresh=force_refresh,
                stale_token=stale_access_token,
                reload=self._reload_validated,
                token_of=lambda state: (state.get("tokens") or {}).get(
                    "access_token", ""
                ),
                is_stale=_tokens_stale,
                refresh=refresh,
            )
            tokens = auth_data["tokens"]
            access_token = tokens.get("access_token", "")

        account_id = tokens.get("account_id") or _account_id_from_token(access_token)
        return CodexCredentials(
            access_token=access_token,
            account_id=account_id,
            email=_email_from_id_token(tokens.get("id_token")),
        )

    def _load_auth_file(self) -> dict[str, Any]:
        return load_json_credentials(
            self._auth_file,
            error_cls=CodexAuthError,
            missing_message=(
                f"{self._auth_file} not found; install the Codex CLI and run `codex login`"
            ),
            encoding=None,
        )

    def _reload_validated(self) -> dict[str, Any]:
        auth_data = self._load_auth_file()
        if not isinstance(auth_data.get("tokens"), dict):
            raise CodexAuthError(
                f"no ChatGPT tokens in {self._auth_file}; run `codex login` first"
            )
        return auth_data

    async def _refresh_tokens(self, auth_data: dict[str, Any]) -> dict[str, Any]:
        tokens = auth_data.get("tokens") or {}
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            raise CodexAuthError(
                f"access token expired and no refresh token in {self._auth_file}; "
                "run `codex login` again"
            )

        try:
            response = await self._http_client.post(
                CODEX_TOKEN_URL,
                data={
                    "client_id": CODEX_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "scope": "openid profile email",
                },
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise CodexAuthError(f"token refresh request failed: {exc}") from exc
        if response.status_code != 200:
            raise CodexAuthError(
                f"token refresh failed with status {response.status_code}: {response.text}"
            )

        refreshed = response.json()
        new_tokens = dict(tokens)
        new_tokens["access_token"] = refreshed.get("access_token", "")
        if refreshed.get("refresh_token"):
            new_tokens["refresh_token"] = refreshed["refresh_token"]
        if refreshed.get("id_token"):
            new_tokens["id_token"] = refreshed["id_token"]
        if not new_tokens.get("account_id"):
            account_id = _account_id_from_token(
                refreshed.get("id_token") or new_tokens["access_token"]
            )
            if account_id:
                new_tokens["account_id"] = account_id

        auth_data = dict(auth_data)
        auth_data["tokens"] = new_tokens
        auth_data["last_refresh"] = (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
        replace_credentials_file(
            self._auth_file,
            temp_file=self._auth_file.with_suffix(".json.tmp"),
            text=json.dumps(auth_data, indent=2),
            encoding=None,
        )
        return new_tokens
