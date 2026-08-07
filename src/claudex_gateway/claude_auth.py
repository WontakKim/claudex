"""Claude account credential management for gateway-registered accounts.

The gateway serves Anthropic passthrough traffic with a registered account's
OAuth tokens (design: `.docs/design/multi-account-pool.md` §3). Each manager
owns one account directory produced by `claude_accounts.add_account`::

    accounts/claude/<id>/
    ├── credentials.json      # {"claudeAiOauth": {...}} — the captured
    │                         #   Claude Code credentials file, mode 0600
    └── oauth-account.json    # the .claude.json oauthAccount block

Expired access tokens are refreshed against the same token endpoint and
client id Claude Code itself uses (verified from the Claude Code 2.1.224
bundle): a JSON POST of `grant_type=refresh_token` with the stored scopes,
answered with `access_token`/`refresh_token`/`expires_in`. The rotated
refresh token is persisted atomically before the new access token is ever
returned to a caller — refresh tokens are single-use, so a crash between
rotation and persistence must lose the *new* token, never the recorded one.

The gateway is the only refresher for a registered account (each account is
its own OAuth login session, never shared with the user's real `claude`
CLI), so no cross-process coordination is needed beyond the per-manager
asyncio lock. Credential values never appear in log lines or exceptions.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

CLAUDE_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
# The public Claude Code OAuth client id (bundle constant CLIENT_ID).
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# Refresh the access token this many seconds before its recorded expiry.
_EXPIRY_SKEW_SECONDS = 300

_CREDENTIALS_FILENAME = "credentials.json"
_OAUTH_ACCOUNT_FILENAME = "oauth-account.json"


class ClaudeAccountAuthError(Exception):
    """Raised when account credentials are missing, malformed, or cannot be
    refreshed. Messages never contain credential values — only paths and a
    description of what went wrong."""


@dataclass(frozen=True)
class ClaudeAccountCredentials:
    access_token: str
    # The Anthropic account uuid from oauth-account.json, used to rewrite
    # `metadata.user_id` so a request never names a different account than
    # the one serving it. None when the capture did not record it.
    account_uuid: str | None


class ClaudeAccountAuthManager:
    """Loads, refreshes, and persists one registered account's credentials."""

    def __init__(self, account_dir: Path, http_client: httpx.AsyncClient) -> None:
        self._credentials_file = account_dir / _CREDENTIALS_FILENAME
        self._oauth_account_file = account_dir / _OAUTH_ACCOUNT_FILENAME
        self._http_client = http_client
        self._refresh_lock = asyncio.Lock()

    async def get_credentials(self, force_refresh: bool = False) -> ClaudeAccountCredentials:
        file_data, oauth_blob = self._load_credentials_file()
        access_token = _access_token(oauth_blob)
        if force_refresh or not access_token or _is_expiring(oauth_blob):
            # Remember which token looked stale: concurrent 401 retries all
            # force-refresh, and only the first one holding the lock should
            # actually rotate — with single-use refresh tokens a second POST
            # for the same generation invalidates the fresh credentials.
            stale_access_token = access_token
            async with self._refresh_lock:
                file_data, oauth_blob = self._load_credentials_file()
                access_token = _access_token(oauth_blob)
                if (
                    (force_refresh and access_token == stale_access_token)
                    or not access_token
                    or _is_expiring(oauth_blob)
                ):
                    oauth_blob = await self._refresh_tokens(file_data, oauth_blob)
                    access_token = _access_token(oauth_blob)
        return ClaudeAccountCredentials(
            access_token=access_token,
            account_uuid=self._load_account_uuid(),
        )

    def _load_credentials_file(self) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            raw = self._credentials_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ClaudeAccountAuthError(
                f"no account credentials at {self._credentials_file}; "
                "was the account removed? Re-add it with `claudex-gateway account add`"
            ) from exc
        try:
            file_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ClaudeAccountAuthError(
                f"{self._credentials_file} is not valid JSON: {exc}"
            ) from exc
        oauth_blob = file_data.get("claudeAiOauth") if isinstance(file_data, dict) else None
        if not isinstance(oauth_blob, dict):
            raise ClaudeAccountAuthError(
                f"{self._credentials_file} has no claudeAiOauth object; "
                "re-add the account with `claudex-gateway account add`"
            )
        return file_data, oauth_blob

    def _load_account_uuid(self) -> str | None:
        # A missing or malformed oauth-account.json never blocks a request;
        # the metadata rewrite degrades to stripping the account uuid.
        try:
            parsed = json.loads(self._oauth_account_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(parsed, dict):
            return None
        account_uuid = parsed.get("accountUuid")
        return account_uuid if isinstance(account_uuid, str) and account_uuid else None

    async def _refresh_tokens(
        self, file_data: dict[str, Any], oauth_blob: dict[str, Any]
    ) -> dict[str, Any]:
        refresh_token = oauth_blob.get("refreshToken")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise ClaudeAccountAuthError(
                f"access token expired and no refresh token in {self._credentials_file}; "
                "re-add the account with `claudex-gateway account add`"
            )

        # The exact request Claude Code itself sends: a JSON body carrying the
        # stored scopes and client id (falling back to the public one).
        client_id = oauth_blob.get("clientId")
        payload: dict[str, Any] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id if isinstance(client_id, str) and client_id else CLAUDE_CLIENT_ID,
        }
        scopes = oauth_blob.get("scopes")
        if isinstance(scopes, list) and scopes and all(isinstance(s, str) for s in scopes):
            payload["scope"] = " ".join(scopes)

        try:
            response = await self._http_client.post(
                CLAUDE_TOKEN_URL,
                json=payload,
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise ClaudeAccountAuthError(f"token refresh request failed: {exc}") from exc
        if response.status_code != 200:
            raise ClaudeAccountAuthError(
                f"token refresh failed with status {response.status_code}: {response.text}"
            )
        try:
            refreshed = response.json()
        except ValueError as exc:
            raise ClaudeAccountAuthError("token refresh response is not valid JSON") from exc

        access_token = refreshed.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ClaudeAccountAuthError("token refresh returned no access token")
        new_blob = dict(oauth_blob)
        new_blob["accessToken"] = access_token
        if isinstance(refreshed.get("refresh_token"), str) and refreshed["refresh_token"]:
            new_blob["refreshToken"] = refreshed["refresh_token"]
        if isinstance(refreshed.get("scope"), str) and refreshed["scope"]:
            new_blob["scopes"] = refreshed["scope"].split(" ")
        expires_in = refreshed.get("expires_in")
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
            # Claude Code's record shape: absolute expiry in epoch milliseconds.
            new_blob["expiresAt"] = int((time.time() + float(expires_in)) * 1000)
        else:
            # Without a fresh expiry the old one would claim the new token is
            # already stale; drop it and let the 401 retry path catch decay.
            new_blob.pop("expiresAt", None)
        new_file_data = dict(file_data)
        new_file_data["claudeAiOauth"] = new_blob
        self._persist_credentials_file(new_file_data)
        return new_blob

    def _persist_credentials_file(self, file_data: dict[str, Any]) -> None:
        temp_file = self._credentials_file.with_name(self._credentials_file.name + ".tmp")
        temp_file.write_text(json.dumps(file_data, indent=2) + "\n", encoding="utf-8")
        os.chmod(temp_file, 0o600)
        os.replace(temp_file, self._credentials_file)


def _access_token(oauth_blob: dict[str, Any]) -> str:
    access_token = oauth_blob.get("accessToken")
    return access_token if isinstance(access_token, str) else ""


def _is_expiring(oauth_blob: dict[str, Any]) -> bool:
    """True when expiresAt (epoch milliseconds) is within the refresh skew.

    A missing expiresAt counts as valid — refreshing on absence would loop
    forever when the token endpoint omits expires_in, and the 401 retry path
    already recovers from silent expiry. A non-numeric value refreshes once,
    which rewrites the record into a well-formed one.
    """
    expires_at = oauth_blob.get("expiresAt")
    if expires_at is None:
        return False
    if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
        return True
    return float(expires_at) / 1000.0 - time.time() < _EXPIRY_SKEW_SECONDS
