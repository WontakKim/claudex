"""Read the local Claude Code login as an ambient balanced-pool account.

The provider is strictly read-through: it observes the identity and current
access token owned by the local Claude Code CLI, while the CLI remains the sole
refresher. In particular, this module never writes credentials or exchanges a
refresh token. That ownership boundary avoids racing the CLI over a single-use
refresh token, which could otherwise log one of the two processes out.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .claude.account_profile import compute_account_profile_fingerprint
from .claude.accounts import AccountRecord
from .claude.auth import ClaudeAccountAuthError, ClaudeAccountCredentials
from .claude.capture_model import CaptureError, KeychainBackend
from .claude.keychain import LEGACY_KEYCHAIN_SERVICE, SecurityKeychainBackend

AMBIENT_ID_NAMESPACE = uuid.UUID("49d01c38-08d0-4df4-a742-13aa3c192f3e")

_CACHE_TTL_SECONDS = 30.0

__all__ = (
    "AMBIENT_ID_NAMESPACE",
    "AmbientAccountProvider",
    "AmbientClaudeAuthManager",
    "AmbientPoolMember",
    "is_duplicate_identity",
)


@dataclass(frozen=True)
class AmbientPoolMember:
    """A runtime-only pool candidate synthesized from the local CLI login."""

    record: AccountRecord
    oauth_account: dict[str, Any]
    profile_fingerprint: str | None


@dataclass(frozen=True)
class _AmbientSource:
    """One combined observation of identity and credential sources."""

    oauth_account: dict[str, Any] | None
    oauth_credentials: dict[str, Any] | None


@dataclass(frozen=True)
class _CacheEntry:
    """A source observation and the monotonic instant when it completed."""

    source: _AmbientSource
    read_at: float


class AmbientAccountProvider:
    """Caches read-only observations of the machine's Claude Code login.

    Identity and credentials form one cache entry, including unsuccessful
    observations. A forced read bypasses that entry but never mutates either
    source. On macOS, omitting ``keychain`` selects the same production
    security-CLI backend used by credential capture; callers may inject a
    backend to isolate or replace that blocking source.
    """

    def __init__(
        self,
        keychain: KeychainBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._keychain = (
            SecurityKeychainBackend()
            if keychain is None and sys.platform == "darwin"
            else keychain
        )
        self._clock = clock
        self._cache: _CacheEntry | None = None

    def pool_member(self) -> AmbientPoolMember | None:
        """Return the current eligible ambient candidate, or ``None``.

        Missing or malformed identity, unreadable or tokenless credentials,
        and an already-expired access token all make the ambient login
        ineligible without affecting registered accounts.
        """
        source = self._get_source()
        if not _has_access_token(source.oauth_credentials):
            return None
        if _is_expired(source.oauth_credentials):
            return None
        return _make_pool_member(source.oauth_account)

    def is_ambient_account_id(self, account_id: str) -> bool:
        """Return whether ``account_id`` belongs to the current CLI identity.

        Credential availability does not change identity. This lets routing
        continue to recognize an already-selected ambient id while a later
        credential read reports that the local CLI login is stale.
        """
        record = _make_account_record(self._get_source().oauth_account)
        return record is not None and record.id == account_id

    def auth_manager(self) -> AmbientClaudeAuthManager:
        """Build the read-through auth manager for this provider."""
        return AmbientClaudeAuthManager(self)

    def _get_source(self, *, force_refresh: bool = False) -> _AmbientSource:
        cached = self._cached_source(force_refresh=force_refresh)
        if cached is not None:
            return cached
        source = self._read_source()
        self._cache = _CacheEntry(source=source, read_at=self._clock())
        return source

    async def _get_source_async(self, *, force_refresh: bool = False) -> _AmbientSource:
        """Read through the cache without blocking the event loop.

        The uncached operation is moved as a unit to a worker thread so a
        possible security-CLI Keychain subprocess never runs on the async
        request path's event-loop thread.
        """
        cached = self._cached_source(force_refresh=force_refresh)
        if cached is not None:
            return cached
        source = await asyncio.to_thread(self._read_source)
        self._cache = _CacheEntry(source=source, read_at=self._clock())
        return source

    def _cached_source(self, *, force_refresh: bool) -> _AmbientSource | None:
        if force_refresh or self._cache is None:
            return None
        if self._clock() - self._cache.read_at >= _CACHE_TTL_SECONDS:
            return None
        return self._cache.source

    def _read_source(self) -> _AmbientSource:
        oauth_account = _read_oauth_account(_identity_file())
        if oauth_account is None:
            return _AmbientSource(oauth_account=None, oauth_credentials=None)
        oauth_credentials = self._read_oauth_credentials(_credentials_file())
        return _AmbientSource(
            oauth_account=oauth_account,
            oauth_credentials=oauth_credentials,
        )

    def _read_oauth_credentials(self, credentials_file: Path) -> dict[str, Any] | None:
        try:
            raw = credentials_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            if self._keychain is None:
                return None
            try:
                raw = self._keychain.read(LEGACY_KEYCHAIN_SERVICE, getpass.getuser())
            except (CaptureError, OSError):
                return None
            if raw is None:
                return None
        except (OSError, UnicodeDecodeError):
            return None

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        oauth_credentials = parsed.get("claudeAiOauth")
        if not isinstance(oauth_credentials, dict):
            return None
        if not _has_valid_expiry(oauth_credentials):
            return None
        return oauth_credentials


class AmbientClaudeAuthManager:
    """Duck-typed auth manager for the read-only ambient CLI credential.

    ``force_refresh`` means re-read, not OAuth refresh. If the source still
    exposes the token rejected by the upstream request, the caller receives a
    normal auth error directing it back to the CLI that owns refresh lineage.
    """

    def __init__(self, provider: AmbientAccountProvider) -> None:
        self._provider = provider

    async def get_credentials(
        self, force_refresh: bool = False
    ) -> ClaudeAccountCredentials:
        """Return the current CLI token or raise a recoverable auth error.

        A forced call first identifies the token generation currently cached,
        then bypasses the TTL. Serving the same generation again cannot repair
        an expiry or rejection, so it fails rather than attempting refresh.
        """
        if force_refresh:
            stale_source = await self._provider._get_source_async()
            source = await self._provider._get_source_async(force_refresh=True)
        else:
            stale_source = None
            source = await self._provider._get_source_async()

        access_token = _access_token(source.oauth_credentials)
        account_uuid = _identity_account_uuid(source.oauth_account)
        if not access_token or source.oauth_account is None or _is_expired(
            source.oauth_credentials
        ):
            raise _ambient_auth_error()

        if force_refresh and access_token == _access_token(
            stale_source.oauth_credentials if stale_source is not None else None
        ):
            raise _ambient_auth_error()

        return ClaudeAccountCredentials(
            access_token=access_token,
            account_uuid=account_uuid,
        )


def is_duplicate_identity(
    member: AmbientPoolMember, records: Iterable[AccountRecord]
) -> bool:
    """Return whether a registered record supersedes ``member``.

    The primary identity is case-folded email plus canonical organization UUID.
    A canonical upstream account UUID is a secondary guard when both sides
    carry one.
    """
    member_identity = (
        member.record.email.casefold(),
        _canonical_identity_uuid(member.record.organization_uuid),
    )
    member_upstream_uuid = _canonical_uuid_or_none(member.record.upstream_account_uuid)

    for record in records:
        record_identity = (
            record.email.casefold(),
            _canonical_identity_uuid(record.organization_uuid),
        )
        if record_identity == member_identity:
            return True
        record_upstream_uuid = _canonical_uuid_or_none(record.upstream_account_uuid)
        if (
            member_upstream_uuid is not None
            and record_upstream_uuid is not None
            and record_upstream_uuid == member_upstream_uuid
        ):
            return True
    return False


def _identity_file() -> Path:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir is not None:
        return Path(config_dir).expanduser() / ".claude.json"
    return Path.home() / ".claude.json"


def _credentials_file() -> Path:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir is not None:
        return Path(config_dir).expanduser() / ".credentials.json"
    return Path.home() / ".claude" / ".credentials.json"


def _read_oauth_account(identity_file: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(identity_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    oauth_account = parsed.get("oauthAccount")
    if not isinstance(oauth_account, dict):
        return None
    email = oauth_account.get("emailAddress")
    if not isinstance(email, str) or not email:
        return None
    return oauth_account


def _make_pool_member(oauth_account: dict[str, Any] | None) -> AmbientPoolMember | None:
    record = _make_account_record(oauth_account)
    if record is None or oauth_account is None:
        return None
    return AmbientPoolMember(
        record=record,
        oauth_account=oauth_account,
        profile_fingerprint=compute_account_profile_fingerprint(oauth_account),
    )


def _make_account_record(oauth_account: dict[str, Any] | None) -> AccountRecord | None:
    if oauth_account is None:
        return None
    email = oauth_account.get("emailAddress")
    if not isinstance(email, str) or not email:
        return None

    organization_uuid = _optional_string(oauth_account.get("organizationUuid"))
    identity_key = f"{email.casefold()}\x00{_canonical_identity_uuid(organization_uuid)}"
    return AccountRecord(
        id=str(uuid.uuid5(AMBIENT_ID_NAMESPACE, "account:" + identity_key)),
        email=email,
        organization_uuid=organization_uuid,
        organization_name=_optional_string(oauth_account.get("organizationName")),
        created_at=0,
        updated_at=0,
        last_authenticated_at=0,
        state="ready",
        account_incarnation_id=str(
            uuid.uuid5(AMBIENT_ID_NAMESPACE, "incarnation:" + identity_key)
        ),
        upstream_account_uuid=_optional_string(oauth_account.get("accountUuid")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _canonical_identity_uuid(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    canonical = _canonical_uuid_or_none(value)
    return canonical if canonical is not None else value


def _canonical_uuid_or_none(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _identity_account_uuid(oauth_account: dict[str, Any] | None) -> str | None:
    if oauth_account is None:
        return None
    return _optional_string(oauth_account.get("accountUuid"))


def _access_token(oauth_credentials: dict[str, Any] | None) -> str:
    if oauth_credentials is None:
        return ""
    return _optional_string(oauth_credentials.get("accessToken")) or ""


def _has_access_token(oauth_credentials: dict[str, Any] | None) -> bool:
    return bool(_access_token(oauth_credentials))


def _has_valid_expiry(oauth_credentials: dict[str, Any]) -> bool:
    expires_at = oauth_credentials.get("expiresAt")
    if expires_at is None:
        return True
    return (
        isinstance(expires_at, (int, float))
        and not isinstance(expires_at, bool)
        and math.isfinite(float(expires_at))
    )


def _is_expired(oauth_credentials: dict[str, Any] | None) -> bool:
    if oauth_credentials is None:
        return False
    expires_at = oauth_credentials.get("expiresAt")
    if expires_at is None:
        return False
    if not _has_valid_expiry(oauth_credentials):
        return True
    return float(expires_at) <= time.time() * 1000.0


def _ambient_auth_error() -> ClaudeAccountAuthError:
    return ClaudeAccountAuthError(
        "the ambient CLI login is unavailable or stale; use or re-login to the "
        "local Claude Code CLI so it can refresh its credentials"
    )
