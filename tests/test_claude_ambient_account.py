"""Tests for the read-only ambient Claude Code CLI account provider."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path
from typing import Any

import pytest

from claudex_gateway.claude import ambient_account as ambient_module
from claudex_gateway.claude.accounts import AccountRecord
from claudex_gateway.claude.ambient_account import (
    AmbientAccountProvider,
    AmbientPoolMember,
    is_duplicate_identity,
)
from claudex_gateway.claude.auth import ClaudeAccountAuthError
from claudex_gateway.claude.capture_model import KeychainBackend

_ACCOUNT_UUID = "226432ad-b115-4a79-a152-8420aacbd4b2"
_ORGANIZATION_UUID = "8ccaa9d3-2bb1-4447-a09b-b35467e47a9e"
_OTHER_ACCOUNT_UUID = "d16a1603-6ca0-4939-8f4c-b5eaa833aa08"
_OTHER_ORGANIZATION_UUID = "135e9673-7685-4de2-9e03-a4cedade4680"


class FakeKeychainBackend:
    """In-memory KeychainBackend that records every lookup."""

    def __init__(self, secret: str | None = None) -> None:
        self.secret = secret
        self.reads: list[tuple[str, str]] = []

    def read(self, service: str, account: str) -> str | None:
        self.reads.append((service, account))
        return self.secret

    def delete(self, service: str, account: str) -> None:
        raise AssertionError("ambient credential access must never delete a Keychain item")


class FakeClock:
    """Manually advanced monotonic clock for cache tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


def _write_identity(
    home: Path,
    *,
    email: str = "ambient@example.com",
    organization_uuid: str = _ORGANIZATION_UUID,
    account_uuid: str = _ACCOUNT_UUID,
) -> dict[str, Any]:
    oauth_account = {
        "emailAddress": email,
        "organizationUuid": organization_uuid,
        "organizationName": "Ambient Organization",
        "accountUuid": account_uuid,
        "organizationRateLimitTier": "default_claude_max_5x",
    }
    (home / ".claude.json").write_text(
        json.dumps({"oauthAccount": oauth_account}), encoding="utf-8"
    )
    return oauth_account


def _write_credentials(
    home: Path,
    *,
    access_token: str = "access-1",
    expires_at: float | None = None,
) -> None:
    oauth_credentials: dict[str, Any] = {"accessToken": access_token}
    if expires_at is not None:
        oauth_credentials["expiresAt"] = expires_at
    credentials_dir = home / ".claude"
    credentials_dir.mkdir(parents=True, exist_ok=True)
    (credentials_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": oauth_credentials}), encoding="utf-8"
    )


def _credentials_secret(access_token: str = "keychain-access") -> str:
    return json.dumps({"claudeAiOauth": {"accessToken": access_token}})


def _provider(
    keychain: KeychainBackend | None = None,
    *,
    clock: FakeClock | None = None,
) -> AmbientAccountProvider:
    return AmbientAccountProvider(
        keychain=keychain if keychain is not None else FakeKeychainBackend(),
        clock=clock if clock is not None else time.monotonic,
    )


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _record(
    *,
    email: str = "registered@example.com",
    organization_uuid: str | None = _OTHER_ORGANIZATION_UUID,
    upstream_account_uuid: str | None = _OTHER_ACCOUNT_UUID,
) -> AccountRecord:
    return AccountRecord(
        id="registered-id",
        email=email,
        organization_uuid=organization_uuid,
        organization_name="Registered Organization",
        created_at=1,
        updated_at=2,
        last_authenticated_at=3,
        state="ready",
        account_incarnation_id="registered-incarnation-id",
        upstream_account_uuid=upstream_account_uuid,
    )


def test_parses_identity_into_complete_pool_member(tmp_path: Path) -> None:
    oauth_account = _write_identity(tmp_path)
    _write_credentials(tmp_path)

    provider = _provider()
    member = provider.pool_member()

    assert member is not None
    assert provider.is_ambient_account_id(member.record.id) is True
    assert provider.is_ambient_account_id("registered-id") is False
    assert member.oauth_account == oauth_account
    assert member.profile_fingerprint is not None
    assert member.record.email == "ambient@example.com"
    assert member.record.organization_uuid == _ORGANIZATION_UUID
    assert member.record.organization_name == "Ambient Organization"
    assert member.record.upstream_account_uuid == _ACCOUNT_UUID
    assert member.record.created_at == 0
    assert member.record.updated_at == 0
    assert member.record.last_authenticated_at == 0
    assert member.record.state == "ready"


def test_derived_ids_are_deterministic_and_identity_specific(tmp_path: Path) -> None:
    _write_identity(tmp_path, organization_uuid=_ORGANIZATION_UUID.upper())
    _write_credentials(tmp_path)
    first = _provider().pool_member()

    _write_identity(tmp_path, organization_uuid=_ORGANIZATION_UUID)
    same_identity = _provider().pool_member()

    _write_identity(tmp_path, email="different@example.com")
    different_identity = _provider().pool_member()

    assert first is not None
    assert same_identity is not None
    assert different_identity is not None
    assert first.record.id == same_identity.record.id
    assert first.record.account_incarnation_id == same_identity.record.account_incarnation_id
    assert first.record.id != different_identity.record.id
    assert (
        first.record.account_incarnation_id
        != different_identity.record.account_incarnation_id
    )


def test_get_credentials_reads_file_source(tmp_path: Path) -> None:
    _write_identity(tmp_path)
    _write_credentials(tmp_path, access_token="file-access")

    credentials = _run(_provider().auth_manager().get_credentials())

    assert credentials.access_token == "file-access"
    assert credentials.account_uuid == _ACCOUNT_UUID


def test_get_credentials_falls_back_to_keychain_when_file_is_absent(
    tmp_path: Path,
) -> None:
    _write_identity(tmp_path)
    keychain = FakeKeychainBackend(_credentials_secret())

    credentials = _run(_provider(keychain).auth_manager().get_credentials())

    assert credentials.access_token == "keychain-access"
    assert len(keychain.reads) == 1
    assert keychain.reads[0][0] == "Claude Code-credentials"


def test_file_source_prevents_keychain_lookup(tmp_path: Path) -> None:
    _write_identity(tmp_path)
    _write_credentials(tmp_path, access_token="file-access")
    keychain = FakeKeychainBackend(_credentials_secret())

    credentials = _run(_provider(keychain).auth_manager().get_credentials())

    assert credentials.access_token == "file-access"
    assert keychain.reads == []


def test_pool_member_is_none_when_identity_is_missing() -> None:
    keychain = FakeKeychainBackend(_credentials_secret())

    assert _provider(keychain).pool_member() is None
    assert keychain.reads == []


def test_pool_member_is_none_when_credentials_are_missing(tmp_path: Path) -> None:
    _write_identity(tmp_path)

    assert _provider(FakeKeychainBackend()).pool_member() is None


def test_pool_member_is_none_when_access_token_is_expired(tmp_path: Path) -> None:
    _write_identity(tmp_path)
    _write_credentials(tmp_path, expires_at=(time.time() - 1) * 1000)

    assert _provider().pool_member() is None


def test_failed_read_is_cached_within_ttl(tmp_path: Path) -> None:
    clock = FakeClock()
    provider = _provider(clock=clock)

    assert provider.pool_member() is None
    _write_identity(tmp_path)
    _write_credentials(tmp_path)
    clock.now = 29.9

    assert provider.pool_member() is None

    clock.now = 30.0
    assert provider.pool_member() is not None


def test_second_read_within_ttl_serves_cached_credentials(tmp_path: Path) -> None:
    _write_identity(tmp_path)
    _write_credentials(tmp_path, access_token="access-1")
    clock = FakeClock()
    provider = _provider(clock=clock)

    first = _run(provider.auth_manager().get_credentials())
    _write_credentials(tmp_path, access_token="access-2")
    clock.now = 29.9
    second = _run(provider.auth_manager().get_credentials())

    assert first.access_token == "access-1"
    assert second.access_token == "access-1"


def test_force_refresh_rereads_credentials(tmp_path: Path) -> None:
    _write_identity(tmp_path)
    _write_credentials(tmp_path, access_token="access-1")
    provider = _provider()
    assert _run(provider.auth_manager().get_credentials()).access_token == "access-1"
    _write_credentials(tmp_path, access_token="access-2")

    refreshed = _run(provider.auth_manager().get_credentials(force_refresh=True))

    assert refreshed.access_token == "access-2"


def test_force_refresh_rejects_same_expired_token(tmp_path: Path) -> None:
    _write_identity(tmp_path)
    _write_credentials(
        tmp_path,
        access_token="stale-access",
        expires_at=(time.time() - 1) * 1000,
    )
    manager = _provider().auth_manager()

    with pytest.raises(ClaudeAccountAuthError, match="local Claude Code CLI"):
        _run(manager.get_credentials(force_refresh=True))


def test_duplicate_identity_matches_primary_or_upstream_identity(tmp_path: Path) -> None:
    _write_identity(tmp_path)
    _write_credentials(tmp_path)
    member = _provider().pool_member()
    assert member is not None

    primary_match = _record(
        email="AMBIENT@EXAMPLE.COM",
        organization_uuid=_ORGANIZATION_UUID.upper(),
    )
    upstream_match = _record(upstream_account_uuid=_ACCOUNT_UUID.upper())
    non_match = _record()

    assert is_duplicate_identity(member, [primary_match]) is True
    assert is_duplicate_identity(member, [upstream_match]) is True
    assert is_duplicate_identity(member, [non_match]) is False


def test_module_has_no_http_client_refresh_path() -> None:
    source = inspect.getsource(ambient_module)

    assert "httpx" not in source
    assert "refreshToken" not in source
