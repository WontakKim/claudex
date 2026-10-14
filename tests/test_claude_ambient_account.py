"""Tests for the read-only ambient Claude Code CLI account provider."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from claudex.claude import ambient_account as ambient_module
from claudex.claude.accounts import AccountRecord
from claudex.claude.ambient_account import (
    AmbientAccountProvider,
    AmbientPoolMember,
    is_duplicate_identity,
)
from claudex.claude.auth import ClaudeAccountAuthError
from claudex.claude.capture_model import KeychainBackend
from claudex.claude.keychain import LEGACY_KEYCHAIN_SERVICE, scoped_keychain_service

_ACCOUNT_UUID = "226432ad-b115-4a79-a152-8420aacbd4b2"
_ORGANIZATION_UUID = "8ccaa9d3-2bb1-4447-a09b-b35467e47a9e"
_OTHER_ACCOUNT_UUID = "d16a1603-6ca0-4939-8f4c-b5eaa833aa08"
_OTHER_ORGANIZATION_UUID = "135e9673-7685-4de2-9e03-a4cedade4680"
_KEYCHAIN_ACCOUNT = "ambient-test-user"


class FakeKeychainBackend:
    """In-memory KeychainBackend that records every lookup."""

    def __init__(self, secret: str | None = None) -> None:
        self.store: dict[tuple[str, str], str] = {}
        if secret is not None:
            self.store[(LEGACY_KEYCHAIN_SERVICE, _KEYCHAIN_ACCOUNT)] = secret
        self.reads: list[tuple[str, str]] = []

    def read(self, service: str, account: str) -> str | None:
        self.reads.append((service, account))
        return self.store.get((service, account))

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
    monkeypatch.setenv("USER", _KEYCHAIN_ACCOUNT)
    monkeypatch.setenv("USERNAME", _KEYCHAIN_ACCOUNT)
    monkeypatch.setenv("LOGNAME", _KEYCHAIN_ACCOUNT)
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


def test_explicit_profile_uses_scoped_keychain_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_identity(profile_dir)
    raw_config_dir = str(profile_dir)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", raw_config_dir)
    monkeypatch.setenv("USER", "profile-keychain-user")
    monkeypatch.setenv("LOGNAME", "legacy-keychain-user")
    service = scoped_keychain_service(raw_config_dir)
    keychain = FakeKeychainBackend()
    keychain.store = {
        (service, "profile-keychain-user"): _credentials_secret("scoped-profile-access"),
        (LEGACY_KEYCHAIN_SERVICE, "legacy-keychain-user"): _credentials_secret(
            "unrelated-legacy-access"
        ),
    }
    initial_store = keychain.store.copy()
    provider = _provider(keychain)

    member = provider.pool_member()
    credentials = _run(provider.auth_manager().get_credentials())

    assert member is not None
    assert member.record.upstream_account_uuid == _ACCOUNT_UUID
    assert credentials.access_token == "scoped-profile-access"
    assert credentials.account_uuid == _ACCOUNT_UUID
    assert keychain.reads == [(service, "profile-keychain-user")]
    assert keychain.store == initial_store


def test_explicit_profile_with_only_scoped_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_identity(profile_dir)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile_dir))
    service = scoped_keychain_service(str(profile_dir))
    keychain = FakeKeychainBackend()
    keychain.store[(service, _KEYCHAIN_ACCOUNT)] = _credentials_secret("scoped-access")
    provider = _provider(keychain)

    member = provider.pool_member()
    credentials = _run(provider.auth_manager().get_credentials())

    assert member is not None
    assert credentials.access_token == "scoped-access"
    assert credentials.account_uuid == _ACCOUNT_UUID
    assert keychain.reads == [(service, _KEYCHAIN_ACCOUNT)]
    assert is_duplicate_identity(member, [_record(upstream_account_uuid=_ACCOUNT_UUID)])


@pytest.mark.parametrize(
    "scoped_secret",
    [
        None,
        "not-json",
        "[]",
        json.dumps({"claudeAiOauth": {}}),
        json.dumps({"claudeAiOauth": {"accessToken": "expired-access", "expiresAt": 1}}),
    ],
    ids=["missing", "malformed", "non-object", "tokenless", "expired"],
)
def test_unavailable_scoped_credentials_never_borrow_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scoped_secret: str | None
) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_identity(profile_dir)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile_dir))
    service = scoped_keychain_service(str(profile_dir))
    keychain = FakeKeychainBackend(_credentials_secret("unrelated-legacy-access"))
    if scoped_secret is not None:
        keychain.store[(service, _KEYCHAIN_ACCOUNT)] = scoped_secret
    initial_store = keychain.store.copy()
    provider = _provider(keychain)

    assert provider.pool_member() is None
    with pytest.raises(ClaudeAccountAuthError, match="local Claude Code CLI"):
        _run(provider.auth_manager().get_credentials())
    assert keychain.reads == [(service, _KEYCHAIN_ACCOUNT)]
    assert keychain.store == initial_store


@pytest.mark.parametrize(
    "path_form",
    [
        "absolute",
        "trailing-slash",
        "repeated-slash",
        "dot-component",
        "tilde",
        "explicit-default",
        "relative",
        "empty",
        "unicode-nfc",
        "unicode-nfd",
    ],
)
def test_scoped_lookup_uses_raw_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_form: str
) -> None:
    monkeypatch.chdir(tmp_path)
    base_path = str(tmp_path / "profile")
    raw_config_dir = {
        "absolute": base_path,
        "trailing-slash": base_path + "/",
        "repeated-slash": str(tmp_path) + "//profile",
        "dot-component": str(tmp_path) + "/./profile",
        "tilde": "~/profile",
        "explicit-default": "~/.claude",
        "relative": "profile",
        "empty": "",
        "unicode-nfc": str(tmp_path / "profile-café"),
        "unicode-nfd": unicodedata.normalize("NFD", str(tmp_path / "profile-café")),
    }[path_form]
    profile_dir = Path(raw_config_dir).expanduser()
    profile_dir.mkdir(parents=True, exist_ok=True)
    _write_identity(profile_dir)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", raw_config_dir)
    service = scoped_keychain_service(raw_config_dir)
    resolved_service = scoped_keychain_service(str(profile_dir.resolve()))
    keychain = FakeKeychainBackend(_credentials_secret("unrelated-legacy-access"))
    if resolved_service != service:
        keychain.store[(resolved_service, _KEYCHAIN_ACCOUNT)] = _credentials_secret(
            "wrong-resolved-path-access"
        )
    keychain.store[(service, _KEYCHAIN_ACCOUNT)] = _credentials_secret("raw-path-access")

    credentials = _run(_provider(keychain).auth_manager().get_credentials())

    assert credentials.access_token == "raw-path-access"
    assert credentials.account_uuid == _ACCOUNT_UUID
    assert keychain.reads == [(service, _KEYCHAIN_ACCOUNT)]


def test_scoped_keychain_account_uses_username_when_user_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_identity(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("USER")
    monkeypatch.setenv("USERNAME", "profile-username")
    service = scoped_keychain_service(str(tmp_path))
    keychain = FakeKeychainBackend(_credentials_secret("unrelated-legacy-access"))
    keychain.store[(service, "profile-username")] = _credentials_secret("scoped-access")

    credentials = _run(_provider(keychain).auth_manager().get_credentials())

    assert credentials.access_token == "scoped-access"
    assert keychain.reads == [(service, "profile-username")]


def test_missing_scoped_keychain_account_does_not_guess_or_borrow_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_identity(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("USER")
    monkeypatch.delenv("USERNAME")
    keychain = FakeKeychainBackend(_credentials_secret("unrelated-legacy-access"))
    keychain.store[(scoped_keychain_service(str(tmp_path)), _KEYCHAIN_ACCOUNT)] = (
        _credentials_secret("scoped-access")
    )
    provider = _provider(keychain)

    assert provider.pool_member() is None
    with pytest.raises(ClaudeAccountAuthError, match="local Claude Code CLI"):
        _run(provider.auth_manager().get_credentials())
    assert keychain.reads == []


def test_unset_profile_preserves_legacy_getpass_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_identity(tmp_path)
    monkeypatch.setenv("USER", "different-user")
    keychain = FakeKeychainBackend(_credentials_secret("legacy-access"))

    credentials = _run(_provider(keychain).auth_manager().get_credentials())

    assert credentials.access_token == "legacy-access"
    assert keychain.reads == [(LEGACY_KEYCHAIN_SERVICE, _KEYCHAIN_ACCOUNT)]


def test_scoped_credentials_preserve_cache_expiry_and_force_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_identity(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    service = scoped_keychain_service(str(tmp_path))
    source_key = (service, _KEYCHAIN_ACCOUNT)
    keychain = FakeKeychainBackend()
    keychain.store[source_key] = _credentials_secret("access-1")
    clock = FakeClock()
    provider = _provider(keychain, clock=clock)
    manager = provider.auth_manager()
    assert provider.pool_member() is not None
    assert _run(manager.get_credentials()).access_token == "access-1"

    keychain.store[source_key] = _credentials_secret("access-2")
    clock.now = 29.9
    assert _run(manager.get_credentials()).access_token == "access-1"
    assert keychain.reads == [source_key]

    clock.now = 30.0
    assert _run(manager.get_credentials()).access_token == "access-2"
    assert keychain.reads == [source_key, source_key]

    with pytest.raises(ClaudeAccountAuthError, match="local Claude Code CLI"):
        _run(manager.get_credentials(force_refresh=True))
    keychain.store[source_key] = _credentials_secret("access-3")
    assert _run(manager.get_credentials(force_refresh=True)).access_token == "access-3"
    assert keychain.reads == [source_key] * 4


def test_explicit_profile_file_prevents_keychain_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_identity(profile_dir)
    (profile_dir / ".credentials.json").write_text(
        _credentials_secret("file-access"), encoding="utf-8"
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile_dir))
    keychain = FakeKeychainBackend(_credentials_secret("legacy-access"))
    keychain.store[(scoped_keychain_service(str(profile_dir)), _KEYCHAIN_ACCOUNT)] = (
        _credentials_secret("scoped-access")
    )

    credentials = _run(_provider(keychain).auth_manager().get_credentials())

    assert credentials.access_token == "file-access"
    assert credentials.account_uuid == _ACCOUNT_UUID
    assert keychain.reads == []


@pytest.mark.parametrize("raw_credentials", [b"not-json", b"[]", b"{}", b"\xff"])
def test_malformed_profile_file_does_not_fall_back_to_keychain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_credentials: bytes
) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_identity(profile_dir)
    (profile_dir / ".credentials.json").write_bytes(raw_credentials)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile_dir))
    keychain = FakeKeychainBackend(_credentials_secret("legacy-access"))
    keychain.store[(scoped_keychain_service(str(profile_dir)), _KEYCHAIN_ACCOUNT)] = (
        _credentials_secret("scoped-access")
    )
    provider = _provider(keychain)

    assert provider.pool_member() is None
    with pytest.raises(ClaudeAccountAuthError, match="local Claude Code CLI"):
        _run(provider.auth_manager().get_credentials())
    assert keychain.reads == []


def test_unreadable_profile_file_does_not_fall_back_to_keychain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_identity(profile_dir)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile_dir))
    credentials_file = profile_dir / ".credentials.json"
    original_read_text = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == credentials_file:
            raise PermissionError("synthetic unreadable credential file")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    keychain = FakeKeychainBackend(_credentials_secret())

    assert _provider(keychain).pool_member() is None
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
