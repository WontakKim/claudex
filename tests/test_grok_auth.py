"""Tests for Grok CLI credential loading, OIDC refresh, and refresh coalescing."""

from __future__ import annotations

import asyncio
import json
import stat
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from claudex_gateway.grok_auth import XAI_ISSUER, GrokAuthError, GrokAuthManager

_SCOPE = f"{XAI_ISSUER}::client-1"
_TOKEN_ENDPOINT = "https://auth.x.ai/oauth/token"
_DISCOVERY_URL = f"{XAI_ISSUER}/.well-known/openid-configuration"


def _rfc3339_in(seconds: float) -> str:
    expiry = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return expiry.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_store(auth_file: Path, key: str, expires_at: str | None) -> None:
    entry: dict[str, Any] = {
        "key": key,
        "auth_mode": "oidc",
        "create_time": _rfc3339_in(-3600),
        "user_id": "user-1",
        "email": "user@example.com",
        "principal_type": "User",
        "principal_id": "user-1",
        "refresh_token": "refresh-1",
        "oidc_issuer": XAI_ISSUER,
        "oidc_client_id": "client-1",
    }
    if expires_at is not None:
        entry["expires_at"] = expires_at
    auth_file.write_text(json.dumps({_SCOPE: entry}), encoding="utf-8")


def _refresh_transport(counter: dict[str, int], new_access_token: str) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == _DISCOVERY_URL:
            return httpx.Response(200, json={"token_endpoint": _TOKEN_ENDPOINT})
        assert str(request.url) == _TOKEN_ENDPOINT
        form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
        assert form["grant_type"] == "refresh_token"
        assert form["refresh_token"] == "refresh-1"
        assert form["client_id"] == "client-1"
        # Team principals refresh under their principal identity.
        assert form["principal_type"] == "User"
        assert form["principal_id"] == "user-1"
        counter["posts"] += 1
        # Suspend while holding the refresh lock so a concurrent caller
        # genuinely waits on it instead of racing past.
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "access_token": new_access_token,
                "refresh_token": "refresh-2",
                "expires_in": 21600,
                "token_type": "Bearer",
            },
        )

    return httpx.MockTransport(handler)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_valid_token_is_returned_without_refresh(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_store(auth_file, "valid-token", _rfc3339_in(3600))
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await GrokAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 0
    assert credentials.access_token == "valid-token"
    assert credentials.email == "user@example.com"
    assert credentials.is_api_key is False


def test_expiring_token_is_refreshed_and_persisted(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_store(auth_file, "stale-token", _rfc3339_in(60))
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            return await GrokAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 1
    assert credentials.access_token == "rotated-token"
    persisted = json.loads(auth_file.read_text(encoding="utf-8"))[_SCOPE]
    assert persisted["key"] == "rotated-token"
    assert persisted["refresh_token"] == "refresh-2"
    assert persisted["expires_at"] > _rfc3339_in(3600)
    # Untouched identity fields survive the rotation.
    assert persisted["email"] == "user@example.com"
    assert persisted["oidc_client_id"] == "client-1"


def test_concurrent_forced_refreshes_rotate_only_once(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_store(auth_file, "stale-token", _rfc3339_in(3600))
    counter = {"posts": 0}

    async def scenario() -> list[Any]:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            manager = GrokAuthManager(auth_file, http_client)
            # Two requests saw a 401 with the same stale token and retry.
            return list(
                await asyncio.gather(
                    manager.get_credentials(force_refresh=True),
                    manager.get_credentials(force_refresh=True),
                )
            )

    first, second = _run(scenario())

    assert counter["posts"] == 1
    assert first.access_token == "rotated-token"
    assert second.access_token == "rotated-token"


def test_missing_auth_file_raises_with_guidance(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await GrokAuthManager(tmp_path / "auth.json", http_client).get_credentials()

    with pytest.raises(GrokAuthError, match="grok login"):
        _run(scenario())


def test_api_key_entry_is_used_without_refresh(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({"grok::api_key": {"key": "grok-api-key", "auth_mode": "api_key"}}),
        encoding="utf-8",
    )

    async def failing_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP expected for API-key credentials")

    async def scenario() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(failing_handler)) as http_client:
            return await GrokAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert credentials.access_token == "grok-api-key"
    assert credentials.is_api_key is True
    assert credentials.email is None


def test_legacy_web_login_entry_is_rejected(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {"https://accounts.x.ai/sign-in": {"key": "old", "auth_mode": "web_login"}}
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await GrokAuthManager(auth_file, http_client).get_credentials()

    with pytest.raises(GrokAuthError, match="grok login"):
        _run(scenario())


def test_expired_entry_without_refresh_token_raises_with_guidance(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_store(auth_file, "stale-token", _rfc3339_in(-60))
    store = json.loads(auth_file.read_text(encoding="utf-8"))
    del store[_SCOPE]["refresh_token"]
    auth_file.write_text(json.dumps(store), encoding="utf-8")

    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await GrokAuthManager(auth_file, http_client).get_credentials()

    with pytest.raises(GrokAuthError, match="grok login"):
        _run(scenario())


def test_refresh_failure_surfaces_status_and_body(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_store(auth_file, "stale-token", _rfc3339_in(60))

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == _DISCOVERY_URL:
            return httpx.Response(200, json={"token_endpoint": _TOKEN_ENDPOINT})
        return httpx.Response(400, json={"error": "invalid_grant"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await GrokAuthManager(auth_file, http_client).get_credentials()

    with pytest.raises(GrokAuthError, match="status 400.*invalid_grant"):
        _run(scenario())


def test_persisted_store_is_owner_only(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_store(auth_file, "stale-token", _rfc3339_in(60))
    counter = {"posts": 0}

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            await GrokAuthManager(auth_file, http_client).get_credentials()

    _run(scenario())

    assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600
