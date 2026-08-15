"""Tests for Kimi Code CLI credential loading, refresh, and refresh coalescing."""

from __future__ import annotations

import asyncio
import base64
import json
import stat
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from claudex_gateway.providers.kimi_auth import (
    KIMI_CLIENT_ID,
    KIMI_TOKEN_URL,
    KimiAuthError,
    KimiAuthManager,
)


def _fake_jwt(claims: dict[str, Any]) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def _write_credentials(home: Path, access_token: str, expires_at: float | None) -> None:
    auth_data: dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": "refresh-1",
        "scope": "kimi-code",
        "token_type": "Bearer",
        "expires_in": 900,
    }
    if expires_at is not None:
        auth_data["expires_at"] = expires_at
    credentials_dir = home / "credentials"
    credentials_dir.mkdir(parents=True)
    (credentials_dir / "kimi-code.json").write_text(json.dumps(auth_data), encoding="utf-8")
    (home / "device_id").write_text("device-1\n", encoding="utf-8")


def _refresh_transport(counter: dict[str, int], new_access_token: str) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == KIMI_TOKEN_URL
        form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
        assert form["client_id"] == KIMI_CLIENT_ID
        assert form["grant_type"] == "refresh_token"
        assert form["refresh_token"] == "refresh-1"
        assert request.headers["X-Msh-Device-Id"] == "device-1"
        counter["posts"] += 1
        # Suspend while holding the refresh lock so a concurrent caller
        # genuinely waits on it instead of racing past.
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "access_token": new_access_token,
                "refresh_token": "refresh-2",
                "expires_in": 900,
                "scope": "kimi-code",
                "token_type": "Bearer",
            },
        )

    return httpx.MockTransport(handler)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_concurrent_forced_refreshes_rotate_only_once(tmp_path: Path) -> None:
    _write_credentials(tmp_path, "stale-token", time.time() + 3600)
    counter = {"posts": 0}

    async def scenario() -> list[Any]:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            manager = KimiAuthManager(tmp_path, http_client)
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
    credentials_file = tmp_path / "credentials" / "kimi-code.json"
    persisted_text = credentials_file.read_text(encoding="utf-8")
    persisted = json.loads(persisted_text)
    assert persisted["access_token"] == "rotated-token"
    assert persisted["refresh_token"] == "refresh-2"
    # The CLI's record shape is preserved: epoch expiry plus the granted TTL.
    assert persisted["expires_at"] > time.time()
    assert persisted["expires_in"] == 900
    assert persisted["scope"] == "kimi-code"
    assert persisted_text == json.dumps(persisted, indent=2) + "\n"
    assert stat.S_IMODE(credentials_file.stat().st_mode) == 0o600


def test_sequential_forced_refreshes_each_rotate(tmp_path: Path) -> None:
    _write_credentials(tmp_path, "stale-token", time.time() + 3600)
    counter = {"posts": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
        counter["posts"] += 1
        # The second rotation must spend the refresh token the first one wrote.
        assert form["refresh_token"] == f"refresh-{counter['posts']}"
        return httpx.Response(
            200,
            json={
                "access_token": f"rotated-{counter['posts']}",
                "refresh_token": f"refresh-{counter['posts'] + 1}",
                "expires_in": 900,
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            manager = KimiAuthManager(tmp_path, http_client)
            await manager.get_credentials(force_refresh=True)
            # A later 401 against the rotated token is a new stale generation.
            await manager.get_credentials(force_refresh=True)

    _run(scenario())

    assert counter["posts"] == 2


def test_expiring_token_is_refreshed_without_force(tmp_path: Path) -> None:
    _write_credentials(tmp_path, "stale-token", time.time() + 60)
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            return await KimiAuthManager(tmp_path, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 1
    assert credentials.access_token == "rotated-token"
    assert credentials.device_id == "device-1"


def test_valid_token_is_returned_without_refresh(tmp_path: Path) -> None:
    _write_credentials(tmp_path, "valid-token", time.time() + 3600)
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await KimiAuthManager(tmp_path, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 0
    assert credentials.access_token == "valid-token"


def test_missing_expiry_counts_as_valid(tmp_path: Path) -> None:
    # Refreshing on absence would loop forever when the token endpoint omits
    # expires_in; the 401 retry path covers silent expiry instead.
    _write_credentials(tmp_path, "valid-token", None)
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await KimiAuthManager(tmp_path, http_client).get_credentials()

    assert _run(scenario()).access_token == "valid-token"
    assert counter["posts"] == 0


def test_missing_credentials_file_raises_with_guidance(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await KimiAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(KimiAuthError, match="kimi login"):
        _run(scenario())


def test_missing_device_id_file_yields_no_device_id(tmp_path: Path) -> None:
    _write_credentials(tmp_path, "valid-token", time.time() + 3600)
    (tmp_path / "device_id").unlink()

    async def scenario() -> Any:
        async with httpx.AsyncClient() as http_client:
            return await KimiAuthManager(tmp_path, http_client).get_credentials()

    assert _run(scenario()).device_id is None


def test_account_is_decoded_from_the_access_token(tmp_path: Path) -> None:
    token = _fake_jwt({"user_id": "kimi-user-1", "sub": "fallback", "exp": 0})
    _write_credentials(tmp_path, token, time.time() + 3600)

    async def scenario() -> Any:
        async with httpx.AsyncClient() as http_client:
            return await KimiAuthManager(tmp_path, http_client).get_credentials()

    assert _run(scenario()).account == "kimi-user-1"


def test_account_is_none_for_a_non_jwt_token(tmp_path: Path) -> None:
    _write_credentials(tmp_path, "opaque-token", time.time() + 3600)

    async def scenario() -> Any:
        async with httpx.AsyncClient() as http_client:
            return await KimiAuthManager(tmp_path, http_client).get_credentials()

    assert _run(scenario()).account is None


def test_refresh_failure_surfaces_status_and_body(tmp_path: Path) -> None:
    _write_credentials(tmp_path, "stale-token", time.time() + 60)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await KimiAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(KimiAuthError, match="status 400.*invalid_grant"):
        _run(scenario())
