"""Tests for Kimi device-flow login, credential loading, and refresh coalescing."""

from __future__ import annotations

import asyncio
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from claudex_gateway import kimi_auth
from claudex_gateway.kimi_auth import (
    KIMI_CLIENT_ID,
    KIMI_DEVICE_AUTH_URL,
    KIMI_TOKEN_URL,
    DeviceAuthorization,
    KimiAuthError,
    KimiAuthManager,
    poll_device_token,
    request_device_authorization,
    write_auth_file,
)


def _rfc3339_in(seconds: float) -> str:
    expiry = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return expiry.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_credentials(path: Path, access_token: str, expires_at: str | None) -> None:
    auth_data: dict[str, Any] = {
        "type": "kimi",
        "access_token": access_token,
        "refresh_token": "refresh-1",
        "token_type": "Bearer",
        "scope": "all",
        "device_id": "device-1",
    }
    if expires_at is not None:
        auth_data["expires_at"] = expires_at
    path.write_text(json.dumps(auth_data), encoding="utf-8")


def _refresh_transport(counter: dict[str, int], new_access_token: str) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == KIMI_TOKEN_URL
        form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
        assert form["client_id"] == KIMI_CLIENT_ID
        assert form["grant_type"] == "refresh_token"
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
                "expires_in": 3600,
            },
        )

    return httpx.MockTransport(handler)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_concurrent_forced_refreshes_rotate_only_once(tmp_path: Path) -> None:
    auth_file = tmp_path / "kimi-auth.json"
    _write_credentials(auth_file, "stale-token", _rfc3339_in(3600))
    counter = {"posts": 0}

    async def scenario() -> list[Any]:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            manager = KimiAuthManager(auth_file, http_client)
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
    persisted = json.loads(auth_file.read_text(encoding="utf-8"))
    assert persisted["access_token"] == "rotated-token"
    assert persisted["refresh_token"] == "refresh-2"
    assert isinstance(persisted["expires_at"], str)
    assert isinstance(persisted["last_refresh"], str)


def test_sequential_forced_refreshes_each_rotate(tmp_path: Path) -> None:
    auth_file = tmp_path / "kimi-auth.json"
    _write_credentials(auth_file, "stale-token", _rfc3339_in(3600))
    counter = {"posts": 0}

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            manager = KimiAuthManager(auth_file, http_client)
            await manager.get_credentials(force_refresh=True)
            # A later 401 against the rotated token is a new stale generation.
            await manager.get_credentials(force_refresh=True)

    _run(scenario())

    assert counter["posts"] == 2


def test_expiring_token_is_refreshed_without_force(tmp_path: Path) -> None:
    auth_file = tmp_path / "kimi-auth.json"
    _write_credentials(auth_file, "stale-token", _rfc3339_in(60))
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            return await KimiAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 1
    assert credentials.access_token == "rotated-token"
    assert credentials.device_id == "device-1"


def test_unparseable_expiry_is_refreshed(tmp_path: Path) -> None:
    auth_file = tmp_path / "kimi-auth.json"
    _write_credentials(auth_file, "stale-token", "not-a-timestamp")
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            return await KimiAuthManager(auth_file, http_client).get_credentials()

    assert _run(scenario()).access_token == "rotated-token"
    assert counter["posts"] == 1


def test_valid_token_is_returned_without_refresh(tmp_path: Path) -> None:
    auth_file = tmp_path / "kimi-auth.json"
    _write_credentials(auth_file, "valid-token", _rfc3339_in(3600))
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await KimiAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 0
    assert credentials.access_token == "valid-token"


def test_missing_expiry_counts_as_valid(tmp_path: Path) -> None:
    # Refreshing on absence would loop forever when the token endpoint omits
    # expires_in; the 401 retry path covers silent expiry instead.
    auth_file = tmp_path / "kimi-auth.json"
    _write_credentials(auth_file, "valid-token", None)
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await KimiAuthManager(auth_file, http_client).get_credentials()

    assert _run(scenario()).access_token == "valid-token"
    assert counter["posts"] == 0


def test_missing_auth_file_raises_with_guidance(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await KimiAuthManager(tmp_path / "kimi-auth.json", http_client).get_credentials()

    with pytest.raises(KimiAuthError, match="login kimi"):
        _run(scenario())


def test_refresh_failure_surfaces_status_and_body(tmp_path: Path) -> None:
    auth_file = tmp_path / "kimi-auth.json"
    _write_credentials(auth_file, "stale-token", _rfc3339_in(60))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await KimiAuthManager(auth_file, http_client).get_credentials()

    with pytest.raises(KimiAuthError, match="status 400.*invalid_grant"):
        _run(scenario())


def test_write_auth_file_is_owner_only(tmp_path: Path) -> None:
    auth_file = tmp_path / "nested" / "kimi-auth.json"
    write_auth_file(auth_file, {"type": "kimi", "access_token": "token"})
    assert json.loads(auth_file.read_text(encoding="utf-8"))["access_token"] == "token"
    assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600


class TestDeviceFlow:
    def test_device_authorization_parses_and_generates_device_id(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == KIMI_DEVICE_AUTH_URL
            assert request.content.decode() == f"client_id={KIMI_CLIENT_ID}"
            assert request.headers["X-Msh-Platform"] == "claudex-gateway"
            assert request.headers["X-Msh-Device-Id"]
            return httpx.Response(
                200,
                json={
                    "device_code": "dev-code",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://kimi.com/activate",
                    "verification_uri_complete": "https://kimi.com/activate?code=ABCD-1234",
                    "interval": 1,
                    "expires_in": 300,
                },
            )

        async def scenario() -> DeviceAuthorization:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await request_device_authorization(http_client)

        authorization = _run(scenario())

        assert authorization.device_code == "dev-code"
        assert authorization.user_code == "ABCD-1234"
        assert authorization.verification_uri == "https://kimi.com/activate"
        assert authorization.verification_uri_complete == (
            "https://kimi.com/activate?code=ABCD-1234"
        )
        assert authorization.device_id

    @staticmethod
    def _authorization(interval: float = 0.0, expires_in: float = 60.0) -> DeviceAuthorization:
        return DeviceAuthorization(
            device_code="dev-code",
            user_code="ABCD-1234",
            verification_uri="https://kimi.com/activate",
            verification_uri_complete=None,
            interval=interval,
            expires_in=expires_in,
            device_id="device-1",
        )

    def test_poll_waits_through_authorization_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(kimi_auth, "_MIN_POLL_INTERVAL_SECONDS", 0.0)
        counter = {"posts": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            counter["posts"] += 1
            if counter["posts"] < 3:
                return httpx.Response(400, json={"error": "authorization_pending"})
            return httpx.Response(
                200,
                json={
                    "access_token": "granted-token",
                    "refresh_token": "refresh-1",
                    "expires_in": 3600,
                    "scope": "all",
                },
            )

        async def scenario() -> dict[str, Any]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await poll_device_token(http_client, self._authorization())

        auth_data = _run(scenario())

        assert counter["posts"] == 3
        assert auth_data["access_token"] == "granted-token"
        assert auth_data["device_id"] == "device-1"
        assert auth_data["type"] == "kimi"
        assert "expires_at" in auth_data

    def test_poll_slow_down_grows_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kimi_auth, "_MIN_POLL_INTERVAL_SECONDS", 0.0)
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(kimi_auth.asyncio, "sleep", fake_sleep)
        counter = {"posts": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            counter["posts"] += 1
            if counter["posts"] == 1:
                return httpx.Response(400, json={"error": "slow_down"})
            return httpx.Response(200, json={"access_token": "granted-token"})

        async def scenario() -> dict[str, Any]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await poll_device_token(http_client, self._authorization(expires_in=600))

        _run(scenario())

        assert sleeps == [5.0]

    @pytest.mark.parametrize("error", ["access_denied", "expired_token"])
    def test_poll_terminal_errors_raise(
        self, monkeypatch: pytest.MonkeyPatch, error: str
    ) -> None:
        monkeypatch.setattr(kimi_auth, "_MIN_POLL_INTERVAL_SECONDS", 0.0)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": error})

        async def scenario() -> None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                await poll_device_token(http_client, self._authorization())

        with pytest.raises(KimiAuthError, match=error):
            _run(scenario())

    def test_poll_deadline_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kimi_auth, "_MIN_POLL_INTERVAL_SECONDS", 0.0)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "authorization_pending"})

        async def scenario() -> None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                await poll_device_token(
                    http_client, self._authorization(interval=10.0, expires_in=1.0)
                )

        with pytest.raises(KimiAuthError, match="timed out"):
            _run(scenario())
