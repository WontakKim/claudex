"""Tests for gptpro storage-state persistence and static validation."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from claudex import paths
from claudex.gptpro import session
from claudex.providers import auth_support


def _storage_state(
    *, expires: float | int | None = 4_000_000_000, token: str = "secret-token"
) -> dict[str, object]:
    cookie: dict[str, object] = {
        "name": f"{session.AUTH_COOKIE_PREFIX}.0",
        "value": token,
        "domain": ".chatgpt.com",
        "path": "/",
    }
    if expires is not None:
        cookie["expires"] = expires
    return {"cookies": [cookie], "origins": []}


def _write_state(path: Path, state: object) -> None:
    path.write_text(json.dumps(state), encoding="utf-8")


def test_auth_cookie_policy_selects_the_first_prefixed_cookie() -> None:
    expected = {
        "name": f"{session.AUTH_COOKIE_PREFIX}.1",
        "value": "selected",
        "domain": ".chatgpt.com",
    }
    cookies = [
        {"name": "unrelated", "value": "ignored", "domain": ".chatgpt.com"},
        expected,
        {"name": session.AUTH_COOKIE_PREFIX, "value": "later", "domain": ".chatgpt.com"},
    ]

    assert session.find_auth_cookie(cookies) is expected
    assert session.AUTH_COOKIE_NAME_PATTERN.match(expected["name"])
    assert session.AUTH_COOKIE_NAME_PATTERN.match("unrelated") is None


def test_auth_cookie_policy_rejects_valueless_cookie() -> None:
    cookies = [{"name": session.AUTH_COOKIE_PREFIX, "domain": ".chatgpt.com"}]

    assert session.find_auth_cookie(cookies) is None


def test_auth_cookie_policy_rejects_foreign_domain_cookie() -> None:
    cookies = [
        {
            "name": session.AUTH_COOKIE_PREFIX,
            "value": "foreign-secret",
            "domain": ".example.com",
        }
    ]

    assert session.find_auth_cookie(cookies) is None


def test_load_auth_cookie_expiry_returns_first_prefixed_cookie(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    state = _storage_state(expires=1234)
    state["cookies"] = [
        {"name": "unrelated", "expires": 1},
        *state["cookies"],
        {
            "name": session.AUTH_COOKIE_PREFIX,
            "expires": 9999,
        },
    ]
    _write_state(path, state)

    assert session.load_auth_cookie_expiry(path) == 1234.0


@pytest.mark.parametrize(
    "contents",
    [
        "not-json",
        json.dumps([]),
        json.dumps({"cookies": [], "origins": []}),
        json.dumps({"cookies": [], "origins": {}}),
    ],
)
def test_load_auth_cookie_expiry_rejects_broken_or_missing_state(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "session.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(session.GptProSessionError):
        session.load_auth_cookie_expiry(path)


def test_invalid_utf8_session_is_reported_as_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "session.json"
    path.write_bytes(b"\xff\xfe")
    monkeypatch.setattr(session.paths, "gptpro_session_file", lambda: path)

    with pytest.raises(session.GptProSessionError):
        session.load_auth_cookie_expiry(path)

    status = session.session_status()
    assert status["exists"] is True
    assert status["valid"] is False
    assert "invalid" in str(status["message"])


def test_load_auth_cookie_expiry_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(session.GptProSessionError):
        session.load_auth_cookie_expiry(tmp_path / "missing.json")


def test_load_auth_cookie_expiry_accepts_a_session_cookie(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    _write_state(path, _storage_state(expires=None))

    assert session.load_auth_cookie_expiry(path) is None


@pytest.mark.parametrize(
    ("expires", "now", "expected"),
    [
        (100.0, 101.0, True),
        (100.0, 100.0, False),
        (101.0, 100.0, False),
        (0.0, 100.0, False),
        (-1.0, 100.0, False),
        (None, 100.0, False),
    ],
)
def test_is_expired_preserves_the_plugin_rule(
    expires: float | None, now: float, expected: bool
) -> None:
    assert session.is_expired(expires, now) is expected


class _FakeStorageContext:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state

    async def storage_state(self) -> dict[str, object]:
        return self.state


def test_save_storage_state_is_atomic_private_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    state = _storage_state()
    path = tmp_path / "gptpro" / "session.json"

    asyncio.run(session.save_storage_state(_FakeStorageContext(state), path))

    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert list(path.parent.iterdir()) == [path]


def test_private_json_writer_fsyncs_file_then_replace_then_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "runtime" / "gptpro" / "session.json"
    events: list[str] = []
    real_replace = auth_support.os.replace

    def recording_fsync(file_descriptor: int) -> None:
        mode = os.fstat(file_descriptor).st_mode
        events.append("directory-fsync" if stat.S_ISDIR(mode) else "file-fsync")

    def recording_replace(source: Path, destination: Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(auth_support.os, "fsync", recording_fsync)
    monkeypatch.setattr(auth_support.os, "replace", recording_replace)

    auth_support.write_private_json_atomic(path, _storage_state())

    assert events == ["file-fsync", "replace", "directory-fsync"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700
    assert list(path.parent.iterdir()) == [path]


def test_session_status_reports_missing_valid_expired_and_invalid_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / ".claudex" / "gptpro" / "session.json"
    monkeypatch.setattr(session.paths, "gptpro_session_file", lambda: path)

    missing = session.session_status()
    assert missing["exists"] is False
    assert missing["valid"] is False
    assert "run claudex-gateway gptpro login" in str(missing["message"])

    path.parent.mkdir(parents=True)
    _write_state(path, _storage_state(expires=4_000_000_000))
    valid = session.session_status()
    assert valid["has_auth_cookie"] is True
    assert valid["expired"] is False
    assert valid["valid"] is True

    _write_state(path, _storage_state(expires=1))
    expired = session.session_status()
    assert expired["expired"] is True
    assert expired["valid"] is False

    path.write_text("broken", encoding="utf-8")
    invalid = session.session_status()
    assert invalid["exists"] is True
    assert invalid["has_auth_cookie"] is False
    assert invalid["valid"] is False


def test_session_status_rejects_cookie_playwright_cannot_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = paths.gptpro_session_file()
    path.parent.mkdir(parents=True)

    _write_state(path, {"cookies": [{"name": session.AUTH_COOKIE_PREFIX}], "origins": []})
    valueless = session.session_status()
    assert valueless["has_auth_cookie"] is False
    assert valueless["valid"] is False

    _write_state(
        path,
        {
            "cookies": [
                {
                    "name": session.AUTH_COOKIE_PREFIX,
                    "value": "foreign-secret",
                    "domain": ".example.com",
                }
            ],
            "origins": [],
        },
    )
    foreign = session.session_status()
    assert foreign["has_auth_cookie"] is False
    assert foreign["valid"] is False
