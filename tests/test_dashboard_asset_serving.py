"""Runtime tests for packaged dashboard asset serving."""

from __future__ import annotations

import asyncio
import importlib.resources
import json
from collections.abc import Awaitable, Callable

import pytest
from starlette.responses import Response

import claudex.admin.system as admin_system
import claudex.server_support as server_support


_DashboardHandler = Callable[..., Awaitable[Response]]
_DASHBOARD_CASES = [
    (admin_system._handle_dashboard, "dashboard.html", "text/html; charset=utf-8"),
    (admin_system._handle_dashboard_css, "dashboard.css", "text/css"),
    (admin_system._handle_dashboard_js, "dashboard.js", "application/javascript"),
]


@pytest.mark.parametrize(
    ("handler", "basename", "media_type"),
    _DASHBOARD_CASES,
    ids=["html", "css", "javascript"],
)
def test_dashboard_wrappers_load_expected_utf8_assets(
    monkeypatch: pytest.MonkeyPatch,
    handler: _DashboardHandler,
    basename: str,
    media_type: str,
) -> None:
    sentinel_body = f"sentinel body for {basename}"
    anchors: list[str] = []
    joined_paths: list[tuple[str, ...]] = []

    class FakeAsset:
        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return sentinel_body

    class FakePackage:
        def joinpath(self, *parts: str) -> FakeAsset:
            joined_paths.append(parts)
            return FakeAsset()

    def fake_files(anchor: str) -> FakePackage:
        anchors.append(anchor)
        return FakePackage()

    monkeypatch.setattr(importlib.resources, "files", fake_files)

    response = asyncio.run(handler(None))

    assert anchors == ["claudex"]
    assert joined_paths == [("dashboard", basename)]
    assert response.status_code == 200
    assert response.body.decode("utf-8") == sentinel_body
    assert response.media_type == media_type


@pytest.mark.parametrize(
    ("handler", "basename", "media_type"),
    _DASHBOARD_CASES,
    ids=["html", "css", "javascript"],
)
def test_dashboard_wrappers_handle_joinpath_oserror(
    monkeypatch: pytest.MonkeyPatch,
    handler: _DashboardHandler,
    basename: str,
    media_type: str,
) -> None:
    class JoinFailurePackage:
        def joinpath(self, *parts: str) -> object:
            raise OSError(f"cannot join {parts!r}")

    monkeypatch.setattr(
        importlib.resources,
        "files",
        lambda _anchor: JoinFailurePackage(),
    )

    response = asyncio.run(handler(None))

    assert response.status_code == 500
    assert json.loads(response.body) == server_support._openai_error_body(
        "server_error", f"{basename} is missing from the package"
    )


@pytest.mark.parametrize(
    ("handler", "basename", "media_type"),
    _DASHBOARD_CASES,
    ids=["html", "css", "javascript"],
)
def test_dashboard_wrappers_handle_read_text_oserror(
    monkeypatch: pytest.MonkeyPatch,
    handler: _DashboardHandler,
    basename: str,
    media_type: str,
) -> None:
    class ReadFailureAsset:
        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            raise OSError("cannot read dashboard asset")

    class FakePackage:
        def joinpath(self, *parts: str) -> ReadFailureAsset:
            return ReadFailureAsset()

    monkeypatch.setattr(
        importlib.resources,
        "files",
        lambda _anchor: FakePackage(),
    )

    response = asyncio.run(handler(None))

    assert response.status_code == 500
    assert json.loads(response.body) == server_support._openai_error_body(
        "server_error", f"{basename} is missing from the package"
    )
