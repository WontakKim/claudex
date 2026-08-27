"""Tests for ChatGPT Pro plain-text attachment uploads."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from claudex.gptpro import attachments


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += seconds


class _FakeRequest:
    method = "POST"


class _FakeResponse:
    request = _FakeRequest()
    url = "https://chatgpt.com/backend-api/files"
    status = 201

    async def finished(self) -> None:
        return None


class _FakeHandle:
    def __init__(self, label: str) -> None:
        self.label = label
        self.dispose_calls = 0

    async def dispose(self) -> None:
        self.dispose_calls += 1


class _FakePage:
    def __init__(self, body_text: str) -> None:
        self.body_text = body_text
        self.listeners: dict[str, list[Callable[[Any], None]]] = {
            "response": []
        }
        self.evaluate_handle_calls: list[tuple[str, Any]] = []
        self.evaluate_calls: list[tuple[str, Any]] = []
        self.handles: list[_FakeHandle] = []

    def on(self, event: str, listener: Callable[[Any], None]) -> None:
        self.listeners[event].append(listener)

    def remove_listener(
        self, event: str, listener: Callable[[Any], None]
    ) -> None:
        self.listeners[event].remove(listener)

    async def evaluate_handle(self, script: str, argument: Any) -> _FakeHandle:
        self.evaluate_handle_calls.append((script, argument))
        handle = _FakeHandle(f"handle-{len(self.handles)}")
        self.handles.append(handle)
        return handle

    async def evaluate(self, script: str, argument: Any = None) -> Any:
        self.evaluate_calls.append((script, argument))
        if script == attachments.DISPATCH_ATTACHMENT_DROP_JS:
            file_count = len(self.evaluate_handle_calls) - 1
            for _ in range(file_count):
                for listener in tuple(self.listeners["response"]):
                    listener(_FakeResponse())
            return "#thread-bottom-container"
        if script == attachments.READ_BODY_INNER_TEXT_JS:
            return self.body_text
        raise AssertionError("unexpected page evaluation")


def test_empty_attachment_list_is_noop() -> None:
    asyncio.run(attachments.attach_files(object(), []))


def test_attachment_count_limit_is_validated_before_reading() -> None:
    paths = ["missing.txt"] * (attachments.MAX_ATTACHMENTS_PER_ASK + 1)

    with pytest.raises(ValueError, match="At most 10 attachments"):
        asyncio.run(attachments.attach_files(object(), paths))


def test_total_attachment_size_is_limited(tmp_path: Path) -> None:
    attachment = tmp_path / "large.txt"
    attachment.write_bytes(b"x" * (attachments.MAX_TOTAL_ATTACHMENT_BYTES + 1))

    with pytest.raises(ValueError, match="1200001 bytes.*1200000-byte limit"):
        asyncio.run(attachments.attach_files(object(), [str(attachment)]))


def test_attachment_must_be_valid_utf8(tmp_path: Path) -> None:
    attachment = tmp_path / "invalid.txt"
    attachment.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="UTF-8 plain text.*invalid UTF-8"):
        asyncio.run(attachments.attach_files(object(), [str(attachment)]))


def test_attachment_must_not_contain_nul_bytes(tmp_path: Path) -> None:
    attachment = tmp_path / "nul.txt"
    attachment.write_bytes(b"before\x00after")

    with pytest.raises(ValueError, match="UTF-8 plain text.*NUL bytes"):
        asyncio.run(attachments.attach_files(object(), [str(attachment)]))


def test_attach_files_dispatches_drop_and_waits_for_settlement(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first body", encoding="utf-8")
    second.write_text("second body", encoding="utf-8")
    page = _FakePage("first.txt\nsecond.txt")

    asyncio.run(
        attachments.attach_files(page, [str(first), str(second)])
    )

    assert [call[0] for call in page.evaluate_handle_calls] == [
        attachments.CREATE_ATTACHMENT_FILE_JS,
        attachments.CREATE_ATTACHMENT_FILE_JS,
        attachments.CREATE_ATTACHMENT_DATA_TRANSFER_JS,
    ]
    first_descriptor = page.evaluate_handle_calls[0][1]
    second_descriptor = page.evaluate_handle_calls[1][1]
    assert base64.b64decode(first_descriptor["bytesBase64"]) == b"first body"
    assert base64.b64decode(second_descriptor["bytesBase64"]) == b"second body"
    assert (first_descriptor["name"], first_descriptor["mime"]) == (
        "first.txt",
        "text/plain",
    )
    assert (second_descriptor["name"], second_descriptor["mime"]) == (
        "second.txt",
        "text/plain",
    )
    assert page.evaluate_handle_calls[2][1] == page.handles[:2]
    assert page.evaluate_calls[0] == (
        attachments.DISPATCH_ATTACHMENT_DROP_JS,
        page.handles[2],
    )
    assert page.evaluate_calls[1][0] == attachments.READ_BODY_INNER_TEXT_JS
    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)


def test_attach_files_timeout_reports_settle_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "missing-chip.txt"
    attachment.write_text("body", encoding="utf-8")
    page = _FakePage("composer without the filename")
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(
            attachments.attach_files(
                page,
                [str(attachment)],
                timeout_seconds=0.01,
            )
        )

    message = str(raised.value)
    assert "1/1 completed POST /backend-api/files responses" in message
    assert "expected filename chips for 'missing-chip.txt'" in message
    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)
