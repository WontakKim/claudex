"""Tests for ChatGPT Pro plain-text attachment uploads."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import async_playwright

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
    def __init__(
        self,
        *,
        states: list[dict[str, Any]] | None = None,
        responses: list[Any] | None = None,
    ) -> None:
        self.states = iter(states or [])
        self.responses = responses
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
            responses = self.responses
            if responses is None:
                responses = [_FakeResponse() for _ in range(file_count)]
            for response in responses:
                for listener in tuple(self.listeners["response"]):
                    listener(response)
            return "form[data-chatgpt-composer]"
        if script == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS:
            return next(
                self.states,
                {"ready": {}, "processing": {}, "failed": {}},
            )
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
    page = _FakePage(
        states=[
            {"ready": {}, "processing": {}, "failed": {}},
            {
                "ready": {"first.txt": 1, "second.txt": 1},
                "processing": {},
                "failed": {},
            },
        ],
    )

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
    assert (
        attachments.DISPATCH_ATTACHMENT_DROP_JS, page.handles[2]
    ) in page.evaluate_calls
    assert any(
        script == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
        for script, _ in page.evaluate_calls
    )
    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)


def test_drop_dispatch_targets_redesigned_composer_form() -> None:
    drop_js = attachments.DISPATCH_ATTACHMENT_DROP_JS

    assert (
        "const preferredSelector = 'form[data-chatgpt-composer]'" in drop_js
    )
    assert "#thread-bottom-container" not in drop_js
    assert "const fallbackSelector = 'main'" in drop_js


def test_attach_files_timeout_reports_settle_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "missing-chip.txt"
    attachment.write_text("body", encoding="utf-8")
    page = _FakePage()
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


def _attachment_state(
    *,
    ready: dict[str, int] | None = None,
    processing: dict[str, int] | None = None,
    failed: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "ready": ready or {},
        "processing": processing or {},
        "failed": failed or {},
    }


def test_upload_receipt_without_composer_chip_does_not_confirm_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "history.txt"
    attachment.write_text("body", encoding="utf-8")
    page = _FakePage(states=[_attachment_state()] * 3)
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError):
        asyncio.run(
            attachments.attach_files(
                page, [str(attachment)], timeout_seconds=0.5
            )
        )

    assert page.evaluate_calls


def test_completed_upload_waits_for_delayed_ready_chip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "delayed.txt"
    attachment.write_text("body", encoding="utf-8")
    page = _FakePage(
        states=[
            _attachment_state(),
            _attachment_state(processing={"delayed.txt": 1}),
            _attachment_state(processing={"delayed.txt": 1}),
            _attachment_state(ready={"delayed.txt": 1}),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    asyncio.run(attachments.attach_files(page, [str(attachment)], timeout_seconds=1))

    probes = [
        call for call in page.evaluate_calls
        if call[0] == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
    ]
    assert len(probes) == 4


@pytest.mark.parametrize(
    ("state", "expected_error"),
    [
        ("processing", attachments.AttachmentSettleTimeoutError),
        ("failed", attachments.AttachmentUploadFailedError),
    ],
)
def test_unready_chip_does_not_confirm_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: str,
    expected_error: type[Exception],
) -> None:
    attachment = tmp_path / "stuck.txt"
    attachment.write_text("body", encoding="utf-8")
    page = _FakePage(states=[
        _attachment_state(),
        *[_attachment_state(**{state: {"stuck.txt": 1}})] * 4,
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(expected_error) as raised:
        asyncio.run(
            attachments.attach_files(
                page, [str(attachment)], timeout_seconds=0.5
            )
        )

    if state == "processing":
        assert raised.value.completed_file_create_responses == 1
        assert raised.value.ready_attachments == 0
    else:
        assert "stuck.txt" in str(raised.value)


def test_duplicate_filenames_require_distinct_ready_chips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first" / "same.txt"
    second = tmp_path / "second" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    page = _FakePage(states=[
        _attachment_state(),
        _attachment_state(ready={"same.txt": 1}),
        _attachment_state(ready={"same.txt": 2}),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    asyncio.run(attachments.attach_files(page, [str(first), str(second)]))

    assert len([
        call for call in page.evaluate_calls
        if call[0] == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
    ]) == 3


def test_existing_composer_chip_is_not_new_upload_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "existing.txt"
    attachment.write_text("new body", encoding="utf-8")
    page = _FakePage(states=[
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 2}),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    asyncio.run(attachments.attach_files(page, [str(attachment)]))

    assert len([
        call for call in page.evaluate_calls
        if call[0] == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
    ]) == 3


def test_ready_chip_without_upload_receipt_is_not_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "orphan.txt"
    attachment.write_text("body", encoding="utf-8")
    page = _FakePage(states=[
        _attachment_state(),
        *[_attachment_state(ready={"orphan.txt": 1})] * 3,
    ], responses=[])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(
            attachments.attach_files(
                page, [str(attachment)], timeout_seconds=0.5
            )
        )

    assert raised.value.completed_file_create_responses == 0
    assert raised.value.ready_attachments == 1


def test_upload_receipts_only_count_trusted_file_posts() -> None:
    class Response:
        request = _FakeRequest()
        status = 201

        def __init__(self, url: str) -> None:
            self.url = url

    assert attachments._is_completed_file_create_response(
        Response("https://chatgpt.com/backend-api/files")
    )
    assert not attachments._is_completed_file_create_response(
        Response("https://example.com/backend-api/files")
    )
    assert not attachments._is_completed_file_create_response(
        Response("https://chatgpt.com.evil.test/backend-api/files")
    )


def _probe_attachment_markup(markup: str, filenames: list[str]) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        async with async_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                cache = Path.home() / "Library/Caches/ms-playwright"
                candidates = sorted(cache.glob(
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/"
                    "Contents/MacOS/Google Chrome for Testing"
                ))
                if not candidates:
                    pytest.skip("No local Chromium executable available")
                executable = candidates[-1]
            browser = await playwright.chromium.launch(
                executable_path=str(executable), headless=True
            )
            try:
                page = await browser.new_page()
                await page.set_content(markup)
                return await page.evaluate(
                    attachments.READ_COMPOSER_ATTACHMENT_STATE_JS, filenames
                )
            finally:
                await browser.close()

    return asyncio.run(run())


def test_nested_processing_indicator_blocks_filename_readiness() -> None:
    state = _probe_attachment_markup(
        '<form data-chatgpt-composer>'
        '<div class="chip" aria-busy="true"><div><span>nested.txt</span></div>'
        '<div role="progressbar"></div></div></form>',
        ["nested.txt"],
    )

    assert state == _attachment_state(processing={"nested.txt": 1})


def test_outer_chip_processing_overrides_nested_action() -> None:
    state = _probe_attachment_markup(
        '<form data-chatgpt-composer>'
        '<div aria-busy="true"><div><span>nested-action.txt</span>'
        '<button type="button" aria-label="Remove attachment"></button></div>'
        '<div role="progressbar"></div></div></form>',
        ["nested-action.txt"],
    )

    assert state == _attachment_state(processing={"nested-action.txt": 1})


def test_same_chip_filename_label_and_action_count_once() -> None:
    state = _probe_attachment_markup(
        '<form data-chatgpt-composer><div class="chip">'
        '<span title="same.txt">same.txt</span>'
        '<button type="button" aria-label="same.txt"></button>'
        '</div></form>',
        ["same.txt"],
    )

    assert state == _attachment_state(ready={"same.txt": 1})


def test_unrelated_composer_text_or_neighboring_busy_chip_is_not_ready() -> None:
    state = _probe_attachment_markup(
        '<form data-chatgpt-composer>'
        '<p>arbitrary.txt</p>'
        '<div><div><span>neighbor.txt</span></div>'
        '<button type="button" aria-label="Remove attachment"></button></div>'
        '<div aria-busy="true"><span>other.txt</span>'
        '<div role="progressbar"></div></div>'
        '</form>',
        ["arbitrary.txt", "neighbor.txt"],
    )

    assert state == _attachment_state(ready={"neighbor.txt": 1})


def test_preexisting_processing_chip_becoming_ready_is_not_new_attachment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "existing.txt"
    attachment.write_text("new body", encoding="utf-8")
    page = _FakePage(states=[
        _attachment_state(processing={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 2}),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    asyncio.run(attachments.attach_files(page, [str(attachment)]))

    assert len([
        call for call in page.evaluate_calls
        if call[0] == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
    ]) == 3


def test_attachment_probe_runs_in_local_synthetic_dom() -> None:
    state = _probe_attachment_markup(
        """
        <main>
          <div data-chatgpt-search-unit-key="old:user">history.txt</div>
          <form data-chatgpt-composer>
            <div contenteditable="true">draft.txt</div>
            <div class="chip"><span>split</span><span>.txt</span><button type="button" aria-label="Remove attachment"></button></div>
            <div class="chip" title="attribute.txt"></div>
            <div class="chip" data-filename="data.txt"></div>
            <div class="chip" title="title-pending.txt"><span>Processing</span></div>
            <div class="chip"><span>same.txt</span><button type="button" aria-label="Remove attachment"></button></div>
            <div class="chip"><span>same.txt</span><button type="button" aria-label="Remove attachment"></button></div>
            <div class="chip" aria-busy="true"><span>busy.txt</span></div>
            <div class="chip"><span>progress.txt</span><div role="progressbar"></div></div>
            <div class="chip" data-state="failed"><span>failed.txt</span></div>
            <div class="chip"><span>uploading.txt</span><span>Uploading</span></div>
            <div class="chip"><span>error.txt</span><span>Upload failed</span></div>
            <div class="chip"><span>history.txt.bak</span></div>
          </form>
        </main>
        """,
        [
            "history.txt", "draft.txt", "split.txt", "attribute.txt",
            "data.txt", "same.txt", "busy.txt", "progress.txt", "failed.txt",
            "uploading.txt", "error.txt", "title-pending.txt",
        ],
    )

    assert state == _attachment_state(
        ready={
            "split.txt": 1, "attribute.txt": 1,
            "data.txt": 1, "same.txt": 2,
        },
        processing={
            "busy.txt": 1, "progress.txt": 1,
            "uploading.txt": 1, "title-pending.txt": 1,
        },
        failed={"failed.txt": 1, "error.txt": 1},
    )
