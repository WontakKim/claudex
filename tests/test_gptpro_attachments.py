"""Tests for ChatGPT Pro plain-text attachment uploads."""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from playwright._impl._network import Request as ImplementationRequest
from playwright._impl._network import Response as ImplementationResponse
from playwright.async_api import Error, Response, async_playwright

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


class _FakeBindingTracker(_FakeHandle):
    def __init__(self, states: list[dict[str, Any]], baseline_valid: bool) -> None:
        super().__init__("binding-tracker")
        self.states = iter(states)
        self.baseline_valid = baseline_valid
        self.last_state = {"valid": True, "conflict": False, "bindings": [], "failed_aliases": []}

    async def evaluate(self, script: str, argument: Any = None) -> Any:
        if script == "(tracker) => tracker.baseline_valid":
            return self.baseline_valid
        if script == attachments.READ_ATTACHMENT_BINDINGS_JS:
            self.last_state = next(self.states, self.last_state)
            return self.last_state
        raise AssertionError("unexpected binding tracker evaluation")


class _FakeDropPreflight(_FakeHandle):
    def __init__(self, page: _FakePage, expected: list[dict[str, Any]], data_transfer: Any) -> None:
        super().__init__("drop-preflight")
        self.page = page
        self.expected = expected
        self.data_transfer = data_transfer

    async def evaluate(self, script: str, argument: Any = None) -> Any:
        if script == "(preflight) => preflight.read()":
            return {"state": "eligible", "reason": "observed-capture-path", "receiver_relationship": "ancestor",
                    "file_count": len(self.expected), "item_count": len(self.expected),
                    "file_sizes": [file["size"] for file in self.expected]}
        if script == "(preflight) => preflight.dispatch()":
            await self.page.evaluate(attachments.DISPATCH_ATTACHMENT_DROP_JS, self.data_transfer)
            return {"target": "intended-form", "events": []}
        raise AssertionError("unexpected preflight evaluation")


class _FakeComposerFormHandle(_FakeHandle):
    def __init__(
        self, page: _FakePage, *, invalidate_after_state: bool = False,
        composer_selector: str = attachments.COMPOSER_SELECTOR,
    ) -> None:
        super().__init__("composer-form")
        self.page = page
        self.invalidate_after_state = invalidate_after_state
        self.composer_selector = composer_selector
        self.is_current = True
        self.calls: list[tuple[str, Any]] = []

    def as_element(self) -> _FakeComposerFormHandle:
        return self

    async def evaluate(self, script: str, argument: Any = None) -> Any:
        self.calls.append((script, argument))
        if script == attachments.COMPOSER_FORM_CURRENT_JS:
            return self.is_current
        if script == attachments.READ_EXISTING_ATTACHMENT_NAMES_FOR_FORM_JS:
            return self.page.existing_names
        if script == attachments.READ_COMPOSER_ATTACHMENT_STATE_FOR_FORM_JS:
            state = await self.page.evaluate(attachments.READ_COMPOSER_ATTACHMENT_STATE_JS, argument)
            if self.invalidate_after_state:
                self.is_current = False
            return state
        raise AssertionError("unexpected composer form evaluation")


class _FakePage:
    def __init__(
        self,
        *,
        states: list[dict[str, Any]] | None = None,
        responses: list[Any] | None = None,
        existing_names: list[str] | None = None,
        bindings: list[dict[str, Any]] | None = None,
        baseline_valid: bool = True,
    ) -> None:
        self.binding_states = bindings or []
        self.baseline_valid = baseline_valid
        self.states = iter(states or [])
        self.responses = responses
        self.existing_names = existing_names or []
        self.listeners: dict[str, list[Callable[[Any], None]]] = {
            "response": []
        }
        self.evaluate_handle_calls: list[tuple[str, Any]] = []
        self.evaluate_calls: list[tuple[str, Any]] = []
        self.handles: list[_FakeHandle] = []

    def on(self, event: str, listener: Callable[[Any], None]) -> None:
        self.listeners.setdefault(event, []).append(listener)

    def remove_listener(
        self, event: str, listener: Callable[[Any], None]
    ) -> None:
        self.listeners[event].remove(listener)
        if event != "response" and not self.listeners[event]:
            del self.listeners[event]

    async def evaluate_handle(self, script: str, argument: Any) -> _FakeHandle:
        self.evaluate_handle_calls.append((script, argument))
        if script == attachments.CAPTURE_COMPOSER_FORM_HANDLE_JS:
            handle = _FakeComposerFormHandle(self, composer_selector=argument)
        elif script == attachments.CREATE_ATTACHMENT_BINDING_TRACKER_JS:
            handle = _FakeBindingTracker(self.binding_states, self.baseline_valid)
        elif script == attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS:
            handle = _FakeDropPreflight(self, argument["expected"], argument["dataTransfer"])
        else:
            handle = _FakeHandle(f"handle-{len(self.handles)}")
        self.handles.append(handle)
        return handle

    async def evaluate(self, script: str, argument: Any = None) -> Any:
        self.evaluate_calls.append((script, argument))
        if script == attachments.DISPATCH_ATTACHMENT_DROP_JS:
            file_count = sum(script == attachments.CREATE_ATTACHMENT_FILE_JS for script, _ in self.evaluate_handle_calls)
            responses = self.responses
            if responses is None:
                responses = [_FakeResponse() for _ in range(file_count)]
            for response in responses:
                for listener in tuple(self.listeners["response"]):
                    listener(response)
            return "form[data-chatgpt-composer]"
        if script == attachments.READ_EXISTING_ATTACHMENT_NAMES_JS:
            return self.existing_names
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


class _BoundedAttachmentFile(BytesIO):
    def __init__(self, data: bytes, max_read_bytes: int) -> None:
        super().__init__(data)
        self.max_read_bytes = max_read_bytes
        self.read_sizes: list[int] = []
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        assert 0 < size <= self.max_read_bytes, f"Unbounded attachment read({size})"
        data = super().read(size)
        self.bytes_read += len(data)
        return data


def test_oversized_attachment_read_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attachments, "MAX_TOTAL_ATTACHMENT_BYTES", 8)
    attachment_file = _BoundedAttachmentFile(b"x" * 40, max_read_bytes=9)
    monkeypatch.setattr(Path, "open", lambda self, mode: attachment_file)

    with pytest.raises(ValueError, match="at least 9 bytes.*8-byte limit"):
        attachments._load_descriptors(["large.txt"])

    assert attachment_file.read_sizes == [9]
    assert attachment_file.bytes_read == 9
    assert attachment_file.closed


@pytest.mark.parametrize(("first_size", "remaining_read_bytes"), [(7, 2), (8, 1)])
def test_aggregate_attachment_reads_use_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
    first_size: int,
    remaining_read_bytes: int,
) -> None:
    monkeypatch.setattr(attachments, "MAX_TOTAL_ATTACHMENT_BYTES", 8)
    files = {
        "first.txt": _BoundedAttachmentFile(b"x" * first_size, max_read_bytes=9),
        "large.txt": _BoundedAttachmentFile(b"y" * 40, max_read_bytes=remaining_read_bytes),
        "unread.txt": _BoundedAttachmentFile(b"z", max_read_bytes=1),
    }
    monkeypatch.setattr(Path, "open", lambda self, mode: files[str(self)])

    with pytest.raises(ValueError, match="at least 9 bytes.*8-byte limit"):
        attachments._load_descriptors(list(files))

    assert files["first.txt"].read_sizes == [9]
    assert files["large.txt"].read_sizes == [remaining_read_bytes]
    assert files["unread.txt"].read_sizes == []
    assert sum(attachment_file.bytes_read for attachment_file in files.values()) == 9
    assert files["first.txt"].closed
    assert files["large.txt"].closed


@pytest.mark.parametrize("contents", [
    [b"abcdefg"],
    [b"abcdefgh"],
    [b"abcd", b"efgh"],
    [b"", b"abcd", b"", b"efgh", b""],
    [b""] * attachments.MAX_ATTACHMENTS_PER_ASK,
    [b"\xc3\xa9", b"\xe2\x82\xacabc"],
])
def test_attachments_within_byte_budget_preserve_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: list[bytes],
) -> None:
    monkeypatch.setattr(attachments, "MAX_TOTAL_ATTACHMENT_BYTES", 8)
    paths = [tmp_path / f"file-{index}.txt" for index in range(len(contents))]
    for path, data in zip(paths, contents):
        path.write_bytes(data)

    descriptors = attachments._load_descriptors([str(path) for path in paths])

    assert [base64.b64decode(item["bytesBase64"]) for item in descriptors] == contents
    assert [item["name"] for item in descriptors] == [path.name for path in paths]
    assert all(item["mime"] == "text/plain" for item in descriptors)


def test_total_attachment_size_is_limited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(attachments, "MAX_TOTAL_ATTACHMENT_BYTES", 8)
    attachment = tmp_path / "large.txt"
    attachment.write_bytes(b"x" * 9)

    with pytest.raises(ValueError, match="at least 9 bytes.*8-byte limit"):
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


def test_plain_text_attachments_preserve_original_filenames(
    tmp_path: Path,
) -> None:
    names = [
        "effect.ts", "effect.ts.txt", "effect.ts", "data.json",
        "caps.TXT", "name with [brackets].md",
    ]
    paths = []
    for index, name in enumerate(names):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / name
        path.write_text(f"synthetic-{index}", encoding="utf-8")
        paths.append(str(path))

    descriptors = attachments._load_descriptors(paths)

    assert [descriptor["name"] for descriptor in descriptors] == names
    assert [base64.b64decode(item["bytesBase64"]) for item in descriptors] == [
        f"synthetic-{index}".encode() for index in range(len(names))
    ]
    assert all(item["mime"] == "text/plain" for item in descriptors)


def test_attach_files_dispatches_drop_and_waits_for_settlement(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first body", encoding="utf-8")
    second.write_text("second body", encoding="utf-8")
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "first-card", "first.txt", "ready", True), ("upload-2", "second-card", "second.txt", "ready", True))],
        states=[
            {"ready": {}, "processing": {}, "failed": {}},
            {
                "ready": {"first.txt": 1, "second.txt": 1},
                "processing": {},
                "failed": {},
            },
        ],
        responses=[
            _NamedCreateResponse("first.txt", "file_first"),
            _NamedCreateResponse("second.txt", "file_second"),
        ],
    )

    asyncio.run(
        attachments.attach_files(page, [str(first), str(second)])
    )

    assert [call[0] for call in page.evaluate_handle_calls] == [
        attachments.CAPTURE_COMPOSER_FORM_HANDLE_JS,
        attachments.CREATE_ATTACHMENT_FILE_JS,
        attachments.CREATE_ATTACHMENT_FILE_JS,
        attachments.CREATE_ATTACHMENT_DATA_TRANSFER_JS,
        attachments.CREATE_ATTACHMENT_BINDING_TRACKER_JS,
        attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS,
    ]
    first_descriptor = page.evaluate_handle_calls[1][1]
    second_descriptor = page.evaluate_handle_calls[2][1]
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
    assert page.evaluate_handle_calls[3][1] == page.handles[1:3]
    assert (
        attachments.DISPATCH_ATTACHMENT_DROP_JS, page.handles[3]
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


def test_form_replacement_after_baseline_blocks_attachment_drop(
    tmp_path: Path,
) -> None:
    attachment = tmp_path / "bound.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    page = _FakePage(states=[_attachment_state()])
    composer_selector = "form[data-chatgpt-composer] div.ProseMirror"
    form_handle = _FakeComposerFormHandle(
        page, invalidate_after_state=True, composer_selector=composer_selector,
    )

    with pytest.raises(RuntimeError, match="intended composer form changed"):
        asyncio.run(attachments.attach_files(
            page,
            [str(attachment)],
            composer_form_handle=form_handle,
            composer_selector=composer_selector,
        ))

    assert any(script == attachments.READ_EXISTING_ATTACHMENT_NAMES_FOR_FORM_JS
               for script, _ in form_handle.calls)
    assert any(script == attachments.READ_COMPOSER_ATTACHMENT_STATE_FOR_FORM_JS
               for script, _ in form_handle.calls)
    assert not any(script == attachments.DISPATCH_ATTACHMENT_DROP_JS
                   for script, _ in page.evaluate_calls)
    assert page.listeners == {"response": []}


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
    assert "conversation not established" not in message
    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)


def _attachment_state(
    *,
    ready: dict[str, int] | None = None,
    processing: dict[str, int] | None = None,
    failed: dict[str, int] | None = None,
    unknown: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "ready": ready or {},
        "processing": processing or {},
        "failed": failed or {},
        "unknown": unknown or {},
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
        bindings=[_bound_cards(("upload-1", "delayed-card", "delayed.txt", "processing", True)), _bound_cards(("upload-1", "delayed-card", "delayed.txt", "processing", True)), _bound_cards(("upload-1", "delayed-card", "delayed.txt", "ready", True))],
        states=[
            _attachment_state(),
            _attachment_state(processing={"delayed.txt": 1}),
            _attachment_state(processing={"delayed.txt": 1}),
            _attachment_state(ready={"delayed.txt": 1}),
        ],
        responses=[_NamedCreateResponse("delayed.txt", "file_delayed")],
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
        assert "0/1 observed composer-ready attachments" in str(raised.value)
        assert "0 pending processing responses" in str(raised.value)
        assert "processing unavailable (no attributable receipt file ID)" in str(raised.value)


def test_timeout_reports_file_statuses_and_unattributed_receipt_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class IdentifiedResponse(_FakeResponse):
        def __init__(self, file_id: str) -> None:
            self.file_id = file_id

        async def json(self) -> dict[str, str]:
            return {"file_id": self.file_id, "status": "success"}

    names = ["ready.txt", "pending.txt", "missing.txt"]
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text("synthetic", encoding="utf-8")
        paths.append(str(path))
    page = _FakePage(
        states=[
            _attachment_state(),
            *[
                {**_attachment_state(ready={"ready.txt": 1},
                                     processing={"pending.txt": 1}),
                 "unknown": {"missing.txt": 1}}
            ] * 3,
        ],
        responses=[IdentifiedResponse(f"file_{index}") for index in (2, 0, 1)],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(page, paths, timeout_seconds=0.5))

    message = str(raised.value)
    for name, status in (
        ("ready.txt", "ready"),
        ("pending.txt", "processing"),
        ("missing.txt", "unknown"),
    ):
        assert f"{name!r}: {status}" in message
    assert "upload-1" in message and "upload-2" in message and "upload-3" in message
    assert "unattributed" in message
    assert raised.value.completed_file_create_responses == 3
    assert raised.value.ready_attachments == 0
    assert "1/3 observed composer-ready attachments" in message


def test_out_of_order_receipt_ids_match_request_file_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class NamedResponse(_FakeResponse):
        def __init__(self, filename: str, file_id: str) -> None:
            self.request = type("NamedRequest", (), {
                "method": "POST", "post_data_json": {"file_name": filename},
            })()
            self.file_id = file_id

        async def json(self) -> dict[str, str]:
            return {"file_id": self.file_id, "status": "success"}

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(
            ready={"first.txt": 1}, unknown={"second.txt": 1},
        )] * 3],
        responses=[
            NamedResponse("second.txt", "file_second"),
            NamedResponse("first.txt", "file_first"),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(first), str(second)], timeout_seconds=0.5,
        ))

    assert "'first.txt': ready" in str(raised.value)
    assert "'second.txt': unknown" in str(raised.value)
    message = str(raised.value)
    assert "receipt file ID upload-2" in message.split("'second.txt': unknown")[0]
    assert "receipt file ID upload-1" in message.split("'second.txt': unknown")[1]


class _NamedCreateResponse(_FakeResponse):
    def __init__(self, filename: str, file_id: str) -> None:
        self.request = type("NamedRequest", (), {
            "method": "POST", "post_data_json": {"file_name": filename},
        })()
        self.file_id = file_id

    async def json(self) -> dict[str, str]:
        return {"file_id": self.file_id, "status": "success"}


class _ProcessingResponse(_FakeResponse):
    url = "https://chatgpt.com/backend-api/files/process_upload_stream"
    status = 200

    def __init__(self, frames: list[dict[str, Any]], *, url: str | None = None) -> None:
        self.request = type("ProcessingRequest", (), {"method": "POST"})()
        if url is not None:
            self.url = url
        self.frames = frames

    async def text(self) -> str:
        return "\n".join(json.dumps(frame) for frame in self.frames)


def _processing_frames(file_id: str, display_name: str) -> list[dict[str, Any]]:
    return [
        {"file_id": file_id, "event": "file.indexing.completed", "extra": {
            "library_file_name": display_name,
        }},
        {"file_id": file_id, "event": "file.processing.completed"},
    ]


@pytest.mark.parametrize(("frames", "expected_processing"), [
    pytest.param(
        _processing_frames("file_effect", "effect.ts")[:1],
        "observed=yes, completion=no, indexed display name=yes, missing=completion",
        id="missing-completion",
    ),
    pytest.param(
        _processing_frames("file_effect", "effect.ts")[1:],
        "observed=yes, completion=yes, indexed display name=no, missing=indexed display name",
        id="missing-indexed-name",
    ),
])
def test_ready_chip_timeout_explains_incomplete_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    frames: list[dict[str, Any]], expected_processing: str,
) -> None:
    original = tmp_path / "effect.ts"
    original.write_text("synthetic", encoding="utf-8")
    page = _FakePage(states=[
        _attachment_state(),
        *[_attachment_state(ready={"effect.ts": 1})] * 3,
    ], responses=[
        _NamedCreateResponse("effect.ts", "file_effect"),
        _ProcessingResponse(frames),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)
    progress = []

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(original)], timeout_seconds=0.5,
            on_progress=lambda receipts, ready: progress.append((receipts, ready)),
        ))

    assert raised.value.completed_file_create_responses == 1
    assert raised.value.ready_attachments == 0
    assert progress[0] == (0, 0)
    assert progress[-1] == (1, 0)
    assert clock.value == 0.5
    message = str(raised.value)
    assert "0/1 processing-qualified ready attachments" in message
    assert "1/1 observed composer-ready attachments" in message
    assert "0 pending processing responses" in message
    assert "'effect.ts': ready (composer has a file-specific action)" in message
    assert f"processing upload-1: {expected_processing}" in message


def test_deferred_indexing_started_name_does_not_satisfy_processing_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "deferred.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[
            _NamedCreateResponse(attachment.name, "file_deferred"),
            _ProcessingResponse([{
                "file_id": "file_deferred",
                "event": "file.indexing.started",
                "extra": {
                    "library_file_name": attachment.name,
                    "metadata_object_id": "metadata_object_1",
                    "retrieval_indexing_deferred": True,
                },
            }]),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    assert raised.value.ready_attachments == 0
    assert "processing upload-1: observed=yes, completion=no" in str(raised.value)
    assert "no send click attempted by attachment upload" in str(raised.value)


def test_deferred_indexing_name_is_candidate_not_card_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "deferred-name.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    frames = [
        {"file_id": "file_deferred", "event": "file.processing.started"},
        {"file_id": "file_deferred", "event": "file.processing.file_ready"},
        {
            "file_id": "file_deferred",
            "event": "file.indexing.started",
            "extra": {
                "library_file_name": attachment.name,
                "retrieval_indexing_deferred": True,
            },
        },
        {"file_id": "file_deferred", "event": "file.processing.completed"},
    ]
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[
            _NamedCreateResponse(attachment.name, "file_deferred"),
            _ProcessingResponse(frames),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    message = str(raised.value)
    assert raised.value.ready_attachments == 0
    assert "name source candidate=deferred-indexing-started" in message
    assert "card binding=unverified" in message
    assert "processing upload-1: observed=yes, completion=yes" in message


def test_invalid_deferred_name_fails_without_becoming_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "deferred-invalid.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    private_name = "../private-name.txt"
    frames = [
        {
            "file_id": "file_deferred_invalid",
            "event": "file.indexing.started",
            "extra": {
                "library_file_name": private_name,
                "retrieval_indexing_deferred": True,
            },
        },
        {"file_id": "file_deferred_invalid", "event": "file.processing.completed"},
    ]
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[
            _NamedCreateResponse(attachment.name, "file_deferred_invalid"),
            _ProcessingResponse(frames),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentUploadFailedError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    message = str(raised.value)
    assert "name source candidate=deferred-indexing-started-invalid" in message
    assert private_name not in message


def test_conflicting_deferred_names_fail_for_the_correlated_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "deferred-conflict.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    conflicting_name = "alternate-name.txt"
    file_id = "file_deferred_conflict"
    frames = [
        {
            "file_id": file_id,
            "event": "file.indexing.started",
            "extra": {
                "library_file_name": attachment.name,
                "retrieval_indexing_deferred": True,
            },
        },
        {
            "file_id": file_id,
            "event": "file.indexing.started",
            "extra": {
                "library_file_name": conflicting_name,
                "retrieval_indexing_deferred": True,
            },
        },
        {"file_id": file_id, "event": "file.processing.completed"},
    ]
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[
            _NamedCreateResponse(attachment.name, file_id),
            _ProcessingResponse(frames),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentUploadFailedError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    message = str(raised.value)
    assert "name source candidate=deferred-indexing-started-conflicting" in message
    assert conflicting_name not in message


def test_timeout_distinguishes_unobserved_and_incomplete_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = []
    for name in ("unobserved.ts", "incomplete.ts"):
        path = tmp_path / name
        path.write_text("synthetic", encoding="utf-8")
        paths.append(str(path))
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "incomplete-card", "incomplete.ts", "ready", True), ("upload-2", "unobserved-card", "unobserved.ts", "ready", True))],
        states=[
        _attachment_state(),
        *[_attachment_state(ready={"unobserved.ts": 1, "incomplete.ts": 1})] * 3,
    ], responses=[
        _NamedCreateResponse("incomplete.ts", "file_incomplete"),
        _ProcessingResponse([{
            "file_id": "file_incomplete", "event": "file.processing.started",
        }]),
        _NamedCreateResponse("unobserved.ts", "file_unobserved"),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(page, paths, timeout_seconds=0.5))

    assert raised.value.completed_file_create_responses == 2
    assert raised.value.ready_attachments == 1
    message = str(raised.value)
    assert "1/2 processing-qualified ready attachments" in message
    assert "2/2 observed composer-ready attachments" in message
    assert (
        "processing upload-2: observed=no, completion=no, "
        "indexed display name=no, missing=not required without observation"
    ) in message
    assert (
        "processing upload-1: observed=yes, completion=no, "
        "indexed display name=no, missing=completion, indexed display name"
    ) in message


def test_ready_chip_timeout_reports_pending_processing_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class PendingProcessingResponse(_ProcessingResponse):
        async def finished(self) -> None:
            await asyncio.Event().wait()

    original = tmp_path / "effect.ts"
    original.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "pending-card", "effect.ts", "ready", True))],
        states=[
        _attachment_state(),
        *[_attachment_state(ready={"effect.ts": 1})] * 3,
    ], responses=[
        _NamedCreateResponse("effect.ts", "file_effect"),
        PendingProcessingResponse([]),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)
    progress = []

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(original)], timeout_seconds=0.5,
            on_progress=lambda receipts, ready: progress.append((receipts, ready)),
        ))

    assert raised.value.completed_file_create_responses == 1
    assert raised.value.ready_attachments == 1
    assert progress[-1] == (1, 1)
    message = str(raised.value)
    assert "1/1 processing-qualified ready attachments" in message
    assert "1/1 observed composer-ready attachments" in message
    assert "1 pending processing responses" in message
    assert "processing upload-1: observed=no" in message
    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)


def test_server_renamed_attachment_settles_under_authoritative_display_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original = tmp_path / "effect.ts"
    original.write_text("synthetic", encoding="utf-8")
    renamed = "effect(20260927-120843).ts"
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "renamed-card", renamed, "ready", True))],
        states=[_attachment_state(), _attachment_state(processing={"effect.ts": 1}),
                *[_attachment_state(ready={renamed: 1})] * 3],
        responses=[
            _ProcessingResponse(_processing_frames("file_effect", renamed)),
            _NamedCreateResponse("effect.ts", "file_effect"),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    progress = []
    counts = asyncio.run(attachments.attach_files(
        page, [str(original)], timeout_seconds=0.5,
        on_progress=lambda receipts, ready: progress.append((receipts, ready)),
    ))

    assert counts == {renamed: (0, 1)}
    assert progress[0] == (0, 0)
    assert progress[-1] == (1, 1)
    assert original.name == "effect.ts"
    probes = [argument for script, argument in page.evaluate_calls
              if script == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS]
    assert [renamed] in probes


@pytest.mark.parametrize("frames", [
    _processing_frames("file_unrelated", "effect(20260927-120843).ts"),
    _processing_frames("file_effect", "effect(20260927-120843).ts")[:1],
    [{"file_id": "file_effect", "event": "file.processing.completed"}],
])
def test_unattributed_or_incomplete_rename_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: list[dict[str, Any]],
) -> None:
    original = tmp_path / "effect.ts"
    original.write_text("synthetic", encoding="utf-8")
    renamed = "effect(20260927-120843).ts"
    page = _FakePage(states=[
        _attachment_state(), *[_attachment_state(ready={renamed: 1})] * 3,
    ], responses=[
        _NamedCreateResponse("effect.ts", "file_effect"),
        _ProcessingResponse(frames),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(original)], timeout_seconds=0.5,
        ))

    assert raised.value.ready_attachments == 0
    assert "upload-1" in str(raised.value)


def test_preexisting_renamed_chip_is_not_new_upload_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original = tmp_path / "effect.ts"
    original.write_text("synthetic", encoding="utf-8")
    renamed = "effect(20260927-120843).ts"
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "prior-card", renamed, "ready", False))],
        states=[
        _attachment_state(ready={renamed: 1}),
        *[_attachment_state(ready={renamed: 1})] * 3,
    ], existing_names=[renamed], responses=[
        _NamedCreateResponse("effect.ts", "file_effect"),
        _ProcessingResponse(_processing_frames("file_effect", renamed)),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(original)], timeout_seconds=0.5,
        ))

    assert raised.value.ready_attachments == 0
    assert "0/1 observed composer-ready attachments" in str(raised.value)
    assert (
        "processing upload-1: observed=yes, completion=yes, "
        "indexed display name=yes, missing=none"
    ) in str(raised.value)


def test_processing_failure_blocks_even_ready_original_chip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original = tmp_path / "effect.ts"
    original.write_text("private attachment contents", encoding="utf-8")
    receipt = _NamedCreateResponse("effect.ts", "file_effect")
    receipt.request.post_data_json["credential"] = "synthetic request credential"
    page = _FakePage(states=[
        _attachment_state(), *[_attachment_state(ready={"effect.ts": 1})] * 3,
    ], responses=[
        receipt,
        _ProcessingResponse([{
            "file_id": "file_effect", "event": "file.processing.failed",
            "message": "do not leak raw server text",
            "extra": {"detail": "private event detail"},
        }]),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentUploadFailedError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(original)], timeout_seconds=0.5,
        ))

    assert "do not leak raw server text" not in str(raised.value)
    assert "upload-1" in str(raised.value)

    message = str(raised.value)
    for private_value in (
        "private attachment contents", "synthetic request credential",
        "private event detail", "file.processing.failed",
    ):
        assert private_value not in message
    assert (
        "processing upload-1: observed=yes, completion=no, "
        "indexed display name=no, missing=completion, indexed display name"
    ) in message
    assert "1/1 observed composer-ready attachments" in message
    assert "0 pending processing responses" in message


def test_duplicate_file_id_cannot_count_as_two_requested_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = []
    for name in ("first.ts", "second.ts"):
        file = tmp_path / name
        file.write_text("synthetic", encoding="utf-8")
        paths.append(str(file))
    page = _FakePage(states=[
        _attachment_state(),
        *[_attachment_state(ready={"first.ts": 1, "second.ts": 1})] * 3,
    ], responses=[
        _NamedCreateResponse("first.ts", "file_reused"),
        _NamedCreateResponse("second.ts", "file_reused"),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(
        attachments.AttachmentUploadFailedError,
        match="duplicate receipt file ID upload-1",
    ):
        asyncio.run(attachments.attach_files(page, paths, timeout_seconds=0.5))


def test_stream_identity_change_during_composer_probe_requires_new_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    file = tmp_path / "effect.ts"
    file.write_text("synthetic", encoding="utf-8")
    renamed = "effect(20260927-120843).ts"
    stream = _ProcessingResponse(_processing_frames("file_effect", renamed))

    class ChangingPage(_FakePage):
        changed = False

        async def evaluate(self, script: str, argument: Any = None) -> Any:
            if (script == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
                    and argument == ["effect.ts"] and not self.changed
                    and any(script == attachments.CREATE_ATTACHMENT_FILE_JS for script, _ in self.evaluate_handle_calls)):
                self.changed = True
                for listener in tuple(self.listeners["response"]):
                    listener(stream)
                await asyncio.sleep(0)
                return _attachment_state(ready={"effect.ts": 1})
            return await super().evaluate(script, argument)

    page = ChangingPage(
        bindings=[_bound_cards(("upload-1", "renamed-card", renamed, "ready", True))],
        states=[
        _attachment_state(),
        *[_attachment_state(ready={renamed: 1})] * 3,
    ], responses=[_NamedCreateResponse("effect.ts", "file_effect")])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    counts = asyncio.run(attachments.attach_files(page, [str(file)], timeout_seconds=1))

    assert counts == {renamed: (0, 1)}


def test_seven_renamed_files_correlate_out_of_order_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original_names = [f"synthetic-{index}.ts" for index in range(7)]
    renamed = [f"synthetic-{index}(20260927-120843).ts" for index in range(7)]
    paths = []
    for name in original_names:
        path = tmp_path / name
        path.write_text("synthetic", encoding="utf-8")
        paths.append(str(path))
    page = _FakePage(
        bindings=[_bound_cards(*[(f"upload-{index + 1}", f"card-{index}", renamed[index], "ready", True) for index in range(7)])],
        states=[
        _attachment_state(),
        *[_attachment_state(ready={name: 1 for name in renamed})] * 3,
    ], responses=[
        *[_ProcessingResponse(_processing_frames(f"file_{index}", renamed[index]))
          for index in range(6, -1, -1)],
        *[_NamedCreateResponse(original_names[index], f"file_{index}")
          for index in range(7)],
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    counts = asyncio.run(attachments.attach_files(page, paths, timeout_seconds=1))

    assert counts == {name: (0, 1) for name in renamed}


def test_receipt_json_failure_reports_optional_metadata_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class InvalidJsonResponse(_FakeResponse):
        async def json(self) -> dict[str, str]:
            raise ValueError("do not expose response body")

    file = tmp_path / "pending.txt"
    file.write_text("synthetic", encoding="utf-8")
    page = _FakePage(responses=[InvalidJsonResponse()])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(file)], timeout_seconds=0.5,
        ))

    assert "receipt metadata unavailable (ValueError)" in str(raised.value)
    assert raised.value.completed_file_create_responses == 1


def test_duplicate_filenames_require_distinct_ready_chips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first" / "same.txt"
    second = tmp_path / "second" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "first-card", "same.txt", "ready", True)), _bound_cards(("upload-1", "first-card", "same.txt", "ready", True), ("upload-2", "second-card", "same.txt", "ready", True))],
        states=[
        _attachment_state(),
        _attachment_state(ready={"same.txt": 1}),
        _attachment_state(ready={"same.txt": 2}),
    ], responses=[
        _NamedCreateResponse("same.txt", "file_same_first"),
        _NamedCreateResponse("same.txt", "file_same_second"),
    ])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    asyncio.run(attachments.attach_files(page, [str(first), str(second)]))

    assert len([
        call for call in page.evaluate_calls
        if call[0] == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
    ]) == 3


def test_duplicate_filename_partial_readiness_is_not_claimed_per_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        file = directory / "same.txt"
        file.write_text(f"synthetic-{index}", encoding="utf-8")
        paths.append(str(file))
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(ready={"same.txt": 1})] * 3],
        responses=[_NamedCreateResponse("same.txt", "file_first"), _NamedCreateResponse("same.txt", "file_second")],
        bindings=[_bound_cards(("upload-1", "first-card", "same.txt", "ready", True))],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(page, paths, timeout_seconds=0.5))

    assert "'same.txt': ambiguous (1/2 ready)" in str(raised.value)
    assert raised.value.ready_attachments == 1
    assert "1/2 observed composer-ready attachments" in str(raised.value)


def test_settled_upload_returns_total_ready_counts_for_pre_send_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "existing.txt"
    attachment.write_text("new body", encoding="utf-8")
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "new-card", "existing.txt", "ready", True))],
        states=[
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 2}),
    ], responses=[_NamedCreateResponse("existing.txt", "file_existing")])

    expected_ready_counts = asyncio.run(
        attachments.attach_files(page, [str(attachment)])
    )

    assert expected_ready_counts == {"existing.txt": (1, 1)}


def test_existing_composer_chip_is_not_new_upload_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "existing.txt"
    attachment.write_text("new body", encoding="utf-8")
    page = _FakePage(
        bindings=[_bound_cards(), _bound_cards(("upload-1", "new-card", "existing.txt", "ready", True))],
        states=[
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 2}),
    ], responses=[_NamedCreateResponse("existing.txt", "file_existing")])
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
    assert raised.value.ready_attachments == 0


def test_processing_events_only_count_trusted_exact_endpoint() -> None:
    frame = _processing_frames("file_effect", "effect-renamed.ts")
    assert attachments._is_file_processing_response(_ProcessingResponse(frame))
    assert not attachments._is_file_processing_response(_ProcessingResponse(
        frame, url="https://chatgpt.com.evil.test/backend-api/files/process_upload_stream",
    ))
    assert not attachments._is_file_processing_response(_ProcessingResponse(
        frame, url="https://chatgpt.com/backend-api/files/process_upload_stream_extra",
    ))


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

    assert state == _attachment_state(
        ready={"neighbor.txt": 1}, unknown={"arbitrary.txt": 1},
    )


def test_preexisting_processing_chip_becoming_ready_is_not_new_attachment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attachment = tmp_path / "existing.txt"
    attachment.write_text("new body", encoding="utf-8")
    page = _FakePage(
        bindings=[_bound_cards(), _bound_cards(("upload-1", "new-card", "existing.txt", "ready", True))],
        states=[
        _attachment_state(processing={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 2}),
    ], responses=[_NamedCreateResponse("existing.txt", "file_existing")])
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    asyncio.run(attachments.attach_files(page, [str(attachment)]))

    assert len([
        call for call in page.evaluate_calls
        if call[0] == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS
    ]) == 3


def test_observed_file_without_action_is_unknown_until_ready() -> None:
    filename = "synthetic.txt"
    pending = _probe_attachment_markup(
        '<form data-chatgpt-composer><span><span><svg></svg>'
        '<span>synthetic.txt</span></span><span></span><span></span>'
        '</span></form>',
        [filename],
    )
    ready = _probe_attachment_markup(
        '<form data-chatgpt-composer><span><span><svg></svg>'
        '<span>synthetic.txt</span></span><span></span><span></span>'
        '<button type="button" aria-label="Remove attachment"></button>'
        '</span></form>',
        [filename],
    )

    assert pending == {
        "ready": {}, "processing": {}, "failed": {},
        "unknown": {filename: 1},
    }
    assert ready == {
        "ready": {filename: 1}, "processing": {}, "failed": {},
        "unknown": {},
    }


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


def _create_lifecycle_response(closed: asyncio.Future[Any]) -> Response:
    """Use installed Playwright lifecycle methods without a connection or browser."""
    request = ImplementationRequest.__new__(ImplementationRequest)
    page = SimpleNamespace(_closed_or_crashed_future=closed)
    request._initializer = {
        "frame": SimpleNamespace(_object=SimpleNamespace(_page=page)),
    }
    implementation = ImplementationResponse.__new__(ImplementationResponse)
    implementation._loop = asyncio.get_running_loop()
    implementation._request = request
    implementation._finished_future = implementation._loop.create_future()
    return Response(implementation)


async def _flush_response_completion() -> None:
    for _ in range(8):
        await asyncio.sleep(0)
    gc.collect()
    await asyncio.sleep(0)


async def _assert_response_completion_cleanup(
    responses: list[Response], closed: asyncio.Future[Any],
    initial_tasks: set[asyncio.Task[Any]], warnings: list[str],
) -> None:
    await _flush_response_completion()
    residual_tasks = [
        task.get_coro().__qualname__ for task in asyncio.all_tasks() - initial_tasks
    ]
    callbacks = [len(closed._callbacks or ())] + [
        len(response._impl_obj._finished_future._callbacks or ())
        for response in responses
    ]
    assert not closed.cancelled()
    assert all(not response._impl_obj._finished_future.cancelled() for response in responses)
    if not closed.done():
        closed.set_result(Error("synthetic late close"))
    await _flush_response_completion()
    assert not residual_tasks
    assert callbacks == [0] * (len(responses) + 1)
    assert asyncio.all_tasks() == initial_tasks
    assert warnings == []


@pytest.mark.parametrize("endpoint", ["create", "processing"])
@pytest.mark.parametrize("mode", [
    "prefinished", "success", "cancel", "timeout", "close", "completion_error",
])
def test_response_completion_attachment_call_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, mode: str,
) -> None:
    file = tmp_path / "upload.txt"
    file.write_text("attachment", encoding="utf-8")

    async def poll_sleep(seconds: float) -> None:
        await asyncio.sleep(min(seconds, 0.001))

    monkeypatch.setattr(attachments, "_sleep", poll_sleep)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        warnings: list[str] = []
        loop.set_exception_handler(lambda loop, context: warnings.append(context["message"]))
        initial_tasks = asyncio.all_tasks()
        closed = loop.create_future()
        response = _create_lifecycle_response(closed)
        if mode == "prefinished":
            response._impl_obj._finished_future.set_result(None)
        create = _NamedCreateResponse(file.name, "file_upload")
        processing = _ProcessingResponse(_processing_frames("file_upload", file.name))
        observed = create if endpoint == "create" else processing
        observed._impl_obj = response._impl_obj
        observed.finished = response.finished
        page = _FakePage(
            bindings=[_bound_cards(("upload-1", "upload-card", file.name, "ready", True))],
            responses=[create, processing],
            states=[_attachment_state()] + [_attachment_state(ready={file.name: 1})] * 20,
        )
        progress: list[tuple[int, int]] = []
        task = asyncio.create_task(attachments.attach_files(
            page, [str(file)], timeout_seconds=0.5,
            on_progress=lambda receipts, ready: progress.append((receipts, ready)),
        ))
        await _flush_response_completion()
        if mode == "success":
            response._impl_obj._finished_future.set_result(None)
        elif mode == "cancel":
            task.cancel()
        elif mode == "close":
            closed.set_result(Error("synthetic close"))
        elif mode == "completion_error":
            response._impl_obj._finished_future.set_exception(ValueError("completion failed"))
        if mode in ("prefinished", "success"):
            assert await task == {file.name: (0, 1)}
            assert progress[-1] == (1, 1)
        elif mode == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        elif endpoint == "processing" and mode in ("close", "completion_error"):
            with pytest.raises(attachments.AttachmentUploadFailedError, match="processing observation failed"):
                await task
            assert progress[-1][0] == 1
        else:
            with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
                await task
            assert raised.value.completed_file_create_responses == (0 if endpoint == "create" else 1)
        del task
        assert page.listeners["response"] == []
        assert all(handle.dispose_calls == 1 for handle in page.handles)
        await _assert_response_completion_cleanup([response], closed, initial_tasks, warnings)

    asyncio.run(run())


@pytest.mark.parametrize("mode", [
    "success", "prefinished", "cancel", "timeout", "close", "already_closed",
    "simultaneous", "shared_success", "shared_close", "shared_cancel", "completion_error", "close_error",
])
def test_response_completion_lifecycle(mode: str) -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        warnings: list[str] = []
        loop.set_exception_handler(lambda loop, context: warnings.append(context["message"]))
        initial_tasks = asyncio.all_tasks()
        closed = loop.create_future()
        responses = [_create_lifecycle_response(closed) for _ in range(3 if mode.startswith("shared") else 1)]
        if mode in ("prefinished", "simultaneous"):
            responses[0]._impl_obj._finished_future.set_result(None)
        if mode in ("already_closed", "simultaneous"):
            closed.set_result(Error("synthetic close"))
        tasks = [asyncio.create_task(attachments._wait_for_response_completion(response))
                 for response in responses]
        await _flush_response_completion()
        if mode in ("success", "shared_success", "shared_cancel"):
            if mode == "shared_cancel":
                tasks[0].cancel()
            for response in responses:
                response._impl_obj._finished_future.set_result(None)
        elif mode == "cancel":
            tasks[0].cancel()
        elif mode == "timeout":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(tasks[0], timeout=0.001)
        elif mode in ("close", "shared_close"):
            closed.set_result(Error("synthetic close"))
        elif mode == "completion_error":
            responses[0]._impl_obj._finished_future.set_exception(ValueError("completion failed"))
        elif mode == "close_error":
            closed.set_exception(ValueError("close failed"))
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        if mode in ("close", "already_closed", "simultaneous", "shared_close"):
            assert all(isinstance(outcome, Error) and str(outcome) == "Target closed"
                       for outcome in outcomes)
        elif mode in ("cancel", "timeout"):
            assert isinstance(outcomes[0], asyncio.CancelledError)
        elif mode == "shared_cancel":
            assert isinstance(outcomes[0], asyncio.CancelledError)
            assert outcomes[1:] == [None, None]
        elif mode.endswith("error"):
            assert isinstance(outcomes[0], ValueError)
            assert str(outcomes[0]) == ("completion failed" if mode == "completion_error" else "close failed")
        else:
            assert outcomes == [None] * len(responses)
        del outcomes, tasks
        if mode in ("success", "prefinished", "shared_success"):
            for response in responses:
                await attachments._wait_for_response_completion(response)
        await _assert_response_completion_cleanup(responses, closed, initial_tasks, warnings)
        if mode in ("success", "prefinished", "shared_success"):
            # Reusing completed responses after close must still observe target closure.
            for response in responses:
                with pytest.raises(Error, match="Target closed"):
                    await attachments._wait_for_response_completion(response)
            await _assert_response_completion_cleanup(responses, closed, initial_tasks, warnings)

    asyncio.run(run())


def test_response_completion_public_boundary() -> None:
    class FailedResponse(_FakeResponse):
        async def finished(self) -> None:
            raise Error("public completion failed")

    async def run() -> None:
        assert await attachments._wait_for_response_completion(_FakeResponse()) is None
        with pytest.raises(Error, match="public completion failed"):
            await attachments._wait_for_response_completion(FailedResponse())
        response = _create_lifecycle_response(asyncio.get_running_loop().create_future())
        del response._impl_obj._finished_future
        with pytest.raises(AttributeError, match="_finished_future"):
            await attachments._wait_for_response_completion(response)
        response._impl_obj = object()
        with pytest.raises(TypeError, match="Unsupported Playwright response"):
            await attachments._wait_for_response_completion(response)

    asyncio.run(run())


def test_unattributed_receipt_does_not_use_original_name_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class ReceiptWithoutRequestName(_FakeResponse):
        async def json(self) -> dict[str, str]:
            return {"file_id": "file_candidate"}

    attachment = tmp_path / "candidate.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[ReceiptWithoutRequestName()],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    assert "upload-1" in str(raised.value)
    assert "receipt metadata unavailable" in str(raised.value)


def test_surplus_distinct_receipt_id_is_not_truncated_to_requested_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "one.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[
            _NamedCreateResponse(attachment.name, "file_one"),
            _NamedCreateResponse(attachment.name, "file_surplus"),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    assert "surplus receipt IDs" in str(raised.value)
    assert "receipt attribution=failed" in str(raised.value)


def test_pending_receipt_attribution_blocks_ready_original_name(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class DelayedReceipt(_NamedCreateResponse):
            async def json(self) -> dict[str, str]:
                started.set()
                await release.wait()
                return await super().json()

        attachment = tmp_path / "delayed-receipt.txt"
        attachment.write_text("synthetic", encoding="utf-8")
        page = _FakePage(
            bindings=[_bound_cards(("upload-1", "delayed-card", attachment.name, "ready", True))],
            states=[
                _attachment_state(),
                *[_attachment_state(ready={attachment.name: 1})] * 3,
            ],
            responses=[DelayedReceipt(attachment.name, "file_delayed_receipt")],
        )
        task = asyncio.create_task(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=1,
        ))

        await asyncio.wait_for(started.wait(), timeout=0.5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
        release.set()
        assert await task == {attachment.name: (0, 1)}
        assert page.listeners == {"response": []}
        assert all(handle.dispose_calls == 1 for handle in page.handles)

    asyncio.run(run())


def test_distinct_receipt_names_converging_to_one_display_name_keep_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = []
    for name in ("first.txt", "second.txt"):
        attachment = tmp_path / name
        attachment.write_text("synthetic", encoding="utf-8")
        paths.append(str(attachment))
    display_name = "combined.txt"
    page = _FakePage(
        bindings=[_bound_cards(("upload-1", "first-card", display_name, "ready", True), ("upload-2", "second-card", display_name, "ready", True))],
        states=[
            _attachment_state(),
            *[_attachment_state(ready={display_name: 2})] * 3,
        ],
        responses=[
            _ProcessingResponse(_processing_frames("file_first", display_name)),
            _NamedCreateResponse("first.txt", "file_first"),
            _ProcessingResponse(_processing_frames("file_second", display_name)),
            _NamedCreateResponse("second.txt", "file_second"),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    assert asyncio.run(attachments.attach_files(
        page, paths, timeout_seconds=1,
    )) == {display_name: (0, 2)}


def test_surplus_receipt_diagnostics_remain_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "bounded.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[
            _NamedCreateResponse(attachment.name, f"file_surplus_{index}")
            for index in range(100)
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    assert len(str(raised.value)) <= 4_096
    assert "surplus" in str(raised.value)


def test_attachment_diagnostics_use_attempt_aliases_for_receipt_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    attachment = tmp_path / "diagnostic.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    file_id = "file_private_diagnostic"
    page = _FakePage(
        states=[
            _attachment_state(),
            *[_attachment_state(ready={attachment.name: 1})] * 3,
        ],
        responses=[
            _NamedCreateResponse(attachment.name, file_id),
            _ProcessingResponse([{
                "file_id": file_id, "event": "file.processing.started",
            }]),
        ],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.5,
        ))

    message = str(raised.value)
    assert file_id not in message
    assert "upload-1" in message


def test_settlement_observer_catches_late_failure_before_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    async def run() -> None:
        attachment = tmp_path / "observed.txt"
        attachment.write_text("synthetic", encoding="utf-8")
        page = _FakePage(
            bindings=[_bound_cards(("upload-1", "observed-card", attachment.name, "ready", True))],
            states=[
                _attachment_state(),
                _attachment_state(ready={attachment.name: 1}),
                _attachment_state(ready={attachment.name: 1}),
            ],
            responses=[
                _NamedCreateResponse(attachment.name, "file_observed"),
                _ProcessingResponse(_processing_frames("file_observed", attachment.name)),
            ],
        )
        observations = []
        snapshots = []
        counts = await attachments.attach_files(
            page,
            [str(attachment)],
            timeout_seconds=1,
            on_settlement=observations.append,
            on_diagnostics=snapshots.append,
        )

        assert counts == {attachment.name: (0, 1)}
        assert len(observations) == 1
        assert len(page.listeners["response"]) == 1
        assert await observations[0].revalidate_before_submit() is None
        assert snapshots[-1].pre_submit_result == "ready"
        late_failure = _ProcessingResponse([{
            "file_id": "file_observed", "event": "file.processing.failed",
        }])
        for listener in tuple(page.listeners["response"]):
            listener(late_failure)

        assert await observations[0].revalidate_before_submit() == "processing-failed"
        assert snapshots[-1].pre_submit_result == "processing-failed"
        assert snapshots[-1].outcome == "blocked"
        await observations[0].close()
        assert page.listeners["response"] == []

    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)
    asyncio.run(run())


_BINDING_GRAPH_PROGRAM = r"""
(function () {
  globalThis.File ??= require('node:buffer').File;
  const fs = require('fs');
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const createTracker = new Function(`return (${input.source});`)();
  class Element {
    constructor(tag, attributes = {}) {
      this.tagName = tag.toUpperCase();
      this.attributes = Object.entries(attributes).map(([name, value]) => ({ name, value }));
      this.children = [];
      this.parentElement = null;
      this.parentNode = null;
      this.isConnected = true;
      this.textContent = '';
      this.style = { display: 'block', visibility: 'visible', opacity: '1' };
    }
    append(child) { child.parentElement = this; child.parentNode = this; this.children.push(child); return child; }
    getAttribute(name) { return this.attributes.find((item) => item.name === name)?.value ?? null; }
    contains(node) { return this === node || this.children.some((child) => child.contains(node)); }
    getBoundingClientRect() { return { width: this.boxless ? 0 : 10, height: this.boxless ? 0 : 10 }; }
    getClientRects() { return this.boxless ? [] : [this.getBoundingClientRect()]; }
    matches(selector) {
      return selector === 'form[data-chatgpt-composer]' && this.tagName === 'FORM' &&
        this.getAttribute('data-chatgpt-composer') !== null;
    }
    closest(selector) {
      for (let node = this; node; node = node.parentElement) {
        if (selector === 'form' && node.tagName === 'FORM') return node;
        if (selector === 'button' && node.tagName === 'BUTTON') return node;
        if (selector === '[role="dialog"], dialog, nav, aside' &&
            (['DIALOG', 'NAV', 'ASIDE'].includes(node.tagName) || node.getAttribute('role') === 'dialog')) return node;
        if (selector.includes('[role="search"]') &&
            (node.tagName === 'SEARCH' || node.getAttribute('role') === 'search' ||
             node.getAttribute('data-message-author-role') !== null)) return node;
        if (selector === '[contenteditable], [role="textbox"]' &&
            (node.getAttribute('contenteditable') !== null || node.getAttribute('role') === 'textbox')) return node;
      }
      return null;
    }
    querySelectorAll(selector) {
      const result = [];
      const visit = (parent) => parent.children.forEach((child) => { result.push(child); visit(child); });
      visit(this);
      return selector === '*' ? result : result.filter((node) => selector === 'button[aria-label]' &&
        node.tagName === 'BUTTON' && node.getAttribute('aria-label') !== null);
    }
  }
  const container = new Element('main');
  const form = container.append(new Element('form', { 'data-chatgpt-composer': '' }));
  const composer = form.append(new Element('textarea', { class: 'composer' }));
  globalThis.document = {
    querySelectorAll: (selector) => selector === '.composer' ? [composer]
      : selector === 'form[data-chatgpt-composer]' ? container.querySelectorAll('*').filter((node) => node.matches(selector)) : [],
    querySelector: (selector) => document.querySelectorAll(selector)[0] || null,
  };
  globalThis.getComputedStyle = (node) => node.style;
  const fiber = (tag, stateNode) => ({ tag, stateNode, flags: 0, return: null, alternate: null, child: null, sibling: null });
  const pair = (tag, stateNode) => {
    const current = fiber(tag, stateNode); const alternate = fiber(tag, stateNode);
    current.alternate = alternate; alternate.alternate = current;
    return { current, alternate };
  };
  const children = (parent, items) => {
    for (const branch of ['current', 'alternate']) {
      parent[branch].child = items[0]?.[branch] ?? null;
      items.forEach((item, index) => { item[branch].return = parent[branch]; item[branch].sibling = items[index + 1]?.[branch] ?? null; });
    }
  };
  const rootState = { current: null, containerInfo: container };
  const root = pair(3, rootState); rootState.current = root.current;
  const formPair = pair(5, form); children(root, [formPair]);
  const cardPairs = [];
  function addCard(name, identifier, branch = 'current', state = 'ready') {
    const wrapper3 = form.append(new Element('div'));
    const wrapper2 = wrapper3.append(new Element('div'));
    const wrapper1 = wrapper2.append(new Element('div'));
    const card = wrapper1.append(new Element('div', state === 'processing' ? { 'aria-busy': 'true' } :
      state === 'failed' ? { 'data-state': 'failed' } : {}));
    const label = card.append(new Element('span', { title: name })); label.textContent = name;
    if (state !== 'unknown') card.append(new Element('button', { 'aria-label': 'Remove attachment' }));
    const wrapper3Pair = pair(5, wrapper3); const wrapper2Pair = pair(5, wrapper2);
    const wrapper1Pair = pair(5, wrapper1); const cardPair = pair(5, card);
    const owners = Array.from({ length: 6 }, () => pair(0, null));
    children(wrapper3Pair, [wrapper2Pair]);
    children(wrapper2Pair, [owners[5]]);
    for (let index = 5; index > 0; index -= 1) children(owners[index], [owners[index - 1]]);
    children(owners[0], [wrapper1Pair]); children(wrapper1Pair, [cardPair]);
    owners[2][branch].memoizedProps = { attachment: { id: identifier } };
    owners[3][branch].memoizedProps = { attachment: { id: identifier } };
    for (const owner of owners) {
      owner.current.memoizedProps ??= {}; owner.alternate.memoizedProps ??= {};
    }
    Object.defineProperty(wrapper1, '__reactFiber$test', { value: wrapper1Pair.current });
    cardPairs.push(wrapper3Pair); children(formPair, cardPairs);
    return { card, label, wrapper1, wrapper1Pair, wrapper2Pair, owners };
  }
  const mode = input.mode;
  const priorState = mode.startsWith('baseline-') ? mode.slice('baseline-'.length) : null;
  if (mode === 'control-title') { const control = form.append(new Element('button')); control.append(new Element('span', { title: 'Voice input' })); }
  if (priorState) {
    const prior = addCard(['unlabelled-unknown', 'opaque'].includes(priorState) ? 'other-prior.txt' : 'prior.txt',
      'file_current', 'current', ['unlabelled-unknown', 'opaque'].includes(priorState) ? 'unknown' : priorState);
    if (priorState === 'opaque') prior.label.attributes = [];
    if (priorState === 'unlabelled-unknown') {
      prior.label.attributes = [];
      const icon = new Element('span'); icon.append(new Element('svg'));
      const name = new Element('span'); name.append(new Element('span')); name.append(prior.label);
      prior.card.children = []; prior.card.append(icon); prior.card.append(name);
    }
  }
  if (mode === 'generic-form' || mode === 'form-standard-replacement') form.attributes = [];
  if (mode === 'boxless-form') { form.boxless = true; form.style.display = 'contents'; }
  const tracker = createTracker({ form, composerSelector: '.composer', names: ['upload.txt', 'prior.txt'] });
  if (priorState === 'opaque') {
    if (tracker.baseline_valid !== false) throw new Error('an unclassifiable prior attachment was treated as an empty baseline');
    return;
  }
  if (tracker.baseline_valid !== true) throw new Error('synthetic baseline was not captured');
  if (priorState) {
    for (const node of form.children.slice(1)) node.isConnected = false;
    form.children = [composer]; cardPairs.length = 0;
  }
  const currentBranch = mode === 'alternate-current' || mode === 'stale-marker-only' ? 'alternate' : 'current';
  rootState.current = root[currentBranch];
  const idBranch = mode === 'stale-marker-only' ? 'current' : mode === 'stale-alternate-only' ? 'alternate' : currentBranch;
  const item = addCard(priorState ? 'renamed.txt' : 'upload.txt', 'file_current', idBranch);
  if (mode === 'conflicting-owners') item.owners[3].current.memoizedProps.attachment.id = 'file_other';
  if (mode === 'duplicate-cards') addCard('upload.txt', 'file_current');
  if (mode === 'wrong-field') {
    item.owners[2].current.memoizedProps = { attachment: { metadata_object_id: 'file_current' } };
    item.owners[3].current.memoizedProps = { attachment: { file_id: 'file_current' } };
    item.wrapper1Pair.current.key = 'file_current';
  }
  if (mode === 'card-limit') for (let index = 0; index < 100; index += 1) addCard(`other-${index}.txt`, `file_other_${index}`);
  if (mode === 'unrelated-root') rootState.containerInfo = {};
  if (mode === 'disconnected-form') form.isConnected = false;
  if (mode === 'mismatched-alternate-host') item.wrapper1Pair.alternate.stateNode = {};
  if (mode === 'unrelated-owner') item.owners[2].current.child = null;
  if (mode === 'hidden-card') item.card.style.visibility = 'hidden';
  if (mode === 'boxless-wrapper') item.wrapper1.boxless = true;
  if (mode === 'processing-card') item.card.attributes.push({ name: 'aria-busy', value: 'true' });
  if (mode === 'failed-card') item.card.attributes.push({ name: 'data-state', value: 'failed' });
  if (mode === 'return-cycle') item.owners[5].current.return = item.owners[5].current;
  let getterCalls = 0;
  if (mode === 'accessor') Object.defineProperty(item.owners[2].current, 'memoizedProps', {
    get() { getterCalls += 1; return { attachment: { id: 'file_current' } }; },
  });
  if (mode === 'depth-cutoff') {
    let parent = root;
    for (let index = 0; index < 520; index += 1) { const next = pair(0, null); children(parent, [next]); parent = next; }
    children(parent, [formPair]);
  }
  const options = { aliases: { file_current: 'upload-1', file_other: 'upload-2' }, names: ['upload.txt', 'renamed.txt'] };
  let snapshot = tracker.read(options);
  if (mode.startsWith('form-')) {
    const strongCheck = new Function(`return (${input.form_current});`)();
    if (!strongCheck(form, '.composer') || !snapshot.valid || snapshot.bindings.length !== 1 ||
        snapshot.bindings[0].state !== 'ready' || snapshot.bindings[0].alias !== 'upload-1') {
      throw new Error('coherent form regression requires one valid current ready bound card before mutation');
    }
    if (mode === 'form-hidden') form.style.visibility = 'hidden';
    else if (mode === 'form-search') composer.attributes.push({ name: 'role', value: 'search' });
    else if (mode === 'form-standard-replacement') container.append(new Element('form', { 'data-chatgpt-composer': '' }));
    else {
      const ancestor = new Element(mode.slice('form-'.length));
      container.children[container.children.indexOf(form)] = ancestor;
      ancestor.parentElement = container; ancestor.parentNode = container;
      ancestor.append(form);
    }
    const after = strongCheck(form, '.composer');
    snapshot = tracker.read(options);
    if (after || snapshot.valid || snapshot.bindings.length !== 0) {
      throw new Error(`form became ineligible between evaluations but its new ready-card snapshot qualified: ${JSON.stringify({ strong: after, snapshot })}`);
    }
  }
  if (mode === 'reassigned-card') {
    item.owners[2].current.memoizedProps.attachment.id = 'file_other';
    item.owners[3].current.memoizedProps.attachment.id = 'file_other';
    snapshot = tracker.read(options);
  }
  if (mode === 'failed-card') {
    item.card.attributes = [];
    snapshot = tracker.read(options);
  }
  const ready = snapshot.bindings.filter((binding) => binding.alias === 'upload-1' && binding.state === 'ready' &&
    binding.visible && binding.is_new && !snapshot.failed_aliases.includes(binding.alias));
  const positives = ['marker-current', 'alternate-current', 'boxless-wrapper', 'control-title', 'generic-form', 'boxless-form'];
  if (positives.includes(mode)) {
    if (!snapshot.valid || snapshot.conflict || ready.length !== 1 || snapshot.bindings.length !== 1) {
      throw new Error(`expected one current bound card: ${JSON.stringify(snapshot)}`);
    }
  } else if (ready.length !== 0 && !snapshot.conflict) {
    throw new Error(`uncertain/stale/baseline card qualified: ${JSON.stringify(snapshot)}`);
  }
  if (['conflicting-owners', 'duplicate-cards', 'reassigned-card'].includes(mode) && !snapshot.conflict) {
    throw new Error('conflicting current ID/card assignment was not latched');
  }
  if (mode === 'failed-card' && !snapshot.failed_aliases.includes('upload-1')) throw new Error('failure latch was lost');
  if (getterCalls !== 0) throw new Error('application getter was invoked');
  console.log(JSON.stringify({ valid: snapshot.valid, conflict: snapshot.conflict, ready: ready.length }));
})();
"""


@pytest.mark.parametrize("mode", [
    "marker-current", "alternate-current", "boxless-wrapper", "control-title", "generic-form", "boxless-form",
    "form-dialog", "form-nav", "form-aside", "form-hidden", "form-search", "form-standard-replacement",
    "stale-marker-only", "stale-alternate-only", "unrelated-owner", "mismatched-alternate-host",
    "conflicting-owners", "duplicate-cards", "reassigned-card",
    "hidden-card", "processing-card", "failed-card", "return-cycle", "accessor",
    "depth-cutoff", "wrong-field", "card-limit", "unrelated-root", "disconnected-form",
    "baseline-ready", "baseline-processing", "baseline-failed", "baseline-unknown", "baseline-unlabelled-unknown", "baseline-opaque",
])
def test_current_attachment_binding_runs_in_isolated_javascript(mode: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for executable attachment identity tests")
    completed = subprocess.run(
        [node, "-e", _BINDING_GRAPH_PROGRAM],
        input=json.dumps({"source": attachments.CREATE_ATTACHMENT_BINDING_TRACKER_JS, "mode": mode,
                          "form_current": attachments.COMPOSER_FORM_CURRENT_JS}),
        text=True, capture_output=True, check=False, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_production_renderer_currentness_parent_sets_and_depth_are_executable() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for executable attachment identity tests")
    program = r"""
(function () {
  const fs = require('fs');
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const boundary = input.source.indexOf('  const getCardsInForm =');
  if (boundary < 0) throw new Error('production identity helpers not found');
  const createHelpers = new Function(`return (${input.source.slice(0, boundary)}
    return { resolveCurrentFiber, ownsSelectedHostByChildren };
  });`)();
  const container = {};
  const form = { isConnected: true, parentNode: container };
  const helpers = createHelpers({ form, composerSelector: '.composer', names: [] });
  const pair = (tag, stateNode) => {
    const a = { tag, stateNode, flags: 0, return: null, alternate: null, child: null, sibling: null };
    const b = { ...a }; a.alternate = b; b.alternate = a;
    return [a, b];
  };
  function graph(hops, sharedChild = false) {
    const state = { current: null, containerInfo: container };
    const root = pair(3, state); state.current = root[0];
    const pairs = [root];
    for (let depth = 0; depth < hops; depth += 1) {
      const next = pair(0, null); const parent = pairs[pairs.length - 1];
      for (const branch of [0, 1]) { parent[branch].child = next[branch]; next[branch].return = parent[branch]; }
      pairs.push(next);
    }
    if (sharedChild) pairs[pairs.length - 2][1].child = pairs[pairs.length - 1][0];
    return { state, root, pairs, leaf: pairs[pairs.length - 1] };
  }
  for (const childIndex of [0, 1]) {
    const item = graph(2);
    item.state.current = item.root[1];
    const parent = item.pairs[1]; const candidate = item.leaf;
    candidate[0].return = parent[0]; candidate[1].return = parent[0];
    parent[0].child = null; parent[1].child = candidate[childIndex];
    const result = helpers.resolveCurrentFiber(candidate[0], form);
    if (result.state !== 'resolved' || result.candidate_is_current !== (childIndex === 0)) {
      throw new Error(`parent-B-only mapping reversed: ${JSON.stringify({ childIndex, state: result.state, current: result.candidate_is_current })}`);
    }
  }
  const sharedReturn = graph(2);
  sharedReturn.leaf[1].return = sharedReturn.pairs[1][0];
  if (!helpers.resolveCurrentFiber(sharedReturn.leaf[0], form).candidate_is_current) throw new Error('shared return failed');
  const sharedChild = graph(322, true);
  sharedChild.state.current = sharedChild.root[1];
  if (!helpers.resolveCurrentFiber(sharedChild.leaf[0], form).candidate_is_current) throw new Error('shared child mounted subwalk failed');
  const deep = graph(512);
  if (!helpers.resolveCurrentFiber(deep.leaf[0], form).candidate_is_current) throw new Error('valid bounded deep chain failed');
  deep.state.current = deep.root[1];
  if (helpers.resolveCurrentFiber(deep.leaf[0], form).candidate_is_current) throw new Error('deep stale branch promoted');
  const overBudget = graph(513);
  if (helpers.resolveCurrentFiber(overBudget.leaf[0], form).state !== 'inconclusive') throw new Error('unbounded root walk');
  const mountedOverBudget = graph(515, true);
  if (helpers.resolveCurrentFiber(mountedOverBudget.leaf[0], form).state !== 'inconclusive') throw new Error('unbounded mounted subwalk');
  const disconnected = graph(2);
  disconnected.state.containerInfo = {};
  if (helpers.resolveCurrentFiber(disconnected.leaf[0], form).state !== 'inconclusive') throw new Error('unrelated renderer root accepted');
  const cycle = graph(2); cycle.leaf[0].return = cycle.leaf[0];
  if (helpers.resolveCurrentFiber(cycle.leaf[0], form).state !== 'inconclusive') throw new Error('cycle accepted');
  const accessor = graph(2); let getterCalls = 0;
  Object.defineProperty(accessor.leaf[0], 'alternate', { get() { getterCalls += 1; return null; } });
  if (helpers.resolveCurrentFiber(accessor.leaf[0], form).state !== 'inconclusive' || getterCalls) throw new Error('accessor read');
})();
"""
    completed = subprocess.run(
        [node, "-e", program], input=json.dumps({"source": attachments.CREATE_ATTACHMENT_BINDING_TRACKER_JS}),
        text=True, capture_output=True, check=False, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def _bound_cards(*cards: tuple[str, str, str, str, bool]) -> dict[str, Any]:
    return {
        "valid": True,
        "conflict": False,
        "failed_aliases": [],
        "bindings": [
            {"alias": alias, "card": card, "name": name, "state": state, "visible": True, "is_new": is_new}
            for alias, card, name, state, is_new in cards
        ],
    }


def test_deferred_name_and_completed_receipt_settle_with_current_ready_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "deferred-bound.txt"
    path.write_text("synthetic", encoding="utf-8")
    file_id = "file_deferred_bound"
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(ready={path.name: 1})] * 3],
        responses=[_NamedCreateResponse(path.name, file_id), _ProcessingResponse([
            {"file_id": file_id, "event": "file.indexing.started", "extra": {
                "retrieval_indexing_deferred": True, "library_file_name": path.name,
            }},
            {"file_id": file_id, "event": "file.processing.completed"},
        ])],
        bindings=[_bound_cards(("upload-1", "current-card", path.name, "ready", True))],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    result = asyncio.run(attachments.attach_files(page, [str(path)], timeout_seconds=0.5))

    assert result == {path.name: (0, 1)}
    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)


@pytest.mark.parametrize("frames", [[], [{"file_id": "file_absent", "event": "file.processing.completed"}]])
def test_absent_name_policy_requires_current_receipt_bound_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: list[dict[str, Any]],
) -> None:
    path = tmp_path / "absent.txt"
    path.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(ready={path.name: 1})] * 3],
        responses=[_NamedCreateResponse(path.name, "file_absent"), *([_ProcessingResponse(frames)] if frames else [])],
        bindings=[_bound_cards(("upload-1", "current-card", path.name, "ready", True))],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    assert asyncio.run(attachments.attach_files(page, [str(path)], timeout_seconds=0.5)) == {path.name: (0, 1)}


@pytest.mark.parametrize("binding", [
    _bound_cards(),
    _bound_cards(("upload-1", "replacement", "mixed.txt", "processing", True)),
    _bound_cards(("upload-2", "replacement", "mixed.txt", "ready", True)),
    _bound_cards(("upload-1", "prior-card", "mixed.txt", "ready", False)),
])
def test_name_counts_cannot_lend_readiness_to_a_different_current_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, binding: dict[str, Any],
) -> None:
    path = tmp_path / "mixed.txt"
    path.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(ready={path.name: 1})] * 3],
        responses=[_NamedCreateResponse(path.name, "file_mixed")],
        bindings=[binding],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(page, [str(path)], timeout_seconds=0.5))

    assert raised.value.ready_attachments == 0
    assert "1/1 observed composer-ready attachments" in str(raised.value)
    assert "0/1 processing-qualified ready attachments" in str(raised.value)


@pytest.mark.parametrize("frames", [
    [{"file_id": "file_other", "event": "file.processing.completed"}],
    [
        {"file_id": "file_requested", "event": "file.indexing.started", "extra": {
            "library_file_name": "requested.txt", "retrieval_indexing_deferred": False,
        }},
        {"file_id": "file_requested", "event": "file.processing.completed"},
    ],
])
def test_requested_name_fallback_rejects_unknown_attribution_and_unsupported_present_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: list[dict[str, Any]],
) -> None:
    path = tmp_path / "requested.txt"
    path.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(ready={path.name: 1})] * 3],
        responses=[_NamedCreateResponse(path.name, "file_requested"), _ProcessingResponse(frames)],
        bindings=[_bound_cards(("upload-1", "current-card", path.name, "ready", True))],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(page, [str(path)], timeout_seconds=0.5))

    assert raised.value.ready_attachments == 0
    assert "1/1 observed composer-ready attachments" in str(raised.value)


@pytest.mark.parametrize("frames", [
    _processing_frames("file_requested", "renamed.txt"),
    [
        {"file_id": "file_requested", "event": "file.indexing.started", "extra": {
            "library_file_name": "renamed.txt", "retrieval_indexing_deferred": True,
        }},
        {"file_id": "file_requested", "event": "file.processing.completed"},
    ],
])
def test_authoritative_rename_cannot_borrow_requested_name_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: list[dict[str, Any]],
) -> None:
    path = tmp_path / "requested.txt"
    path.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(ready={"renamed.txt": 1})] * 3],
        responses=[_NamedCreateResponse(path.name, "file_requested"), _ProcessingResponse(frames)],
        bindings=[_bound_cards(("upload-1", "current-card", path.name, "ready", True))],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentSettleTimeoutError) as raised:
        asyncio.run(attachments.attach_files(page, [str(path)], timeout_seconds=0.5))

    assert raised.value.ready_attachments == 0


@pytest.mark.parametrize("changed", [
    _bound_cards(("upload-1", "replacement", "observed.txt", "processing", True)),
    _bound_cards(("upload-2", "replacement", "observed.txt", "ready", True)),
    _bound_cards(("upload-1", "replacement", "other.txt", "ready", True)),
    _bound_cards(),
])
def test_pre_submit_current_binding_rejects_same_name_replacement(
    tmp_path: Path, changed: dict[str, Any],
) -> None:
    async def run() -> None:
        path = tmp_path / "observed.txt"
        path.write_text("synthetic", encoding="utf-8")
        current = _bound_cards(("upload-1", "settled-card", path.name, "ready", True))
        page = _FakePage(
            states=[_attachment_state(), *[_attachment_state(ready={path.name: 1})] * 5],
            responses=[_NamedCreateResponse(path.name, "file_observed")],
            bindings=[current],
        )
        observations = []
        await attachments.attach_files(page, [str(path)], timeout_seconds=1, on_settlement=observations.append)
        tracker = next(handle for handle in page.handles if isinstance(handle, _FakeBindingTracker))
        tracker.states = iter([changed])

        assert await observations[0].revalidate_before_submit() == "attachments-not-ready"
        assert page.listeners["response"]
        await observations[0].close()
        assert page.listeners == {"response": []}
        assert tracker.dispose_calls == 1
        assert await observations[0].revalidate_before_submit() == "observer-closed"

    asyncio.run(run())


@pytest.mark.parametrize("snapshot", [
    {**_bound_cards(), "conflict": True},
    {**_bound_cards(), "failed_aliases": ["upload-1"]},
    _bound_cards(("upload-1", "same-card", "conflict.txt", "ready", True),
                 ("upload-2", "same-card", "conflict.txt", "ready", True)),
    _bound_cards(("upload-1", "card-one", "conflict.txt", "ready", True),
                 ("upload-1", "card-two", "conflict.txt", "ready", True)),
])
def test_current_card_conflicts_or_failure_never_count_as_partial_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, snapshot: dict[str, Any],
) -> None:
    path = tmp_path / "conflict.txt"
    path.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[_attachment_state(), *[_attachment_state(ready={path.name: 1})] * 3],
        responses=[_NamedCreateResponse(path.name, "file_conflict")], bindings=[snapshot],
    )
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)

    with pytest.raises(attachments.AttachmentUploadFailedError):
        asyncio.run(attachments.attach_files(page, [str(path)], timeout_seconds=0.5))

    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)


def test_standalone_upload_pins_generic_form_before_local_drop(tmp_path: Path) -> None:
    path = tmp_path / "local.txt"
    path.write_text("synthetic", encoding="utf-8")

    async def run() -> None:
        async with async_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                candidates = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
                ))
                if not candidates:
                    pytest.skip("No local Chromium executable available")
                executable = candidates[-1]
            browser = await playwright.chromium.launch(executable_path=str(executable), headless=True)
            try:
                page = await browser.new_page()
                await page.set_content('<main id="fallback"><form id="other"></form><form id="intended"><textarea id="composer"></textarea></form></main>')
                await page.evaluate("""() => {
                  window.dropTargets = [];
                  Object.defineProperty(document.querySelector('#intended'), '__reactProps$test', {value: {onDropCapture() {}}});
                  Object.defineProperty(document.querySelector('main'), '__reactProps$test', {value: {
                    onDragEnterCapture() {}, onDragOverCapture() {}, onDropCapture() {},
                  }});
                  document.querySelector('main').addEventListener('drop', (event) => {
                    event.preventDefault(); window.dropTargets.push(event.target.id);
                  });
                }""")
                with pytest.raises(attachments.AttachmentSettleTimeoutError):
                    await attachments.attach_files(page, [str(path)], composer_selector="#composer", timeout_seconds=0.3)
                assert await page.evaluate("window.dropTargets") == ["intended"]
            finally:
                await browser.close()

    asyncio.run(run())


def test_current_attachment_binding_with_real_dom_visibility_and_replacement() -> None:
    async def run() -> None:
        async with async_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                candidates = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
                ))
                if not candidates:
                    pytest.skip("No local Chromium executable available")
                executable = candidates[-1]
            browser = await playwright.chromium.launch(executable_path=str(executable), headless=True)
            try:
                page = await browser.new_page()
                await page.set_content('<main><form data-chatgpt-composer><textarea class="composer"></textarea></form></main>')
                result = await page.evaluate(r"""(source) => {
                  const form = document.querySelector('form');
                  const create = new Function(`return (${source});`)();
                  const tracker = create({ form, composerSelector: '.composer', names: ['dom.txt'] });
                  const wrapper2 = form.appendChild(document.createElement('div'));
                  wrapper2.style.display = 'contents';
                  wrapper2.innerHTML = '<div style="display:contents"><div><span><svg></svg></span><span><span>Text</span><span>dom.txt</span></span><span>91 B</span><button aria-label="Remove dom.txt">Remove</button></div></div>';
                  const wrapper1 = wrapper2.firstElementChild;
                  const card = wrapper1.firstElementChild;
                  const pair = (tag, stateNode) => {
                    const a = { tag, stateNode, flags: 0, return: null, alternate: null, child: null, sibling: null };
                    const b = { ...a }; a.alternate = b; b.alternate = a; return [a, b];
                  };
                  const rootState = { current: null, containerInfo: document.querySelector('main') };
                  const root = pair(3, rootState); rootState.current = root[0];
                  const host = pair(5, wrapper1);
                  const chain = [root, pair(5, form), pair(5, wrapper2), ...Array.from({ length: 6 }, () => pair(0, null)), host];
                  for (let index = 0; index < chain.length - 1; index += 1) {
                    for (const branch of [0, 1]) { chain[index][branch].child = chain[index + 1][branch]; chain[index + 1][branch].return = chain[index][branch]; }
                  }
                  for (const distance of [3, 4]) {
                    const owner = chain[chain.length - 1 - distance];
                    owner[0].memoizedProps = { attachment: { id: 'file_dom' } };
                    owner[1].memoizedProps = {};
                  }
                  Object.defineProperty(wrapper1, '__reactFiber$test', { value: host[0] });
                  const options = { aliases: { file_dom: 'upload-1' }, names: ['dom.txt'] };
                  const ready = tracker.read(options);
                  card.setAttribute('aria-busy', 'true');
                  const processing = tracker.read(options);
                  card.removeAttribute('aria-busy');
                  card.style.visibility = 'hidden';
                  const hidden = tracker.read(options);
                  card.style.visibility = 'visible';
                  const navigation = document.createElement('nav');
                  const originalParent = form.parentElement;
                  originalParent.appendChild(navigation);
                  navigation.appendChild(form);
                  const ineligibleForm = tracker.read(options);
                  originalParent.appendChild(form);
                  navigation.remove();
                  rootState.current = root[1];
                  const stale = tracker.read(options);
                  return { ready, processing, hidden, ineligibleForm, stale };
                }""", attachments.CREATE_ATTACHMENT_BINDING_TRACKER_JS)
            finally:
                await browser.close()
        ready = result["ready"]
        assert ready["valid"] and not ready["conflict"]
        assert len(ready["bindings"]) == 1
        assert ready["bindings"][0] == {
            "alias": "upload-1", "card": "card-1", "name": "dom.txt",
            "state": "ready", "visible": True, "is_new": True,
        }
        assert result["processing"]["bindings"][0]["state"] == "processing"
        assert result["hidden"]["bindings"][0]["visible"] is False
        assert result["ineligibleForm"]["valid"] is False
        assert result["ineligibleForm"]["bindings"] == []
        assert result["stale"]["bindings"] == []
        assert result["stale"]["valid"] is False

    asyncio.run(run())


def test_processing_failure_during_binding_snapshot_is_not_settled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    async def run() -> None:
        observed = asyncio.Event()
        failed = asyncio.Event()

        class CompletedProcessing(_ProcessingResponse):
            async def text(self) -> str:
                result = await super().text()
                observed.set()
                return result

        class LateFailure(_ProcessingResponse):
            async def text(self) -> str:
                result = await super().text()
                failed.set()
                return result

        class LateFailurePage(_FakePage):
            async def evaluate(self, script: str, argument: Any = None) -> Any:
                if script == attachments.READ_COMPOSER_ATTACHMENT_STATE_JS and any(
                    expression == attachments.CREATE_ATTACHMENT_FILE_JS for expression, _ in self.evaluate_handle_calls
                ):
                    await observed.wait()
                return await super().evaluate(script, argument)

            async def evaluate_handle(self, script: str, argument: Any) -> _FakeHandle:
                handle = await super().evaluate_handle(script, argument)
                if isinstance(handle, _FakeBindingTracker):
                    read = handle.evaluate

                    async def read_with_failure(expression: str, options: Any = None) -> Any:
                        snapshot = await read(expression, options)
                        if expression == attachments.READ_ATTACHMENT_BINDINGS_JS and not failed.is_set():
                            response = LateFailure([{"file_id": "file_race", "event": "file.processing.failed"}])
                            for listener in tuple(self.listeners["response"]):
                                listener(response)
                            await failed.wait()
                            await asyncio.sleep(0)
                        return snapshot

                    handle.evaluate = read_with_failure
                return handle

        path = tmp_path / "race.txt"
        path.write_text("synthetic", encoding="utf-8")
        page = LateFailurePage(
            states=[_attachment_state(), *[_attachment_state(ready={path.name: 1})] * 5],
            responses=[_NamedCreateResponse(path.name, "file_race"), CompletedProcessing(_processing_frames("file_race", path.name))],
            bindings=[_bound_cards(("upload-1", "race-card", path.name, "ready", True))],
        )
        with pytest.raises(attachments.AttachmentUploadFailedError):
            await attachments.attach_files(page, [str(path)], timeout_seconds=1)
        assert failed.is_set()
        assert page.listeners == {"response": []}

    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["no_request", "rejected", "transport_failed", "pending", "setup_exhausted"])
def test_upload_diagnostics_distinguish_unobserved_and_failed_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str,
) -> None:
    attachment = tmp_path / "private-name.txt"
    attachment.write_text("private attachment text", encoding="utf-8")
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)
    snapshots = []

    class Page(_FakePage):
        async def evaluate_handle(self, script: str, argument: Any) -> _FakeHandle:
            if mode == "setup_exhausted" and script == attachments.CREATE_ATTACHMENT_FILE_JS:
                clock.value += 1
            return await super().evaluate_handle(script, argument)

        async def evaluate(self, script: str, argument: Any = None) -> Any:
            if script != attachments.DISPATCH_ATTACHMENT_DROP_JS:
                return await super().evaluate(script, argument)
            if mode in ("rejected", "transport_failed", "pending"):
                request = SimpleNamespace(method="POST", url="https://chatgpt.com/backend-api/files")
                for callback in tuple(self.listeners.get("request", [])):
                    callback(request)
                if mode == "rejected":
                    response = SimpleNamespace(request=request, url=request.url, status=429)
                    for callback in tuple(self.listeners["response"]):
                        callback(response)
                    for callback in tuple(self.listeners.get("requestfinished", [])):
                        callback(request)
                elif mode == "transport_failed":
                    for callback in tuple(self.listeners.get("requestfailed", [])):
                        callback(request)
            return "intended-form"

    page = Page(responses=[])
    error = attachments.AttachmentPreflightError if mode == "setup_exhausted" else attachments.AttachmentSettleTimeoutError
    with pytest.raises(error) as raised:
        asyncio.run(attachments.attach_files(
            page, [str(attachment)], timeout_seconds=0.01, on_diagnostics=snapshots.append,
        ))
    diagnostic = snapshots[-1]
    assert diagnostic.outcome == ("failed" if mode == "setup_exhausted" else "timeout")
    if mode == "setup_exhausted":
        assert diagnostic.drop_started_ms is diagnostic.drop_returned_ms is None
        assert not any(script == attachments.DISPATCH_ATTACHMENT_DROP_JS for script, _ in page.evaluate_calls)
    else:
        assert diagnostic.drop_started_ms is not None
        assert diagnostic.drop_returned_ms is not None
    assert diagnostic.file_sizes == (len(attachment.read_bytes()),)
    if mode != "setup_exhausted":
        assert raised.value.completed_file_create_responses == 0
        assert raised.value.ready_attachments == 0
    assert page.listeners == {"response": []}
    events = [event.event for event in diagnostic.events]
    if mode == "rejected":
        assert "requestfinished" in events
        assert any(event.status == 429 for event in diagnostic.events)
    elif mode == "transport_failed":
        assert "requestfailed" in events
        assert "response" not in events
    elif mode == "pending":
        assert "request" in events
        assert "response" not in events
        assert "requestfinished" not in events
    else:
        assert "request" not in events
    if mode == "setup_exhausted":
        assert diagnostic.poll_attempts == diagnostic.poll_completions == 0
        assert diagnostic.last_poll_ms is None
    else:
        assert diagnostic.poll_completions > 0
    serialized = json.dumps(asdict(diagnostic))
    assert str(tmp_path) not in serialized
    assert "private-name" not in serialized
    assert "private attachment text" not in serialized


def test_pre_submit_diagnostics_record_current_binding_conflict(tmp_path: Path) -> None:
    async def run() -> None:
        attachment = tmp_path / "synthetic.txt"
        attachment.write_text("synthetic", encoding="utf-8")
        page = _FakePage(
            states=[_attachment_state(), *[_attachment_state(ready={attachment.name: 1})] * 3],
            responses=[_NamedCreateResponse(attachment.name, "file_synthetic")],
            bindings=[_bound_cards(("upload-1", "card", attachment.name, "ready", True))],
        )
        observations = []
        snapshots = []
        await attachments.attach_files(
            page, [str(attachment)], timeout_seconds=1,
            on_settlement=observations.append, on_diagnostics=snapshots.append,
        )
        assert snapshots[-1].outcome == "settled"
        assert snapshots[-1].binding_conflict is False
        tracker = next(handle for handle in page.handles if isinstance(handle, _FakeBindingTracker))
        tracker.states = iter([{**_bound_cards(), "conflict": True}])
        assert await observations[0].revalidate_before_submit() == "attachment-binding-failed"
        assert snapshots[-1].phase == "pre_submit"
        assert snapshots[-1].binding_conflict is True
        assert snapshots[-1].pre_submit_result == "attachment-binding-failed"
        assert snapshots[-1].outcome == "blocked"
        await observations[0].close()
        assert not snapshots[-1].observer_active
        assert snapshots[-1].outcome == "blocked"
        assert page.listeners == {"response": []}

    asyncio.run(run())


def test_drop_preflight_waits_for_observed_capture_path_in_production_javascript() -> None:
    program = _BINDING_GRAPH_PROGRAM.split("  const fiber =", 1)[0] + r"""
  const assert = require('assert');
  const file = new File(['synthetic bytes'], 'original.txt', {type: 'text/plain'});
  const dataTransfer = {files: [file], items: [{kind: 'file', type: 'text/plain', getAsFile: () => file}]};
  const tracker = createTracker({form, composerSelector: '.composer', dataTransfer,
    files: [file], expected: [{name: file.name, mime: file.type, size: file.size}]});
  assert.deepEqual(tracker.read().state, 'waiting');
  assert.equal(tracker.read().reason, 'capture-path-uninitialized');
  Object.defineProperty(form, '__reactProps$test', {value: {onDropCapture() {}}});
  Object.defineProperty(container, '__reactProps$test', {value: {
    onDragEnterCapture() {}, onDragOverCapture() {}, onDropCapture() {},
  }});
  const ready = tracker.read();
  assert.equal(ready.state, 'eligible');
  assert.equal(ready.receiver_relationship, 'ancestor');
  assert.equal(ready.file_count, 1);
  assert.equal(ready.item_count, 1);
  assert.deepEqual(ready.file_sizes, [15]);
})();
"""
    completed = subprocess.run(
        ["node", "-e", program], text=True, capture_output=True, check=False, timeout=10,
        input=json.dumps({"source": attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS}),
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mutation, reason", [
    ("form.isConnected = false", "form-changed"),
    ("const replacement = form.append(new Element('textarea')); document.querySelectorAll = () => [replacement]", "form-changed"),
    ("for (let index = 0; index < 513; index++) form.append(new Element('div'))", "descendant-limit"),
    ("let parent = container; for (let index = 0; index < 129; index++) { parent.parentElement = new Element('div'); parent = parent.parentElement; }", "ancestor-limit"),
    ("Object.defineProperty(container, '__reactProps$other', {value: receiverProps})", "handler-metadata-unavailable"),
    ("dataTransfer.files.length = 0", "payload-invalid"),
    ("dataTransfer.items = null", "payload-invalid"),
    ("container.parentElement = new Element('main'); Object.defineProperty(container.parentElement, '__reactProps$test', {value: {onDropCapture() {}}})", "ambiguous-capture-path"),
    ("dataTransfer.items[0].kind = 'string'", "payload-invalid"),
    ("dataTransfer.files[0] = new File(['other'], 'original.txt', {type: 'text/plain'})", "payload-invalid"),
    ("form.parentElement = new Element('main'); Object.defineProperty(form.parentElement, '__reactProps$test', {value: receiverProps})", "capture-path-changed"),
    ("receiverProps.onDropCapture = () => {}", "capture-path-changed"),
    ("formProps.onDropCapture = () => {}", "capture-path-changed"),
    ("Object.defineProperty(composer, '__reactProps$test', {value: {onDrop() {}}})", "ambiguous-capture-path"),
    ("Object.defineProperty(receiverProps, 'onDropCapture', {get() {throw Error('PRIVATE_SENTINEL')}})", "handler-metadata-unavailable"),
])
def test_drop_preflight_rejects_invalid_payload_or_changed_receivers(
    mutation: str, reason: str,
) -> None:
    program = _BINDING_GRAPH_PROGRAM.split("  const fiber =", 1)[0] + r"""
  const assert = require('assert');
  const file = new File(['synthetic bytes'], 'original.txt', {type: 'text/plain'});
  const dataTransfer = {files: [file], items: [{kind: 'file', type: 'text/plain', getAsFile: () => file}]};
  const formProps = {onDropCapture() {}};
  const receiverProps = {onDragEnterCapture() {}, onDragOverCapture() {}, onDropCapture() {}};
  Object.defineProperty(form, '__reactProps$test', {value: formProps});
  Object.defineProperty(container, '__reactProps$test', {value: receiverProps});
  const tracker = createTracker({form, composerSelector: '.composer', dataTransfer,
    files: [file], expected: [{name: file.name, mime: file.type, size: file.size}]});
  assert.equal(tracker.read().state, 'eligible');
  """ + mutation + r""";
  const blocked = tracker.read();
  assert.equal(blocked.state, 'blocked');
  assert.equal(blocked.reason, input.reason);
  assert(!JSON.stringify(blocked).includes('PRIVATE_SENTINEL'));
})();
"""
    completed = subprocess.run(
        ["node", "-e", program], text=True, capture_output=True, check=False, timeout=10,
        input=json.dumps({"source": attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS, "reason": reason}),
    )
    assert completed.returncode == 0, completed.stderr


def test_drop_preflight_dispatches_original_payload_once_on_the_observed_capture_path(
    tmp_path: Path,
) -> None:
    attachment = tmp_path / "original.txt"
    original = "synthetic UTF-8 payload: Ω\n".encode()
    attachment.write_bytes(original)

    async def run() -> None:
        async with async_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                candidates = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
                ))
                if not candidates:
                    pytest.skip("No local Chromium executable available")
                executable = candidates[-1]
            browser = await playwright.chromium.launch(executable_path=str(executable), headless=True)
            try:
                page = await browser.new_page()
                await page.set_content('<main><form><textarea class="composer"></textarea></form></main>')
                await page.evaluate("""() => {
                  const form = document.querySelector('form');
                  const receiver = document.querySelector('main');
                  Object.defineProperty(form, '__reactProps$test', {value: {onDropCapture() {}}});
                  Object.defineProperty(receiver, '__reactProps$test', {value: {
                    onDragEnterCapture() {}, onDragOverCapture() {}, onDropCapture() {},
                  }});
                  window.events = [];
                  for (const type of ['dragenter', 'dragover', 'drop']) {
                    receiver.addEventListener(type, event => {
                      event.preventDefault();
                      window.events.push({type, targetIsForm: event.target === form,
                        bubbles: event.bubbles, composed: event.composed, cancelable: event.cancelable});
                      if (type === 'drop') window.droppedBytes = event.dataTransfer.files[0].arrayBuffer().then(value => Array.from(new Uint8Array(value)));
                    }, true);
                  }
                }""")
                form = await page.query_selector('form')
                file = await page.evaluate_handle(attachments.CREATE_ATTACHMENT_FILE_JS, attachments._load_descriptors([str(attachment)])[0])
                transfer = await page.evaluate_handle(attachments.CREATE_ATTACHMENT_DATA_TRANSFER_JS, [file])
                preflight = await page.evaluate_handle(attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS, {
                    "form": form, "composerSelector": ".composer", "files": [file], "dataTransfer": transfer,
                    "expected": [{"name": attachment.name, "mime": "text/plain", "size": len(original)}],
                })
                assert (await preflight.evaluate("(preflight) => preflight.read()"))["state"] == "eligible"
                outcome = await preflight.evaluate("(preflight) => preflight.dispatch()")
                assert outcome == {"target": "intended-form", "events": [
                    {"type": event, "dispatch_returned": False, "default_prevented": True}
                    for event in ("dragenter", "dragover", "drop")
                ]}
                assert await page.evaluate("window.events") == [
                    {"type": event, "targetIsForm": True, "bubbles": True, "composed": True, "cancelable": True}
                    for event in ("dragenter", "dragover", "drop")
                ]
                assert bytes(await page.evaluate("window.droppedBytes")) == original
                with pytest.raises(Exception, match="already-attempted"):
                    await preflight.evaluate("(preflight) => preflight.dispatch()")
                assert len(await page.evaluate("window.events")) == 3
                await page.close()
                fresh_page = await browser.new_page()
                await fresh_page.set_content('<main><form><textarea class="composer"></textarea></form></main>')
                fresh_form = await fresh_page.query_selector('form')
                fresh_file = await fresh_page.evaluate_handle(attachments.CREATE_ATTACHMENT_FILE_JS, attachments._load_descriptors([str(attachment)])[0])
                fresh_transfer = await fresh_page.evaluate_handle(attachments.CREATE_ATTACHMENT_DATA_TRANSFER_JS, [fresh_file])
                fresh_preflight = await fresh_page.evaluate_handle(attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS, {
                    "form": fresh_form, "composerSelector": ".composer", "files": [fresh_file], "dataTransfer": fresh_transfer,
                    "expected": [{"name": attachment.name, "mime": "text/plain", "size": len(original)}],
                })
                assert (await fresh_preflight.evaluate("(preflight) => preflight.read()"))["state"] == "waiting"
                with pytest.raises(Exception, match="capture-path-uninitialized"):
                    await fresh_preflight.evaluate("(preflight) => preflight.dispatch()")
            finally:
                await browser.close()

    asyncio.run(run())


@pytest.mark.parametrize("is_delayed", [False, True])
def test_attach_files_waits_for_preflight_within_original_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_delayed: bool,
) -> None:
    attachment = tmp_path / "preflight.txt"
    attachment.write_text("synthetic", encoding="utf-8")
    page = _FakePage(
        states=[_attachment_state(), _attachment_state(ready={attachment.name: 1})],
        bindings=[_bound_cards(("upload-1", "new-card", attachment.name, "ready", True))],
        responses=[_NamedCreateResponse(attachment.name, "file_preflight")],
    )
    reads = []
    snapshots = []
    tracker = _FakeHandle("preflight")
    async def evaluate(script: str, argument: Any = None) -> Any:
        if "preflight.read()" in script:
            reads.append(True)
            if is_delayed and len(reads) > 1:
                return {"state": "eligible", "reason": "observed-capture-path", "receiver_relationship": "ancestor",
                        "file_count": 1, "item_count": 1, "file_sizes": [9]}
            return {"state": "waiting", "reason": "capture-path-uninitialized"}
        if "preflight.dispatch()" in script:
            assert len(reads) >= 2
            await page.evaluate(attachments.DISPATCH_ATTACHMENT_DROP_JS)
            return {"target": "intended-form", "events": []}
        raise AssertionError("unexpected preflight operation")
    tracker.evaluate = evaluate
    original_evaluate_handle = page.evaluate_handle
    async def evaluate_handle(script: str, argument: Any = None) -> Any:
        if script == attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS:
            page.handles.append(tracker)
            return tracker
        return await original_evaluate_handle(script, argument)
    page.evaluate_handle = evaluate_handle
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)
    if is_delayed:
        assert asyncio.run(attachments.attach_files(page, [str(attachment)], timeout_seconds=0.5, on_diagnostics=snapshots.append)) == {
            attachment.name: (0, 1),
        }
        assert len(reads) >= 2
        assert sum(script == attachments.DISPATCH_ATTACHMENT_DROP_JS for script, _ in page.evaluate_calls) == 1
    else:
        with pytest.raises(RuntimeError, match="preflight"):
            asyncio.run(attachments.attach_files(page, [str(attachment)], timeout_seconds=0.5, on_diagnostics=snapshots.append))
        assert not any(script == attachments.DISPATCH_ATTACHMENT_DROP_JS for script, _ in page.evaluate_calls)
    assert clock.monotonic() <= 0.5
    assert tracker.dispose_calls == 1
    assert page.listeners == {"response": []}
    assert snapshots[-1].preflight_state == ("eligible" if is_delayed else "waiting")
    assert snapshots[-1].dispatch_result == ("returned" if is_delayed else None)


@pytest.mark.parametrize("change", ["handler", "form"])
def test_drop_preflight_rechecks_receiving_contract_between_events(change: str) -> None:
    program = _BINDING_GRAPH_PROGRAM.split("  const fiber =", 1)[0] + r"""
  const assert = require('assert');
  const file = new File(['synthetic'], 'original.txt', {type: 'text/plain'});
  const dataTransfer = {files: [file], items: [{kind: 'file', type: 'text/plain', getAsFile: () => file}]};
  const receiverProps = {onDragEnterCapture() {}, onDragOverCapture() {}, onDropCapture() {}};
  Object.defineProperty(form, '__reactProps$test', {value: {onDropCapture() {}}});
  Object.defineProperty(container, '__reactProps$test', {value: receiverProps});
  const events = [];
  globalThis.DragEvent = class {constructor(type, options) {this.type = type; Object.assign(this, options); this.defaultPrevented = false;}};
  form.dispatchEvent = event => {
    events.push(event.type);
    if (input.change === 'handler') receiverProps.onDropCapture = () => {};
    else form.isConnected = false;
    return true;
  };
  const tracker = createTracker({form, composerSelector: '.composer', dataTransfer, files: [file],
    expected: [{name: file.name, mime: file.type, size: file.size}]});
  assert.equal(tracker.read().state, 'eligible');
  assert.throws(() => tracker.dispatch(), /preflight rejected/);
  assert.deepEqual(events, ['dragenter']);
  assert.throws(() => tracker.dispatch(), /already-attempted/);
  assert.deepEqual(events, ['dragenter']);
})();
"""
    completed = subprocess.run(
        ["node", "-e", program], text=True, capture_output=True, check=False, timeout=10,
        input=json.dumps({"source": attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS, "change": change}),
    )
    assert completed.returncode == 0, completed.stderr


def test_cancellation_during_preflight_disposes_handles_and_observers_without_drop(tmp_path: Path) -> None:
    async def run() -> None:
        path = tmp_path / "cancel.txt"
        path.write_text("synthetic", encoding="utf-8")
        page = _FakePage(states=[_attachment_state()])
        waiting = asyncio.Event()
        original = page.evaluate_handle
        async def evaluate_handle(script: str, argument: Any = None) -> Any:
            handle = await original(script, argument)
            if script == attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS:
                async def evaluate(_script: str, _argument: Any = None) -> Any:
                    waiting.set()
                    await asyncio.Event().wait()
                handle.evaluate = evaluate
            return handle
        page.evaluate_handle = evaluate_handle
        task = asyncio.create_task(attachments.attach_files(page, [str(path)]))
        await asyncio.wait_for(waiting.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(handle.dispose_calls == 1 for handle in page.handles)
        assert page.listeners == {"response": []}
        assert not any(script == attachments.DISPATCH_ATTACHMENT_DROP_JS for script, _ in page.evaluate_calls)
    asyncio.run(run())


def test_preflight_eligibility_cannot_start_dispatch_after_attachment_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "deadline.txt"
    path.write_text("synthetic", encoding="utf-8")
    clock = _FakeClock()
    monkeypatch.setattr(attachments, "_monotonic", clock.monotonic)
    monkeypatch.setattr(attachments, "_sleep", clock.sleep)
    class Page(_FakePage):
        async def evaluate_handle(self, script: str, argument: Any) -> Any:
            handle = await super().evaluate_handle(script, argument)
            if script == attachments.CREATE_ATTACHMENT_DROP_PREFLIGHT_JS:
                original = handle.evaluate
                async def evaluate(expression: str, options: Any = None) -> Any:
                    result = await original(expression, options)
                    if "preflight.read()" in expression:
                        clock.value = 1.0
                    return result
                handle.evaluate = evaluate
            return handle
    page = Page(states=[_attachment_state()], responses=[])
    with pytest.raises(attachments.AttachmentPreflightError, match="deadline"):
        asyncio.run(attachments.attach_files(page, [str(path)], timeout_seconds=0.5))
    assert not any(script == attachments.DISPATCH_ATTACHMENT_DROP_JS for script, _ in page.evaluate_calls)
    assert all(handle.dispose_calls == 1 for handle in page.handles)
