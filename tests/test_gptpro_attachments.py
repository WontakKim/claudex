"""Tests for ChatGPT Pro plain-text attachment uploads."""

from __future__ import annotations

import asyncio
import base64
import gc
import json
from collections.abc import Callable
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


class _FakePage:
    def __init__(
        self,
        *,
        states: list[dict[str, Any]] | None = None,
        responses: list[Any] | None = None,
        existing_names: list[str] | None = None,
    ) -> None:
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
    assert "file_2" in message and "file_0" in message and "file_1" in message
    assert "unattributed" in message
    assert raised.value.completed_file_create_responses == 3
    assert raised.value.ready_attachments == 1


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
    assert "'first.txt': ready" in str(raised.value).split("file_first")[0]
    assert "file_first" in str(raised.value).split("'second.txt': unknown")[0]
    assert "file_second" in str(raised.value).split("'second.txt': unknown")[1]


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
    assert f"processing file_effect: {expected_processing}" in message


def test_timeout_distinguishes_unobserved_and_incomplete_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = []
    for name in ("unobserved.ts", "incomplete.ts"):
        path = tmp_path / name
        path.write_text("synthetic", encoding="utf-8")
        paths.append(str(path))
    page = _FakePage(states=[
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
        "processing file_unobserved: observed=no, completion=no, "
        "indexed display name=no, missing=not required without observation"
    ) in message
    assert (
        "processing file_incomplete: observed=yes, completion=no, "
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
    page = _FakePage(states=[
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
    assert "processing file_effect: observed=no" in message
    assert page.listeners == {"response": []}
    assert all(handle.dispose_calls == 1 for handle in page.handles)


def test_server_renamed_attachment_settles_under_authoritative_display_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original = tmp_path / "effect.ts"
    original.write_text("synthetic", encoding="utf-8")
    renamed = "effect(20260927-120843).ts"
    page = _FakePage(
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
    assert "file_effect" in str(raised.value)


def test_preexisting_renamed_chip_is_not_new_upload_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original = tmp_path / "effect.ts"
    original.write_text("synthetic", encoding="utf-8")
    renamed = "effect(20260927-120843).ts"
    page = _FakePage(states=[
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
        "processing file_effect: observed=yes, completion=yes, "
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
    assert "file_effect" in str(raised.value)

    message = str(raised.value)
    for private_value in (
        "private attachment contents", "synthetic request credential",
        "private event detail", "file.processing.failed",
    ):
        assert private_value not in message
    assert (
        "processing file_effect: observed=yes, completion=no, "
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

    with pytest.raises(attachments.AttachmentUploadFailedError, match="file_reused"):
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
                    and self.evaluate_handle_calls):
                self.changed = True
                for listener in tuple(self.listeners["response"]):
                    listener(stream)
                await asyncio.sleep(0)
                return _attachment_state(ready={"effect.ts": 1})
            return await super().evaluate(script, argument)

    page = ChangingPage(states=[
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
    page = _FakePage(states=[
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
    page = _FakePage(states=[
        _attachment_state(),
        *[_attachment_state(ready={"same.txt": 1})] * 3,
    ])
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
    page = _FakePage(states=[
        _attachment_state(ready={"existing.txt": 1}),
        _attachment_state(ready={"existing.txt": 2}),
    ])

    expected_ready_counts = asyncio.run(
        attachments.attach_files(page, [str(attachment)])
    )

    assert expected_ready_counts == {"existing.txt": (1, 1)}


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
