"""Metadata-only attachment observation without a browser or network."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest

from claudex.gptpro.attachment_diagnostics import (
    MAX_DIAGNOSTIC_EVENTS,
    MAX_DIAGNOSTIC_REQUESTS,
    AttachmentDiagnosticObserver,
)


class _Page:
    def __init__(self) -> None:
        self.listeners: dict[str, list[Any]] = {}
        self.main_frame = object()

    def on(self, event: str, listener: Any) -> None:
        self.listeners.setdefault(event, []).append(listener)

    def remove_listener(self, event: str, listener: Any) -> None:
        self.listeners[event].remove(listener)

    def emit(self, event: str, value: Any = None) -> None:
        for listener in tuple(self.listeners.get(event, [])):
            listener(value)


def _request(**changes: Any) -> Any:
    values = {
        "method": "POST",
        "url": "https://chatgpt.com/backend-api/files?private=never-record",
        "post_data": "private content",
        "headers": {"Authorization": "private credential"},
        "failure": "private network error text",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_request_lifecycle_preserves_rejection_without_sensitive_data() -> None:
    page = _Page()
    snapshots = []
    clock = [0.0]
    observer = AttachmentDiagnosticObserver(
        page, clock=lambda: clock[0], timeout_seconds=120,
        file_sizes=(1050000,), on_update=snapshots.append,
    )
    observer.install()
    observer.set_phase("dispatch")
    observer.mark_drop_started()
    request = _request()
    page.emit("request", request)
    clock[0] = 0.25
    observer.on_response(SimpleNamespace(request=request, url=request.url, status=429))
    page.emit("requestfinished", request)
    observer.mark_drop_returned()
    observer.finish("timeout")
    asyncio.run(observer.close())

    snapshot = snapshots[-1]
    events = [event for event in snapshot.events if event.request_id is not None]
    assert [event.event for event in events] == ["request", "response", "requestfinished"]
    assert {event.request_id for event in events} == {1}
    assert events[1].status == 429
    assert events[1].elapsed_ms == 250
    assert events[1].request_started_before_drop is False
    assert snapshot.drop_returned_ms == 250
    assert snapshot.outcome == "timeout"
    assert snapshot.file_sizes == (1050000,)
    assert not snapshot.observer_active
    assert all(not listeners for listeners in page.listeners.values())
    serialized = json.dumps(asdict(snapshot))
    for secret in ("never-record", "private content", "private credential", "private network error text", "https://"):
        assert secret not in serialized


def test_response_without_observed_start_is_not_claimed_as_dispatched() -> None:
    observer = AttachmentDiagnosticObserver(_Page(), file_sizes=(1,), timeout_seconds=1)
    request = _request()
    observer.on_response(SimpleNamespace(request=request, url=request.url, status=201))
    event = observer.snapshot().events[-1]
    assert event.event == "response"
    assert event.request_started_before_drop is None


def test_pre_drop_request_and_transport_failure_remain_distinct() -> None:
    page = _Page()
    observer = AttachmentDiagnosticObserver(page, file_sizes=(1,), timeout_seconds=1)
    observer.install()
    request = _request()
    page.emit("request", request)
    observer.mark_drop_started()
    observer.on_response(SimpleNamespace(request=request, url=request.url, status=503))
    page.emit("requestfailed", request)
    events = observer.snapshot().events
    assert events[-1].event == "requestfailed"
    assert events[-1].request_started_before_drop is True
    assert events[-2].status == 503
    asyncio.run(observer.close())


def test_unrelated_endpoints_and_subframe_navigation_are_not_recorded() -> None:
    page = _Page()
    observer = AttachmentDiagnosticObserver(page, file_sizes=(1,), timeout_seconds=1)
    observer.install()
    baseline = len(observer.snapshot().events)
    for request in (
        _request(method="GET"),
        _request(url="https://chatgpt.com/backend-api/conversation"),
        _request(url="https://chatgpt.com.evil.test/backend-api/files"),
    ):
        page.emit("request", request)
        observer.on_response(SimpleNamespace(request=request, url=request.url, status=403))
    page.emit("framenavigated", object())
    assert len(observer.snapshot().events) == baseline
    page.emit("framenavigated", page.main_frame)
    page.emit("crash")
    page.emit("close")
    assert [event.event for event in observer.snapshot().events[-3:]] == ["navigation", "crash", "close"]
    asyncio.run(observer.close())


def test_event_and_request_retention_are_bounded() -> None:
    page = _Page()
    observer = AttachmentDiagnosticObserver(page, file_sizes=(1,), timeout_seconds=1)
    observer.install()
    for _ in range(MAX_DIAGNOSTIC_EVENTS + MAX_DIAGNOSTIC_REQUESTS):
        page.emit("request", _request())
    snapshot = observer.snapshot()
    assert len(snapshot.events) == MAX_DIAGNOSTIC_EVENTS
    assert snapshot.events_dropped > 0
    assert snapshot.untracked_request_events > 0
    assert max(event.request_id or 0 for event in snapshot.events) <= MAX_DIAGNOSTIC_REQUESTS
    asyncio.run(observer.close())


def test_snapshot_is_immutable_and_reports_zero_post_drop_probes() -> None:
    clock = [0.0]
    observer = AttachmentDiagnosticObserver(
        _Page(), file_sizes=(1,), timeout_seconds=1, clock=lambda: clock[0],
    )
    observer.set_phase("baseline")
    before = observer.snapshot()
    clock[0] = 2.0
    observer.mark_drop_started()
    observer.mark_drop_returned()
    observer.set_phase("settlement")
    observer.finish("timeout")
    after = observer.snapshot()
    assert before.phase == "baseline"
    assert before.drop_started_ms is None
    assert after.poll_attempts == after.poll_completions == 0
    assert after.last_poll_ms is None
    assert after.drop_started_ms == 2000
    assert after.elapsed_ms > after.timeout_ms


def test_optional_callback_failure_does_not_escape(caplog: pytest.LogCaptureFixture) -> None:
    def unavailable(_snapshot: Any) -> None:
        raise RuntimeError("private observer failure")

    observer = AttachmentDiagnosticObserver(_Page(), file_sizes=(1,), timeout_seconds=1, on_update=unavailable)
    observer.finish("timeout")
    assert observer.snapshot().outcome == "timeout"
    assert "attachment diagnostic observer failed" in caplog.text
    assert "private observer failure" not in caplog.text


def test_closed_observer_ignores_late_response_callbacks() -> None:
    observer = AttachmentDiagnosticObserver(
        _Page(), file_sizes=(1,), timeout_seconds=1, clock=lambda: 0.0,
    )
    observer.install()
    asyncio.run(observer.close())
    closed = observer.snapshot()
    for _ in range(MAX_DIAGNOSTIC_REQUESTS + 1):
        request = _request()
        observer.on_response(SimpleNamespace(request=request, url=request.url, status=201))
    assert observer.snapshot() == closed


def test_optional_browser_metadata_failure_does_not_block_observation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Page(_Page):
        @property
        def context(self) -> Any:
            raise RuntimeError("private metadata exception")

    observer = AttachmentDiagnosticObserver(Page(), file_sizes=(1,), timeout_seconds=1)
    assert observer.snapshot().browser_version is None
    assert "private metadata exception" not in caplog.text


def test_partial_listener_registration_is_visible_and_cleanup_is_bounded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Page(_Page):
        def on(self, event: str, listener: Any) -> None:
            if event == "request":
                raise RuntimeError("private registration exception")
            super().on(event, listener)

    page = Page()
    observer = AttachmentDiagnosticObserver(page, file_sizes=(1,), timeout_seconds=1)
    observer.install()
    snapshot = observer.snapshot()
    assert not snapshot.observer_active
    assert any(event.event == "listener_registration_failed" for event in snapshot.events)
    asyncio.run(observer.close())
    assert all(not callbacks for callbacks in page.listeners.values())
    assert "private registration exception" not in caplog.text


@pytest.mark.parametrize("endpoint", ["create", "processing"])
def test_network_failure_summary_survives_event_and_request_saturation(endpoint: str) -> None:
    page = _Page()
    observer = AttachmentDiagnosticObserver(page, file_sizes=(1,) * 10, timeout_seconds=120)
    observer.install()
    observer.mark_drop_started()
    url = "https://chatgpt.com/backend-api/files"
    if endpoint == "processing":
        url += "/process_upload_stream"
    for _ in range(MAX_DIAGNOSTIC_EVENTS + 1):
        page.emit("request", _request(url=url))
    request = _request(url=url)
    observer.on_response(SimpleNamespace(request=request, url=url, status=429))
    page.emit("requestfailed", request)
    snapshot = observer.snapshot()
    summary = snapshot.create_network if endpoint == "create" else snapshot.processing_network
    assert len(snapshot.events) == MAX_DIAGNOSTIC_EVENTS
    assert snapshot.events_dropped > 0
    assert summary.requests_started == MAX_DIAGNOSTIC_EVENTS + 1
    assert summary.responses_received == 1
    assert summary.responses_without_attributed_start == 1
    assert summary.non_success_responses == 1
    assert summary.last_non_success_status == 429
    assert summary.requests_failed == 1
    asyncio.run(observer.close())


def test_repeated_phase_does_not_consume_event_capacity() -> None:
    observer = AttachmentDiagnosticObserver(_Page(), file_sizes=(1,), timeout_seconds=1)
    observer.set_phase("pre_submit")
    before = observer.snapshot()
    for _ in range(MAX_DIAGNOSTIC_EVENTS):
        observer.set_phase("pre_submit")
    assert observer.snapshot().events == before.events
    assert observer.snapshot().events_dropped == 0


def test_preflight_and_dispatch_diagnostics_are_bounded_and_sanitized() -> None:
    observer = AttachmentDiagnosticObserver(_Page(), timeout_seconds=120, file_sizes=(9,))
    waiting = {"state": "waiting", "reason": "capture-path-uninitialized"}
    for _ in range(1000):
        observer.record_preflight(waiting)
    observer.record_preflight({"state": "eligible", "reason": "observed-capture-path",
                               "receiver_relationship": "ancestor", "file_count": 1, "item_count": 1,
                               "file_sizes": [9], "private": "PRIVATE_SENTINEL"})
    observer.record_dispatch_result({"target": "intended-form", "events": [
        {"type": event, "dispatch_returned": False, "default_prevented": True}
        for event in ("dragenter", "dragover", "drop")
    ], "private": "PRIVATE_SENTINEL"})
    snapshot = observer.snapshot()
    assert snapshot.preflight_state == "eligible"
    assert snapshot.preflight_reason == "observed-capture-path"
    assert snapshot.receiver_relationship == "ancestor"
    assert snapshot.payload_file_count == snapshot.payload_item_count == 1
    assert snapshot.payload_file_sizes == (9,)
    assert snapshot.dispatch_result == "returned"
    assert snapshot.dispatch_events == (("dragenter", False, True), ("dragover", False, True), ("drop", False, True))
    assert snapshot.request_observation_scope == "this-page-trusted-create-and-processing-posts"
    assert len(snapshot.events) <= 3
    assert "PRIVATE_SENTINEL" not in json.dumps(asdict(snapshot))
    observer.record_preflight({"state": "PRIVATE_SENTINEL", "reason": "PRIVATE_SENTINEL"})
    observer.record_dispatch_result({"events": [{"type": "PRIVATE_SENTINEL"}]})
    assert "PRIVATE_SENTINEL" not in json.dumps(asdict(observer.snapshot()))
