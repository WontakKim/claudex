"""Bounded metadata for one attachment attempt, independent of send eligibility."""

from __future__ import annotations

import inspect
import logging
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

from claudex import __version__
from claudex.gptpro.conversation import is_trusted_origin_url

FILE_CREATE_PATH = "/backend-api/files"
FILE_PROCESSING_PATH = "/backend-api/files/process_upload_stream"
MAX_DIAGNOSTIC_EVENTS = 64
MAX_DIAGNOSTIC_REQUESTS = 40

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttachmentDiagnosticEvent:
    event: str
    elapsed_ms: int
    phase: str
    endpoint: str | None = None
    request_id: int | None = None
    # None means no request-start event can be attributed, not that no request was sent.
    request_started_before_drop: bool | None = None
    status: int | None = None


@dataclass(frozen=True)
class AttachmentNetworkSummary:
    """Counters survive truncation of the detailed event and request histories."""

    requests_started: int = 0
    requests_before_drop: int = 0
    responses_received: int = 0
    responses_without_attributed_start: int = 0
    requests_finished: int = 0
    requests_failed: int = 0
    non_success_responses: int = 0
    last_non_success_status: int | None = None
    last_non_success_started_before_drop: bool | None = None


@dataclass(frozen=True)
class AttachmentDiagnostics:
    """Attachment-stage observations, not proof of application acceptance or send."""

    phase: str = "baseline"
    outcome: str = "pending"
    file_sizes: tuple[int, ...] = ()
    timeout_ms: int = 0
    elapsed_ms: int = 0
    drop_started_ms: int | None = None
    drop_returned_ms: int | None = None
    poll_attempts: int = 0
    poll_completions: int = 0
    last_poll_ms: int | None = None
    form_current: bool | None = None
    binding_valid: bool | None = None
    binding_conflict: bool | None = None
    observer_active: bool = False
    preflight_state: str | None = None
    preflight_reason: str | None = None
    receiver_relationship: str | None = None
    payload_file_count: int | None = None
    payload_item_count: int | None = None
    payload_file_sizes: tuple[int, ...] = ()
    dispatch_result: str | None = None
    dispatch_events: tuple[tuple[str, bool, bool], ...] = ()
    request_observation_scope: str = "this-page-trusted-create-and-processing-posts"
    pre_submit_result: str | None = None
    create_network: AttachmentNetworkSummary = field(default_factory=AttachmentNetworkSummary)
    processing_network: AttachmentNetworkSummary = field(default_factory=AttachmentNetworkSummary)
    events: tuple[AttachmentDiagnosticEvent, ...] = ()
    events_dropped: int = 0
    untracked_request_events: int = 0
    gateway_version: str = __version__
    playwright_version: str | None = None
    browser_version: str | None = None
    python_version: str = platform.python_version()


@lru_cache(maxsize=1)
def _playwright_version() -> str | None:
    try:
        return version("playwright")
    except PackageNotFoundError:
        return None


def _endpoint(url: str, method: str) -> str | None:
    if method != "POST" or not is_trusted_origin_url(url):
        return None
    return {
        FILE_CREATE_PATH: "create",
        FILE_PROCESSING_PATH: "processing",
    }.get(urlsplit(url).path.rstrip("/"))


class AttachmentDiagnosticObserver:
    """Observe only this page's upload endpoints; never read payloads or headers."""

    def __init__(
        self, page: Any, *, file_sizes: tuple[int, ...], timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        started_at: float | None = None,
        on_update: Callable[[AttachmentDiagnostics], None] | None = None,
    ) -> None:
        self._page = page
        self._clock = clock
        self._started = clock() if started_at is None else started_at
        self._on_update = on_update
        self._events: list[AttachmentDiagnosticEvent] = []
        # Keep bounded strong references so request identities cannot be reused.
        self._requests: dict[int, tuple[Any, int, bool | None]] = {}
        self._listeners: list[tuple[str, Any]] = []
        self._closed = False
        browser_version = None
        try:
            browser = getattr(getattr(page, "context", None), "browser", None)
            browser_version = getattr(browser, "version", None)
        except Exception:
            logger.warning("attachment browser version unavailable; continuing without version metadata")
        self._state = AttachmentDiagnostics(
            file_sizes=file_sizes, timeout_ms=round(timeout_seconds * 1000),
            playwright_version=_playwright_version(),
            browser_version=browser_version[:128] if isinstance(browser_version, str) else None,
        )

    def _elapsed_ms(self) -> int:
        return max(0, round((self._clock() - self._started) * 1000))

    def snapshot(self) -> AttachmentDiagnostics:
        return replace(self._state, elapsed_ms=self._elapsed_ms(), events=tuple(self._events))

    def _report(self) -> None:
        if self._on_update is None:
            return
        try:
            self._on_update(self.snapshot())
        except Exception:
            logger.warning("attachment diagnostic observer failed; upload checks remain unchanged")

    def _record(
        self, event: str, *, endpoint: str | None = None,
        request_id: int | None = None, request_started_before_drop: bool | None = None,
        status: int | None = None,
    ) -> None:
        if self._closed:
            return
        if len(self._events) < MAX_DIAGNOSTIC_EVENTS:
            self._events.append(AttachmentDiagnosticEvent(
                event, self._elapsed_ms(), self._state.phase, endpoint,
                request_id, request_started_before_drop, status,
            ))
        else:
            self._state = replace(self._state, events_dropped=self._state.events_dropped + 1)
        self._report()

    def set_phase(self, phase: str) -> None:
        if self._state.phase == phase:
            return
        self._state = replace(self._state, phase=phase)
        self._record("phase")

    def record_preflight(self, snapshot: Any) -> None:
        if not isinstance(snapshot, dict):
            return
        state = snapshot.get("state")
        reason = snapshot.get("reason")
        state = state if state in ("waiting", "blocked", "eligible") else "unavailable"
        reason = reason if reason in (
            "capture-path-uninitialized", "observed-capture-path", "payload-invalid", "form-changed",
            "descendant-limit", "ancestor-limit", "ambiguous-capture-path", "capture-path-changed",
            "handler-metadata-unavailable",
        ) else "unavailable"
        changed = (state, reason) != (self._state.preflight_state, self._state.preflight_reason)
        values: dict[str, Any] = {"preflight_state": state, "preflight_reason": reason}
        if snapshot.get("receiver_relationship") == "ancestor":
            values["receiver_relationship"] = "ancestor"
        for key, field_name in (("file_count", "payload_file_count"), ("item_count", "payload_item_count")):
            value = snapshot.get(key)
            if type(value) is int and 0 <= value <= 10:
                values[field_name] = value
        sizes = snapshot.get("file_sizes")
        if isinstance(sizes, list) and len(sizes) <= 10 and all(type(size) is int and size >= 0 for size in sizes):
            values["payload_file_sizes"] = tuple(sizes)
        self._state = replace(self._state, **values)
        if changed:
            self._record("preflight")

    def record_dispatch_result(self, outcome: Any) -> None:
        events = outcome.get("events") if isinstance(outcome, dict) else None
        is_valid = isinstance(events, list) and len(events) <= 3 and all(
            isinstance(event, dict) and event.get("type") in ("dragenter", "dragover", "drop") and
            type(event.get("dispatch_returned")) is bool and type(event.get("default_prevented")) is bool
            for event in events
        )
        self._state = replace(
            self._state, dispatch_result="returned" if is_valid else "unavailable",
            dispatch_events=tuple((event["type"], event["dispatch_returned"], event["default_prevented"])
                                  for event in events) if is_valid else (),
        )
        self._record("dispatch_result")

    def mark_drop_started(self) -> None:
        self._state = replace(self._state, drop_started_ms=self._elapsed_ms())
        self._record("drop_started")

    def mark_drop_returned(self) -> None:
        self._state = replace(self._state, drop_returned_ms=self._elapsed_ms())
        self._record("drop_returned")

    def mark_form_current(self, is_current: bool) -> None:
        self._state = replace(self._state, form_current=is_current)
        if not is_current:
            self._record("form_not_current")

    def mark_poll_started(self) -> None:
        self._state = replace(self._state, poll_attempts=self._state.poll_attempts + 1)
        self._report()

    def mark_poll_completed(self) -> None:
        self._state = replace(
            self._state, poll_completions=self._state.poll_completions + 1,
            last_poll_ms=self._elapsed_ms(),
        )
        self._report()

    def mark_binding(self, snapshot: Any) -> None:
        if isinstance(snapshot, dict):
            self._state = replace(
                self._state, binding_valid=snapshot.get("valid") is True,
                binding_conflict=snapshot.get("conflict") is True,
            )
        self._report()

    def record_pre_submit_result(self, blocker: str | None) -> None:
        self._state = replace(
            self._state, pre_submit_result=blocker or "ready",
            outcome="blocked" if blocker is not None else "settled",
        )
        self._record("pre_submit_blocked" if blocker is not None else "pre_submit_ready")

    def finish(self, outcome: str) -> None:
        self._state = replace(self._state, outcome=outcome)
        self._record(outcome)

    def _request_identity(self, request: Any, event: str) -> tuple[int | None, bool | None]:
        key = id(request)
        existing = self._requests.get(key)
        if existing is None:
            if len(self._requests) >= MAX_DIAGNOSTIC_REQUESTS:
                self._state = replace(
                    self._state, untracked_request_events=self._state.untracked_request_events + 1,
                )
                return None, None
            existing = (request, len(self._requests) + 1, None)
        before_drop = existing[2]
        if event == "request":
            before_drop = self._state.drop_started_ms is None
        self._requests[key] = (request, existing[1], before_drop)
        return existing[1], before_drop

    def _capture(self, event: str, request: Any, url: str, status: int | None = None) -> None:
        if self._closed:
            return
        endpoint = _endpoint(url, request.method)
        if endpoint is None:
            return
        request_id, before_drop = self._request_identity(request, event)
        summary_field = "create_network" if endpoint == "create" else "processing_network"
        summary = getattr(self._state, summary_field)
        if event == "request":
            summary = replace(
                summary, requests_started=summary.requests_started + 1,
                requests_before_drop=summary.requests_before_drop + (self._state.drop_started_ms is None),
            )
        elif event == "response":
            summary = replace(
                summary, responses_received=summary.responses_received + 1,
                responses_without_attributed_start=summary.responses_without_attributed_start + (before_drop is None),
            )
            if type(status) is int and not 200 <= status < 300:
                summary = replace(
                    summary, non_success_responses=summary.non_success_responses + 1,
                    last_non_success_status=status,
                    last_non_success_started_before_drop=before_drop,
                )
        elif event == "requestfinished":
            summary = replace(summary, requests_finished=summary.requests_finished + 1)
        elif event == "requestfailed":
            summary = replace(summary, requests_failed=summary.requests_failed + 1)
        self._state = replace(self._state, **{summary_field: summary})
        self._record(
            event, endpoint=endpoint, request_id=request_id,
            request_started_before_drop=before_drop,
            status=status if type(status) is int else None,
        )

    def _on_request(self, event: str, request: Any) -> None:
        try:
            self._capture(event, request, request.url)
        except Exception:
            self._record("network_observation_error")

    def on_response(self, response: Any) -> None:
        try:
            self._capture("response", response.request, response.url, response.status)
        except Exception:
            self._record("network_observation_error")

    def _on_navigation(self, frame: Any) -> None:
        if frame is self._page.main_frame:
            self._record("navigation")

    def install(self) -> None:
        callbacks = [
            ("request", lambda request: self._on_request("request", request)),
            ("requestfinished", lambda request: self._on_request("requestfinished", request)),
            ("requestfailed", lambda request: self._on_request("requestfailed", request)),
            ("framenavigated", self._on_navigation),
            ("close", lambda *_: self._record("close")),
            ("crash", lambda *_: self._record("crash")),
        ]
        for event, callback in callbacks:
            try:
                self._page.on(event, callback)
            except Exception:
                self._record("listener_registration_failed")
                logger.warning("attachment diagnostic listener registration failed for %s", event)
            else:
                self._listeners.append((event, callback))
        self._state = replace(self._state, observer_active=len(self._listeners) == len(callbacks))
        self._record("listeners_installed")

    async def close(self) -> None:
        if self._closed:
            return
        for event, callback in self._listeners:
            try:
                remove = getattr(self._page, "remove_listener", None) or self._page.off
                result = remove(event, callback)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                self._record("listener_cleanup_failed")
                logger.warning("attachment diagnostic listener cleanup failed for %s", event)
        self._listeners.clear()
        self._requests.clear()
        self._state = replace(self._state, observer_active=False)
        self._record("listeners_removed")
        self._closed = True
