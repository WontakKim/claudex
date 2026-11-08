"""Current-page passive processing-stream authority for an attachment drop."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from claudex.gptpro.attachment_diagnostics import FILE_PROCESSING_PATH
from claudex.gptpro.conversation import is_trusted_origin_url

logger = logging.getLogger(__name__)


def _is_processing(url: str, method: str = "POST") -> bool:
    return (
        method == "POST" and is_trusted_origin_url(url)
        and urlsplit(url).path.rstrip("/") == FILE_PROCESSING_PATH
    )


class ProcessingStreamObserver:
    """CDP IDs own transport; public responses only guard observation coverage."""

    def __init__(
        self, session: Any, on_event: Callable[[Any], None], byte_limit: int, request_limit: int,
    ) -> None:
        self.session = session
        self.on_event = on_event
        self.byte_limit = byte_limit
        self.request_limit = request_limit
        self.is_armed = False
        self.failure: str | None = None
        self.requests: dict[str, dict[str, Any]] = {}
        self.public_requests: dict[int, Any] = {}
        self.public_starts: dict[int, Any] = {}
        self.public_aborts: dict[int, Any] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self.handlers = [
            ("Network.requestWillBeSent", self.on_request),
            ("Network.responseReceived", self.on_response),
            ("Network.dataReceived", self.on_data),
            ("Network.loadingFinished", self.on_finished),
            ("Network.loadingFailed", self.on_failed),
            ("close", self.on_close),
        ]

    @classmethod
    async def open(
        cls, page: Any, on_event: Callable[[Any], None], byte_limit: int, request_limit: int,
    ) -> ProcessingStreamObserver | None:
        create_session = getattr(getattr(page, "context", None), "new_cdp_session", None)
        if not callable(create_session):
            return None
        observer = None
        try:
            # This session owns only the current page target. Public page events
            # guard against processing outside that coverage (for example OOPIFs).
            session = await create_session(page)
            observer = cls(session, on_event, byte_limit, request_limit)
            for event, handler in observer.handlers:
                session.on(event, handler)
            await session.send("Network.enable")
            return observer
        except asyncio.CancelledError:
            if observer is not None:
                await observer.close()
            raise
        except Exception as exc:
            if observer is not None:
                await observer.close()
            logger.warning("Processing CDP unavailable (%s); using response completion observation", type(exc).__name__)
            return None

    def on_close(self, *_: Any) -> None:
        self.failure = "processing stream coverage lost"

    def arm(self) -> None:
        self.is_armed = True

    def on_request(self, event: dict[str, Any]) -> None:
        state = self.requests.get(event["requestId"])
        if state is not None and state["is_current"] and "redirectResponse" in event:
            self.failure = "processing stream redirected outside its request scope"
            return
        request = event["request"]
        if not _is_processing(request["url"]):
            return
        identifier = event["requestId"]
        if identifier in self.requests:
            return
        if len(self.requests) >= self.request_limit:
            self.failure = "processing stream request limit exceeded"
            return
        self.requests[identifier] = {
            "is_current": self.is_armed and request["method"] == "POST", "has_response": False, "is_seeded": False,
            "is_terminal": False, "files": set(), "completed": set(), "has_failed": False, "is_aborted": False,
            "bytes": 0, "buffer": b"", "pending": b"",
        }

    def on_response(self, event: dict[str, Any]) -> None:
        state = self.requests.get(event["requestId"])
        if state is None:
            if self.is_armed and _is_processing(event["response"]["url"]):
                self.failure = "processing stream response without observed start"
            return
        if not state["is_current"] or state["has_response"]:
            return
        state["has_response"] = True
        if not 200 <= event["response"]["status"] < 300:
            self.failure = "processing stream HTTP failure"
            return
        task = asyncio.create_task(self.subscribe(event["requestId"], state))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def subscribe(self, identifier: str, state: dict[str, Any]) -> None:
        try:
            result = await self.session.send("Network.streamResourceContent", {"requestId": identifier})
            # CDP enables dataReceived before replying. Seed earlier buffered
            # bytes first, then consume chunks queued while the reply was pending.
            self.append(state, self.decode(state, result["bufferedData"]))
            state["is_seeded"] = True
            self.append(state, state["pending"])
            state["pending"] = b""
            if state["is_terminal"]:
                self.finish_buffer(state)
        except Exception:
            self.failure = "processing stream observation failed"

    def decode(self, state: dict[str, Any], encoded: str) -> bytes:
        if not isinstance(encoded, str) or len(encoded) > 4 * ((self.byte_limit + 2) // 3):
            raise ValueError("oversized or invalid chunk")
        data = base64.b64decode(encoded, validate=True)
        state["bytes"] += len(data)
        if state["bytes"] > self.byte_limit:
            raise ValueError("oversized stream")
        return data

    def append(self, state: dict[str, Any], data: bytes) -> None:
        state["buffer"] += data
        while b"\n" in state["buffer"]:
            line, state["buffer"] = state["buffer"].split(b"\n", 1)
            self.record_line(state, line)

    def record_line(self, state: dict[str, Any], line: bytes) -> None:
        if not line.strip():
            return
        event = json.loads(line)
        self.on_event(event)
        state["files"].add(event["file_id"])
        if event["event"] == "file.processing.completed":
            state["completed"].add(event["file_id"])
        elif event["event"].endswith((".failed", ".error")):
            state["has_failed"] = True

    def on_data(self, event: dict[str, Any]) -> None:
        state = self.requests.get(event["requestId"])
        if state is None or not state["is_current"] or self.failure:
            return
        if "data" not in event:
            if state["is_seeded"] and event.get("dataLength", 0) > 0:
                self.failure = "processing stream bytes unavailable"
            return
        if state["is_terminal"]:
            self.failure = "processing stream data after terminal event"
            return
        try:
            data = self.decode(state, event["data"])
            if state["is_seeded"]:
                self.append(state, data)
            else:
                state["pending"] += data
        except Exception:
            self.failure = "processing stream observation failed"

    def finish_buffer(self, state: dict[str, Any]) -> None:
        if state["buffer"].strip():
            try:
                self.record_line(state, state["buffer"])
            except Exception:
                self.failure = "processing stream observation failed"
        state["buffer"] = b""

    def on_finished(self, event: dict[str, Any]) -> None:
        state = self.requests.get(event["requestId"])
        if state is not None and state["is_current"]:
            self.finish_buffer(state)
            state["is_terminal"] = True

    def on_failed(self, event: dict[str, Any]) -> None:
        state = self.requests.get(event["requestId"])
        if state is None or not state["is_current"]:
            return
        self.finish_buffer(state)
        state["is_terminal"] = True
        if not (event.get("canceled") is True and event.get("errorText") == "net::ERR_ABORTED"
                and not event.get("blockedReason") and not event.get("corsErrorStatus")
                and state["is_seeded"] and not state["has_failed"] and state["files"] and state["files"] == state["completed"]):
            self.failure = "processing stream transport failed"
        else:
            state["is_aborted"] = True

    def observe_public_request(self, request: Any) -> None:
        if len(self.public_starts) >= self.request_limit and id(request) not in self.public_starts:
            self.failure = "processing public request limit exceeded"
            return
        self.public_starts[id(request)] = request

    def observe_public_failure(self, request: Any) -> None:
        try:
            if (self.public_requests.get(id(request)) is not request
                    and not _is_processing(request.url, request.method)):
                return
            if request.failure != "net::ERR_ABORTED":
                self.failure = "processing public request transport failed"
                return
        except (AttributeError, TypeError, ValueError):
            self.failure = "processing public request failure unavailable"
            return
        if len(self.public_aborts) >= self.request_limit and id(request) not in self.public_aborts:
            self.failure = "processing public failure limit exceeded"
            return
        self.public_aborts[id(request)] = request

    def observe_public_response(self, response: Any) -> bool:
        request = response.request
        if self.public_requests.get(id(request)) is request:
            return False
        if len(self.public_requests) >= self.request_limit:
            self.failure = "processing public response limit exceeded"
            return False
        self.public_requests[id(request)] = request
        return True

    @property
    def pending_response_count(self) -> int:
        return sum(
            state["is_current"] and state["has_response"]
            and (not state["is_terminal"] or not state["is_seeded"])
            for state in self.requests.values()
        )

    def blocker(self, receipt_ids: set[str]) -> str | None:
        if self.failure:
            return self.failure
        current = [state for state in self.requests.values() if state["is_current"]]
        # Counts guard coverage only. File identity comes exclusively from parsed
        # CDP frames reconciled with create receipts and current composer cards.
        if (not self.public_starts.keys() <= self.public_requests.keys()
                or len(self.public_requests) != sum(state["has_response"] for state in current)
                or len(self.public_aborts) != sum(state["is_aborted"] for state in current)
                or self.tasks):
            return "processing stream coverage pending"
        for state in current:
            if not state["is_terminal"] or not state["is_seeded"]:
                return "processing stream pending"
            if state["has_failed"] or not state["files"] or state["files"] != state["completed"] or not state["files"] <= receipt_ids:
                return "processing stream evidence incomplete or unattributed"
        return None

    async def close(self) -> None:
        for event, handler in self.handlers:
            self.session.remove_listener(event, handler)
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        self.requests.clear()
        self.public_requests.clear()
        self.public_starts.clear()
        self.public_aborts.clear()
        try:
            await self.session.detach()
        except Exception as exc:
            logger.warning("Processing CDP detach unavailable (%s); local observer released", type(exc).__name__)
