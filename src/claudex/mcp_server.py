"""Lazy streamable HTTP MCP transport for the ChatGPT Pro ask runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from claudex import mcp_tools
from claudex.gptpro import jobs
from claudex.gptpro.ask import AskCallbacks


class LazyAskRuntime:
    """Create the optional ChatGPT Pro runtime on its first background ask."""

    def __init__(self) -> None:
        self._runtime: Any | None = None
        self._job_service: jobs.AskJobService | None = None
        self._thread_registry = jobs.ThreadRegistry()
        self._lock = asyncio.Lock()

    async def ask(
        self,
        question: str,
        *,
        callbacks: AskCallbacks | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> Any:
        runtime = await self._get_runtime()
        return await runtime.ask(
            question,
            callbacks=callbacks,
            conversation_id=conversation_id,
            timeout_seconds=timeout_seconds,
            attachment_paths=attachment_paths,
        )

    def start_ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        on_thread_ref: Callable[[str], None] | None = None,
        attachment_paths: Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> jobs.AskJob:
        if self._job_service is None:
            self._job_service = jobs.AskJobService(self.ask)
        return self._job_service.start(
            question,
            conversation_id=conversation_id,
            on_thread_ref=on_thread_ref,
            attachment_paths=attachment_paths,
            session_id=session_id,
        )

    def lookup_thread(self, session_id: str) -> str | None:
        return self._thread_registry.lookup(session_id)

    def bind_thread(self, session_id: str, thread_ref: str) -> None:
        self._thread_registry.bind(session_id, thread_ref)

    def job_status(self, ask_id: str) -> jobs.AskJob | None:
        if self._job_service is None:
            return None
        return self._job_service.status(ask_id)

    def job_result(self, ask_id: str) -> jobs.AskJob | None:
        if self._job_service is None:
            return None
        return self._job_service.result(ask_id)

    def has_active_jobs(self) -> bool:
        if self._job_service is None:
            return False
        return self._job_service.has_active_jobs()

    async def release_runtime(self) -> None:
        """Release the warm runtime while preserving the job registry.

        Unlike `aclose()`, this leaves `_job_service` intact so existing job
        status and result lookups remain available. The login child needs the
        Chrome profile lock, so the daemon must first give up any warm ask
        runtime that may own it.
        """
        async with self._lock:
            runtime = self._runtime
            self._runtime = None
            if runtime is not None:
                await runtime.aclose()

    async def aclose(self) -> None:
        async with self._lock:
            job_service = self._job_service
            self._job_service = None
            if job_service is not None:
                await job_service.aclose()

            runtime = self._runtime
            self._runtime = None
            if runtime is not None:
                await runtime.aclose()

    async def _get_runtime(self) -> Any:
        async with self._lock:
            if self._runtime is None:
                from claudex.gptpro.runtime import AskRuntime

                self._runtime = AskRuntime()
            return self._runtime


class McpEndpoint:
    """Start and serve the optional MCP transport only after its first request."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._manager: Any | None = None
        self._manager_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._startup_error: BaseException | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive=receive)
        from claudex.admin.common import _admin_guard

        denied = _admin_guard(request)
        if denied is not None:
            await denied(scope, receive, send)
            return

        try:
            manager = await self._get_manager(request.app)
        except ModuleNotFoundError:
            response = JSONResponse(
                {
                    "error": {
                        "message": (
                            "MCP support is not installed; install the gptpro extra"
                        ),
                        "type": "service_unavailable_error",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return
        await manager.handle_request(scope, receive, send)

    async def aclose(self) -> None:
        async with self._lock:
            task = self._manager_task
            stop = self._stop
            if task is None or stop is None:
                return
            stop.set()
        try:
            await task
        finally:
            async with self._lock:
                if self._manager_task is task:
                    self._manager = None
                    self._manager_task = None
                    self._ready = None
                    self._stop = None
                    self._startup_error = None

    async def _get_manager(self, app: Any) -> Any:
        async with self._lock:
            if self._manager_task is None:
                manager = _create_session_manager(app)
                self._manager = manager
                self._ready = asyncio.Event()
                self._stop = asyncio.Event()
                self._manager_task = asyncio.create_task(
                    self._run_manager(manager, self._ready, self._stop)
                )
            ready = self._ready

        assert ready is not None
        await ready.wait()
        if self._startup_error is not None:
            raise self._startup_error
        assert self._manager is not None
        return self._manager

    async def _run_manager(
        self,
        manager: Any,
        ready: asyncio.Event,
        stop: asyncio.Event,
    ) -> None:
        try:
            async with manager.run():
                ready.set()
                await stop.wait()
        except BaseException as exc:
            self._startup_error = exc
            ready.set()
            raise


def _create_session_manager(app: Any) -> Any:
    """Build the MCP transport without importing the SDK at boot."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    server = mcp_tools.build_gptpro_server(app)
    return StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=False,
    )
