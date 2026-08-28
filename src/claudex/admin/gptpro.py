"""GPT Pro session, login, doctor, and MCP connection admin handlers."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import signal
import sys
import time
from pathlib import Path
from typing import cast

from starlette.requests import Request
from starlette.responses import JSONResponse

from claudex import server_support
from claudex.admin.common import (
    _admin_guard,
    _read_json_object,
    _require_json_content_type,
)
from claudex.config import GatewayConfig
from claudex.gptpro import session as gptpro_session
from claudex.gptpro.login_session import GptProLoginSession


_GPTPRO_DOCTOR_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "claudex",
    "gptpro",
    "doctor",
)
_GPTPRO_DOCTOR_TIMEOUT = 30.0
_GPTPRO_CONNECT_TIMEOUT = 30.0


async def _handle_admin_gptpro_session(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied

    status = gptpro_session.session_status()
    path = cast(Path, status["path"])
    expires_at: float | None = None
    if status["has_auth_cookie"] is True:
        try:
            expires_at = gptpro_session.load_auth_cookie_expiry(path)
        except gptpro_session.GptProSessionError:
            # The optional session may be replaced between the status and expiry reads.
            status = gptpro_session.session_status()
            path = cast(Path, status["path"])

    expires_in_seconds = (
        max(0, int(expires_at - time.time()))
        if expires_at is not None and expires_at > 0
        else None
    )
    return JSONResponse(
        {
            **status,
            "path": str(path),
            "expires_at": expires_at,
            "expires_in_seconds": expires_in_seconds,
        }
    )


async def _handle_admin_gptpro_login_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied

    session: GptProLoginSession | None = request.app.state.gptpro_login_session
    if session is None:
        return JSONResponse({"status": "idle"})
    return JSONResponse(session.status())


async def _handle_admin_gptpro_login_post(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    denied = _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    if body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unexpected keys: {', '.join(sorted(body))}; POST an empty JSON object",
            ),
            status_code=400,
        )

    session: GptProLoginSession | None = request.app.state.gptpro_login_session
    if session is not None and not session.is_terminal:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "a gptpro login session is already active; poll GET /admin/gptpro/login",
                "login-active",
            ),
            status_code=409,
        )

    runtime = request.app.state.gptpro_ask_runtime
    if runtime.has_active_jobs():
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "cannot start login while a gptpro ask is in progress",
                "asks-active",
            ),
            status_code=409,
        )

    await runtime.release_runtime()
    session = request.app.state.gptpro_login_session
    if session is not None and not session.is_terminal:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "a gptpro login session is already active; poll GET /admin/gptpro/login",
                "login-active",
            ),
            status_code=409,
        )

    session = GptProLoginSession()
    session.start()
    request.app.state.gptpro_login_session = session
    return JSONResponse(session.status(), status_code=201)


async def _handle_admin_gptpro_login_delete(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied

    session: GptProLoginSession | None = request.app.state.gptpro_login_session
    if session is None:
        return JSONResponse({"status": "idle"})
    if session.is_terminal:
        request.app.state.gptpro_login_session = None
        return JSONResponse({"status": "idle"})
    session.request_cancel()
    return JSONResponse({"status": "cancelling"})


def _gptpro_mcp_endpoint(config: GatewayConfig) -> str:
    host = "127.0.0.1" if config.host in {"0.0.0.0", "::"} else config.host
    try:
        endpoint_host = f"[{host}]" if ipaddress.ip_address(host).version == 6 else host
    except ValueError:
        endpoint_host = host
    return f"http://{endpoint_host}:{config.port}/mcp"


def _build_gptpro_connect_command(
    endpoint: str, token: str | None
) -> tuple[str, ...]:
    command = (
        "claude",
        "mcp",
        "add",
        "--transport",
        "http",
        "-s",
        "user",
        "claudex-gptpro",
        endpoint,
    )
    if token is not None:
        command += ("--header", f"Authorization: Bearer {token}")
    return command


async def _handle_admin_gptpro_doctor(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    denied = _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    if body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unexpected keys: {', '.join(sorted(body))}; POST an empty JSON object",
            ),
            status_code=400,
        )

    try:
        process = await asyncio.create_subprocess_exec(
            *_GPTPRO_DOCTOR_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        return JSONResponse(
            {
                "ok": False,
                "exit_code": None,
                "output": f"doctor failed to run: {exc}",
            }
        )

    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(), _GPTPRO_DOCTOR_TIMEOUT
        )
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.kill()
        await process.communicate()
        return JSONResponse(
            {
                "ok": False,
                "exit_code": None,
                "output": "doctor failed to run: "
                f"timed out after {_GPTPRO_DOCTOR_TIMEOUT:g}s",
            }
        )

    output = stdout.decode("utf-8", errors="replace")
    return JSONResponse(
        {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "output": output,
        }
    )


async def _handle_admin_gptpro_mcp(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied

    config: GatewayConfig = request.app.state.config
    return JSONResponse(
        {
            "endpoint": _gptpro_mcp_endpoint(config),
            "auth_required": config.local_token is not None,
        }
    )


async def _handle_admin_gptpro_connect(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    denied = _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    if body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unexpected keys: {', '.join(sorted(body))}; POST an empty JSON object",
            ),
            status_code=400,
        )

    config: GatewayConfig = request.app.state.config
    command = _build_gptpro_connect_command(
        _gptpro_mcp_endpoint(config), config.local_token
    )
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        return JSONResponse(
            {
                "ok": False,
                "exit_code": None,
                "output": f"connect failed to run: {exc}; install the Claude Code CLI "
                "and ensure `claude` is on PATH",
            }
        )

    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(), _GPTPRO_CONNECT_TIMEOUT
        )
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.kill()
        await process.communicate()
        return JSONResponse(
            {
                "ok": False,
                "exit_code": None,
                "output": "connect failed to run: "
                f"timed out after {_GPTPRO_CONNECT_TIMEOUT:g}s",
            }
        )

    output = stdout.decode("utf-8", errors="replace")
    return JSONResponse(
        {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "output": output,
        }
    )
