"""Subprocess tests for the background start/stop lifecycle, plus in-process
unit tests for the `compact` command family."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from claudex_gateway import __main__ as gateway_main
from claudex_gateway import paths
from claudex_gateway.config import GatewayConfig
from claudex_gateway.locking import try_file_lock

# Serves the given JSON payload on every GET; "__SELF_PID__": true is
# replaced with the fake server's own pid so identity checks can match.
_FAKE_SERVER = """
import http.server, json, os, sys
payload = json.loads(sys.argv[2])
if payload.pop("__SELF_PID__", False):
    payload["pid"] = os.getpid()
body = json.dumps(payload).encode()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _is_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _gateway_env(tmp_path: Path, port: int) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("CLAUDEX_")}
    env["HOME"] = str(tmp_path)
    env["CLAUDEX_PORT"] = str(port)
    return env


def _run_cli(env: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "claudex_gateway", *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _record_file(tmp_path: Path) -> Path:
    return tmp_path / ".claudex" / "gateway.pid"


def _pool_lock_file(tmp_path: Path) -> Path:
    return tmp_path / ".claudex" / "claude-account-pool" / "balanced-router.lock"


def _write_daemon_record(
    tmp_path: Path, pid: int, port: int, nonce: str, host: str = "127.0.0.1"
) -> None:
    runtime_dir = tmp_path / ".claudex"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _record_file(tmp_path).write_text(
        json.dumps({"pid": pid, "host": host, "port": port, "nonce": nonce}) + "\n",
        encoding="utf-8",
    )


def _kill_leaked_daemon(tmp_path: Path) -> None:
    record_file = _record_file(tmp_path)
    if not record_file.exists():
        return
    pid = None
    with contextlib.suppress(ValueError):
        parsed = json.loads(record_file.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            pid = parsed.get("pid")
        elif isinstance(parsed, int):
            pid = parsed
    if isinstance(pid, int):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _fetch_hello(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/hello", timeout=2) as response:
        return json.load(response)


def _spawn_sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_background_start_is_idempotent_and_stop_works(tmp_path: Path) -> None:
    port = _free_port()
    env = _gateway_env(tmp_path, port)
    try:
        started = _run_cli(env)
        assert started.returncode == 0, started.stderr
        assert "started on" in started.stdout
        assert _is_listening(port)
        record_file = _record_file(tmp_path)
        assert record_file.exists()
        assert (tmp_path / ".claudex" / "gateway.log").exists()

        # The record and the running daemon must agree on identity.
        record = json.loads(record_file.read_text(encoding="utf-8"))
        assert set(record) == {"pid", "host", "port", "nonce"}
        assert record["port"] == port
        hello = _fetch_hello(port)
        assert hello["pid"] == record["pid"]
        assert hello["nonce"] == record["nonce"]

        again = _run_cli(env)
        assert again.returncode == 0, again.stderr
        assert "already running" in again.stdout

        stopped = _run_cli(env, "stop")
        assert stopped.returncode == 0, stopped.stderr
        assert "stopped" in stopped.stdout
        assert not record_file.exists()
        for _ in range(50):
            if not _is_listening(port):
                break
            time.sleep(0.1)
        assert not _is_listening(port)
    finally:
        _kill_leaked_daemon(tmp_path)


def _spawn_fake_server(port: int, payload: dict) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [sys.executable, "-c", _FAKE_SERVER, str(port), json.dumps(payload)]
    )
    # Reap immediately on exit: with pytest as the parent the killed fake
    # would linger as a zombie, which os.kill(pid, 0) reports as alive. A
    # real daemon is reparented to launchd, which reaps promptly.
    threading.Thread(target=process.wait, daemon=True).start()
    for _ in range(50):
        if _is_listening(port):
            return process
        time.sleep(0.1)
    process.kill()
    raise RuntimeError("fake server did not start listening")


def test_start_restarts_a_stale_version_daemon(tmp_path: Path) -> None:
    port = _free_port()
    env = _gateway_env(tmp_path, port)
    # Identifies as the gateway but reports no version — an old install.
    fake = _spawn_fake_server(
        port, {"hello": "claudex-gateway", "__SELF_PID__": True, "nonce": "old-nonce"}
    )
    _write_daemon_record(tmp_path, fake.pid, port, "old-nonce")
    try:
        started = _run_cli(env)
        assert started.returncode == 0, started.stderr
        assert "restarting" in started.stdout
        assert "started on" in started.stdout
        assert fake.poll() is not None
        stopped = _run_cli(env, "stop")
        assert stopped.returncode == 0, stopped.stderr
    finally:
        with contextlib.suppress(ProcessLookupError):
            fake.kill()
        _kill_leaked_daemon(tmp_path)


def test_start_refuses_restart_with_legacy_pid_record(tmp_path: Path) -> None:
    port = _free_port()
    env = _gateway_env(tmp_path, port)
    fake = _spawn_fake_server(port, {"hello": "claudex-gateway"})
    runtime_dir = tmp_path / ".claudex"
    runtime_dir.mkdir(parents=True)
    # A pre-0.4 install recorded a bare pid, which cannot be verified.
    _record_file(tmp_path).write_text(f"{fake.pid}\n", encoding="utf-8")
    try:
        result = _run_cli(env)
        assert result.returncode == 1
        assert "cannot restart automatically" in result.stderr
        assert "stop the old gateway manually" in result.stderr
        assert fake.poll() is None
    finally:
        with contextlib.suppress(ProcessLookupError):
            fake.kill()


def test_start_refuses_a_foreign_port_occupant(tmp_path: Path) -> None:
    port = _free_port()
    env = _gateway_env(tmp_path, port)
    fake = _spawn_fake_server(port, {"service": "something-else"})
    try:
        result = _run_cli(env)
        assert result.returncode == 1
        assert "does not look like claudex-gateway" in result.stderr
        assert fake.poll() is None
    finally:
        with contextlib.suppress(ProcessLookupError):
            fake.kill()


def test_stop_refuses_a_nonce_mismatch(tmp_path: Path) -> None:
    port = _free_port()
    fake = _spawn_fake_server(
        port, {"hello": "claudex-gateway", "__SELF_PID__": True, "nonce": "actual-nonce"}
    )
    _write_daemon_record(tmp_path, fake.pid, port, "recorded-nonce")
    try:
        result = _run_cli(_gateway_env(tmp_path, port), "stop")
        assert result.returncode == 1
        assert "does not match the daemon record" in result.stderr
        assert fake.poll() is None
    finally:
        with contextlib.suppress(ProcessLookupError):
            fake.kill()


def test_stop_refuses_a_live_pid_without_a_gateway(tmp_path: Path) -> None:
    # The recorded pid is alive, but nothing answers on the recorded port —
    # exactly what pid reuse looks like. The process must not be signaled.
    sleeper = _spawn_sleeper()
    port = _free_port()
    _write_daemon_record(tmp_path, sleeper.pid, port, "nonce")
    try:
        result = _run_cli(_gateway_env(tmp_path, port), "stop")
        assert result.returncode == 1
        assert "refusing to signal an unverified process" in result.stderr
        assert sleeper.poll() is None
    finally:
        with contextlib.suppress(ProcessLookupError):
            sleeper.kill()
        sleeper.wait()


def test_stop_refuses_a_legacy_integer_record(tmp_path: Path) -> None:
    sleeper = _spawn_sleeper()
    runtime_dir = tmp_path / ".claudex"
    runtime_dir.mkdir(parents=True)
    _record_file(tmp_path).write_text(f"{sleeper.pid}\n", encoding="utf-8")
    try:
        result = _run_cli(_gateway_env(tmp_path, _free_port()), "stop")
        assert result.returncode == 1
        assert "not a daemon identity record" in result.stderr
        assert sleeper.poll() is None
        # Fail closed: the unverifiable record stays for manual cleanup.
        assert _record_file(tmp_path).exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            sleeper.kill()
        sleeper.wait()


def test_stop_refuses_a_corrupt_record(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".claudex"
    runtime_dir.mkdir(parents=True)
    _record_file(tmp_path).write_text("{not json", encoding="utf-8")
    result = _run_cli(_gateway_env(tmp_path, _free_port()), "stop")
    assert result.returncode == 1
    assert "not a daemon identity record" in result.stderr


def test_stop_removes_a_stale_record_for_a_dead_pid(tmp_path: Path) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    port = _free_port()
    _write_daemon_record(tmp_path, dead.pid, port, "nonce")
    result = _run_cli(_gateway_env(tmp_path, port), "stop")
    assert result.returncode == 1
    assert "not running" in result.stderr
    assert not _record_file(tmp_path).exists()


def test_stop_without_running_daemon_fails(tmp_path: Path) -> None:
    result = _run_cli(_gateway_env(tmp_path, _free_port()), "stop")
    assert result.returncode == 1
    assert "not running" in result.stderr


def test_unknown_argument_prints_usage_and_exits_2(tmp_path: Path) -> None:
    result = _run_cli(_gateway_env(tmp_path, _free_port()), "definitely-not-a-subcommand")
    assert result.returncode == 2
    assert "usage: claudex-gateway" in result.stderr


def test_login_subcommand_is_gone(tmp_path: Path) -> None:
    # Provider logins belong to the CLIs (`codex`/`kimi`/`grok` login); the
    # gateway only reuses their credential stores.
    for arguments in (["login"], ["login", "kimi"], ["login", "grok"]):
        result = _run_cli(_gateway_env(tmp_path, _free_port()), *arguments)
        assert result.returncode == 2
        assert "usage: claudex-gateway" in result.stderr


def test_foreground_start_fails_with_the_pinned_message_when_pool_lock_is_held(
    tmp_path: Path,
) -> None:
    # Every routing mode shares one claude account pool lease (T-9); a
    # process already holding it (in-process here, standing in for another
    # gateway) must make a freshly spawned foreground gateway abort startup
    # instead of silently serving alongside it.
    env = _gateway_env(tmp_path, _free_port())
    holder = try_file_lock(_pool_lock_file(tmp_path))
    assert holder is not None
    try:
        result = subprocess.run(
            [sys.executable, "-m", "claudex_gateway", "--foreground"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "balanced-router.lock held" in result.stdout
    finally:
        holder.release()


# ---------------------------------------------------------------------------
# `compact` command family: unit tests around the daemon-aware helpers
# (probe classification, endpoint resolution, URL bracketing, the
# /admin/settings/compaction HTTP client, envelope validation), plus command-level
# tests for show/set/off across the outcome matrix. These run in-process
# with `urllib.request.urlopen` (and, for the outcome matrix, the probe and
# admin helpers themselves) faked, rather than spawning real subprocess
# daemons -- focused unit coverage per the task's own guidance.
# ---------------------------------------------------------------------------

_HELLO_BODY = json.dumps({"hello": "claudex-gateway", "version": "9.9.9"}).encode()


class _FakeResponse:
    """A stand-in for the `http.client.HTTPResponse` context manager."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _stub_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: object | None = None,
    exception: Exception | None = None,
) -> None:
    def _fake(request: object, timeout: float | None = None) -> object:
        if exception is not None:
            raise exception
        return response

    monkeypatch.setattr(gateway_main, "_urlopen_no_redirect", _fake)


class _RecordingOpener:
    """Fake `urlopen` that records the last request it was called with."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.last_request: urllib.request.Request | None = None

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> object:
        self.last_request = request
        return self.response


def _request_url(request: object) -> str:
    return request if isinstance(request, str) else request.full_url


def _stub_full_stack_urlopen(
    monkeypatch: pytest.MonkeyPatch, admin_status: int, admin_body: bytes
) -> None:
    """Fake `urlopen` answering /api/hello with a valid hello and raising a
    real `urllib.error.HTTPError` for the /admin/settings/compaction call, so both the
    probe and the admin client run their real (unmocked) code paths."""

    def _fake(request: object, timeout: float | None = None) -> object:
        url = _request_url(request)
        if url.endswith("/api/hello"):
            return _FakeResponse(200, _HELLO_BODY)
        raise urllib.error.HTTPError(url, admin_status, "error", None, io.BytesIO(admin_body))

    monkeypatch.setattr(gateway_main, "_urlopen_no_redirect", _fake)


# --- URL bracketing (IPv6-safe URL construction) ---------------------------


def test_bracket_host_leaves_ipv4_and_hostnames_unbracketed() -> None:
    assert gateway_main._bracket_host("127.0.0.1") == "127.0.0.1"
    assert gateway_main._bracket_host("example.com") == "example.com"


def test_bracket_host_brackets_ipv6_literals() -> None:
    assert gateway_main._bracket_host("::1") == "[::1]"


def test_http_url_formats_ipv6_endpoint_with_brackets() -> None:
    assert gateway_main._http_url("::1", 8787, "/api/hello") == "http://[::1]:8787/api/hello"


def test_http_url_formats_ipv4_endpoint_without_brackets() -> None:
    assert (
        gateway_main._http_url("127.0.0.1", 8787, "/api/hello")
        == "http://127.0.0.1:8787/api/hello"
    )


# --- Endpoint resolution: daemon record vs. foreground/no-record discovery -


def _write_record(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields) + "\n", encoding="utf-8")


def test_probe_endpoint_uses_the_daemon_record_when_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_file = tmp_path / "gateway.pid"
    monkeypatch.setattr(paths, "daemon_record_file", lambda: record_file)
    _write_record(record_file, pid=123, host="10.0.0.5", port=9999, nonce="abc")
    config = GatewayConfig(host="127.0.0.1", port=8787, settings_file=tmp_path / "settings.json")

    assert gateway_main._probe_endpoint(config) == ("10.0.0.5", 9999)


def test_probe_endpoint_falls_back_to_config_when_no_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No record file exists: a foreground daemon without a record must
    # remain discoverable via the resolved config host/port.
    monkeypatch.setattr(paths, "daemon_record_file", lambda: tmp_path / "gateway.pid")
    config = GatewayConfig(host="192.168.1.5", port=9001, settings_file=tmp_path / "settings.json")

    assert gateway_main._probe_endpoint(config) == ("192.168.1.5", 9001)


def test_probe_endpoint_treats_a_corrupt_record_as_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_file = tmp_path / "gateway.pid"
    monkeypatch.setattr(paths, "daemon_record_file", lambda: record_file)
    record_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.write_text("42\n", encoding="utf-8")  # legacy bare-pid record
    config = GatewayConfig(host="127.0.0.1", port=8787, settings_file=tmp_path / "settings.json")

    assert gateway_main._probe_endpoint(config) == ("127.0.0.1", 8787)


def test_probe_endpoint_converts_wildcard_record_host_via_connect_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_file = tmp_path / "gateway.pid"
    monkeypatch.setattr(paths, "daemon_record_file", lambda: record_file)
    _write_record(record_file, pid=123, host="0.0.0.0", port=9999, nonce="abc")
    config = GatewayConfig(settings_file=tmp_path / "settings.json")

    assert gateway_main._probe_endpoint(config) == ("127.0.0.1", 9999)


def test_probe_endpoint_converts_wildcard_config_host_when_no_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "daemon_record_file", lambda: tmp_path / "gateway.pid")
    config = GatewayConfig(host="::", port=8787, settings_file=tmp_path / "settings.json")

    assert gateway_main._probe_endpoint(config) == ("127.0.0.1", 8787)


# --- Probe classification: the four-way outcome grammar ---------------------


def test_classify_daemon_identified_on_valid_hello(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_urlopen(monkeypatch, response=_FakeResponse(200, _HELLO_BODY))
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.IDENTIFIED


def test_classify_daemon_no_listener_on_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_urlopen(
        monkeypatch, exception=urllib.error.URLError(ConnectionRefusedError())
    )
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.NO_LISTENER


def test_classify_daemon_foreign_on_http_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = urllib.error.HTTPError(
        "http://127.0.0.1:8787/api/hello", 500, "boom", None, io.BytesIO(b"oops")
    )
    _stub_urlopen(monkeypatch, exception=exc)
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.FOREIGN


def test_classify_daemon_foreign_on_non_hello_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"service": "something-else"}).encode()
    _stub_urlopen(monkeypatch, response=_FakeResponse(200, body))
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.FOREIGN


def test_classify_daemon_foreign_on_invalid_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_urlopen(monkeypatch, response=_FakeResponse(200, b"not json"))
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.FOREIGN


def test_classify_daemon_ambiguous_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_urlopen(monkeypatch, exception=urllib.error.URLError(TimeoutError("timed out")))
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.AMBIGUOUS


def test_classify_daemon_ambiguous_on_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_urlopen(
        monkeypatch, exception=urllib.error.URLError(socket.gaierror("name not known"))
    )
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.AMBIGUOUS


def test_classify_daemon_ambiguous_on_connection_reset_before_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ResetResponse:
        def __enter__(self) -> "_ResetResponse":
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

        def read(self) -> bytes:
            raise ConnectionResetError("reset")

    _stub_urlopen(monkeypatch, response=_ResetResponse())
    assert gateway_main._classify_daemon("127.0.0.1", 8787) is gateway_main.ProbeOutcome.AMBIGUOUS


# --- The /admin/settings/compaction HTTP helper: auth, content type, error handling -


def test_admin_request_success_returns_status_body_and_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps({"model": None, "env_locked": False, "last_reroute": None}).encode()
    _stub_urlopen(monkeypatch, response=_FakeResponse(200, body))

    response = gateway_main._admin_request(
        "127.0.0.1", 8787, "GET", "/admin/settings/compaction", local_token=None
    )

    assert response.status == 200
    assert response.body == {"model": None, "env_locked": False, "last_reroute": None}


def test_admin_request_attaches_bearer_header_when_local_token_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RecordingOpener(_FakeResponse(200, b"{}"))
    monkeypatch.setattr(gateway_main, "_urlopen_no_redirect", opener)

    gateway_main._admin_request(
        "127.0.0.1", 8787, "GET", "/admin/settings/compaction", local_token="local_token-value"
    )

    assert opener.last_request is not None
    assert opener.last_request.get_header("Authorization") == "Bearer local_token-value"


def test_admin_request_omits_bearer_header_when_no_local_token_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RecordingOpener(_FakeResponse(200, b"{}"))
    monkeypatch.setattr(gateway_main, "_urlopen_no_redirect", opener)

    gateway_main._admin_request(
        "127.0.0.1", 8787, "GET", "/admin/settings/compaction", local_token=None
    )

    assert opener.last_request is not None
    assert opener.last_request.get_header("Authorization") is None


def test_admin_request_sets_json_content_type_on_put_but_not_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RecordingOpener(_FakeResponse(200, b"{}"))
    monkeypatch.setattr(gateway_main, "_urlopen_no_redirect", opener)

    gateway_main._admin_request(
        "127.0.0.1",
        8787,
        "PUT",
        "/admin/settings/compaction",
        local_token=None,
        json_body={"model": None},
    )
    assert opener.last_request is not None
    assert opener.last_request.get_header("Content-type") == "application/json"
    assert opener.last_request.get_method() == "PUT"

    gateway_main._admin_request(
        "127.0.0.1", 8787, "GET", "/admin/settings/compaction", local_token=None
    )
    assert opener.last_request is not None
    assert opener.last_request.get_header("Content-type") is None
    assert opener.last_request.get_method() == "GET"


def test_admin_request_http_error_with_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    error_body = json.dumps(
        {"error": {"message": "no local token", "type": "authentication_error"}}
    ).encode()
    exc = urllib.error.HTTPError(
        "http://127.0.0.1:8787/admin/settings/compaction", 401, "Unauthorized", None, io.BytesIO(error_body)
    )
    _stub_urlopen(monkeypatch, exception=exc)

    response = gateway_main._admin_request(
        "127.0.0.1", 8787, "GET", "/admin/settings/compaction", local_token=None
    )

    assert response.status == 401
    assert response.body == {"error": {"message": "no local token", "type": "authentication_error"}}
    assert response.detail == "no local token"


def test_admin_request_http_error_with_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = urllib.error.HTTPError(
        "http://127.0.0.1:8787/admin/settings/compaction",
        404,
        "Not Found",
        None,
        io.BytesIO(b"<html>not found</html>"),
    )
    _stub_urlopen(monkeypatch, exception=exc)

    response = gateway_main._admin_request(
        "127.0.0.1", 8787, "GET", "/admin/settings/compaction", local_token=None
    )

    assert response.status == 404
    assert response.body is None
    assert "not found" in response.detail.lower()


def test_admin_request_transport_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_urlopen(monkeypatch, exception=urllib.error.URLError(TimeoutError("timed out")))

    with pytest.raises(gateway_main._AdminTransportError):
        gateway_main._admin_request(
            "127.0.0.1", 8787, "GET", "/admin/settings/compaction", local_token=None
        )


# --- Envelope validation: malformed 2xx bodies are treated as failures -----


def test_parse_compaction_envelope_accepts_a_well_formed_body() -> None:
    envelope = {"model": "claude:claude-opus-5", "env_locked": False, "last_reroute": None}
    assert gateway_main._parse_compaction_envelope(envelope) == envelope


def test_parse_compaction_envelope_rejects_a_non_dict_body() -> None:
    assert gateway_main._parse_compaction_envelope(["not", "a", "dict"]) is None


def test_parse_compaction_envelope_rejects_a_missing_field() -> None:
    assert gateway_main._parse_compaction_envelope({"model": None, "env_locked": False}) is None


def test_parse_compaction_envelope_rejects_a_wrongly_typed_model() -> None:
    body = {"model": 123, "env_locked": False, "last_reroute": None}
    assert gateway_main._parse_compaction_envelope(body) is None


def test_parse_compaction_envelope_rejects_a_wrongly_typed_env_locked() -> None:
    body = {"model": None, "env_locked": "yes", "last_reroute": None}
    assert gateway_main._parse_compaction_envelope(body) is None


def test_parse_compaction_envelope_rejects_a_wrongly_typed_last_reroute() -> None:
    body = {"model": None, "env_locked": False, "last_reroute": "nope"}
    assert gateway_main._parse_compaction_envelope(body) is None


# --- `compact` command dispatch: unsupported forms and validation ordering -


def test_compact_test_is_an_unsupported_form(capsys: pytest.CaptureFixture[str]) -> None:
    # "compact test" (unlike `compact`/`compact set`/`compact off`) was
    # deliberately never implemented; every unsupported argument form,
    # including this one, exits 2 with a usage message.
    exit_code = gateway_main._compact_main(["test"])
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "usage: claudex-gateway compact" in err


@pytest.mark.parametrize(
    "argv",
    [["bogus"], ["set"], ["set", "claude:x", "extra"], ["off", "extra"]],
    ids=["unknown-subcommand", "set-no-value", "set-too-many-args", "off-with-extra-arg"],
)
def test_compact_unsupported_argument_shapes_exit_2(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    exit_code = gateway_main._compact_main(argv)
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "usage: claudex-gateway compact" in err


def test_compact_set_invalid_model_syntax_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = gateway_main._compact_main(["set", "gpt-5"])
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "claude:" in err


def test_compact_set_validates_before_any_config_daemon_network_or_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not be called for an invalid compact set argument")

    monkeypatch.setattr(GatewayConfig, "load", staticmethod(_fail))
    monkeypatch.setattr(gateway_main, "_read_daemon_record", _fail)
    monkeypatch.setattr(gateway_main, "_classify_daemon", _fail)
    monkeypatch.setattr(gateway_main, "update_settings_file", _fail)

    exit_code = gateway_main._compact_main(["set", "not-a-claude-model"])

    assert exit_code == 2


# --- `compact` command behavior across the outcome matrix -------------------


def _compact_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: "gateway_main.ProbeOutcome",
    **config_overrides: object,
) -> GatewayConfig:
    """Wire a compact command up to a fixed probe outcome and config, with no
    real daemon record, network call, or config-file read involved."""
    config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_overrides)
    monkeypatch.setattr(GatewayConfig, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(gateway_main, "_read_daemon_record", lambda: (None, "not running"))
    monkeypatch.setattr(gateway_main, "_classify_daemon", lambda host, port: outcome)
    return config


def _stub_admin_request(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: "gateway_main._AdminHttpResponse | None" = None,
    exception: Exception | None = None,
    calls: list[dict[str, object]] | None = None,
) -> None:
    def _fake(
        host: str,
        port: int,
        method: str,
        path: str,
        *,
        local_token: str | None,
        json_body: dict[str, object] | None = None,
    ) -> "gateway_main._AdminHttpResponse":
        if calls is not None:
            calls.append(
                {
                    "host": host,
                    "port": port,
                    "method": method,
                    "path": path,
                    "local_token": local_token,
                    "json_body": json_body,
                }
            )
        if exception is not None:
            raise exception
        assert response is not None
        return response

    monkeypatch.setattr(gateway_main, "_admin_request", _fake)


def test_compact_show_no_listener_reads_settings_file_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _compact_env(
        monkeypatch,
        tmp_path,
        gateway_main.ProbeOutcome.NO_LISTENER,
        compaction_model="claude:claude-opus-5",
    )
    exit_code = gateway_main._compact_main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "compaction: enabled (target claude:claude-opus-5)" in out


def test_compact_show_foreign_reads_settings_file_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.FOREIGN)
    exit_code = gateway_main._compact_main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "compaction: disabled" in out


def test_compact_set_no_listener_writes_settings_file_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.NO_LISTENER)
    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])
    assert exit_code == 0
    saved = json.loads(config.settings_file.read_text(encoding="utf-8"))
    assert saved == {"compaction.model": "claude:claude-opus-5"}


def test_compact_set_foreign_writes_settings_file_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.FOREIGN)
    exit_code = gateway_main._compact_main(["set", "claude:claude-sonnet-5"])
    assert exit_code == 0
    saved = json.loads(config.settings_file.read_text(encoding="utf-8"))
    assert saved == {"compaction.model": "claude:claude-sonnet-5"}


def test_compact_off_no_listener_writes_settings_file_directly_via_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"compaction.model": "claude:claude-old-5", "port": 9000}), encoding="utf-8"
    )
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.NO_LISTENER)
    exit_code = gateway_main._compact_main(["off"])
    assert exit_code == 0
    saved = json.loads(config.settings_file.read_text(encoding="utf-8"))
    assert saved == {"port": 9000}
    assert "compaction.model" not in saved


def test_compact_off_foreign_writes_settings_file_directly_via_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"compaction.model": "claude:claude-old-5"}), encoding="utf-8"
    )
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.FOREIGN)
    exit_code = gateway_main._compact_main(["off"])
    assert exit_code == 0
    saved = json.loads(config.settings_file.read_text(encoding="utf-8"))
    assert "compaction.model" not in saved
    # `off` deletes the key rather than ever persisting a JSON null.
    assert "null" not in settings_file.read_text(encoding="utf-8")


def test_compact_show_ambiguous_reads_file_with_unreachable_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _compact_env(
        monkeypatch,
        tmp_path,
        gateway_main.ProbeOutcome.AMBIGUOUS,
        compaction_model="claude:claude-opus-5",
    )
    exit_code = gateway_main._compact_main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "unreachable" in captured.err
    assert "compaction: enabled (target claude:claude-opus-5)" in captured.out


def test_compact_set_ambiguous_refuses_to_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.AMBIGUOUS)
    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])
    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_off_ambiguous_refuses_to_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.AMBIGUOUS)
    exit_code = gateway_main._compact_main(["off"])
    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_show_identified_success_prints_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    envelope = {
        "model": "claude:claude-opus-5",
        "env_locked": False,
        "last_reroute": {
            "outcome": "rerouted",
            "timestamp": "2026-01-01T00:00:00Z",
            "target_model": "claude-opus-5",
            "mapped_model": "codex:gpt-5.1-codex-max",
            "estimated_prompt_tokens": 4096,
            "context_window": 4000,
            "detail": None,
        },
    }
    _stub_admin_request(
        monkeypatch, response=gateway_main._AdminHttpResponse(status=200, body=envelope, detail="")
    )

    exit_code = gateway_main._compact_main([])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "compaction: enabled (target claude:claude-opus-5)" in out
    assert "outcome=rerouted" in out
    assert "mapped_model=codex:gpt-5.1-codex-max" in out
    assert "estimated_prompt_tokens=4096" in out
    assert "context_window=4000" in out


def test_compact_show_identified_disabled_prints_no_reroute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    envelope = {"model": None, "env_locked": False, "last_reroute": None}
    _stub_admin_request(
        monkeypatch, response=gateway_main._AdminHttpResponse(status=200, body=envelope, detail="")
    )

    exit_code = gateway_main._compact_main([])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "compaction: disabled" in out
    assert "last reroute: none" in out


def test_compact_set_identified_success_calls_put_with_the_local_bearer_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(
        monkeypatch,
        tmp_path,
        gateway_main.ProbeOutcome.IDENTIFIED,
        local_token="local_token-abc",
    )
    envelope = {"model": "claude:claude-opus-5", "env_locked": False, "last_reroute": None}
    calls: list[dict[str, object]] = []
    _stub_admin_request(
        monkeypatch,
        response=gateway_main._AdminHttpResponse(status=200, body=envelope, detail=""),
        calls=calls,
    )

    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])

    assert exit_code == 0
    assert not config.settings_file.exists()  # the admin call succeeded; no direct write
    [call] = calls
    assert call["method"] == "PUT"
    assert call["path"] == "/admin/settings/compaction"
    assert call["json_body"] == {"model": "claude:claude-opus-5"}
    assert call["local_token"] == "local_token-abc"


@pytest.mark.parametrize("status", [400, 401, 409, 500])
def test_compact_set_identified_admin_http_error_exits_nonzero_without_file_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    _stub_admin_request(
        monkeypatch,
        response=gateway_main._AdminHttpResponse(status=status, body=None, detail="boom"),
    )

    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])

    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_off_identified_admin_http_error_exits_nonzero_without_file_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"compaction.model": "claude:claude-old-5"}), encoding="utf-8"
    )
    config = _compact_env(
        monkeypatch,
        tmp_path,
        gateway_main.ProbeOutcome.IDENTIFIED,
        compaction_model="claude:claude-old-5",
    )
    _stub_admin_request(
        monkeypatch,
        response=gateway_main._AdminHttpResponse(status=409, body=None, detail="env locked"),
    )

    exit_code = gateway_main._compact_main(["off"])

    assert exit_code != 0
    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    assert saved == {"compaction.model": "claude:claude-old-5"}


def test_compact_set_identified_404_non_json_falls_back_with_restart_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = GatewayConfig(settings_file=tmp_path / "settings.json")
    monkeypatch.setattr(GatewayConfig, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(gateway_main, "_read_daemon_record", lambda: (None, "not running"))
    _stub_full_stack_urlopen(monkeypatch, 404, b"<html>not found</html>")

    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])
    err = capsys.readouterr().err

    assert exit_code == 0
    assert "restart" in err.lower()
    saved = json.loads(config.settings_file.read_text(encoding="utf-8"))
    assert saved == {"compaction.model": "claude:claude-opus-5"}


def test_compact_off_identified_405_non_json_falls_back_with_restart_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"compaction.model": "claude:claude-old-5"}), encoding="utf-8"
    )
    config = GatewayConfig(settings_file=settings_file, compaction_model="claude:claude-old-5")
    monkeypatch.setattr(GatewayConfig, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(gateway_main, "_read_daemon_record", lambda: (None, "not running"))
    _stub_full_stack_urlopen(monkeypatch, 405, b"method not allowed")

    exit_code = gateway_main._compact_main(["off"])
    err = capsys.readouterr().err

    assert exit_code == 0
    assert "restart" in err.lower()
    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "compaction.model" not in saved


def test_compact_show_identified_405_falls_back_with_may_differ_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = GatewayConfig(
        settings_file=tmp_path / "settings.json", compaction_model="claude:claude-opus-5"
    )
    monkeypatch.setattr(GatewayConfig, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(gateway_main, "_read_daemon_record", lambda: (None, "not running"))
    _stub_full_stack_urlopen(monkeypatch, 405, b"method not allowed")

    exit_code = gateway_main._compact_main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "may differ" in captured.err.lower()
    assert "compaction: enabled (target claude:claude-opus-5)" in captured.out


def test_compact_set_identified_non_json_500_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(settings_file=tmp_path / "settings.json")
    monkeypatch.setattr(GatewayConfig, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(gateway_main, "_read_daemon_record", lambda: (None, "not running"))
    _stub_full_stack_urlopen(monkeypatch, 500, b"internal server error, not json")

    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])

    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_show_identified_malformed_get_envelope_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    _stub_admin_request(
        monkeypatch,
        response=gateway_main._AdminHttpResponse(
            status=200,
            body={"model": None, "env_locked": "not-a-bool", "last_reroute": None},
            detail="",
        ),
    )

    exit_code = gateway_main._compact_main([])

    assert exit_code != 0


def test_compact_set_identified_malformed_put_envelope_exits_nonzero_without_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    _stub_admin_request(
        monkeypatch,
        response=gateway_main._AdminHttpResponse(
            status=200, body={"model": 123, "env_locked": False, "last_reroute": None}, detail=""
        ),
    )

    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])

    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_off_identified_missing_field_envelope_exits_nonzero_without_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    _stub_admin_request(
        monkeypatch,
        response=gateway_main._AdminHttpResponse(
            status=200, body={"model": None, "env_locked": False}, detail=""
        ),
    )

    exit_code = gateway_main._compact_main(["off"])

    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_set_identified_transport_failure_after_probe_exits_nonzero_no_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    _stub_admin_request(monkeypatch, exception=gateway_main._AdminTransportError("timed out"))

    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])

    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_off_identified_transport_failure_after_probe_leaves_settings_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"compaction.model": "claude:claude-old-5"}), encoding="utf-8"
    )
    _compact_env(
        monkeypatch,
        tmp_path,
        gateway_main.ProbeOutcome.IDENTIFIED,
        compaction_model="claude:claude-old-5",
    )
    _stub_admin_request(monkeypatch, exception=gateway_main._AdminTransportError("connection reset"))

    exit_code = gateway_main._compact_main(["off"])

    assert exit_code != 0
    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    assert saved == {"compaction.model": "claude:claude-old-5"}


def _malformed_reroute_envelope() -> "gateway_main._AdminHttpResponse":
    # Envelope-level fields are fine, but the nested last_reroute is not the
    # pinned seven-key record — this must be an admin failure, not success.
    return gateway_main._AdminHttpResponse(
        status=200,
        body={"model": None, "env_locked": False, "last_reroute": {}},
        detail="",
    )


def test_parse_compaction_envelope_rejects_malformed_nested_reroute_record() -> None:
    assert (
        gateway_main._parse_compaction_envelope(
            {"model": None, "env_locked": False, "last_reroute": {}}
        )
        is None
    )


@pytest.mark.parametrize(
    "record_override",
    [
        {"outcome": "exploded"},
        {"timestamp": 12345},
        {"estimated_prompt_tokens": True},
        {"context_window": "big"},
        {"detail": "sql_injection"},
        {"detail": "http_50x"},
        {"outcome": "rerouted", "detail": "read_error"},
        {"outcome": "midstream_error", "detail": "connect_error"},
        {"outcome": "fallback_mapped", "detail": None},
        {"sequence": 7},
    ],
)
def test_parse_reroute_record_rejects_schema_violations(
    record_override: dict[str, object],
) -> None:
    record: dict[str, object] = {
        "outcome": "fallback_mapped",
        "timestamp": "2026-08-07T00:00:00.000+00:00",
        "target_model": "claude-opus-5",
        "mapped_model": "codex:gpt-5.1-codex-max",
        "estimated_prompt_tokens": 250000,
        "context_window": 200000,
        "detail": "http_401",
    }
    record.update(record_override)
    assert gateway_main._parse_reroute_record(record) is None


def test_parse_reroute_record_accepts_each_valid_outcome_shape() -> None:
    base = {
        "timestamp": "2026-08-07T00:00:00.000+00:00",
        "target_model": "claude-opus-5",
        "mapped_model": "grok:grok-code-fast",
        "estimated_prompt_tokens": 250000,
        "context_window": 200000,
    }
    for outcome, detail in [
        ("rerouted", None),
        ("skipped_no_credentials", None),
        ("fallback_mapped", "connect_error"),
        ("fallback_mapped", "http_503"),
        ("midstream_error", "read_error"),
    ]:
        record = dict(base, outcome=outcome, detail=detail)
        assert gateway_main._parse_reroute_record(record) == record


def test_compact_show_malformed_nested_record_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    _stub_admin_request(monkeypatch, response=_malformed_reroute_envelope())

    exit_code = gateway_main._compact_main([])

    assert exit_code != 0
    assert "malformed" in capsys.readouterr().err


def test_compact_set_malformed_nested_record_exits_nonzero_without_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _compact_env(monkeypatch, tmp_path, gateway_main.ProbeOutcome.IDENTIFIED)
    _stub_admin_request(monkeypatch, response=_malformed_reroute_envelope())

    exit_code = gateway_main._compact_main(["set", "claude:claude-opus-5"])

    assert exit_code != 0
    assert not config.settings_file.exists()


def test_compact_off_malformed_nested_record_exits_nonzero_without_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"compaction.model": "claude:claude-old-5"}), encoding="utf-8"
    )
    _compact_env(
        monkeypatch,
        tmp_path,
        gateway_main.ProbeOutcome.IDENTIFIED,
        compaction_model="claude:claude-old-5",
    )
    _stub_admin_request(monkeypatch, response=_malformed_reroute_envelope())

    exit_code = gateway_main._compact_main(["off"])

    assert exit_code != 0
    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    assert saved == {"compaction.model": "claude:claude-old-5"}


class _RedirectTarget:
    """Second endpoint that records whether it was ever contacted."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []


def _serve_redirect_pair() -> "tuple[object, object, int, _RedirectTarget]":
    """Start server A (redirects everything to server B) and recording server B."""
    import http.server

    target = _RedirectTarget()

    class TargetHandler(http.server.BaseHTTPRequestHandler):
        def _record(self) -> None:
            target.requests.append((self.path, dict(self.headers)))
            body = b'{"hello": "claudex-gateway"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _record
        do_PUT = _record

        def log_message(self, *args: object) -> None:
            pass

    target_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_port = target_server.server_address[1]

    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def _redirect(self) -> None:
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{target_port}{self.path}"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = _redirect
        do_PUT = _redirect

        def log_message(self, *args: object) -> None:
            pass

    redirect_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    redirect_port = redirect_server.server_address[1]

    for server in (target_server, redirect_server):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    return redirect_server, target_server, redirect_port, target


def test_probe_and_admin_never_follow_redirects_or_leak_the_bearer() -> None:
    redirect_server, target_server, redirect_port, target = _serve_redirect_pair()
    try:
        # Probe: the 3xx is the final answer from the port occupant — FOREIGN,
        # never classified from the redirect target.
        outcome = gateway_main._classify_daemon("127.0.0.1", redirect_port)
        assert outcome is gateway_main.ProbeOutcome.FOREIGN

        # Admin: the 3xx is a completed non-2xx response — FAILURE, and the
        # bearer-carrying request must never reach the redirect target.
        admin_outcome, envelope, detail = gateway_main._run_admin_compaction(
            "127.0.0.1",
            redirect_port,
            "GET",
            "secret-local-token",
            None,
        )
        assert admin_outcome is gateway_main._AdminCallOutcome.FAILURE
        assert envelope is None
        assert "302" in detail

        assert target.requests == []
    finally:
        redirect_server.shutdown()
        target_server.shutdown()
        redirect_server.server_close()
        target_server.server_close()
