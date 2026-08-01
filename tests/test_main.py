"""Subprocess tests for the background start/stop lifecycle."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

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
    assert "login kimi" in result.stderr


def test_login_requires_a_known_provider(tmp_path: Path) -> None:
    for arguments in (["login"], ["login", "gemini"]):
        result = _run_cli(_gateway_env(tmp_path, _free_port()), *arguments)
        assert result.returncode == 2
        assert "login kimi" in result.stderr


def test_login_kimi_runs_device_flow_and_writes_credentials(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import claudex_gateway.__main__ as main_module
    from claudex_gateway.config import GatewayConfig
    from claudex_gateway.kimi_auth import DeviceAuthorization

    authorization = DeviceAuthorization(
        device_code="dev-code",
        user_code="ABCD-1234",
        verification_uri="https://kimi.com/activate",
        verification_uri_complete="https://kimi.com/activate?code=ABCD-1234",
        interval=1.0,
        expires_in=300.0,
        device_id="device-1",
    )

    async def fake_request_device_authorization(http_client):
        return authorization

    async def fake_poll_device_token(http_client, granted):
        assert granted is authorization
        return {"type": "kimi", "access_token": "token", "device_id": "device-1"}

    monkeypatch.setattr(
        main_module, "request_device_authorization", fake_request_device_authorization
    )
    monkeypatch.setattr(main_module, "poll_device_token", fake_poll_device_token)

    auth_file = tmp_path / "kimi-auth.json"
    main_module._login_kimi(GatewayConfig(kimi_auth_file=auth_file))

    output = capsys.readouterr().out
    assert "https://kimi.com/activate?code=ABCD-1234" in output
    assert "ABCD-1234" in output
    assert json.loads(auth_file.read_text(encoding="utf-8"))["access_token"] == "token"
    assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600
