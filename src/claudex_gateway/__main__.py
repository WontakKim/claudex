"""Entrypoint for `claudex-gateway` / `python -m claudex_gateway`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn

import claudex_gateway
from claudex_gateway import claude_accounts, claude_capture, paths
from claudex_gateway.config import ConfigError, GatewayConfig
from claudex_gateway.server import create_app

_READY_TIMEOUT = 15.0
_STOP_TIMEOUT = 10.0

# Carries the launcher-generated daemon nonce into the spawned foreground
# process so /api/hello can prove which daemon a record refers to.
_DAEMON_NONCE_ENV = "CLAUDEX_DAEMON_NONCE"


def _load_config() -> GatewayConfig:
    try:
        return GatewayConfig.load()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _connect_host(host: str) -> str:
    """Return the address to probe: wildcard binds answer on loopback."""
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def _is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _gateway_hello(host: str, port: int) -> dict[str, Any] | None:
    """Fetch /api/hello, or None when the occupant is not claudex-gateway."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/hello", timeout=2) as response:
            payload = json.load(response)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("hello") != "claudex-gateway":
        return None
    return payload


def _run_foreground(config: GatewayConfig) -> None:
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(
        f"claudex-gateway listening on http://{config.host}:{config.port}\n"
        f"Claude Code -> mapped models -> Codex, Kimi, or Grok, others -> Anthropic passthrough\n"
        f"point Claude Code at it with:\n"
        f"  ANTHROPIC_BASE_URL=http://{config.host}:{config.port} claude"
    )
    app = create_app(config, daemon_nonce=os.environ.get(_DAEMON_NONCE_ENV))
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


def _start_background(config: GatewayConfig) -> None:
    probe_host = _connect_host(config.host)
    if _is_listening(probe_host, config.port):
        hello = _gateway_hello(probe_host, config.port)
        if hello is None:
            print(
                f"port {config.port} is in use by something that does not look like "
                f"claudex-gateway; free the port or change CLAUDEX_PORT",
                file=sys.stderr,
            )
            raise SystemExit(1)
        # Daemons predating the version field report as "unknown", which
        # still counts as a mismatch worth restarting.
        running_version = hello.get("version")
        if not isinstance(running_version, str):
            running_version = "unknown"
        if running_version == claudex_gateway.__version__:
            print(f"claudex-gateway already running on http://{config.host}:{config.port}")
            return
        # A daemon from a previous install survived the package update; its
        # code (and possibly its files) are gone, so replace it.
        print(
            f"claudex-gateway {running_version} is running but this launcher is "
            f"{claudex_gateway.__version__}; restarting"
        )
        stopped, message = _stop_recorded_daemon()
        if not stopped:
            print(
                f"cannot restart automatically: {message}; "
                f"stop the old gateway manually and retry",
                file=sys.stderr,
            )
            raise SystemExit(1)
        deadline = time.monotonic() + _STOP_TIMEOUT
        while time.monotonic() < deadline and _is_listening(probe_host, config.port):
            time.sleep(0.1)

    paths.runtime_dir().mkdir(parents=True, exist_ok=True)
    log_file = paths.log_file()
    daemon_nonce = secrets.token_hex(16)
    child_env = dict(os.environ)
    child_env[_DAEMON_NONCE_ENV] = daemon_nonce
    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "claudex_gateway", "--foreground"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_env,
        )
    _write_daemon_record(process.pid, config.host, config.port, daemon_nonce)

    deadline = time.monotonic() + _READY_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            paths.daemon_record_file().unlink(missing_ok=True)
            print(
                f"claudex-gateway exited during startup; last log lines from {log_file}:\n"
                f"{_log_tail(log_file)}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        # Readiness means the spawned daemon itself answers, not merely that
        # some process appears on the port.
        hello = _gateway_hello(probe_host, config.port)
        if (
            hello is not None
            and hello.get("pid") == process.pid
            and hello.get("nonce") == daemon_nonce
        ):
            print(
                f"claudex-gateway started on http://{config.host}:{config.port} "
                f"(pid {process.pid}, log {log_file})"
            )
            return
        time.sleep(0.1)

    process.terminate()
    paths.daemon_record_file().unlink(missing_ok=True)
    print(
        f"claudex-gateway did not become ready within {_READY_TIMEOUT:.0f}s; "
        f"check {log_file}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _log_tail(log_file: Path, lines: int = 10) -> str:
    try:
        return "\n".join(log_file.read_text(encoding="utf-8").splitlines()[-lines:])
    except OSError:
        return "(log file unreadable)"


def _write_daemon_record(pid: int, host: str, port: int, nonce: str) -> None:
    paths.daemon_record_file().write_text(
        json.dumps({"pid": pid, "host": host, "port": port, "nonce": nonce}) + "\n",
        encoding="utf-8",
    )


def _read_daemon_record() -> tuple[dict[str, Any] | None, str]:
    """Read the daemon identity record; (None, reason) when unusable.

    Legacy bare-pid files and corrupt content are unusable on purpose: a pid
    alone cannot prove which process it belongs to, so signaling from it
    could kill an unrelated process after pid reuse."""
    record_file = paths.daemon_record_file()
    try:
        raw = record_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "not running (no daemon record)"
    except OSError as exc:
        return None, f"cannot read {record_file}: {exc}"
    try:
        record = json.loads(raw)
    except ValueError:
        record = None
    if (
        isinstance(record, dict)
        and isinstance(record.get("pid"), int)
        and not isinstance(record.get("pid"), bool)
        and isinstance(record.get("host"), str)
        and record["host"]
        and isinstance(record.get("port"), int)
        and not isinstance(record.get("port"), bool)
        and isinstance(record.get("nonce"), str)
        and record["nonce"]
    ):
        return record, ""
    return None, (
        f"{record_file} is not a daemon identity record (left by an older "
        f"gateway?); stop that gateway manually and delete the file"
    )


def _stop_recorded_daemon() -> tuple[bool, str]:
    """Verify the recorded daemon's identity via /api/hello, then SIGTERM it.

    Returns (stopped, detail): the stopped pid on success, the reason on
    failure. Never signals a pid whose identity cannot be verified."""
    record, reason = _read_daemon_record()
    if record is None:
        return False, reason
    pid = record["pid"]
    probe_host = _connect_host(record["host"])
    hello = _gateway_hello(probe_host, record["port"])
    if hello is None:
        # The recorded gateway is not answering. A dead pid means the record
        # is stale; a live one may be an unrelated process after pid reuse.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            paths.daemon_record_file().unlink(missing_ok=True)
            return False, "not running (stale daemon record removed)"
        return False, (
            f"pid {pid} is alive but no gateway answers on "
            f"{probe_host}:{record['port']}; refusing to signal an unverified process"
        )
    if hello.get("pid") != pid or hello.get("nonce") != record["nonce"]:
        return False, (
            f"the gateway on {probe_host}:{record['port']} does not match the "
            f"daemon record; refusing to signal pid {pid}"
        )

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        paths.daemon_record_file().unlink(missing_ok=True)
        return False, "not running (stale daemon record removed)"

    deadline = time.monotonic() + _STOP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            paths.daemon_record_file().unlink(missing_ok=True)
            return True, str(pid)
        time.sleep(0.1)
    return False, f"pid {pid} did not exit within {_STOP_TIMEOUT:.0f}s"


def _stop_background() -> None:
    stopped, detail = _stop_recorded_daemon()
    if not stopped:
        print(f"claudex-gateway: {detail}", file=sys.stderr)
        raise SystemExit(1)
    print(f"claudex-gateway stopped (pid {detail})")


# ---------------------------------------------------------------------------
# `account` subcommands: manage ~/.claudex account files only, never touching
# GatewayConfig. Every mutation stays inside claude_accounts/claude_capture;
# this module only parses argv, prints outcomes, and maps exceptions to exit
# codes.
# ---------------------------------------------------------------------------

_ACCOUNT_ADD_USAGE = "usage: claudex-gateway account add [--from <dir>]"
_ACCOUNT_REMOVE_USAGE = "usage: claudex-gateway account remove <account-id> [--yes]"


def _build_account_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claudex-gateway account")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--from", dest="from_dir", metavar="<dir>", default=None)

    subparsers.add_parser("list")

    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("account_id", metavar="<account-id>")
    remove_parser.add_argument("--yes", action="store_true")

    return parser


def _account_add(from_dir: str | None) -> int:
    if from_dir is None and not sys.stdin.isatty():
        print(_ACCOUNT_ADD_USAGE, file=sys.stderr)
        print(
            "interactive login requires a terminal; pass --from <dir> to import "
            "an already-completed login instead",
            file=sys.stderr,
        )
        return 2

    try:
        if from_dir is not None:
            captured = claude_capture.capture_from_config_dir(from_dir)
        else:
            captured = claude_capture.capture_interactive()
    except claude_capture.CaptureCancelled:
        return 130
    except claude_capture.CaptureError as exc:
        print(f"account add failed: {exc}", file=sys.stderr)
        return 1

    try:
        record = claude_accounts.add_account(
            captured.email,
            captured.organization_uuid,
            captured.organization_name,
            captured.credentials_json,
            captured.oauth_account_json,
        )
    except claude_accounts.AccountRegistryError as exc:
        print(f"account add failed: {exc}", file=sys.stderr)
        return 1

    print(f"added account {record.email} ({record.id})")
    return 0


def _format_added_timestamp(created_at_millis: int) -> str:
    return datetime.fromtimestamp(created_at_millis / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _print_account_table(records: list[claude_accounts.AccountRecord]) -> None:
    header = ("ID", "EMAIL", "ORGANIZATION", "ADDED")
    rows = [
        (
            record.id,
            record.email,
            record.organization_name or record.organization_uuid or "-",
            _format_added_timestamp(record.created_at),
        )
        for record in records
    ]
    widths = [
        max(len(header[column]), *(len(row[column]) for row in rows))
        for column in range(len(header) - 1)
    ]
    for line in (header, *rows):
        padded = (value.ljust(width) for value, width in zip(line, widths))
        print("  ".join((*padded, line[-1])))


def _account_list() -> int:
    try:
        records = claude_accounts.list_accounts()
    except claude_accounts.AccountRegistryError as exc:
        print(f"account list failed: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("no accounts registered")
        return 0

    _print_account_table(records)
    return 0


def _account_remove(account_id: str, assume_yes: bool) -> int:
    if not assume_yes and not sys.stdin.isatty():
        print(_ACCOUNT_REMOVE_USAGE, file=sys.stderr)
        print(
            "removal without confirmation requires --yes when stdin is not a terminal",
            file=sys.stderr,
        )
        return 2

    try:
        records = claude_accounts.list_accounts()
    except claude_accounts.AccountRegistryError as exc:
        print(f"account remove failed: {exc}", file=sys.stderr)
        return 1

    record = next((candidate for candidate in records if candidate.id == account_id), None)
    if record is None:
        print(
            f"account remove failed: no account registered with id {account_id!r}",
            file=sys.stderr,
        )
        return 1

    if not assume_yes:
        try:
            answer = input(f"Remove account {record.email} ({record.id})? [y/N] ")
        except EOFError:
            return 0
        if answer.strip().lower() not in ("y", "yes"):
            return 0

    try:
        claude_accounts.remove_account(record.id)
    except claude_accounts.AccountRegistryError as exc:
        print(f"account remove failed: {exc}", file=sys.stderr)
        return 1

    print(f"removed account {record.email} ({record.id})")
    return 0


def _account_main(argv: list[str]) -> int:
    parser = _build_account_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        if args.command == "add":
            return _account_add(args.from_dir)
        if args.command == "list":
            return _account_list()
        return _account_remove(args.account_id, args.yes)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # last-resort safety net: never a bare traceback
        print(f"unexpected error: {type(exc).__name__}", file=sys.stderr)
        return 1


def main() -> None:
    arguments = sys.argv[1:]
    # stop must work even when the current configuration is broken.
    if arguments == ["stop"]:
        _stop_background()
        return

    # Account management only touches ~/.claudex account files, so it must
    # work even when the current gateway configuration is broken.
    if arguments and arguments[0] == "account":
        result = _account_main(arguments[1:])
        if result != 0:
            raise SystemExit(result)
        return

    config = _load_config()
    if not arguments:
        _start_background(config)
        return
    if arguments == ["--foreground"]:
        _run_foreground(config)
        return

    print("usage: claudex-gateway [--foreground|stop]", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
