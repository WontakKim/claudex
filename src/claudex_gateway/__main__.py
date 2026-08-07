"""Entrypoint for `claudex-gateway` / `python -m claudex_gateway`."""

from __future__ import annotations

import argparse
import enum
import ipaddress
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn

import claudex_gateway
from claudex_gateway import claude_accounts, claude_capture, paths
from claudex_gateway.config import (
    ConfigError,
    GatewayConfig,
    parse_compaction_model,
    update_settings_file,
)
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
# `account` subcommands. add/list/remove manage ~/.claudex account files
# only, never touching GatewayConfig, so they work even with a broken
# configuration. `use` selects which registered account serves Anthropic
# passthrough — that is a settings change, so it loads the config and goes
# through the same daemon-aware channel selection as `compact` (see the
# account-use section below). Every registry mutation stays inside
# claude_accounts/claude_capture; this module only parses argv, prints
# outcomes, and maps exceptions to exit codes.
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

    use_parser = subparsers.add_parser("use")
    use_parser.add_argument("target", nargs="?", default=None, metavar="<id|email|off>")

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
            answer = input(f"Remove account {record.email} ({record.id})? [y/N]")
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
        if args.command == "use":
            return _account_use(args.target)
        return _account_remove(args.account_id, args.yes)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # last-resort safety net: never a bare traceback
        print(f"unexpected error: {type(exc).__name__}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# `compact` subcommands: show/set/disable the compaction reroute target
# (settings.json's "compaction.model") through whichever daemon-aware channel
# is safe. A running daemon is preferred (the admin API keeps its in-memory
# config and the settings file in sync); a settings-file write is only used
# when no live daemon can be confirmed, or when the confirmed daemon predates
# the admin compaction API.
# ---------------------------------------------------------------------------

_ADMIN_COMPACTION_PATH = "/admin/compaction"
_PROBE_TIMEOUT = 2.0
_ADMIN_TIMEOUT = 5.0

_COMPACT_USAGE = "usage: claudex-gateway compact [set claude:<id>|off]"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow any redirect: the 3xx surfaces as an HTTPError.

    Following a redirect would re-send the request headers — including the
    local bearer token on admin calls — to whatever host the Location header
    names, and would classify the probe from the redirect target instead of
    the actual port occupant.
    """

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _urlopen_no_redirect(request: urllib.request.Request | str, timeout: float) -> Any:
    """`urlopen` for probe/admin calls; every 3xx is a final response."""
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


class ProbeOutcome(enum.Enum):
    """The four-way result of probing GET /api/hello for a compact command.

    IDENTIFIED means a claudex-gateway answered; NO_LISTENER means nothing is
    listening at all; FOREIGN means something answered but is not
    claudex-gateway (wrong port occupant, or an HTTP error status); AMBIGUOUS
    covers everything else (timeout, DNS failure, connection reset before a
    response) where liveness genuinely cannot be determined.
    """

    IDENTIFIED = "identified"
    NO_LISTENER = "no_listener"
    FOREIGN = "foreign"
    AMBIGUOUS = "ambiguous"


def _bracket_host(host: str) -> str:
    """Bracket an IPv6 literal for URL use; IPv4 addresses/hostnames pass through."""
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return host
    return f"[{host}]" if parsed.version == 6 else host


def _http_url(host: str, port: int, path: str) -> str:
    return f"http://{_bracket_host(host)}:{port}{path}"


def _probe_endpoint(config: GatewayConfig) -> tuple[str, int]:
    """Resolve the host/port to probe for a compact command.

    A structurally valid daemon record wins over the config: it names the
    process the launcher itself started, which the config's host/port need
    not match (e.g. CLAUDEX_HOST changed after the daemon started). Absent a
    valid record, the resolved config host/port keeps a foreground daemon
    that never wrote a record file discoverable.
    """
    record, _ = _read_daemon_record()
    if record is not None:
        return _connect_host(record["host"]), record["port"]
    return _connect_host(config.host), config.port


def _classify_daemon(host: str, port: int) -> ProbeOutcome:
    """Probe GET /api/hello and classify the occupant.

    Read-only: this never signals a process (no os.kill), it only issues one
    GET request with a short timeout.
    """
    try:
        with _urlopen_no_redirect(
            _http_url(host, port, "/api/hello"), timeout=_PROBE_TIMEOUT
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError:
        # A completed HTTP response, but an error or redirect status: not a
        # valid hello (redirects are never followed, so a 3xx lands here).
        return ProbeOutcome.FOREIGN
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return ProbeOutcome.NO_LISTENER
        # Timeout, DNS failure, connection reset before a response, etc.
        return ProbeOutcome.AMBIGUOUS
    except OSError:
        # e.g. a connection reset while reading the body, after the connect
        # itself succeeded (so it never went through URLError above).
        return ProbeOutcome.AMBIGUOUS

    try:
        payload = json.loads(raw)
    except ValueError:
        return ProbeOutcome.FOREIGN
    if isinstance(payload, dict) and payload.get("hello") == "claudex-gateway":
        return ProbeOutcome.IDENTIFIED
    return ProbeOutcome.FOREIGN


class _AdminTransportError(Exception):
    """The admin request never got an HTTP response (DNS, reset, timeout, ...)."""


@dataclass(frozen=True)
class _AdminHttpResponse:
    status: int
    body: Any  # parsed JSON value when the body decodes, else None
    detail: str  # bounded diagnostic text: an error message, or raw body text


def _parse_admin_body(raw: bytes) -> tuple[Any, str]:
    """Decode an admin response body; never raises on non-JSON content."""
    text = raw.decode("utf-8", errors="replace")
    bounded = text[:500]
    try:
        body = json.loads(text)
    except ValueError:
        return None, bounded
    detail = bounded
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            detail = error["message"]
        elif isinstance(body.get("detail"), str):
            detail = body["detail"]
    return body, detail


def _admin_request(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    local_token: str | None,
    json_body: dict[str, Any] | None = None,
) -> _AdminHttpResponse:
    """GET/PUT an admin endpoint; raises only for a transport-level failure.

    An HTTP error status is a completed response, not a transport failure:
    urllib.error.HTTPError is caught here and turned into a normal
    _AdminHttpResponse so the caller can inspect a 4xx/5xx body (JSON or
    not) instead of it propagating as an exception.
    """
    data = None
    headers = {"Accept": "application/json"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if local_token:
        headers["Authorization"] = f"Bearer {local_token}"
    request = urllib.request.Request(
        _http_url(host, port, path), data=data, headers=headers, method=method
    )
    try:
        with _urlopen_no_redirect(request, timeout=_ADMIN_TIMEOUT) as response:
            body, detail = _parse_admin_body(response.read())
            return _AdminHttpResponse(status=response.status, body=body, detail=detail)
    except urllib.error.HTTPError as exc:
        body, detail = _parse_admin_body(exc.read())
        return _AdminHttpResponse(status=exc.code, body=body, detail=detail)
    except (urllib.error.URLError, OSError) as exc:
        raise _AdminTransportError(str(exc)) from exc


# The pinned seven-key last_reroute record (see server's
# _assign_compaction_reroute) and its detail grammar/outcome constraints.
_REROUTE_RECORD_KEYS = frozenset(
    {
        "outcome",
        "timestamp",
        "target_model",
        "mapped_model",
        "estimated_prompt_tokens",
        "context_window",
        "detail",
    }
)
_REROUTE_OUTCOMES = frozenset(
    {"rerouted", "fallback_mapped", "skipped_no_credentials", "midstream_error"}
)
_REROUTE_SIMPLE_DETAILS = frozenset({"connect_error", "read_error", "invalid_json"})


def _is_real_int(value: Any) -> bool:
    # bool is an int subclass; True/False must not pass as token counts.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_valid_reroute_detail(detail: Any) -> bool:
    if detail is None:
        return True
    if not isinstance(detail, str):
        return False
    if detail in _REROUTE_SIMPLE_DETAILS:
        return True
    return (
        detail.startswith("http_")
        and len(detail) == len("http_") + 3
        and all(char in "0123456789" for char in detail[len("http_"):])
    )


def _parse_reroute_record(record: Any) -> dict[str, Any] | None:
    """Validate a non-null last_reroute against the pinned record schema."""
    if not isinstance(record, dict) or set(record) != _REROUTE_RECORD_KEYS:
        return None
    outcome = record["outcome"]
    if outcome not in _REROUTE_OUTCOMES:
        return None
    for key in ("timestamp", "target_model", "mapped_model"):
        if not isinstance(record[key], str):
            return None
    if not _is_real_int(record["estimated_prompt_tokens"]):
        return None
    if not _is_real_int(record["context_window"]):
        return None
    detail = record["detail"]
    if not _is_valid_reroute_detail(detail):
        return None
    if outcome in ("rerouted", "skipped_no_credentials") and detail is not None:
        return None
    if outcome == "midstream_error" and detail != "read_error":
        return None
    if outcome == "fallback_mapped" and detail is None:
        return None
    return record


def _parse_compaction_envelope(body: Any) -> dict[str, Any] | None:
    """Validate the pinned {model, env_locked, last_reroute} envelope.

    Returns the body unchanged when every required field is present and
    correctly typed (model: str|None, env_locked: bool, last_reroute: None or
    a record matching the pinned seven-key schema, detail grammar, and
    outcome constraints); otherwise None, so the caller treats a malformed
    2xx body as an admin failure rather than trusting it.
    """
    if not isinstance(body, dict):
        return None
    if not {"model", "env_locked", "last_reroute"} <= set(body):
        return None
    model = body["model"]
    if model is not None and not isinstance(model, str):
        return None
    if not isinstance(body["env_locked"], bool):
        return None
    last_reroute = body["last_reroute"]
    if last_reroute is not None and _parse_reroute_record(last_reroute) is None:
        return None
    return body


class _AdminCallOutcome(enum.Enum):
    SUCCESS = "success"
    OLDER_DAEMON = "older_daemon"  # 404/405: the daemon predates the endpoint
    FAILURE = "failure"


def _run_admin_envelope(
    host: str,
    port: int,
    method: str,
    path: str,
    parse_envelope: Any,
    local_token: str | None,
    json_body: dict[str, Any] | None,
) -> tuple[_AdminCallOutcome, dict[str, Any] | None, str]:
    """Run one GET/PUT admin call and classify the result.

    Returns (outcome, envelope, detail): envelope is the parse_envelope-
    validated body only on SUCCESS; detail is diagnostic text for FAILURE
    (including a malformed 2xx envelope, any non-404/405 error status, or a
    transport failure).
    """
    try:
        response = _admin_request(
            host,
            port,
            method,
            path,
            local_token=local_token,
            json_body=json_body,
        )
    except _AdminTransportError as exc:
        return _AdminCallOutcome.FAILURE, None, f"admin request failed: {exc}"

    if response.status in (404, 405):
        return _AdminCallOutcome.OLDER_DAEMON, None, ""
    if 200 <= response.status < 300:
        envelope = parse_envelope(response.body)
        if envelope is not None:
            return _AdminCallOutcome.SUCCESS, envelope, ""
        return (
            _AdminCallOutcome.FAILURE,
            None,
            f"admin returned a malformed response (status {response.status})",
        )
    return (
        _AdminCallOutcome.FAILURE,
        None,
        f"admin request failed (status {response.status}): {response.detail}",
    )


def _run_admin_compaction(
    host: str,
    port: int,
    method: str,
    local_token: str | None,
    json_body: dict[str, Any] | None,
) -> tuple[_AdminCallOutcome, dict[str, Any] | None, str]:
    return _run_admin_envelope(
        host,
        port,
        method,
        _ADMIN_COMPACTION_PATH,
        _parse_compaction_envelope,
        local_token,
        json_body,
    )


def _print_compact_state(model: str | None, diagnostics: dict[str, Any] | None) -> None:
    if model is None:
        print("compaction: disabled")
    else:
        print(f"compaction: enabled (target {model})")
    if diagnostics is None:
        print("last reroute: none")
        return
    print(
        "last reroute: "
        f"outcome={diagnostics.get('outcome')} "
        f"timestamp={diagnostics.get('timestamp')} "
        f"target_model={diagnostics.get('target_model')} "
        f"mapped_model={diagnostics.get('mapped_model')} "
        f"estimated_prompt_tokens={diagnostics.get('estimated_prompt_tokens')} "
        f"context_window={diagnostics.get('context_window')} "
        f"detail={diagnostics.get('detail')}"
    )


def _write_compact_settings(config: GatewayConfig, value: str | None) -> int:
    try:
        if value is None:
            # A disabled setting is represented by the key's absence, so a
            # JSON null is never persisted.
            update_settings_file(config.settings_file, {}, deletions=("compaction.model",))
        else:
            update_settings_file(config.settings_file, {"compaction.model": value})
    except ConfigError as exc:
        print(f"compact: could not persist settings: {exc}", file=sys.stderr)
        return 1
    _print_compact_state(value, None)
    return 0


def _compact_show() -> int:
    config = _load_config()
    host, port = _probe_endpoint(config)
    outcome = _classify_daemon(host, port)

    if outcome is ProbeOutcome.IDENTIFIED:
        admin_outcome, envelope, detail = _run_admin_compaction(
            host, port, "GET", config.local_token, None
        )
        if admin_outcome is _AdminCallOutcome.SUCCESS:
            _print_compact_state(envelope["model"], envelope["last_reroute"])
            return 0
        if admin_outcome is _AdminCallOutcome.OLDER_DAEMON:
            print(
                "compact: the running daemon predates the admin compaction API; "
                "showing settings-file state, which may differ from live runtime state",
                file=sys.stderr,
            )
            _print_compact_state(config.compaction_model, None)
            return 0
        print(f"compact: {detail}", file=sys.stderr)
        return 1

    if outcome in (ProbeOutcome.NO_LISTENER, ProbeOutcome.FOREIGN):
        _print_compact_state(config.compaction_model, None)
        return 0

    # AMBIGUOUS: liveness could not be confirmed either way, but a read is
    # harmless, so show the best-known (settings-file) state with a note.
    print(
        "compact: could not reach claudex-gateway to confirm live runtime state "
        "(unreachable); showing settings-file state",
        file=sys.stderr,
    )
    _print_compact_state(config.compaction_model, None)
    return 0


def _compact_apply(value: str | None) -> int:
    config = _load_config()
    host, port = _probe_endpoint(config)
    outcome = _classify_daemon(host, port)

    if outcome is ProbeOutcome.IDENTIFIED:
        admin_outcome, envelope, detail = _run_admin_compaction(
            host, port, "PUT", config.local_token, {"model": value}
        )
        if admin_outcome is _AdminCallOutcome.SUCCESS:
            _print_compact_state(envelope["model"], None)
            return 0
        if admin_outcome is _AdminCallOutcome.OLDER_DAEMON:
            print(
                "compact: the running daemon predates the admin compaction API; "
                "writing the settings file directly. Restart claudex-gateway for "
                "the change to take effect.",
                file=sys.stderr,
            )
            return _write_compact_settings(config, value)
        print(f"compact: {detail}", file=sys.stderr)
        return 1

    if outcome in (ProbeOutcome.NO_LISTENER, ProbeOutcome.FOREIGN):
        return _write_compact_settings(config, value)

    # AMBIGUOUS: fail closed. Writing here risks clobbering a live daemon's
    # settings behind its back, or racing a daemon that is actually up.
    print(
        "compact: could not confirm whether claudex-gateway is running "
        "(unreachable); refusing to modify settings without a live daemon",
        file=sys.stderr,
    )
    return 1


def _compact_main(arguments: list[str]) -> int:
    # Argument validation happens before any config load, daemon-record
    # read, network request, or settings-file read: an invalid `compact set`
    # argument must fail on syntax alone, with no other side effect.
    if not arguments:
        return _compact_show()
    if arguments == ["off"]:
        return _compact_apply(None)
    if len(arguments) == 2 and arguments[0] == "set":
        try:
            parse_compaction_model(arguments[1])
        except ConfigError as exc:
            print(f"compact set failed: {exc}", file=sys.stderr)
            return 2
        return _compact_apply(arguments[1])
    print(_COMPACT_USAGE, file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# `account use`: show/select/clear the registered account serving Anthropic
# passthrough (settings.json's "claude_account.id"), through the same
# daemon-aware channel selection as `compact`: a confirmed daemon is managed
# via the admin API, a settings-file write is only used when no live daemon
# can be confirmed (or the confirmed daemon predates the endpoint), and an
# ambiguous probe refuses to write.
# ---------------------------------------------------------------------------

_ADMIN_CLAUDE_ACCOUNT_PATH = "/admin/claude-account"

_ACCOUNT_USE_USAGE = "usage: claudex-gateway account use [<id|email>|off]"


def _parse_claude_account_envelope(body: Any) -> dict[str, Any] | None:
    """Validate the pinned {account_id, env_locked} envelope; None if malformed."""
    if not isinstance(body, dict):
        return None
    if not {"account_id", "env_locked"} <= set(body):
        return None
    account_id = body["account_id"]
    if account_id is not None and not isinstance(account_id, str):
        return None
    if not isinstance(body["env_locked"], bool):
        return None
    return body


def _run_admin_claude_account(
    host: str,
    port: int,
    method: str,
    local_token: str | None,
    json_body: dict[str, Any] | None,
) -> tuple[_AdminCallOutcome, dict[str, Any] | None, str]:
    return _run_admin_envelope(
        host,
        port,
        method,
        _ADMIN_CLAUDE_ACCOUNT_PATH,
        _parse_claude_account_envelope,
        local_token,
        json_body,
    )


def _account_email_for(account_id: str) -> str | None:
    """Best-effort local-registry lookup for display; None on any failure."""
    try:
        records = claude_accounts.list_accounts()
    except claude_accounts.AccountRegistryError:
        return None
    record = next((candidate for candidate in records if candidate.id == account_id), None)
    return record.email if record is not None else None


def _print_account_use_state(account_id: str | None) -> None:
    if account_id is None:
        print("account use: off (passthrough forwards client credentials)")
        return
    email = _account_email_for(account_id)
    if email is None:
        print(f"account use: {account_id}")
        print(
            f"warning: account {account_id} is not in the local registry; "
            "passthrough requests will fail until it is registered or deselected",
            file=sys.stderr,
        )
        return
    print(f"account use: {email} ({account_id})")


def _write_account_use_settings(config: GatewayConfig, value: str | None) -> int:
    try:
        if value is None:
            # A disabled setting is represented by the key's absence, so a
            # JSON null is never persisted.
            update_settings_file(
                config.settings_file, {}, deletions=("claude_account.id",)
            )
        else:
            update_settings_file(config.settings_file, {"claude_account.id": value})
    except ConfigError as exc:
        print(f"account use: could not persist settings: {exc}", file=sys.stderr)
        return 1
    _print_account_use_state(value)
    return 0


def _account_use_show() -> int:
    config = _load_config()
    host, port = _probe_endpoint(config)
    outcome = _classify_daemon(host, port)

    if outcome is ProbeOutcome.IDENTIFIED:
        admin_outcome, envelope, detail = _run_admin_claude_account(
            host, port, "GET", config.local_token, None
        )
        if admin_outcome is _AdminCallOutcome.SUCCESS:
            _print_account_use_state(envelope["account_id"])
            return 0
        if admin_outcome is _AdminCallOutcome.OLDER_DAEMON:
            print(
                "account use: the running daemon predates the admin claude-account "
                "API; showing settings-file state, which may differ from live "
                "runtime state",
                file=sys.stderr,
            )
            _print_account_use_state(config.claude_account_id)
            return 0
        print(f"account use: {detail}", file=sys.stderr)
        return 1

    if outcome in (ProbeOutcome.NO_LISTENER, ProbeOutcome.FOREIGN):
        _print_account_use_state(config.claude_account_id)
        return 0

    # AMBIGUOUS: liveness could not be confirmed either way, but a read is
    # harmless, so show the best-known (settings-file) state with a note.
    print(
        "account use: could not reach claudex-gateway to confirm live runtime "
        "state (unreachable); showing settings-file state",
        file=sys.stderr,
    )
    _print_account_use_state(config.claude_account_id)
    return 0


def _account_use_apply(value: str | None) -> int:
    config = _load_config()
    host, port = _probe_endpoint(config)
    outcome = _classify_daemon(host, port)

    if outcome is ProbeOutcome.IDENTIFIED:
        admin_outcome, envelope, detail = _run_admin_claude_account(
            host, port, "PUT", config.local_token, {"account_id": value}
        )
        if admin_outcome is _AdminCallOutcome.SUCCESS:
            _print_account_use_state(envelope["account_id"])
            return 0
        if admin_outcome is _AdminCallOutcome.OLDER_DAEMON:
            print(
                "account use: the running daemon predates the admin claude-account "
                "API; writing the settings file directly. Restart claudex-gateway "
                "for the change to take effect.",
                file=sys.stderr,
            )
            return _write_account_use_settings(config, value)
        print(f"account use: {detail}", file=sys.stderr)
        return 1

    if outcome in (ProbeOutcome.NO_LISTENER, ProbeOutcome.FOREIGN):
        return _write_account_use_settings(config, value)

    # AMBIGUOUS: fail closed. Writing here risks clobbering a live daemon's
    # settings behind its back, or racing a daemon that is actually up.
    print(
        "account use: could not confirm whether claudex-gateway is running "
        "(unreachable); refusing to modify settings without a live daemon",
        file=sys.stderr,
    )
    return 1


def _resolve_account_use_target(target: str) -> str | None:
    """Resolve an id-or-email target to a registered account id.

    Prints the failure reason and returns None when nothing (or more than
    one email match) resolves; ids win over emails so a uuid-shaped email
    can never shadow a real account id."""
    try:
        records = claude_accounts.list_accounts()
    except claude_accounts.AccountRegistryError as exc:
        print(f"account use failed: {exc}", file=sys.stderr)
        return None

    by_id = next((candidate for candidate in records if candidate.id == target), None)
    if by_id is not None:
        return by_id.id
    # The registry stores emails trimmed and lowercased; match likewise.
    email = target.strip().lower()
    email_matches = [candidate for candidate in records if candidate.email == email]
    if len(email_matches) == 1:
        return email_matches[0].id
    if len(email_matches) > 1:
        ids = ", ".join(candidate.id for candidate in email_matches)
        print(
            f"account use failed: email {target!r} matches multiple accounts "
            f"({ids}); pass an account id instead",
            file=sys.stderr,
        )
        return None
    print(
        f"account use failed: no account registered with id or email {target!r}; "
        "see `claudex-gateway account list`",
        file=sys.stderr,
    )
    return None


def _account_use(target: str | None) -> int:
    if target is None:
        return _account_use_show()
    if target == "off":
        return _account_use_apply(None)
    account_id = _resolve_account_use_target(target)
    if account_id is None:
        return 1
    return _account_use_apply(account_id)


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

    # compact validates its own argument (compact set claude:<id>) before
    # touching configuration, a daemon record, the network, or the settings
    # file, so it is dispatched ahead of the ordinary _load_config() below.
    if arguments and arguments[0] == "compact":
        result = _compact_main(arguments[1:])
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
