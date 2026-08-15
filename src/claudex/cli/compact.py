"""Implementation of the compact command family."""

from __future__ import annotations

import sys
from typing import Any

from claudex.cli import admin_client
from claudex.config import (
    ConfigError,
    GatewayConfig,
    parse_compaction_model,
    update_settings_file,
)

_ADMIN_COMPACTION_PATH = "/admin/settings/compaction"
_COMPACT_USAGE = "usage: claudex-gateway compact [set claude:<id>|off]"

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


def _run_admin_compaction(
    host: str,
    port: int,
    method: str,
    local_token: str | None,
    json_body: dict[str, Any] | None,
) -> tuple[admin_client._AdminCallOutcome, dict[str, Any] | None, str]:
    return admin_client._run_admin_envelope(
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
    config = admin_client.daemon._load_config()
    host, port = admin_client._probe_endpoint(config)
    outcome = admin_client._classify_daemon(host, port)

    if outcome is admin_client.ProbeOutcome.IDENTIFIED:
        admin_outcome, envelope, detail = _run_admin_compaction(
            host, port, "GET", config.local_token, None
        )
        if admin_outcome is admin_client._AdminCallOutcome.SUCCESS:
            _print_compact_state(envelope["model"], envelope["last_reroute"])
            return 0
        if admin_outcome is admin_client._AdminCallOutcome.OLDER_DAEMON:
            print(
                "compact: the running daemon predates the admin compaction API; "
                "showing settings-file state, which may differ from live runtime state",
                file=sys.stderr,
            )
            _print_compact_state(config.compaction_model, None)
            return 0
        print(f"compact: {detail}", file=sys.stderr)
        return 1

    if outcome in (
        admin_client.ProbeOutcome.NO_LISTENER,
        admin_client.ProbeOutcome.FOREIGN,
    ):
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
    config = admin_client.daemon._load_config()
    host, port = admin_client._probe_endpoint(config)
    outcome = admin_client._classify_daemon(host, port)

    if outcome is admin_client.ProbeOutcome.IDENTIFIED:
        admin_outcome, envelope, detail = _run_admin_compaction(
            host, port, "PUT", config.local_token, {"model": value}
        )
        if admin_outcome is admin_client._AdminCallOutcome.SUCCESS:
            _print_compact_state(envelope["model"], None)
            return 0
        if admin_outcome is admin_client._AdminCallOutcome.OLDER_DAEMON:
            print(
                "compact: the running daemon predates the admin compaction API; "
                "writing the settings file directly. Restart claudex-gateway for "
                "the change to take effect.",
                file=sys.stderr,
            )
            return _write_compact_settings(config, value)
        print(f"compact: {detail}", file=sys.stderr)
        return 1

    if outcome in (
        admin_client.ProbeOutcome.NO_LISTENER,
        admin_client.ProbeOutcome.FOREIGN,
    ):
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
