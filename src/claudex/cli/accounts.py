"""Implementation of the account command family."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

from claudex.claude import accounts as claude_accounts
from claudex.claude import capture as claude_capture
from claudex.cli import admin_client
from claudex.config import ConfigError, GatewayConfig, update_settings_file

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

_ACCOUNT_ADD_USAGE = "usage: claudex-gateway account add [--from <dir>] [--yes]"
_ACCOUNT_REMOVE_USAGE = "usage: claudex-gateway account remove <account-id> [--yes]"


def _build_account_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claudex-gateway account")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--from", dest="from_dir", metavar="<dir>", default=None)
    add_parser.add_argument("--yes", action="store_true")

    subparsers.add_parser("list")

    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("account_id", metavar="<account-id>")
    remove_parser.add_argument("--yes", action="store_true")

    use_parser = subparsers.add_parser("use")
    use_parser.add_argument("target", nargs="?", default=None, metavar="<id|email|off>")

    return parser


def _account_add(from_dir: str | None, assume_yes: bool) -> int:
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
    except claude_accounts.DuplicateAccountError:
        return _account_replace_credentials(captured, assume_yes)
    except claude_accounts.AccountRegistryError as exc:
        print(f"account add failed: {exc}", file=sys.stderr)
        return 1

    print(f"added account {record.email} ({record.id})")
    return 0


def _account_replace_credentials(
    captured: claude_capture.CapturedAccount, assume_yes: bool
) -> int:
    """Upsert path for `account add`: the identity is already registered, so
    replace that account's stored credentials in place (re-auth) after the
    same confirmation protocol `account remove` uses. The freshly captured
    credentials are already in memory — confirming never repeats the login.
    """
    if not assume_yes and not sys.stdin.isatty():
        print(_ACCOUNT_ADD_USAGE, file=sys.stderr)
        print(
            f"an account is already registered for {captured.email}; replacing its "
            "stored credentials without confirmation requires --yes when stdin is "
            "not a terminal",
            file=sys.stderr,
        )
        return 2

    if not assume_yes:
        try:
            answer = input(
                f"Account {captured.email} is already registered; replace its "
                "stored credentials? [y/N]"
            )
        except EOFError:
            return 0
        if answer.strip().lower() not in ("y", "yes"):
            return 0

    try:
        record = claude_accounts.update_account_credentials(
            captured.email,
            captured.organization_uuid,
            captured.organization_name,
            captured.credentials_json,
            captured.oauth_account_json,
        )
    except claude_accounts.AccountRegistryError as exc:
        print(f"account add failed: {exc}", file=sys.stderr)
        return 1

    print(f"updated account {record.email} ({record.id}): stored credentials replaced")
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
            return _account_add(args.from_dir, args.yes)
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
# `account use`: show/select/clear the registered account serving Anthropic
# passthrough (settings.json's "claude_account.id"), through the same
# daemon-aware channel selection as `compact`: a confirmed daemon is managed
# via the admin API, a settings-file write is only used when no live daemon
# can be confirmed (or the confirmed daemon predates the endpoint), and an
# ambiguous probe refuses to write.
# ---------------------------------------------------------------------------

_ADMIN_CLAUDE_ACCOUNT_PATH = "/admin/providers/claude/pool/serving"
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
) -> tuple[admin_client._AdminCallOutcome, dict[str, Any] | None, str]:
    return admin_client._run_admin_envelope(
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
    config = admin_client.daemon._load_config()
    host, port = admin_client._probe_endpoint(config)
    outcome = admin_client._classify_daemon(host, port)

    if outcome is admin_client.ProbeOutcome.IDENTIFIED:
        admin_outcome, envelope, detail = _run_admin_claude_account(
            host, port, "GET", config.local_token, None
        )
        if admin_outcome is admin_client._AdminCallOutcome.SUCCESS:
            _print_account_use_state(envelope["account_id"])
            return 0
        if admin_outcome is admin_client._AdminCallOutcome.OLDER_DAEMON:
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

    if outcome in (
        admin_client.ProbeOutcome.NO_LISTENER,
        admin_client.ProbeOutcome.FOREIGN,
    ):
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
    config = admin_client.daemon._load_config()
    host, port = admin_client._probe_endpoint(config)
    outcome = admin_client._classify_daemon(host, port)

    if outcome is admin_client.ProbeOutcome.IDENTIFIED:
        # Selecting pins via PUT; `off` clears the pin via DELETE (the
        # endpoint refuses a null PUT). Both return the state envelope.
        if value is None:
            admin_outcome, envelope, detail = _run_admin_claude_account(
                host, port, "DELETE", config.local_token, None
            )
        else:
            admin_outcome, envelope, detail = _run_admin_claude_account(
                host, port, "PUT", config.local_token, {"account_id": value}
            )
        if admin_outcome is admin_client._AdminCallOutcome.SUCCESS:
            _print_account_use_state(envelope["account_id"])
            return 0
        if admin_outcome is admin_client._AdminCallOutcome.OLDER_DAEMON:
            print(
                "account use: the running daemon predates the admin claude-account "
                "API; writing the settings file directly. Restart claudex-gateway "
                "for the change to take effect.",
                file=sys.stderr,
            )
            return _write_account_use_settings(config, value)
        print(f"account use: {detail}", file=sys.stderr)
        return 1

    if outcome in (
        admin_client.ProbeOutcome.NO_LISTENER,
        admin_client.ProbeOutcome.FOREIGN,
    ):
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
