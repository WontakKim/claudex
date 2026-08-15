"""Tests for the `claudex-gateway account {add,list,remove}` CLI subcommands.

Two testing strategies are used, matching the conventions already
established by `test_claude_capture.py` and `test_main.py`:

- Scenarios that need a real process boundary (signal delivery, a `pty` for
  an interactive prompt, top-level argv routing) spawn
  `python -m claudex_gateway ...` via `subprocess`, with `HOME` isolated to
  `tmp_path` and every `CLAUDEX_*`/`CLAUDE_CONFIG_DIR` variable scrubbed from
  the child's environment.
- Scenarios that need a successful Claude credential capture call
  `claudex_gateway.cli.accounts._account_main(...)` in-process, so a real macOS
  Keychain is never touched: either `sys.platform` is forced to `"linux"`
  (capture then reads plain `.credentials.json`/`.claude.json` files, exactly
  as `test_claude_capture.py`'s own version-gate tests do), or the login
  child never reaches credential capture at all (e.g. it is killed by SIGINT
  while still hanging).
"""

from __future__ import annotations

import contextlib
import json
import os
import pty
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

from claudex_gateway import claude_accounts, paths
from claudex_gateway.claude import capture as claude_capture
from claudex_gateway.cli import accounts, admin_client, daemon

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every test's `~/.claudex` under a throwaway directory, shared by
    both in-process calls and any subprocess spawned with `_clean_env()`."""
    monkeypatch.setenv("HOME", str(tmp_path))


class _FakeTTYStdin:
    def isatty(self) -> bool:
        return True


def _make_stdin_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeTTYStdin())


def _add_record(email: str = "user@example.com", **overrides: object) -> claude_accounts.AccountRecord:
    """Seed the registry directly through the storage layer, bypassing any
    capture flow, for tests that only need an existing account to act on."""
    kwargs: dict[str, object] = {
        "organization_uuid": None,
        "organization_name": None,
        "credentials_json": {"accessToken": "at-1"},
        "oauth_account_json": None,
    }
    kwargs.update(overrides)
    return claude_accounts.add_account(email, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    """`os.environ` with every `CLAUDEX_*`/`CLAUDE_CONFIG_DIR` variable
    scrubbed; `HOME` is already isolated by the autouse fixture above."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CLAUDEX_") and key != "CLAUDE_CONFIG_DIR"
    }


def _run_cli(
    env: dict[str, str],
    *args: str,
    stdin: int | None = None,
    input_text: str | None = None,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, object] = {}
    if input_text is not None:
        kwargs["input"] = input_text
    else:
        kwargs["stdin"] = stdin
    return subprocess.run(
        [sys.executable, "-m", "claudex_gateway", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        **kwargs,
    )


def _write_broken_settings(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".claudex"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "settings.json").write_text("{not valid json", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fake `claude` CLI (PATH-prepended, not in-process) -- silent by default so
# gateway-output assertions never mistake the child's own output for the
# gateway's (Step 7).
# ---------------------------------------------------------------------------

from fake_claude import (  # noqa: E402  (shared fixture module in tests/)
    prepend_fake_claude as _prepend_fake_claude,
    write_fake_claude as _write_fake_claude,
)


def _wait_for_file(path: Path, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            content = path.read_text().strip()
            if content:
                return content
        time.sleep(0.02)
    raise AssertionError(f"{path} did not appear within {timeout}s")


def _assert_process_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} is still alive after {timeout}s")


# ---------------------------------------------------------------------------
# pty helpers for the interactive `remove` confirmation prompt
# ---------------------------------------------------------------------------


def _spawn_pty_cli(env: dict[str, str], *args: str) -> tuple[int, int]:
    """Fork a `python -m claudex_gateway <args>` child on a fresh pty.

    Uses `os.forkpty()` rather than `subprocess.Popen` dup'ing an
    already-open fd: only a real `open()` of the pty slave -- which
    `forkpty()` performs in the child -- makes it that child's controlling
    terminal, and a controlling terminal is required for the line discipline
    to turn Ctrl-C into a real `SIGINT` delivered to the child.
    """
    pid, controller_fd = os.forkpty()
    if pid == 0:
        try:
            os.execvpe(sys.executable, [sys.executable, "-m", "claudex_gateway", *args], env)
        except OSError:
            pass
        os._exit(127)
    return pid, controller_fd


def _wait_exit_code(pid: int, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        finished_pid, status = os.waitpid(pid, os.WNOHANG)
        if finished_pid == pid:
            if os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            if os.WIFSIGNALED(status):
                return -os.WTERMSIG(status)
            raise AssertionError(f"child {pid} left an unexpected wait status {status}")
        time.sleep(0.05)
    raise AssertionError(f"child {pid} did not exit within {timeout}s")


def _read_pty_until(fd: int, marker: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    collected = b""
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        collected += chunk
        if marker.encode() in collected:
            break
    return collected.decode(errors="replace")


# ---------------------------------------------------------------------------
# Routing: top-level argv order is preserved exactly (routing)
# ---------------------------------------------------------------------------


def test_routing_stop_bypasses_broken_config(tmp_path: Path) -> None:
    _write_broken_settings(tmp_path)
    result = _run_cli(_clean_env(), "stop", timeout=10)
    assert result.returncode == 1
    assert "configuration error" not in result.stderr


def test_routing_account_list_bypasses_broken_config_exit_code_0(tmp_path: Path) -> None:
    _write_broken_settings(tmp_path)
    result = _run_cli(_clean_env(), "account", "list", timeout=10)
    assert result.returncode == 0
    assert result.stdout.strip() == "no accounts registered"


def test_routing_unknown_account_syntax_exits_2(tmp_path: Path) -> None:
    result = _run_cli(_clean_env(), "account", "frobnicate", timeout=10)
    assert result.returncode == 2


@pytest.mark.parametrize(
    "argv", [[], ["--foreground"], ["unknown-top-level-arg"]], ids=["empty", "foreground", "unknown-arg"]
)
def test_routing_broken_config_regression_matrix(tmp_path: Path, argv: list[str]) -> None:
    _write_broken_settings(tmp_path)
    result = _run_cli(_clean_env(), *argv, timeout=10)
    assert result.returncode == 1
    assert "configuration error" in result.stderr


# ---------------------------------------------------------------------------
# non-TTY guards on `add` and `remove` (non_tty, exit_code)
# ---------------------------------------------------------------------------


def test_non_tty_add_without_from_exit_code_2(tmp_path: Path) -> None:
    result = _run_cli(_clean_env(), "account", "add", stdin=subprocess.DEVNULL, timeout=10)
    assert result.returncode == 2
    assert "usage: claudex-gateway account add" in result.stderr


def test_non_tty_piped_y_without_yes_exit_code_2_retains_account(tmp_path: Path) -> None:
    record = _add_record()
    result = _run_cli(_clean_env(), "account", "remove", record.id, input_text="y\n", timeout=10)
    assert result.returncode == 2
    assert "usage: claudex-gateway account remove" in result.stderr
    assert [r.id for r in claude_accounts.list_accounts()] == [record.id]


def test_yes_flag_with_stdin_devnull_exit_code_0(tmp_path: Path) -> None:
    record = _add_record()
    result = _run_cli(
        _clean_env(), "account", "remove", record.id, "--yes", stdin=subprocess.DEVNULL, timeout=10
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"removed account {record.email} ({record.id})"
    assert claude_accounts.list_accounts() == []


# ---------------------------------------------------------------------------
# Interactive `remove` prompt through a real pty: accept/decline (decline)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="pty is POSIX-only")
@pytest.mark.parametrize(
    ("answer", "expect_removed"), [("n\n", False), ("y\n", True)], ids=["decline", "accept"]
)
def test_remove_prompt_pty_response(tmp_path: Path, answer: str, expect_removed: bool) -> None:
    record = _add_record()
    pid, controller_fd = _spawn_pty_cli(_clean_env(), "account", "remove", record.id)
    try:
        output = _read_pty_until(controller_fd, "[y/N]")
        assert f"Remove account {record.email} ({record.id})? [y/N]" in output
        os.write(controller_fd, answer.encode())
        returncode = _wait_exit_code(pid)
    finally:
        with contextlib.suppress(OSError):
            os.close(controller_fd)
    assert returncode == 0
    remaining = [r.id for r in claude_accounts.list_accounts()]
    assert remaining == ([] if expect_removed else [record.id])


# ---------------------------------------------------------------------------
# Cancellation: SIGINT/Ctrl-C never leaves a mutation behind (interrupt)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="pty/SIGINT are POSIX-only")
def test_interrupt_ctrl_c_during_remove_prompt_exit_code_130_no_mutation(tmp_path: Path) -> None:
    record = _add_record()
    pid, controller_fd = _spawn_pty_cli(_clean_env(), "account", "remove", record.id)
    try:
        _read_pty_until(controller_fd, "[y/N]")
        os.write(controller_fd, b"\x03")  # Ctrl-C: the pty line discipline raises SIGINT
        returncode = _wait_exit_code(pid)
    finally:
        with contextlib.suppress(OSError):
            os.close(controller_fd)
    assert returncode == 130
    assert [r.id for r in claude_accounts.list_accounts()] == [record.id]


@pytest.mark.skipif(sys.platform == "win32", reason="interactive Claude login is POSIX-only")
def test_interrupt_sigint_during_interactive_login_exit_code_130_no_account_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_stdin_a_tty(monkeypatch)
    _prepend_fake_claude(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")
    leader_pid_file = tmp_path / "leader.pid"
    monkeypatch.setenv("CLAUDEX_TEST_LEADER_PID_FILE", str(leader_pid_file))

    def _deliver_sigint() -> None:
        _wait_for_file(leader_pid_file)
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGINT)

    signaler = threading.Thread(target=_deliver_sigint, daemon=True)
    signaler.start()
    try:
        exit_code = accounts._account_main(["add"])
    finally:
        signaler.join(timeout=5)

    assert exit_code == 130
    assert claude_accounts.list_accounts() == []
    leader_pid = int(leader_pid_file.read_text().strip())
    _assert_process_gone(leader_pid)
    # Cleanup ran to completion: no leftover temp Claude config dir.
    assert not list(Path(tempfile.gettempdir()).glob(f"{claude_capture._TEMP_DIR_PREFIX}*"))


# ---------------------------------------------------------------------------
# add: interactive round trip, --from import, duplicates (all in-process,
# with sys.platform forced to "linux" so capture reads plain files instead
# of touching the real macOS Keychain -- mirrors test_claude_capture.py)
# ---------------------------------------------------------------------------


def test_interactive_add_list_remove_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_stdin_a_tty(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    _prepend_fake_claude(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "success")

    add_exit_code = accounts._account_main(["add"])
    added_out = capsys.readouterr().out
    assert add_exit_code == 0
    assert re.fullmatch(
        r"added account fixture@example\.com \([0-9a-f-]{36}\)\n", added_out
    )

    list_exit_code = accounts._account_main(["list"])
    list_out = capsys.readouterr().out
    assert list_exit_code == 0
    assert "fixture@example.com" in list_out

    [record] = claude_accounts.list_accounts()
    remove_exit_code = accounts._account_main(["remove", record.id, "--yes"])
    removed_out = capsys.readouterr().out
    assert remove_exit_code == 0
    assert removed_out.strip() == f"removed account {record.email} ({record.id})"
    assert claude_accounts.list_accounts() == []


def _write_fixture_config_dir(
    config_dir: Path, *, email: str = "import@example.com", access_token: str = "at-import"
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": access_token, "email": email}}),
        encoding="utf-8",
    )
    (config_dir / ".claude.json").write_text(
        json.dumps(
            {
                "oauthAccount": {
                    "emailAddress": email,
                    "organizationUuid": "org-import",
                    "organizationName": "Import Org",
                }
            }
        ),
        encoding="utf-8",
    )


def test_from_import_uses_fixture_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    config_dir = tmp_path / "fixture-config"
    _write_fixture_config_dir(config_dir)

    exit_code = accounts._account_main(["add", "--from", str(config_dir)])
    out = capsys.readouterr().out
    assert exit_code == 0

    [record] = claude_accounts.list_accounts()
    assert record.email == "import@example.com"
    assert record.organization_name == "Import Org"
    assert out.strip() == f"added account {record.email} ({record.id})"


def _stored_access_token(record: claude_accounts.AccountRecord) -> str:
    credentials_path = paths.accounts_dir("claude") / record.id / "credentials.json"
    return json.loads(credentials_path.read_text())["claudeAiOauth"]["accessToken"]


def test_duplicate_add_non_tty_without_yes_exit_code_2_keeps_stored_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    config_dir = tmp_path / "fixture-config"
    _write_fixture_config_dir(config_dir)

    first_exit_code = accounts._account_main(["add", "--from", str(config_dir)])
    capsys.readouterr()
    _write_fixture_config_dir(config_dir, access_token="at-import-2")
    second_exit_code = accounts._account_main(["add", "--from", str(config_dir)])
    err = capsys.readouterr().err

    assert first_exit_code == 0
    assert second_exit_code == 2
    assert "already registered" in err
    assert "--yes" in err
    assert "Traceback" not in err
    [record] = claude_accounts.list_accounts()
    assert _stored_access_token(record) == "at-import"


def test_duplicate_add_with_yes_replaces_credentials_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    config_dir = tmp_path / "fixture-config"
    _write_fixture_config_dir(config_dir)

    assert accounts._account_main(["add", "--from", str(config_dir)]) == 0
    capsys.readouterr()
    [original] = claude_accounts.list_accounts()

    _write_fixture_config_dir(config_dir, access_token="at-import-2")
    exit_code = accounts._account_main(["add", "--from", str(config_dir), "--yes"])
    out = capsys.readouterr().out

    assert exit_code == 0
    [record] = claude_accounts.list_accounts()
    assert record.id == original.id  # same account, updated in place
    assert record.created_at == original.created_at
    assert _stored_access_token(record) == "at-import-2"
    assert out.strip() == (
        f"updated account {record.email} ({record.id}): stored credentials replaced"
    )


@pytest.mark.parametrize(
    ("answer", "expect_replaced"), [("y", True), ("yes", True), ("n", False), ("", False)]
)
def test_duplicate_add_tty_prompt_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    expect_replaced: bool,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    config_dir = tmp_path / "fixture-config"
    _write_fixture_config_dir(config_dir)

    assert accounts._account_main(["add", "--from", str(config_dir)]) == 0
    capsys.readouterr()

    _make_stdin_a_tty(monkeypatch)
    prompts: list[str] = []

    def _fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    monkeypatch.setattr("builtins.input", _fake_input)
    _write_fixture_config_dir(config_dir, access_token="at-import-2")
    exit_code = accounts._account_main(["add", "--from", str(config_dir)])

    assert exit_code == 0
    assert len(prompts) == 1
    assert "already registered" in prompts[0]
    [record] = claude_accounts.list_accounts()
    expected_token = "at-import-2" if expect_replaced else "at-import"
    assert _stored_access_token(record) == expected_token


# ---------------------------------------------------------------------------
# remove: unknown id, malformed registry (exit_code, traceback)
# ---------------------------------------------------------------------------


def test_unknown_remove_id_exit_code_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = accounts._account_main(["remove", str(uuid.uuid4()), "--yes"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "account remove failed" in err
    assert "Traceback" not in err


def test_malformed_registry_exit_code_1_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    accounts_root = paths.accounts_dir("claude")
    accounts_root.mkdir(parents=True)
    (accounts_root / "registry.json").write_text("{not valid json", encoding="utf-8")

    exit_code = accounts._account_main(["list"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "account list failed" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# list: empty output, table shape, and reading only registry.json
# (does_not_read_credentials, corrupt)
# ---------------------------------------------------------------------------


def test_empty_list_output_exit_code_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = accounts._account_main(["list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.strip() == "no accounts registered"


def test_list_table_headers_and_date_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _add_record("user@example.com", organization_name="Acme")
    exit_code = accounts._account_main(["list"])
    out = capsys.readouterr().out
    lines = out.splitlines()

    assert exit_code == 0
    assert lines[0].split() == ["ID", "EMAIL", "ORGANIZATION", "ADDED"]
    assert record.id in lines[1]
    assert "user@example.com" in lines[1]
    assert "Acme" in lines[1]
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", lines[1])


def test_list_does_not_read_credentials_files_even_when_corrupt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`account list` reads only registry.json: corrupt, missing per-account
    credential files must never affect it (does_not_read_credentials)."""
    record = _add_record()
    account_dir = paths.accounts_dir("claude") / record.id
    (account_dir / "credentials.json").write_text("{not valid json at all", encoding="utf-8")
    (account_dir / "oauth-account.json").unlink()

    exit_code = accounts._account_main(["list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert record.email in out


# ---------------------------------------------------------------------------
# No credential material ever leaves the gateway's own stdout/stderr
# (no_secret_output, sentinel)
# ---------------------------------------------------------------------------


def test_no_secret_output_sentinel_absent_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    sentinel = "sk-ant-oat-sentinel-must-never-leak"
    config_dir = tmp_path / "fixture-config"
    config_dir.mkdir()
    (config_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": sentinel, "email": "user@example.com"}}),
        encoding="utf-8",
    )

    add_exit_code = accounts._account_main(["add", "--from", str(config_dir)])
    added = capsys.readouterr()
    assert add_exit_code == 0
    assert sentinel not in added.out
    assert sentinel not in added.err

    accounts._account_main(["list"])
    listed = capsys.readouterr()
    assert sentinel not in listed.out
    assert sentinel not in listed.err

    failure_exit_code = accounts._account_main(["remove", str(uuid.uuid4()), "--yes"])
    failed = capsys.readouterr()
    assert failure_exit_code == 1
    assert sentinel not in failed.out
    assert sentinel not in failed.err


# ---------------------------------------------------------------------------
# `account use` — serving-account selection through the compact-style
# daemon-aware channel. The probe and admin transport are stubbed exactly as
# test_main.py's compact tests do; settings writes go to the isolated HOME.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _unlocked_account_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ID", raising=False)


def _stub_probe(
    monkeypatch: pytest.MonkeyPatch, outcome: "admin_client.ProbeOutcome"
) -> dict[str, int]:
    calls = {"record": 0, "classify": 0}

    def read_daemon_record() -> tuple[None, str]:
        calls["record"] += 1
        return None, "not running"

    def classify_daemon(host: str, port: int) -> admin_client.ProbeOutcome:
        calls["classify"] += 1
        return outcome

    monkeypatch.setattr(daemon, "_read_daemon_record", read_daemon_record)
    monkeypatch.setattr(admin_client, "_classify_daemon", classify_daemon)
    return calls


def _settings_path() -> Path:
    return paths.settings_file()


def test_account_use_show_prints_off_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_probe(monkeypatch, admin_client.ProbeOutcome.NO_LISTENER)

    exit_code = accounts._account_main(["use"])

    assert exit_code == 0
    assert calls == {"record": 1, "classify": 1}
    assert "account use: off" in capsys.readouterr().out


def test_account_use_writes_settings_when_no_daemon(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _add_record()
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.NO_LISTENER)

    exit_code = accounts._account_main(["use", record.id])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"account use: user@example.com ({record.id})" in output
    saved = json.loads(_settings_path().read_text(encoding="utf-8"))
    assert saved["claude_account"]["id"] == record.id


def test_account_use_resolves_email_to_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _add_record(email="pool@example.com")
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.NO_LISTENER)

    exit_code = accounts._account_main(["use", "Pool@Example.com"])

    assert exit_code == 0
    saved = json.loads(_settings_path().read_text(encoding="utf-8"))
    assert saved["claude_account"]["id"] == record.id


def test_account_use_off_removes_the_settings_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _add_record()
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.NO_LISTENER)
    assert accounts._account_main(["use", record.id]) == 0

    exit_code = accounts._account_main(["use", "off"])

    assert exit_code == 0
    assert "account use: off" in capsys.readouterr().out
    saved = json.loads(_settings_path().read_text(encoding="utf-8"))
    assert "claude_account" not in saved


def test_account_use_unknown_target_fails_without_touching_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.NO_LISTENER)

    exit_code = accounts._account_main(["use", "nobody@example.com"])

    assert exit_code == 1
    assert "no account registered" in capsys.readouterr().err
    assert not _settings_path().exists()


def test_account_use_ambiguous_email_requires_an_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _add_record(email="shared@example.com", organization_uuid="org-1")
    second = _add_record(email="shared@example.com", organization_uuid="org-2")
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.NO_LISTENER)

    exit_code = accounts._account_main(["use", "shared@example.com"])

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "matches multiple accounts" in error
    assert first.id in error and second.id in error
    assert not _settings_path().exists()


def test_account_use_identified_daemon_puts_through_admin_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _add_record()
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.IDENTIFIED)
    calls: list[dict[str, object]] = []

    def fake_admin_request(
        host: str,
        port: int,
        method: str,
        path: str,
        *,
        local_token: str | None,
        json_body: dict[str, object] | None = None,
    ) -> "admin_client._AdminHttpResponse":
        calls.append({"method": method, "path": path, "json_body": json_body})
        return admin_client._AdminHttpResponse(
            status=200,
            body={"account_id": record.id, "env_locked": False},
            detail="",
        )

    monkeypatch.setattr(admin_client, "_admin_request", fake_admin_request)

    exit_code = accounts._account_main(["use", record.id])

    assert exit_code == 0
    assert calls == [
        {
            "method": "PUT",
            "path": "/admin/providers/claude/pool/serving",
            "json_body": {"account_id": record.id},
        }
    ]
    assert f"({record.id})" in capsys.readouterr().out
    # The daemon persisted the change through the admin API; the CLI must
    # not also write the settings file behind its back.
    assert not _settings_path().exists()


def test_account_use_off_identified_daemon_deletes_through_admin_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.IDENTIFIED)
    calls: list[dict[str, object]] = []

    def fake_admin_request(
        host: str,
        port: int,
        method: str,
        path: str,
        *,
        local_token: str | None,
        json_body: dict[str, object] | None = None,
    ) -> "admin_client._AdminHttpResponse":
        calls.append({"method": method, "path": path, "json_body": json_body})
        return admin_client._AdminHttpResponse(
            status=200,
            body={"account_id": None, "env_locked": False},
            detail="",
        )

    monkeypatch.setattr(admin_client, "_admin_request", fake_admin_request)

    exit_code = accounts._account_main(["use", "off"])

    assert exit_code == 0
    assert calls == [
        {
            "method": "DELETE",
            "path": "/admin/providers/claude/pool/serving",
            "json_body": None,
        }
    ]
    assert "account use: off" in capsys.readouterr().out
    assert not _settings_path().exists()


def test_account_use_show_identified_reads_admin_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _add_record()
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.IDENTIFIED)

    def fake_admin_request(*args: object, **kwargs: object) -> "admin_client._AdminHttpResponse":
        return admin_client._AdminHttpResponse(
            status=200,
            body={"account_id": record.id, "env_locked": False},
            detail="",
        )

    monkeypatch.setattr(admin_client, "_admin_request", fake_admin_request)

    exit_code = accounts._account_main(["use"])

    assert exit_code == 0
    assert f"user@example.com ({record.id})" in capsys.readouterr().out


def test_account_use_ambiguous_probe_refuses_to_write(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _add_record()
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.AMBIGUOUS)

    exit_code = accounts._account_main(["use", record.id])

    assert exit_code == 1
    assert "refusing to modify settings" in capsys.readouterr().err
    assert not _settings_path().exists()


def test_account_use_show_warns_about_an_unregistered_configured_account(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    orphan_id = str(uuid.uuid4())
    accounts.update_settings_file(
        _settings_path(), {"claude_account.id": orphan_id}
    )
    _stub_probe(monkeypatch, admin_client.ProbeOutcome.NO_LISTENER)

    exit_code = accounts._account_main(["use"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert orphan_id in captured.out
    assert "not in the local registry" in captured.err
