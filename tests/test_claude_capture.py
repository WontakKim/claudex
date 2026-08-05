"""Tests for the scoped-only Claude credential capture module."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import pytest

from claudex_gateway import claude_capture
from claudex_gateway.claude_capture import CaptureCancelled, CaptureError, CapturedAccount

# ---------------------------------------------------------------------------
# Fixtures and shared test doubles
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every test's `paths.runtime_dir()` (and thus the login lock file)
    under a throwaway directory instead of the real `~/.claudex`."""
    home_dir = tmp_path / "home"
    home_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))


class _FakeTTYStdin:
    def isatty(self) -> bool:
        return True


class _FakeNonTTYStdin:
    def isatty(self) -> bool:
        return False


def _make_stdin_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeTTYStdin())


class _KeychainCall:
    def __init__(self, op: str, service: str, account: str) -> None:
        self.op = op
        self.service = service
        self.account = account


class _FakeKeychainBackend:
    """In-process `KeychainBackend` double: no PATH, no `/usr/bin/security`."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.calls: list[_KeychainCall] = []
        self.read_failure: Exception | None = None
        self.delete_failure: Exception | None = None

    def read(self, service: str, account: str) -> str | None:
        self.calls.append(_KeychainCall("read", service, account))
        if self.read_failure is not None:
            raise self.read_failure
        return self.store.get((service, account))

    def delete(self, service: str, account: str) -> None:
        self.calls.append(_KeychainCall("delete", service, account))
        if self.delete_failure is not None:
            raise self.delete_failure
        self.store.pop((service, account), None)


def _sequential_mkdtemp(root: Path) -> Any:
    """A deterministic `tempfile.mkdtemp` stand-in: `<root>/<prefix><n>`."""
    counter = {"n": 0}

    def _mkdtemp(*, prefix: str = "", **_kwargs: object) -> str:
        counter["n"] += 1
        candidate = root / f"{prefix}{counter['n']}"
        candidate.mkdir()
        return str(candidate)

    return _mkdtemp


# ---------------------------------------------------------------------------
# Fake `claude` CLI (PATH-prepended, not in-process)
# ---------------------------------------------------------------------------

_FAKE_CLAUDE_SCRIPT = r"""
import json
import os
import subprocess
import sys
import time

record_dir = os.environ.get("CLAUDEX_TEST_RECORD_DIR")
argv = sys.argv[1:]

if argv == ["--version"]:
    if record_dir:
        with open(os.path.join(record_dir, "version-invocation.jsonl"), "a") as handle:
            handle.write(json.dumps({"path": sys.argv[0], "env": dict(os.environ)}) + "\n")
    print("__CLAUDE_VERSION__")
    sys.exit(0)

if argv[:3] == ["auth", "login", "--claudeai"]:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR", "")
    record = {"argv": argv, "env": dict(os.environ), "path": sys.argv[0]}
    with open(os.path.join(config_dir, "login-invocation.json"), "w") as handle:
        json.dump(record, handle)
    if record_dir:
        with open(os.path.join(record_dir, "login-invocation.json"), "w") as handle:
            json.dump(record, handle)

    mode = os.environ.get("CLAUDEX_FAKE_CLAUDE_MODE", "success")

    if mode == "success":
        with open(os.path.join(config_dir, ".credentials.json"), "w") as handle:
            json.dump(
                {
                    "claudeAiOauth": {
                        "accessToken": "fake-access-token",
                        "email": "fixture@example.com",
                    }
                },
                handle,
            )
        with open(os.path.join(config_dir, ".claude.json"), "w") as handle:
            json.dump({"oauthAccount": {"emailAddress": "fixture@example.com"}}, handle)
        sys.exit(0)

    if mode == "denied":
        sys.exit(1)

    if mode == "hang":
        with open(os.path.join(config_dir, "leader.pid"), "w") as handle:
            handle.write(str(os.getpid()))
        time.sleep(120)
        sys.exit(0)

    if mode == "hang_with_descendant":
        with open(os.path.join(config_dir, "leader.pid"), "w") as handle:
            handle.write(str(os.getpid()))
        descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open(os.path.join(config_dir, "descendant.pid"), "w") as handle:
            handle.write(str(descendant.pid))
        descendant.wait()
        sys.exit(0)

    sys.exit(1)

sys.exit(1)
"""


def _write_fake_claude(bin_dir: Path, *, version: str = "2.1.222") -> Path:
    claude_path = bin_dir / "claude"
    script = _FAKE_CLAUDE_SCRIPT.replace("__CLAUDE_VERSION__", version)
    claude_path.write_text(f"#!{sys.executable}\n{script}")
    claude_path.chmod(0o755)
    return claude_path


def _prepend_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


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
# Public data types
# ---------------------------------------------------------------------------


def test_captured_account_carries_all_five_fields() -> None:
    account = CapturedAccount(
        credentials_json={"a": 1},
        oauth_account_json={"b": 2},
        email="user@example.com",
        organization_uuid="org-1",
        organization_name="Acme",
    )
    assert account.credentials_json == {"a": 1}
    assert account.oauth_account_json == {"b": 2}
    assert account.email == "user@example.com"
    assert account.organization_uuid == "org-1"
    assert account.organization_name == "Acme"


def test_capture_cancelled_is_a_capture_error() -> None:
    assert issubclass(CaptureCancelled, CaptureError)


# ---------------------------------------------------------------------------
# Scoped selector algorithm (selector_vector)
# ---------------------------------------------------------------------------


def test_selector_vector_nfc_and_nfd_korean_paths_hash_identically() -> None:
    nfc_path = unicodedata.normalize("NFC", "/tmp/한글-설정")
    nfd_path = unicodedata.normalize("NFD", "/tmp/한글-설정")
    assert nfc_path != nfd_path  # sanity: genuinely distinct byte sequences
    assert claude_capture._scoped_keychain_service(
        nfc_path
    ) == claude_capture._scoped_keychain_service(nfd_path)


def test_selector_vector_trailing_slash_hashes_differently() -> None:
    base = "/tmp/claude-config"
    assert claude_capture._scoped_keychain_service(
        base
    ) != claude_capture._scoped_keychain_service(base + "/")


def test_selector_vector_repeated_slash_hashes_differently() -> None:
    assert claude_capture._scoped_keychain_service(
        "/tmp/claude-config"
    ) != claude_capture._scoped_keychain_service("/tmp//claude-config")


def test_selector_vector_dot_component_hashes_differently() -> None:
    assert claude_capture._scoped_keychain_service(
        "/tmp/claude-config"
    ) != claude_capture._scoped_keychain_service("/tmp/./claude-config")


def test_selector_vector_matches_the_documented_sha256_nfc_derivation() -> None:
    raw = "/tmp/claude-config"
    expected_suffix = hashlib.sha256(
        unicodedata.normalize("NFC", raw).encode("utf-8")
    ).hexdigest()[:8]
    assert (
        claude_capture._scoped_keychain_service(raw)
        == f"Claude Code-credentials-{expected_suffix}"
    )


def test_selector_vector_account_prefers_user_over_username(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("USERNAME", "bob")
    assert claude_capture._keychain_account() == "alice"


def test_selector_vector_account_falls_back_to_username(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setenv("USERNAME", "bob")
    assert claude_capture._keychain_account() == "bob"


def test_fail_closed_when_user_and_username_are_both_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    with pytest.raises(CaptureError):
        claude_capture._keychain_account()


# ---------------------------------------------------------------------------
# Production Keychain backend: exit-code classification (no real Keychain)
# ---------------------------------------------------------------------------


def test_production_backend_classifies_security_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"returncode": 0, "stdout": "the-password\n"}

    def _fake_run_security(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, state["returncode"], stdout=state["stdout"], stderr=""
        )

    monkeypatch.setattr(claude_capture, "_run_security", _fake_run_security)
    backend = claude_capture._SecurityKeychainBackend()

    state["returncode"], state["stdout"] = 0, "the-password\n"
    assert backend.read("svc", "acct") == "the-password"

    state["returncode"] = claude_capture._KEYCHAIN_ITEM_NOT_FOUND_STATUS
    assert backend.read("svc", "acct") is None

    state["returncode"] = 1
    with pytest.raises(CaptureError):
        backend.read("svc", "acct")

    state["returncode"] = 0
    backend.delete("svc", "acct")  # must not raise

    state["returncode"] = claude_capture._KEYCHAIN_ITEM_NOT_FOUND_STATUS
    backend.delete("svc", "acct")  # conclusively-missing counts as success

    state["returncode"] = 1
    with pytest.raises(CaptureError):
        backend.delete("svc", "acct")


# ---------------------------------------------------------------------------
# capture_from_config_dir: headless, strictly read-only (headless_read_only)
# ---------------------------------------------------------------------------


def test_headless_read_only_reads_via_scoped_keychain_on_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    monkeypatch.delenv("USERNAME", raising=False)
    config_dir = str(tmp_path / "claude-config")
    Path(config_dir).mkdir()

    claude_json_path = Path(config_dir) / ".claude.json"
    claude_json_path.write_text(
        json.dumps(
            {
                "oauthAccount": {
                    "emailAddress": "user@example.com",
                    "organizationUuid": "org-1",
                    "organizationName": "Acme",
                }
            }
        )
    )
    before_bytes = claude_json_path.read_bytes()
    before_mtime_ns = claude_json_path.stat().st_mtime_ns

    backend = _FakeKeychainBackend()
    service = claude_capture._scoped_keychain_service(config_dir)
    backend.store[(service, "tester")] = json.dumps(
        {"claudeAiOauth": {"accessToken": "tok", "email": "user@example.com"}}
    )
    monkeypatch.setattr(claude_capture, "_default_keychain_backend", lambda: backend)

    def _forbid_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("capture_from_config_dir must never spawn a process")

    monkeypatch.setattr(claude_capture.subprocess, "Popen", _forbid_spawn)
    monkeypatch.setattr(claude_capture.subprocess, "run", _forbid_spawn)

    account = claude_capture.capture_from_config_dir(config_dir)

    assert account.email == "user@example.com"
    assert account.organization_uuid == "org-1"
    assert account.organization_name == "Acme"
    assert [call.op for call in backend.calls] == ["read"]
    assert claude_json_path.read_bytes() == before_bytes
    assert claude_json_path.stat().st_mtime_ns == before_mtime_ns


def test_headless_read_only_reads_credentials_json_file_on_non_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_capture.sys, "platform", "linux")
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    credentials_path = config_dir / ".credentials.json"
    credentials_path.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "tok", "email": "user@example.com"}})
    )
    before_bytes = credentials_path.read_bytes()
    before_mtime_ns = credentials_path.stat().st_mtime_ns

    def _forbid_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("capture_from_config_dir must never spawn a process")

    monkeypatch.setattr(claude_capture.subprocess, "Popen", _forbid_spawn)
    monkeypatch.setattr(claude_capture.subprocess, "run", _forbid_spawn)

    account = claude_capture.capture_from_config_dir(str(config_dir))

    assert account.email == "user@example.com"
    assert credentials_path.read_bytes() == before_bytes
    assert credentials_path.stat().st_mtime_ns == before_mtime_ns


def test_missing_credentials_raises_on_non_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_capture.sys, "platform", "linux")
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    with pytest.raises(CaptureError):
        claude_capture.capture_from_config_dir(str(config_dir))


# ---------------------------------------------------------------------------
# Fail-closed behavior (fail_closed)
# ---------------------------------------------------------------------------


def test_fail_closed_when_scoped_keychain_item_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    backend = _FakeKeychainBackend()
    monkeypatch.setattr(claude_capture, "_default_keychain_backend", lambda: backend)
    config_dir = str(tmp_path / "claude-config")
    Path(config_dir).mkdir()

    with pytest.raises(CaptureError, match="Claude Code-credentials-"):
        claude_capture.capture_from_config_dir(config_dir)


def test_fail_closed_when_keychain_read_operationally_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    backend = _FakeKeychainBackend()
    backend.read_failure = CaptureError("Keychain unavailable")
    monkeypatch.setattr(claude_capture, "_default_keychain_backend", lambda: backend)
    config_dir = str(tmp_path / "claude-config")
    Path(config_dir).mkdir()

    with pytest.raises(CaptureError):
        claude_capture.capture_from_config_dir(config_dir)


# ---------------------------------------------------------------------------
# Identity resolution: precedence and cross-source conflict (conflict)
# ---------------------------------------------------------------------------


def test_identity_precedence_prefers_oauth_account_when_claude_ai_oauth_lacks_a_field() -> None:
    # Both sources agree on email, but only oauthAccount carries the
    # organization fields: those come through untouched, not treated as a
    # conflict, since claudeAiOauth simply has nothing to disagree with.
    email, org_uuid, org_name = claude_capture._resolve_identity(
        {"claudeAiOauth": {"email": "user@example.com"}},
        {
            "emailAddress": "user@example.com",
            "organizationUuid": "org-1",
            "organizationName": "Acme",
        },
    )
    assert (email, org_uuid, org_name) == ("user@example.com", "org-1", "Acme")


def test_identity_falls_back_to_claude_ai_oauth_when_oauth_account_missing() -> None:
    email, org_uuid, org_name = claude_capture._resolve_identity(
        {"claudeAiOauth": {"email": "only@example.com", "organizationUuid": "org-9"}},
        None,
    )
    assert (email, org_uuid, org_name) == ("only@example.com", "org-9", None)


def test_conflict_between_sources_on_email_raises() -> None:
    with pytest.raises(CaptureError, match="conflicting email"):
        claude_capture._resolve_identity(
            {"claudeAiOauth": {"email": "b@example.com"}},
            {"emailAddress": "a@example.com"},
        )


def test_conflict_between_sources_on_organization_uuid_raises() -> None:
    with pytest.raises(CaptureError, match="conflicting organizationUuid"):
        claude_capture._resolve_identity(
            {"claudeAiOauth": {"email": "a@example.com", "organizationUuid": "org-2"}},
            {"emailAddress": "a@example.com", "organizationUuid": "org-1"},
        )


def test_conflict_free_when_values_match_after_nfc_normalization() -> None:
    nfd_email = unicodedata.normalize("NFD", "café@example.com")
    nfc_email = unicodedata.normalize("NFC", "café@example.com")
    email, _org_uuid, _org_name = claude_capture._resolve_identity(
        {"claudeAiOauth": {"email": nfd_email}},
        {"emailAddress": nfc_email},
    )
    assert email == nfc_email  # the oauthAccount value wins once equivalence is proven


def test_missing_email_raises_capture_error() -> None:
    with pytest.raises(CaptureError):
        claude_capture._resolve_identity({"claudeAiOauth": {}}, None)


def test_malformed_claude_json_raises_capture_error(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".claude.json").write_text("{not valid json")
    with pytest.raises(CaptureError):
        claude_capture._read_oauth_account_block(str(config_dir))


def test_missing_claude_json_is_treated_as_absent_not_an_error(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    assert claude_capture._read_oauth_account_block(str(config_dir)) is None


def test_source_file_byte_cap_is_enforced(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    oversized = config_dir / ".claude.json"
    oversized.write_bytes(b"{" + b" " * (claude_capture._SOURCE_FILE_BYTE_CAP + 10) + b"}")
    with pytest.raises(CaptureError, match="byte read cap"):
        claude_capture._read_oauth_account_block(str(config_dir))


# ---------------------------------------------------------------------------
# Non-TTY interactive attempt (non_tty)
# ---------------------------------------------------------------------------


def test_non_tty_interactive_attempt_raises_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeNonTTYStdin())
    with pytest.raises(CaptureError, match="interactive terminal"):
        claude_capture.capture_interactive()


def test_windows_platform_is_unsupported_for_interactive_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_capture.sys, "platform", "win32")
    with pytest.raises(CaptureError, match="Windows"):
        claude_capture.capture_interactive()


# ---------------------------------------------------------------------------
# Version gate (version_gate)
# ---------------------------------------------------------------------------


def test_version_gate_rejects_unparseable_version_before_any_login_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_stdin_a_tty(monkeypatch)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude(bin_dir, version="not-a-version")
    _prepend_path(monkeypatch, bin_dir)

    with pytest.raises(CaptureError, match="parse"):
        claude_capture.capture_interactive(timeout_secs=5)

    assert not list(Path(tempfile.gettempdir()).glob(f"{claude_capture._TEMP_DIR_PREFIX}*"))


def test_version_gate_rejects_old_version_before_any_login_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_stdin_a_tty(monkeypatch)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude(bin_dir, version="2.1.100")
    _prepend_path(monkeypatch, bin_dir)
    monkeypatch.delenv("CLAUDEX_ALLOW_UNTESTED_CLAUDE", raising=False)

    with pytest.raises(CaptureError, match="2.1.100"):
        claude_capture.capture_interactive(timeout_secs=5)

    assert not list(Path(tempfile.gettempdir()).glob(f"{claude_capture._TEMP_DIR_PREFIX}*"))


def test_version_gate_rejects_unlisted_version_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir, version="1.0.0")
    monkeypatch.delenv("CLAUDEX_ALLOW_UNTESTED_CLAUDE", raising=False)

    with pytest.raises(CaptureError, match="1.0.0"):
        claude_capture._check_claude_version(str(claude_path))


def test_version_gate_opt_in_override_allows_untested_version_and_uses_same_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_stdin_a_tty(monkeypatch)
    monkeypatch.setattr(claude_capture.sys, "platform", "linux")

    record_dir = tmp_path / "record"
    record_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir, version="9.9.9")
    _prepend_path(monkeypatch, bin_dir)
    monkeypatch.setenv("CLAUDEX_ALLOW_UNTESTED_CLAUDE", "1")
    monkeypatch.setenv("CLAUDEX_TEST_RECORD_DIR", str(record_dir))
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "success")

    account = claude_capture.capture_interactive(timeout_secs=10)

    assert account.email == "fixture@example.com"

    version_calls = [
        json.loads(line)
        for line in (record_dir / "version-invocation.jsonl").read_text().splitlines()
    ]
    assert len(version_calls) == 1
    assert version_calls[0]["path"] == str(claude_path)
    assert version_calls[0]["env"]["DISABLE_UPDATES"] == "1"

    login_record = json.loads((record_dir / "login-invocation.json").read_text())
    assert login_record["path"] == str(claude_path)
    assert login_record["env"]["DISABLE_UPDATES"] == "1"

    # Cleanup ran: no leftover temp config dir.
    assert not list(Path(tempfile.gettempdir()).glob(f"{claude_capture._TEMP_DIR_PREFIX}*"))


def test_version_gate_aborts_if_the_resolved_executable_changes_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_stdin_a_tty(monkeypatch)
    monkeypatch.setattr(claude_capture.sys, "platform", "linux")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir, version="2.1.222")

    call_count = {"n": 0}
    real_stat_identity = claude_capture._stat_identity

    def _flaky_stat_identity(path: str) -> tuple[int, int, int, int]:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            # Pretend the binary changed underneath us between the version
            # check and the login spawn.
            return (999999, 999999, 0, 0)
        return real_stat_identity(path)

    monkeypatch.setattr(claude_capture, "_stat_identity", _flaky_stat_identity)
    _prepend_path(monkeypatch, bin_dir)

    with pytest.raises(CaptureError, match="changed"):
        claude_capture.capture_interactive(timeout_secs=5)

    assert claude_path.exists()  # sanity: the fake binary itself was untouched


# ---------------------------------------------------------------------------
# Environment scrubbing for the login child (env_scrub)
# ---------------------------------------------------------------------------


def test_env_scrub_removes_forbidden_vars_from_login_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "success")

    forbidden = {
        "ANTHROPIC_API_KEY": "sk-should-not-leak",
        "ANTHROPIC_AUTH_TOKEN": "should-not-leak-auth-token",
        "CLAUDE_CODE_OAUTH_TOKEN": "should-not-leak-oauth-token",
        "AWS_BEARER_TOKEN_BEDROCK": "should-not-leak-bedrock-token",
        "ANTHROPIC_BASE_URL": "https://example.invalid",
        "ANTHROPIC_CUSTOM_HEADERS": "x-secret: should-not-leak-headers",
        "CLAUDE_SECURESTORAGE_CONFIG_DIR": "/tmp/should-not-leak-securestorage",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_USE_FOUNDRY": "1",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS": "1",
        "CLAUDE_CODE_USE_MANTLE": "1",
        "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST": "1",
    }
    for key, value in forbidden.items():
        monkeypatch.setenv(key, value)

    claude_capture._run_login(str(claude_path), str(config_dir), timeout_secs=10)

    record = json.loads((config_dir / "login-invocation.json").read_text())
    assert record["argv"] == ["auth", "login", "--claudeai"]
    serialized_env = json.dumps(record["env"])
    for key, value in forbidden.items():
        assert key not in record["env"]
        # Only the distinctive sentinel values are checked for leakage; the
        # bare "1" selector values are too generic to prove anything on
        # their own once the key itself is confirmed absent.
        if value != "1":
            assert value not in serialized_env
    assert record["env"]["CLAUDE_CONFIG_DIR"] == str(config_dir)
    assert record["env"]["DISABLE_UPDATES"] == "1"
    assert record["env"].get("PATH")
    if "HOME" in os.environ:
        assert record["env"].get("HOME") == os.environ["HOME"]


def test_login_denied_with_nonzero_exit_raises_capture_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "denied")

    with pytest.raises(CaptureError, match="exited with status 1"):
        claude_capture._run_login(str(claude_path), str(config_dir), timeout_secs=10)


def test_token_sentinels_absent_from_login_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "denied")

    with pytest.raises(CaptureError):
        claude_capture._run_login(str(claude_path), str(config_dir), timeout_secs=10)

    record = json.loads((config_dir / "login-invocation.json").read_text())
    assert record["argv"] == ["auth", "login", "--claudeai"]
    assert not any("token" in arg.lower() or "secret" in arg.lower() for arg in record["argv"])


def test_token_sentinels_absent_from_malformed_credentials_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    backend = _FakeKeychainBackend()
    config_dir = str(tmp_path / "claude-config")
    Path(config_dir).mkdir()
    service = claude_capture._scoped_keychain_service(config_dir)
    secret_sentinel = "sk-ant-oat-super-secret-token-value"
    backend.store[(service, "tester")] = f"not valid json but contains {secret_sentinel}"
    monkeypatch.setattr(claude_capture, "_default_keychain_backend", lambda: backend)

    with pytest.raises(CaptureError) as exc_info:
        claude_capture.capture_from_config_dir(config_dir)

    assert secret_sentinel not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Timeout and signal handling kill the process group (timeout_kills_process_group)
# ---------------------------------------------------------------------------


def test_timeout_kills_process_group_and_reaps_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang_with_descendant")

    with pytest.raises(CaptureError, match="timed out"):
        claude_capture._run_login(str(claude_path), str(config_dir), timeout_secs=1)

    leader_pid = int(_wait_for_file(config_dir / "leader.pid"))
    descendant_pid = int(_wait_for_file(config_dir / "descendant.pid"))
    _assert_process_gone(leader_pid)
    _assert_process_gone(descendant_pid)


def test_timeout_kills_process_group_on_sigint_and_reaps_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang_with_descendant")

    def _deliver_sigint() -> None:
        _wait_for_file(config_dir / "descendant.pid")
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGINT)

    signaler = threading.Thread(target=_deliver_sigint, daemon=True)
    signaler.start()
    try:
        with pytest.raises(CaptureCancelled):
            claude_capture._run_login(str(claude_path), str(config_dir), timeout_secs=30)
    finally:
        signaler.join(timeout=5)

    leader_pid = int(_wait_for_file(config_dir / "leader.pid"))
    descendant_pid = int(_wait_for_file(config_dir / "descendant.pid"))
    _assert_process_gone(leader_pid)
    _assert_process_gone(descendant_pid)


def test_sigterm_during_login_wait_translates_to_capture_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")

    def _deliver_sigterm() -> None:
        _wait_for_file(config_dir / "leader.pid")
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    signaler = threading.Thread(target=_deliver_sigterm, daemon=True)
    signaler.start()
    try:
        with pytest.raises(CaptureCancelled):
            claude_capture._run_login(str(claude_path), str(config_dir), timeout_secs=30)
    finally:
        signaler.join(timeout=5)

    leader_pid = int(_wait_for_file(config_dir / "leader.pid"))
    _assert_process_gone(leader_pid)


def test_sighup_during_login_wait_translates_to_capture_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude_path = _write_fake_claude(bin_dir)
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")

    def _deliver_sighup() -> None:
        _wait_for_file(config_dir / "leader.pid")
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGHUP)

    signaler = threading.Thread(target=_deliver_sighup, daemon=True)
    signaler.start()
    try:
        with pytest.raises(CaptureCancelled):
            claude_capture._run_login(str(claude_path), str(config_dir), timeout_secs=30)
    finally:
        signaler.join(timeout=5)

    leader_pid = int(_wait_for_file(config_dir / "leader.pid"))
    _assert_process_gone(leader_pid)


# ---------------------------------------------------------------------------
# Temp config dir minting and its Keychain collision-retry
# ---------------------------------------------------------------------------


def test_mint_temp_config_dir_retries_on_keychain_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    backend = _FakeKeychainBackend()
    mkdtemp_root = tmp_path / "mkdtemp-root"
    mkdtemp_root.mkdir()
    monkeypatch.setattr(claude_capture.tempfile, "mkdtemp", _sequential_mkdtemp(mkdtemp_root))

    first_dir = str(mkdtemp_root / f"{claude_capture._TEMP_DIR_PREFIX}1")
    colliding_service = claude_capture._scoped_keychain_service(first_dir)
    backend.store[(colliding_service, "tester")] = "occupied by another login"

    result = claude_capture._mint_temp_config_dir(backend)

    assert result != first_dir
    assert not Path(first_dir).exists()  # discarded, never touched via delete
    assert Path(result).exists()
    assert all(call.op == "read" for call in backend.calls)  # never deletes a collision


def test_mint_temp_config_dir_gives_up_after_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    backend = _FakeKeychainBackend()
    mkdtemp_root = tmp_path / "mkdtemp-root"
    mkdtemp_root.mkdir()
    counter = {"n": 0}

    def _always_colliding_mkdtemp(*, prefix: str = "", **_kwargs: object) -> str:
        counter["n"] += 1
        candidate = mkdtemp_root / f"{prefix}{counter['n']}"
        candidate.mkdir()
        service = claude_capture._scoped_keychain_service(str(candidate))
        backend.store[(service, "tester")] = "occupied"
        return str(candidate)

    monkeypatch.setattr(claude_capture.tempfile, "mkdtemp", _always_colliding_mkdtemp)

    with pytest.raises(CaptureError):
        claude_capture._mint_temp_config_dir(backend)

    assert counter["n"] == claude_capture._MAX_TEMP_DIR_ATTEMPTS


def test_mint_temp_config_dir_skips_keychain_check_on_non_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_capture.sys, "platform", "linux")
    backend = _FakeKeychainBackend()
    mkdtemp_root = tmp_path / "mkdtemp-root"
    mkdtemp_root.mkdir()
    monkeypatch.setattr(claude_capture.tempfile, "mkdtemp", _sequential_mkdtemp(mkdtemp_root))

    result = claude_capture._mint_temp_config_dir(backend)

    assert Path(result).exists()
    assert backend.calls == []


# ---------------------------------------------------------------------------
# Cleanup: delete-on-missing succeeds; failures aggregate; both attempted
# ---------------------------------------------------------------------------


def test_cleanup_deletes_scoped_item_and_removes_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    backend = _FakeKeychainBackend()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    service = claude_capture._scoped_keychain_service(str(config_dir))
    backend.store[(service, "tester")] = "payload"

    claude_capture._cleanup_temp_config_dir(str(config_dir), backend)

    assert backend.store == {}
    assert not config_dir.exists()


def test_cleanup_attempts_both_actions_and_raises_when_either_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER", "tester")
    backend = _FakeKeychainBackend()
    backend.delete_failure = CaptureError("Keychain delete boom")
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with pytest.raises(CaptureError):
        claude_capture._cleanup_temp_config_dir(str(config_dir), backend)

    # The directory removal was still attempted despite the Keychain failure.
    assert not config_dir.exists()


# ---------------------------------------------------------------------------
# Full end-to-end success via a fake Keychain (mint -> login -> capture -> cleanup)
# ---------------------------------------------------------------------------


def test_capture_interactive_succeeds_end_to_end_with_fake_keychain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_stdin_a_tty(monkeypatch)
    monkeypatch.setenv("USER", "tester")
    monkeypatch.delenv("USERNAME", raising=False)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_claude(bin_dir, version="2.1.222")
    _prepend_path(monkeypatch, bin_dir)

    mkdtemp_root = tmp_path / "mkdtemp-root"
    mkdtemp_root.mkdir()
    monkeypatch.setattr(claude_capture.tempfile, "mkdtemp", _sequential_mkdtemp(mkdtemp_root))
    predicted_config_dir = str(mkdtemp_root / f"{claude_capture._TEMP_DIR_PREFIX}1")
    service = claude_capture._scoped_keychain_service(predicted_config_dir)

    backend = _FakeKeychainBackend()
    monkeypatch.setattr(claude_capture, "_default_keychain_backend", lambda: backend)

    def _fake_run_login(claude_path: str, config_dir: str, timeout_secs: int) -> None:
        assert config_dir == predicted_config_dir
        # Stand in for the real `claude auth login` writing its own scoped
        # Keychain item; this module never performs a Keychain write itself.
        backend.store[(service, "tester")] = json.dumps(
            {"claudeAiOauth": {"accessToken": "fake-access-token", "email": "fixture@example.com"}}
        )
        Path(config_dir, ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "fixture@example.com"}})
        )

    monkeypatch.setattr(claude_capture, "_run_login", _fake_run_login)

    account = claude_capture.capture_interactive(timeout_secs=10)

    assert account.email == "fixture@example.com"
    assert not Path(predicted_config_dir).exists()
    assert backend.store == {}
    assert [call.op for call in backend.calls] == ["read", "read", "delete"]


# ---------------------------------------------------------------------------
# Opt-in real macOS Keychain integration (Step 6)
# ---------------------------------------------------------------------------


def _security_add_generic_password(service: str, account: str, password: str) -> None:
    subprocess.run(
        [
            "/usr/bin/security",
            "add-generic-password",
            "-s",
            service,
            "-a",
            account,
            "-w",
            password,
            "-U",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="real Keychain integration is macOS-only")
@pytest.mark.skipif(
    os.environ.get("CLAUDEX_TEST_REAL_KEYCHAIN") != "1",
    reason="set CLAUDEX_TEST_REAL_KEYCHAIN=1 to exercise the real macOS Keychain",
)
def test_real_keychain_integration_found_missing_and_legacy_item_untouched(
    tmp_path: Path,
) -> None:
    backend = claude_capture._SecurityKeychainBackend()
    account = os.environ.get("USER") or os.environ.get("USERNAME")
    assert account, "USER/USERNAME must be set to run the real-Keychain integration test"

    config_dir = str(tmp_path / f"claudex-test-real-keychain-{uuid.uuid4().hex[:8]}")
    service = claude_capture._scoped_keychain_service(config_dir)

    legacy_service = "Claude Code-credentials"
    legacy_before = backend.read(legacy_service, account)

    # Conclusively missing before anything is added: a failed capture.
    assert backend.read(service, account) is None
    with pytest.raises(CaptureError):
        claude_capture.capture_from_config_dir(config_dir)

    _security_add_generic_password(
        service,
        account,
        json.dumps(
            {"claudeAiOauth": {"accessToken": "fixture-token", "email": "fixture@example.com"}}
        ),
    )
    try:
        assert backend.read(service, account) is not None
        captured = claude_capture.capture_from_config_dir(config_dir)  # a successful capture
        assert captured.email == "fixture@example.com"
    finally:
        backend.delete(service, account)
    assert backend.read(service, account) is None
    # Deleting an already-missing item is still success (tri-state contract).
    backend.delete(service, account)

    assert backend.read(legacy_service, account) == legacy_before

    temp_dir = tempfile.mkdtemp(prefix=claude_capture._TEMP_DIR_PREFIX)
    try:
        assert stat.S_IMODE(os.stat(temp_dir).st_mode) & 0o077 == 0
    finally:
        os.rmdir(temp_dir)
