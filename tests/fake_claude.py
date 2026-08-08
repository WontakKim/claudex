"""Shared fake `claude` CLI for login-capture tests (PATH-prepended, not
in-process). Modes are selected via CLAUDEX_FAKE_CLAUDE_MODE:

- ``success``            write credentials immediately, exit 0 (CLI tests)
- ``hang``               sleep two minutes (timeout/cancel tests)
- ``piped-url-code``     print the authorize URL + paste prompt, gate the
                         credential write on reading ``good-code`` from stdin
- ``piped-autocomplete`` print the URL, then self-complete (the browser's
                         localhost-callback path)
- ``piped-denied``       print the URL and an access_denied line, then hang —
                         proving the session fails fast on denial

The URL uses the real ``claude.com/cai/oauth/authorize`` shape so the login
session's spike-derived regex matches. ``CLAUDEX_TEST_LEADER_PID_FILE`` gets
the child's pid for process-group assertions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

FAKE_CLAUDE_SCRIPT = r"""
import json
import os
import sys
import time

argv = sys.argv[1:]

if argv == ["--version"]:
    print("2.1.222")
    sys.exit(0)


def write_credentials(config_dir, email):
    with open(os.path.join(config_dir, ".credentials.json"), "w") as handle:
        json.dump({"claudeAiOauth": {"accessToken": "fake-token", "email": email}}, handle)
    with open(os.path.join(config_dir, ".claude.json"), "w") as handle:
        json.dump({"oauthAccount": {"emailAddress": email}}, handle)


if argv[:3] == ["auth", "login", "--claudeai"]:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR", "")
    mode = os.environ.get("CLAUDEX_FAKE_CLAUDE_MODE", "success")
    email = os.environ.get("CLAUDEX_FAKE_CLAUDE_EMAIL", "fixture@example.com")
    leader_pid_file = os.environ.get("CLAUDEX_TEST_LEADER_PID_FILE")
    if leader_pid_file:
        with open(leader_pid_file, "w") as handle:
            handle.write(str(os.getpid()))

    if mode == "success":
        write_credentials(config_dir, email)
        sys.exit(0)

    if mode == "hang":
        time.sleep(120)
        sys.exit(0)

    if mode == "piped-url-code":
        print("Opening browser to sign in…")
        print(
            "If the browser didn't open, visit: "
            "https://claude.com/cai/oauth/authorize?code=true&client_id=fixture&state=s1"
        )
        print("Paste code here if prompted > ", end="")
        sys.stdout.flush()
        line = sys.stdin.readline().strip()
        if line == "good-code":
            write_credentials(config_dir, email)
            sys.exit(0)
        print("OAuth error: invalid code")
        sys.exit(1)

    if mode == "piped-autocomplete":
        print("Opening browser to sign in…")
        print(
            "If the browser didn't open, visit: "
            "https://claude.com/cai/oauth/authorize?code=true&client_id=fixture&state=s2"
        )
        sys.stdout.flush()
        time.sleep(0.2)
        write_credentials(config_dir, email)
        sys.exit(0)

    if mode == "piped-denied":
        print(
            "If the browser didn't open, visit: "
            "https://claude.com/cai/oauth/authorize?code=true&client_id=fixture&state=s3"
        )
        print("OAuth error: access_denied")
        sys.stdout.flush()
        time.sleep(120)
        sys.exit(1)

    sys.exit(1)

sys.exit(1)
"""


def write_fake_claude(bin_dir: Path) -> Path:
    claude_path = bin_dir / "claude"
    claude_path.write_text(f"#!{sys.executable}\n{FAKE_CLAUDE_SCRIPT}")
    claude_path.chmod(0o755)
    return claude_path


def prepend_fake_claude(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    write_fake_claude(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
