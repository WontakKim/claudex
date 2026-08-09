"""Live capture probe for Anthropic's unified `anthropic-ratelimit-*` header
wire table (T-14; see the task's Context for the full ruling this exists to
satisfy: unified-header parsing must never be implemented from guessed
header names).

This is a standalone script, deliberately outside `src/claudex_gateway`: it
makes exactly two direct OAuth calls to `https://api.anthropic.com/v1/messages`
(`max_tokens: 1` each) and, on full success, publishes
`tests/fixtures/unified_ratelimit_headers.json` for later unified-header
parsing work to build against.

Model selection is a probe-only choice made once, before either call, by the
`--include-fable` CLI flag -- never a runtime routing knob:

- default: `claude-sonnet-5` then `claude-sonnet-5`
- `--include-fable`: `claude-sonnet-5` then `claude-fable-5` (only for when
  Fable eligibility is already known)

There is no retry and no third/fallback request, regardless of either
call's outcome. The fixture is published if and only if both responses are
2xx and both expose at least one `anthropic-ratelimit-*` header; any other
outcome exits non-zero with a credential-free diagnostic and leaves a
previously absent fixture absent. Authorization values, credential objects,
account profile values, and message content are never printed or persisted
-- only status, requested model, capture time, and verbatim lowercase
`anthropic-ratelimit-*` header names/values ever reach the fixture or stdout.

Reuses the gateway's own `ClaudeAccountAuthManager` against the first ready
registered account, so this probe observes exactly the credentials and
refresh path the gateway itself would use.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from claudex_gateway import claude_accounts, paths
from claudex_gateway.claude_auth import ClaudeAccountAuthError, ClaudeAccountAuthManager

_ANTHROPIC_API_BASE = "https://api.anthropic.com"
_MESSAGES_PATH = "/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_BETA = "oauth-2025-04-20"
_RATELIMIT_HEADER_PREFIX = "anthropic-ratelimit-"

_SONNET_MODEL = "claude-sonnet-5"
_FABLE_MODEL = "claude-fable-5"

# A fixed, minimal, non-sensitive prompt, held only in memory -- never
# logged or persisted, and never read back out of a response.
_PROBE_PROMPT = "Reply with a single word."

_REQUEST_TIMEOUT = httpx.Timeout(30.0)

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "unified_ratelimit_headers.json"


def select_model_pair(include_fable: bool) -> tuple[str, str]:
    """The two models to call, chosen once before either call is made."""
    return (_SONNET_MODEL, _FABLE_MODEL) if include_fable else (_SONNET_MODEL, _SONNET_MODEL)


@dataclass(frozen=True)
class CallResult:
    """One captured `/v1/messages` response -- never carries a credential or
    message-content value, only what the fixture is allowed to record."""

    model: str
    status: int | None  # None means the request never got a response at all.
    captured_at_utc: str
    ratelimit_headers: dict[str, str]

    @property
    def ok(self) -> bool:
        """True iff this call is eligible to contribute to a published fixture:
        a 2xx status exposing at least one ratelimit header."""
        return self.status is not None and 200 <= self.status < 300 and bool(self.ratelimit_headers)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _extract_ratelimit_headers(response: httpx.Response) -> dict[str, str]:
    """Verbatim `anthropic-ratelimit-*` headers, with lowercase names."""
    return {
        name.lower(): value
        for name, value in response.headers.items()
        if name.lower().startswith(_RATELIMIT_HEADER_PREFIX)
    }


async def _post_one_message(http_client: httpx.AsyncClient, access_token: str, model: str) -> CallResult:
    """Issue exactly one `max_tokens: 1` request for `model`. Never retried by
    this function or any caller; a transport failure is captured as a
    status-less `CallResult`, never raised."""
    headers = {
        "authorization": f"Bearer {access_token}",
        "anthropic-version": _ANTHROPIC_VERSION,
        "anthropic-beta": _OAUTH_BETA,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": _PROBE_PROMPT}],
    }
    captured_at_utc = _now_utc_iso()
    try:
        response = await http_client.post(
            f"{_ANTHROPIC_API_BASE}{_MESSAGES_PATH}",
            headers=headers,
            json=body,
            timeout=_REQUEST_TIMEOUT,
        )
    except httpx.HTTPError:
        return CallResult(model=model, status=None, captured_at_utc=captured_at_utc, ratelimit_headers={})
    return CallResult(
        model=model,
        status=response.status_code,
        captured_at_utc=captured_at_utc,
        ratelimit_headers=_extract_ratelimit_headers(response),
    )


async def probe_unified_headers(
    http_client: httpx.AsyncClient, access_token: str, *, include_fable: bool
) -> tuple[CallResult, CallResult]:
    """Issue exactly the two calls the flag selects, in order, unconditionally.

    The internal callable this module exists to make testable: no retry, and
    the second call is made regardless of the first call's outcome -- there
    is never a third request no matter how either of these two resolves.
    """
    model_a, model_b = select_model_pair(include_fable)
    first = await _post_one_message(http_client, access_token, model_a)
    second = await _post_one_message(http_client, access_token, model_b)
    return first, second


def _calls_are_publishable(calls: tuple[CallResult, CallResult]) -> bool:
    return all(call.ok for call in calls)


def _diagnostic_summary(calls: tuple[CallResult, CallResult]) -> str:
    """A credential-free, message-content-free description of why the pair
    of calls did not qualify for publication."""
    parts: list[str] = []
    for index, call in enumerate(calls, start=1):
        if call.status is None:
            parts.append(f"call {index} ({call.model}): no response (network or timeout failure)")
        elif not (200 <= call.status < 300):
            parts.append(f"call {index} ({call.model}): non-2xx status {call.status}")
        elif not call.ratelimit_headers:
            parts.append(
                f"call {index} ({call.model}): status {call.status} carried no "
                f"{_RATELIMIT_HEADER_PREFIX}* header"
            )
        else:
            parts.append(f"call {index} ({call.model}): ok")
    return "; ".join(parts)


def build_fixture_payload(calls: tuple[CallResult, CallResult]) -> dict[str, Any]:
    """The exact fixture shape: `{"calls": [...]}`, two entries, no other keys."""
    return {
        "calls": [
            {
                "model": call.model,
                "status": call.status,
                "captured_at_utc": call.captured_at_utc,
                "ratelimit_headers": dict(call.ratelimit_headers),
            }
            for call in calls
        ]
    }


def publish_fixture_atomic(fixture_path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish `payload` as `fixture_path`'s exact JSON content.

    Writes into a fresh, same-directory staging file and `os.replace`s it
    into place -- the sole commit point -- so a partially written file can
    never be observed under `fixture_path`'s name. Any staging file left
    over from a failed write is removed before the exception propagates.
    """
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = fixture_path.with_name(f".{fixture_path.name}.tmp-{uuid.uuid4().hex}")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        staging_path.write_text(text, encoding="utf-8")
        os.replace(staging_path, fixture_path)
    except BaseException:
        if staging_path.exists():
            staging_path.unlink()
        raise


@dataclass(frozen=True)
class CaptureOutcome:
    success: bool
    calls: tuple[CallResult, CallResult] | None
    diagnostic: str


async def run_capture(
    http_client: httpx.AsyncClient,
    account_dir: Path,
    *,
    include_fable: bool,
    fixture_path: Path = _FIXTURE_PATH,
) -> CaptureOutcome:
    """Run the exact two-call probe against `account_dir`'s credentials and
    publish `fixture_path` if and only if both calls qualify.

    Never raises for an expected failure mode (missing/malformed
    credentials, refresh failure, network failure, non-2xx status, missing
    ratelimit headers): every outcome is reported through the returned
    `CaptureOutcome` instead, and the fixture is left exactly as found unless
    `outcome.success` is true.
    """
    manager = ClaudeAccountAuthManager(account_dir, http_client)
    try:
        credentials = await manager.get_credentials()
    except ClaudeAccountAuthError as exc:
        return CaptureOutcome(
            success=False,
            calls=None,
            diagnostic=f"could not load account credentials ({type(exc).__name__})",
        )

    calls = await probe_unified_headers(http_client, credentials.access_token, include_fable=include_fable)
    if not _calls_are_publishable(calls):
        return CaptureOutcome(success=False, calls=calls, diagnostic=_diagnostic_summary(calls))

    publish_fixture_atomic(fixture_path, build_fixture_payload(calls))
    return CaptureOutcome(success=True, calls=calls, diagnostic="published fixture")


def _first_ready_account_id() -> str | None:
    """The id of the first ready registered Claude account, in the same
    `(createdAt, id)` order `claude_accounts.list_accounts` returns, or
    `None` when the registry is empty, unreadable, or holds no ready row."""
    try:
        records = claude_accounts.list_accounts()
    except claude_accounts.AccountRegistryError:
        return None
    for record in records:
        if record.state == "ready":
            return record.id
    return None


async def _async_main(*, include_fable: bool) -> int:
    account_id = _first_ready_account_id()
    if account_id is None:
        print("error: no ready registered claude account found", file=sys.stderr)
        return 1

    account_dir = paths.accounts_dir("claude") / account_id
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as http_client:
        outcome = await run_capture(http_client, account_dir, include_fable=include_fable)
    return _report_outcome(outcome)


def _report_outcome(outcome: CaptureOutcome) -> int:
    if not outcome.success:
        print(f"error: capture failed: {outcome.diagnostic}", file=sys.stderr)
        return 1
    print(f"published {_FIXTURE_PATH}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="capture_unified_headers",
        description=(
            "Make exactly two direct OAuth /v1/messages calls (max_tokens: 1 "
            "each) and publish tests/fixtures/unified_ratelimit_headers.json "
            "on full success."
        ),
    )
    parser.add_argument(
        "--include-fable",
        action="store_true",
        help=(
            "Probe-only model selection, decided before either call: call "
            f"{_SONNET_MODEL} then {_FABLE_MODEL} instead of the default "
            f"{_SONNET_MODEL} twice. Only for when Fable eligibility is "
            "already known -- never a runtime routing knob."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_async_main(include_fable=args.include_fable))


if __name__ == "__main__":
    raise SystemExit(main())
