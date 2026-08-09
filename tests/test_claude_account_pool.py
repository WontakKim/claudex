"""Tests for the ordered-fallback pool: cooldown tracker, chain, derivation."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from email.utils import format_datetime

from claudex_gateway.claude_account_pool import (
    AccountCooldownTracker,
    build_serving_chain,
    rate_limit_cooldown_seconds,
)
from claudex_gateway.claude_accounts import AccountRecord


class FakeClock:
    """A controllable stand-in for ``time.monotonic`` — no real sleeping."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _record(account_id: str, created_at: int, state: str = "ready") -> AccountRecord:
    return AccountRecord(
        id=account_id,
        email=f"{account_id}@example.com",
        organization_uuid=f"org-{account_id}",
        organization_name=None,
        created_at=created_at,
        updated_at=created_at,
        last_authenticated_at=created_at,
        state=state,
        account_incarnation_id=str(uuid.uuid4()),
        upstream_account_uuid=None,
    )


# ---------------------------------------------------------------------------
# AccountCooldownTracker
# ---------------------------------------------------------------------------


def test_mark_and_is_cooling_until_deadline() -> None:
    clock = FakeClock()
    tracker = AccountCooldownTracker(clock=clock)

    tracker.mark("a", 30.0)
    assert tracker.is_cooling("a")
    clock.advance(29.0)
    assert tracker.is_cooling("a")
    clock.advance(1.5)
    assert not tracker.is_cooling("a")
    assert not tracker.is_cooling("never-marked")


def test_remaining_seconds_counts_down_and_floors_at_zero() -> None:
    clock = FakeClock()
    tracker = AccountCooldownTracker(clock=clock)

    tracker.mark("a", 30.0)
    clock.advance(10.0)
    assert tracker.remaining_seconds("a") == 20.0
    clock.advance(40.0)
    assert tracker.remaining_seconds("a") == 0.0
    assert tracker.remaining_seconds("never-marked") == 0.0


def test_min_remaining_seconds_across_accounts_and_none_when_idle() -> None:
    clock = FakeClock()
    tracker = AccountCooldownTracker(clock=clock)

    assert tracker.min_remaining_seconds() is None
    tracker.mark("a", 30.0)
    tracker.mark("b", 10.0)
    assert tracker.min_remaining_seconds() == 10.0
    clock.advance(15.0)
    assert tracker.min_remaining_seconds() == 15.0
    clock.advance(20.0)
    assert tracker.min_remaining_seconds() is None


def test_later_mark_replaces_the_deadline() -> None:
    clock = FakeClock()
    tracker = AccountCooldownTracker(clock=clock)

    tracker.mark("a", 100.0)
    tracker.mark("a", 5.0)
    clock.advance(6.0)
    assert not tracker.is_cooling("a")


def test_expired_entries_are_pruned_lazily() -> None:
    clock = FakeClock()
    tracker = AccountCooldownTracker(clock=clock)

    tracker.mark("a", 5.0)
    clock.advance(10.0)
    assert tracker.remaining_seconds("a") == 0.0
    assert "a" not in tracker._deadlines


# ---------------------------------------------------------------------------
# build_serving_chain
# ---------------------------------------------------------------------------


def test_chain_puts_serving_account_first_then_created_at_order() -> None:
    tracker = AccountCooldownTracker(clock=FakeClock())
    records = [_record("aaa", 3_000), _record("bbb", 1_000), _record("ccc", 2_000)]

    chain = build_serving_chain("aaa", records, tracker)

    assert [record.id for record in chain.attempts] == ["aaa", "bbb", "ccc"]
    assert chain.cooling_ids == ()
    assert chain.serving_registered
    assert chain.serving_state == "ready"


def test_chain_created_at_tie_breaks_by_id() -> None:
    tracker = AccountCooldownTracker(clock=FakeClock())
    records = [_record("bbb", 1_000), _record("aaa", 1_000), _record("ccc", 1_000)]

    chain = build_serving_chain("ccc", records, tracker)

    assert [record.id for record in chain.attempts] == ["ccc", "aaa", "bbb"]


def test_chain_excludes_needs_reauth_accounts() -> None:
    tracker = AccountCooldownTracker(clock=FakeClock())
    records = [_record("aaa", 1_000), _record("bbb", 2_000, state="needs-reauth")]

    chain = build_serving_chain("aaa", records, tracker)

    assert [record.id for record in chain.attempts] == ["aaa"]
    assert chain.cooling_ids == ()


def test_chain_excludes_cooling_accounts_and_reports_them() -> None:
    clock = FakeClock()
    tracker = AccountCooldownTracker(clock=clock)
    tracker.mark("aaa", 60.0)
    records = [_record("aaa", 1_000), _record("bbb", 2_000), _record("ccc", 3_000)]

    chain = build_serving_chain("aaa", records, tracker)

    assert [record.id for record in chain.attempts] == ["bbb", "ccc"]
    assert chain.cooling_ids == ("aaa",)


def test_chain_readmits_serving_account_after_cooldown_expiry() -> None:
    clock = FakeClock()
    tracker = AccountCooldownTracker(clock=clock)
    tracker.mark("aaa", 60.0)
    records = [_record("aaa", 1_000), _record("bbb", 2_000)]

    clock.advance(61.0)
    chain = build_serving_chain("aaa", records, tracker)

    assert [record.id for record in chain.attempts] == ["aaa", "bbb"]
    assert chain.cooling_ids == ()


def test_chain_reports_unregistered_serving_id() -> None:
    tracker = AccountCooldownTracker(clock=FakeClock())
    records = [_record("bbb", 2_000)]

    chain = build_serving_chain("aaa", records, tracker)

    assert [record.id for record in chain.attempts] == ["bbb"]
    assert not chain.serving_registered
    assert chain.serving_state is None


def test_chain_reports_serving_account_state_when_not_ready() -> None:
    tracker = AccountCooldownTracker(clock=FakeClock())
    records = [_record("aaa", 1_000, state="needs-reauth"), _record("bbb", 2_000)]

    chain = build_serving_chain("aaa", records, tracker)

    assert [record.id for record in chain.attempts] == ["bbb"]
    assert chain.serving_registered
    assert chain.serving_state == "needs-reauth"


# ---------------------------------------------------------------------------
# rate_limit_cooldown_seconds
# ---------------------------------------------------------------------------

_NOW = 1_700_000_000.0


def _wall_clock() -> float:
    return _NOW


def test_cooldown_prefers_retry_after_delta_seconds() -> None:
    envelope = {"weekly": {"used_percent": 100.0, "resets_at": _NOW + 9_999.0}}

    seconds = rate_limit_cooldown_seconds(
        {"Retry-After": "42"}, b"{}", envelope, wall_clock=_wall_clock
    )

    assert seconds == 42.0


def test_cooldown_parses_retry_after_http_date() -> None:
    http_date = format_datetime(datetime.fromtimestamp(_NOW + 120.0, tz=timezone.utc))

    seconds = rate_limit_cooldown_seconds(
        {"retry-after": http_date}, b"{}", None, wall_clock=_wall_clock
    )

    assert abs(seconds - 120.0) < 1.0


def test_cooldown_uses_unified_reset_header_epoch() -> None:
    headers = {"anthropic-ratelimit-unified-reset": str(int(_NOW + 300.0))}

    seconds = rate_limit_cooldown_seconds(headers, b"{}", None, wall_clock=_wall_clock)

    assert seconds == 300.0


def test_cooldown_reads_reset_from_error_body() -> None:
    reset_iso = datetime.fromtimestamp(_NOW + 600.0, tz=timezone.utc).isoformat()
    body = json.dumps({"error": {"type": "rate_limit_error", "resets_at": reset_iso}}).encode()

    seconds = rate_limit_cooldown_seconds({}, body, None, wall_clock=_wall_clock)

    assert abs(seconds - 600.0) < 1.0


def test_cooldown_falls_back_to_min_exhausted_usage_window() -> None:
    envelope = {
        "session": {"used_percent": 0.0, "resets_at": None},
        "weekly": {"used_percent": 87.0, "resets_at": _NOW + 100.0},
        "fable_weekly": {"used_percent": 100.0, "resets_at": _NOW + 7_200.0},
    }

    seconds = rate_limit_cooldown_seconds({}, b"not json", envelope, wall_clock=_wall_clock)

    # weekly is below the exhaustion threshold, so only fable_weekly counts.
    assert seconds == 7_200.0

    envelope["weekly"]["used_percent"] = 100.0
    assert rate_limit_cooldown_seconds({}, b"", envelope, wall_clock=_wall_clock) == 100.0


def test_cooldown_defaults_and_clamps_to_bounds() -> None:
    # No signal anywhere → default.
    assert rate_limit_cooldown_seconds({}, b"", None, wall_clock=_wall_clock) == 60.0
    # Garbage header values fall through to the default instead of raising.
    garbage = {"retry-after": "soon", "anthropic-ratelimit-unified-reset": "tomorrow"}
    assert rate_limit_cooldown_seconds(garbage, b"[]", None, wall_clock=_wall_clock) == 60.0
    # Zero/negative Retry-After clamps up; a huge one clamps down to 7 days.
    assert rate_limit_cooldown_seconds({"retry-after": "0"}, b"", None, wall_clock=_wall_clock) == 5.0
    assert (
        rate_limit_cooldown_seconds({"retry-after": "999999999"}, b"", None, wall_clock=_wall_clock)
        == 7 * 24 * 3600.0
    )
