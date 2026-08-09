"""Tests for domain-separated session-key derivation and HRW routing math."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import Any

import pytest

from claudex_gateway.claude_balanced_router import (
    AccountCandidate,
    ClaudeBalancedRouter,
    NoEligibleAccountError,
    ObservationView,
    PendingDurabilityBarrier,
    PlacementResult,
    SessionKey,
    account_headroom,
    binding_windows,
    derive_session_key,
    derive_stateless_routing_digest,
    emergency_capacity,
    emergency_weight,
    freshness_adjusted_pressure,
    hrw_unit_interval,
    pick_weighted_hrw,
    quota_family,
    resolve_tie_break,
    select_weights,
    unknown_floor,
    warning_factor,
)
from claudex_gateway.claude_pool_runtime_state import (
    ClaudePoolRuntimeStateStore,
    PinRow,
    RestoreResult,
    RestoreValidationContext,
)


def _claude_code_user_id(session_id: str, account_uuid: str = "client-account-uuid") -> str:
    """Build the JSON string Claude Code sends as `metadata.user_id`."""
    return json.dumps(
        {"device_id": "d" * 64, "account_uuid": account_uuid, "session_id": session_id},
        separators=(",", ":"),
    )


def _reference_session_key_digest(seed: bytes, kind: bytes, canonical_utf8: bytes) -> bytes:
    """Reimplements the task's documented HMAC framing, independent of the module."""
    frame = (
        b"claudex-session-key-v1\x00"
        + kind
        + b"\x00"
        + len(canonical_utf8).to_bytes(8, "big")
        + canonical_utf8
    )
    return hmac.new(seed, frame, hashlib.sha256).digest()


def test_uuid_branch_is_case_insensitive_and_yields_the_same_digest() -> None:
    seed = b"seed-case-insensitive"
    lower = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
    upper = lower.upper()
    lower_body = {"metadata": {"user_id": _claude_code_user_id(lower)}}
    upper_body = {"metadata": {"user_id": _claude_code_user_id(upper)}}

    lower_key = derive_session_key(lower_body, seed)
    upper_key = derive_session_key(upper_body, seed)

    assert lower_key is not None and upper_key is not None
    assert lower_key.kind == upper_key.kind == "uuid"
    assert lower_key.digest == upper_key.digest
    assert lower_key.digest == _reference_session_key_digest(
        seed, b"uuid", lower.encode("utf-8")
    )


def test_non_rfc4122_variant_falls_back_to_content_hash_branch() -> None:
    seed = b"seed-non-rfc4122"
    ncs_variant_uuid = "11111111-2222-4333-0444-555555555555"
    body = {
        "metadata": {"user_id": _claude_code_user_id(ncs_variant_uuid)},
        "messages": [{"role": "user", "content": "hello"}],
    }

    key = derive_session_key(body, seed)

    assert key is not None
    assert key.kind == "content_hash"


def test_whitespace_padded_uuid_is_rejected_and_falls_back_to_content_hash() -> None:
    seed = b"seed-whitespace"
    padded_uuid = " 3fa85f64-5717-4562-b3fc-2c963f66afa6 "
    body = {
        "metadata": {"user_id": _claude_code_user_id(padded_uuid)},
        "messages": [{"role": "user", "content": "hello"}],
    }

    key = derive_session_key(body, seed)

    assert key is not None
    assert key.kind == "content_hash"


def test_content_hash_digest_is_deterministic_and_key_order_independent() -> None:
    seed = b"seed-content-hash"
    message_in_order = {"role": "user", "content": "café ☕ unicode test", "id": "m-1"}
    message_reordered = {"id": "m-1", "content": "café ☕ unicode test", "role": "user"}
    body_a = {"messages": [message_in_order]}
    body_b = {"messages": [message_reordered]}

    key_a = derive_session_key(body_a, seed)
    key_b = derive_session_key(body_b, seed)

    assert key_a is not None and key_b is not None
    assert key_a.kind == key_b.kind == "content_hash"
    assert key_a.digest == key_b.digest
    expected_canonical_utf8 = json.dumps(
        message_in_order,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert key_a.digest == _reference_session_key_digest(
        seed, b"content_hash", expected_canonical_utf8
    )


def test_no_user_message_and_no_metadata_yields_no_session_key() -> None:
    seed = b"seed-no-session-key"
    body = {"messages": [{"role": "assistant", "content": "hi there"}]}

    assert derive_session_key(body, seed) is None
    assert derive_session_key({}, seed) is None


def test_uuid_and_content_hash_digests_differ_for_identical_underlying_string() -> None:
    seed = b"seed-domain-separation"
    canonical_utf8 = b"identical-underlying-bytes"

    uuid_domain_digest = _reference_session_key_digest(seed, b"uuid", canonical_utf8)
    content_hash_domain_digest = _reference_session_key_digest(
        seed, b"content_hash", canonical_utf8
    )

    assert uuid_domain_digest != content_hash_domain_digest


def test_hrw_unit_interval_stays_in_the_open_interval() -> None:
    seed = b"seed-hrw-range"
    digest = hashlib.sha256(b"a-session-key-digest").digest()

    for account_id in ["account-1", "account-2", "account-3", ""]:
        sample = hrw_unit_interval(seed, digest, account_id)
        assert 0.0 < sample < 1.0


def test_hrw_unit_interval_is_deterministic_under_a_fixed_seed() -> None:
    seed = b"seed-hrw-deterministic"
    digest = hashlib.sha256(b"a-session-key-digest").digest()

    first = hrw_unit_interval(seed, digest, "account-1")
    second = hrw_unit_interval(seed, digest, "account-1")

    assert first == second


def test_hrw_unit_interval_changes_under_a_rotated_seed() -> None:
    digest = hashlib.sha256(b"a-session-key-digest").digest()

    with_original_seed = hrw_unit_interval(b"seed-hrw-original", digest, "account-1")
    with_rotated_seed = hrw_unit_interval(b"seed-hrw-rotated", digest, "account-1")

    assert with_original_seed != with_rotated_seed


def test_stateless_digest_is_deterministic_for_a_fixed_seed_and_nonce() -> None:
    seed = b"seed-stateless"
    nonce = b"n" * 32

    assert derive_stateless_routing_digest(seed, nonce) == derive_stateless_routing_digest(
        seed, nonce
    )


def test_stateless_digest_differs_across_nonces() -> None:
    seed = b"seed-stateless"

    first = derive_stateless_routing_digest(seed, b"n" * 32)
    second = derive_stateless_routing_digest(seed, b"m" * 32)

    assert first != second


def test_stateless_digest_uses_a_distinct_domain_from_session_key_digests() -> None:
    seed = b"seed-stateless-domain"
    nonce = b"n" * 32

    stateless_digest = derive_stateless_routing_digest(seed, nonce)
    uuid_domain_digest = _reference_session_key_digest(seed, b"uuid", nonce)
    content_hash_domain_digest = _reference_session_key_digest(seed, b"content_hash", nonce)

    assert stateless_digest != uuid_domain_digest
    assert stateless_digest != content_hash_domain_digest


# ==========================================================================
# Balanced picker core: pressures, weighted HRW, pin map (T-7)
# ==========================================================================


class _FakeClock:
    """A manually advanced monotonic-like clock for deterministic freshness/TTL tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _make_router(*, store: Any = None, clock: Any = None, wall_clock: Any = None, **overrides: Any) -> ClaudeBalancedRouter:
    return ClaudeBalancedRouter(
        balanced_epoch_id=overrides.pop("balanced_epoch_id", "epoch-test"),
        store=store,
        clock=clock if clock is not None else _FakeClock(),
        wall_clock=wall_clock if wall_clock is not None else _FakeClock(2_000_000.0),
        **overrides,
    )


def _candidate(account_id: str, **overrides: Any) -> AccountCandidate:
    base: dict[str, Any] = {"account_id": account_id, "account_incarnation_id": f"inc-{account_id}"}
    base.update(overrides)
    return AccountCandidate(**base)


class _FakePendingWrite:
    """A controllable stand-in for `ClaudePoolRuntimeStateStore.PendingWrite`."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._error: BaseException | None = None

    def resolve(self, *, error: BaseException | None = None) -> None:
        self._error = error
        self._event.set()

    async def wait_async(self) -> None:
        await self._event.wait()
        if self._error is not None:
            raise self._error


class _FakeStore:
    """A controllable stand-in store: records every `upsert_pin`/`touch_pin_last_seen` call."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.pending_writes: list[_FakePendingWrite] = []

    def upsert_pin(self, **kwargs: Any) -> _FakePendingWrite:
        self.upsert_calls.append(kwargs)
        pending_write = _FakePendingWrite()
        self.pending_writes.append(pending_write)
        return pending_write

    def touch_pin_last_seen(self, session_key_digest: bytes, last_seen_utc: float) -> _FakePendingWrite:
        pending_write = _FakePendingWrite()
        self.pending_writes.append(pending_write)
        return pending_write


# -- freshness ladder (4/10/20/31 min) --------------------------------------


def test_freshness_ladder_keeps_the_exact_value_within_five_minutes() -> None:
    assert freshness_adjusted_pressure(42.0, 4 * 60, reset_passed=False) == 42.0


def test_freshness_ladder_adds_five_points_between_five_and_fifteen_minutes() -> None:
    assert freshness_adjusted_pressure(42.0, 10 * 60, reset_passed=False) == 47.0


def test_freshness_ladder_adds_ten_points_between_fifteen_and_thirty_minutes() -> None:
    assert freshness_adjusted_pressure(42.0, 20 * 60, reset_passed=False) == 52.0


def test_freshness_ladder_is_unknown_past_thirty_one_minutes() -> None:
    assert freshness_adjusted_pressure(42.0, 31 * 60, reset_passed=False) is None


def test_freshness_ladder_is_unknown_once_the_reset_has_passed_even_if_fresh() -> None:
    assert freshness_adjusted_pressure(42.0, 60, reset_passed=True) is None


# -- unknown_floor ------------------------------------------------------------


def test_unknown_floor_is_zero_with_no_complete_pressures() -> None:
    assert unknown_floor([]) == 0.0


def test_unknown_floor_is_the_minimum_complete_pressure_plus_ten() -> None:
    assert unknown_floor([20.0, 55.0, 90.0]) == 30.0


def test_unknown_floor_is_capped_at_ninety() -> None:
    assert unknown_floor([85.0]) == 90.0
    assert unknown_floor([70.0]) == 80.0


def test_unknown_floor_lets_a_partial_candidate_keep_its_own_known_high_window() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock)
    router.ingest_observation("acct-a", "five_hour", used_percent=95.0, source="usage_api")
    router.ingest_observation("acct-c", "five_hour", used_percent=10.0, source="usage_api")
    now = clock()
    account_ids = ["acct-a", "acct-b", "acct-c"]

    floor = router.candidate_set_unknown_floor(account_ids, "default", now=now)
    assert floor == 20.0  # min(95, 10) + 10

    # acct-a's own known high window survives the max(), unaffected by the lower floor;
    # acct-b (fully unknown) and acct-c's unknown seven_day both fall back to the floor.
    assert router.account_pressure("acct-a", "default", now=now, floor=floor) == 95.0
    assert router.account_pressure("acct-b", "default", now=now, floor=floor) == 20.0
    assert router.account_pressure("acct-c", "default", now=now, floor=floor) == 20.0


def test_observation_view_applies_the_freshness_ladder_and_the_matching_warning_haircut() -> None:
    view = ObservationView()
    view.ingest_window(
        "acct-a", "five_hour", used_percent=30.0, source="usage_api", observed_at=100.0, reset_identity="reset-1"
    )
    assert view.window_pressure("acct-a", "five_hour", now=100.0 + 4 * 60) == 30.0
    assert view.window_pressure("acct-a", "five_hour", now=100.0 + 40 * 60) is None  # stale past 30 min

    assert view.has_active_warning("acct-a", "five_hour", now=200.0) is False
    view.ingest_allowed_warning("acct-a", "five_hour", observed_at=150.0, reset_identity="reset-1")
    assert view.has_active_warning("acct-a", "five_hour", now=150.0 + 60) is True
    assert view.has_active_warning("acct-a", "five_hour", now=150.0 + 10 * 60) is False  # stale (>5 min)

    # a fresh window reading under a new reset identity invalidates the retained warning.
    view.ingest_window(
        "acct-a", "five_hour", used_percent=30.0, source="usage_api", observed_at=100.0, reset_identity="reset-2"
    )
    assert view.has_active_warning("acct-a", "five_hour", now=150.0 + 60) is False


# -- quota_family / fable adjudication G --------------------------------------


def test_quota_family_matches_fable_as_a_bounded_case_insensitive_token() -> None:
    assert quota_family("claude-fable-5") == "fable"
    assert quota_family("FABLE") == "fable"
    assert quota_family("fable") == "fable"
    assert binding_windows("fable") == ("five_hour", "seven_day", "fable_weekly")


def test_quota_family_rejects_fable_as_a_substring_of_a_larger_word() -> None:
    assert quota_family("unfable") == "default"
    assert quota_family("fabled") == "default"
    assert quota_family("claude-sonnet-5") == "default"
    assert binding_windows("default") == ("five_hour", "seven_day")


# -- positive-set rule + amended emergency branch -----------------------------


def test_positive_set_rule_excludes_the_zero_weight_account_when_a_positive_candidate_exists() -> None:
    weights = select_weights(
        ["zero", "positive"],
        pressures={"zero": 100.0, "positive": 20.0},
        warning_factors={"zero": 1.0, "positive": 1.0},
        in_flight={"zero": 0, "positive": 0},
    )
    assert weights == {"positive": 80.0}


def test_positive_set_rule_falls_back_to_the_emergency_branch_once_every_weight_is_zero() -> None:
    weights = select_weights(
        ["a", "b"],
        pressures={"a": 100.0, "b": 100.0},
        warning_factors={"a": 1.0, "b": 1.0},
        in_flight={"a": 0, "b": 0},
    )
    assert set(weights) == {"a", "b"}
    assert all(weight > 0 for weight in weights.values())


def test_emergency_capacity_floors_at_one() -> None:
    assert emergency_capacity(99.0, 1.0) == 1.0  # (100-99)*1 = 1, already at the floor
    assert emergency_capacity(0.0, 1.0) == 100.0
    assert emergency_capacity(50.0, 0.5) == 25.0


def test_emergency_branch_matches_the_amended_c0_formula_for_gate_3_s4() -> None:
    # gate-3 S4: B at P=20, C at P=60, both in_flight-zeroed to an all-zero ordinary headroom
    # (the positive-set is empty) -> the amended emergency branch's C0 = max(1, (100-P)*factor)
    # and W = C0^2/(C0+2M) decide the draw instead.
    pressure_b, pressure_c = 20.0, 60.0
    in_flight_b, in_flight_c = 40, 20  # exactly zeroes each account's ordinary headroom
    factor = 0.5  # both accounts retain a fresh, matching allowed_warning haircut

    assert account_headroom(pressure_b, in_flight_b) == 0.0
    assert account_headroom(pressure_c, in_flight_c) == 0.0

    weights = select_weights(
        ["b", "c"],
        pressures={"b": pressure_b, "c": pressure_c},
        warning_factors={"b": factor, "c": factor},
        in_flight={"b": in_flight_b, "c": in_flight_c},
    )

    C0_b = max(1.0, (100.0 - pressure_b) * factor)
    C0_c = max(1.0, (100.0 - pressure_c) * factor)
    expected_weight_b = C0_b**2 / (C0_b + 2 * in_flight_b)
    expected_weight_c = C0_c**2 / (C0_c + 2 * in_flight_c)
    assert weights["b"] == pytest.approx(expected_weight_b)
    assert weights["c"] == pytest.approx(expected_weight_c)
    assert weights["b"] == pytest.approx(emergency_weight(pressure_b, factor, in_flight_b))
    assert weights["c"] == pytest.approx(emergency_weight(pressure_c, factor, in_flight_c))

    # B's share of the C0-proportional weight: C0_b^2/(C0_b+2M_b) vs C0_c^2/(C0_c+2M_c) ~= 66.67%.
    share_b = weights["b"] / (weights["b"] + weights["c"])
    assert share_b == pytest.approx(2 / 3, abs=0.005)


# -- tie-break determinism -----------------------------------------------------


def test_resolve_tie_break_prefers_the_serving_pin_when_it_is_tied() -> None:
    assert resolve_tie_break(["b", "a", "c"], serving_account_id="c") == "c"


def test_resolve_tie_break_falls_back_to_lexical_order_without_a_tied_serving_pin() -> None:
    assert resolve_tie_break(["b", "a"], serving_account_id=None) == "a"
    assert resolve_tie_break(["b", "a"], serving_account_id="not-tied") == "a"


def test_pick_weighted_hrw_tie_break_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    import claudex_gateway.claude_balanced_router as router_module

    monkeypatch.setattr(router_module, "hrw_unit_interval", lambda seed, digest, account_id: 0.5)
    weights = {"b": 10.0, "a": 10.0, "c": 10.0}

    winner = pick_weighted_hrw(weights=weights, seed=b"seed", session_key_digest=b"digest", serving_account_id="c")
    assert winner == "c"
    again = pick_weighted_hrw(weights=weights, seed=b"seed", session_key_digest=b"digest", serving_account_id="c")
    assert again == winner

    without_serving_tie = pick_weighted_hrw(
        weights=weights, seed=b"seed", session_key_digest=b"digest", serving_account_id="not-in-pool"
    )
    assert without_serving_tie == "a"


# -- N=1 pool trivially converges ----------------------------------------------


def test_single_account_pool_always_converges_regardless_of_session_key() -> None:
    router = _make_router()
    candidates = [_candidate("only-account")]
    for i in range(20):
        session_key = SessionKey(digest=hashlib.sha256(f"n1-{i}".encode()).digest(), kind="content_hash")
        result = router.place_session(
            session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"n1-seed"
        )
        assert result.account_id == "only-account"


# -- cold start / not-cold-start (design v2 §2.4) ------------------------------


def test_cold_start_routes_the_first_session_to_the_serving_pin() -> None:
    router = _make_router()
    candidates = [_candidate("acct-a"), _candidate("acct-b")]
    session_key = SessionKey(digest=b"\x01" * 32, kind="content_hash")

    result = router.place_session(
        session_key=session_key,
        model="claude-sonnet-5",
        candidates=candidates,
        seed=b"cold-start-seed",
        serving_account_id="acct-b",
    )

    assert result.account_id == "acct-b"
    assert result.created is True


def test_cold_start_does_not_apply_once_a_window_is_known() -> None:
    router = _make_router()
    router.ingest_observation("acct-a", "five_hour", used_percent=10.0, source="usage_api")
    candidates = [_candidate("acct-a"), _candidate("acct-b")]
    seed = b"cold-start-known-window"
    session_key = SessionKey(digest=b"\x01" * 32, kind="content_hash")

    result = router.place_session(
        session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=seed, serving_account_id="acct-b"
    )

    floor = unknown_floor([10.0])
    weights = select_weights(
        ["acct-a", "acct-b"],
        pressures={"acct-a": max(10.0, floor), "acct-b": floor},
        warning_factors={"acct-a": 1.0, "acct-b": 1.0},
        in_flight={"acct-a": 0, "acct-b": 0},
    )
    expected_winner = pick_weighted_hrw(
        weights=weights, seed=seed, session_key_digest=session_key.digest, serving_account_id="acct-b"
    )
    assert result.account_id == expected_winner
    assert result.account_id == "acct-a"  # this seed/digest pair genuinely differs from the serving pin


def test_not_cold_start_once_a_live_pin_exists_uses_weighted_hrw() -> None:
    router = _make_router()
    candidates = [_candidate("acct-a"), _candidate("acct-b")]
    seed = b"not-cold-start-seed"

    first_key = SessionKey(digest=b"\x01" * 32, kind="content_hash")
    router.place_session(
        session_key=first_key, model="claude-sonnet-5", candidates=candidates, seed=seed, serving_account_id="acct-b"
    )
    assert router.pin_count() == 1

    second_key = SessionKey(digest=b"\x04" * 32, kind="content_hash")
    result = router.place_session(
        session_key=second_key, model="claude-sonnet-5", candidates=candidates, seed=seed, serving_account_id="acct-b"
    )

    weights = select_weights(
        ["acct-a", "acct-b"],
        pressures={"acct-a": 0.0, "acct-b": 0.0},
        warning_factors={"acct-a": 1.0, "acct-b": 1.0},
        in_flight={"acct-a": 0, "acct-b": 0},
    )
    expected_winner = pick_weighted_hrw(
        weights=weights, seed=seed, session_key_digest=second_key.digest, serving_account_id="acct-b"
    )
    assert result.account_id == expected_winner
    assert result.account_id == "acct-a"  # not forced to the serving pin a second time


def test_restored_pin_for_current_epoch_prevents_cold_start_even_after_it_later_expires() -> None:
    router = _make_router()
    restored_digest = b"\x09" * 32
    restore_result = RestoreResult(
        pins={
            restored_digest: PinRow(
                session_key_digest=restored_digest,
                key_kind="content_hash",
                account_id="acct-a",
                account_incarnation_id="inc-acct-a",
                last_seen_utc=0.0,
                expires_at_utc=0.0,
                generation=0,
                balanced_epoch_id="epoch-test",
            )
        },
        cooldowns={},
        usage_observations={},
        capability_evidence={},
        skip_counts={},
    )
    router.restore_from_store(restore_result, now=0.0, wall_now=0.0)
    assert router.pin_count() == 1

    candidates = [_candidate("acct-a"), _candidate("acct-b")]
    session_key = SessionKey(digest=b"\x0a" * 32, kind="content_hash")
    result = router.place_session(
        session_key=session_key,
        model="claude-sonnet-5",
        candidates=candidates,
        seed=b"seed",
        serving_account_id="acct-b",
    )

    # `place_session` purges the now-expired restored pin itself, but the restore having
    # happened at all still blocks the cold-start rule for good.
    assert router.get_pin(restored_digest) is None
    weights = select_weights(
        ["acct-a", "acct-b"],
        pressures={"acct-a": 0.0, "acct-b": 0.0},
        warning_factors={"acct-a": 1.0, "acct-b": 1.0},
        in_flight={"acct-a": 0, "acct-b": 0},
    )
    expected_winner = pick_weighted_hrw(
        weights=weights, seed=b"seed", session_key_digest=session_key.digest, serving_account_id="acct-b"
    )
    assert result.account_id == expected_winner


def test_place_session_raises_when_no_candidate_is_eligible() -> None:
    router = _make_router()
    candidates = [_candidate("acct-a", ready=False)]
    session_key = SessionKey(digest=b"\x0b" * 32, kind="content_hash")

    with pytest.raises(NoEligibleAccountError):
        router.place_session(session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")


# -- statistical distribution (seeded, deterministic) --------------------------


def _place_many_sessions(
    router: ClaudeBalancedRouter, candidates: list[AccountCandidate], *, seed: bytes, total: int, prefix: str
) -> dict[str, int]:
    counts = {candidate.account_id: 0 for candidate in candidates}
    for i in range(total):
        digest = hashlib.sha256(f"{prefix}-{i}".encode()).digest()
        session_key = SessionKey(digest=digest, kind="content_hash")
        result = router.place_session(
            session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=seed
        )
        counts[result.account_id] += 1
    return counts


def test_weighted_distribution_matches_headroom_proportional_shares_within_tolerance() -> None:
    router = _make_router()
    seed = b"distribution-seed"
    # headroom 80/40/20 -> pressure 20/60/80 (P = 100 - H, zero in-flight, no warning haircut).
    for account_id, used_percent in (("acct-80", 20.0), ("acct-40", 60.0), ("acct-20", 80.0)):
        router.ingest_observation(account_id, "five_hour", used_percent=used_percent, source="usage_api")
        router.ingest_observation(account_id, "seven_day", used_percent=used_percent, source="usage_api")
    candidates = [_candidate("acct-80"), _candidate("acct-40"), _candidate("acct-20")]

    total = 6_000
    counts = _place_many_sessions(router, candidates, seed=seed, total=total, prefix="distribution")
    shares = {account_id: count / total for account_id, count in counts.items()}

    assert shares["acct-80"] == pytest.approx(4 / 7, abs=0.03)
    assert shares["acct-40"] == pytest.approx(2 / 7, abs=0.03)
    assert shares["acct-20"] == pytest.approx(1 / 7, abs=0.03)


def test_weighted_distribution_is_uniform_when_pressures_are_equal() -> None:
    router = _make_router()
    seed = b"uniform-seed"
    for account_id in ("acct-1", "acct-2", "acct-3"):
        router.ingest_observation(account_id, "five_hour", used_percent=40.0, source="usage_api")
        router.ingest_observation(account_id, "seven_day", used_percent=40.0, source="usage_api")
    candidates = [_candidate("acct-1"), _candidate("acct-2"), _candidate("acct-3")]

    total = 6_000
    counts = _place_many_sessions(router, candidates, seed=seed, total=total, prefix="uniform")
    shares = {account_id: count / total for account_id, count in counts.items()}

    for account_id in ("acct-1", "acct-2", "acct-3"):
        assert shares[account_id] == pytest.approx(1 / 3, abs=0.03)


# -- pin-map: TTL, LRU eviction, migration reservation, counters --------------


def test_pin_ttl_expiry_differs_by_key_kind() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock, pin_ttl_uuid_seconds=100.0, pin_ttl_content_hash_seconds=10.0)
    candidates = [_candidate("acct-a")]

    uuid_key = SessionKey(digest=b"\x01" * 32, kind="uuid")
    router.place_session(session_key=uuid_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")
    content_key = SessionKey(digest=b"\x02" * 32, kind="content_hash")
    router.place_session(session_key=content_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

    clock.advance(11.0)  # past the content-hash TTL, still within the uuid TTL
    assert router.purge_expired_pins(now=clock()) == 1
    assert router.get_pin(content_key.digest) is None
    assert router.get_pin(uuid_key.digest) is not None

    clock.advance(100.0)  # now past the uuid TTL too
    assert router.purge_expired_pins(now=clock()) == 1
    assert router.get_pin(uuid_key.digest) is None


def test_lru_eviction_prefers_content_hash_over_uuid() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock, pin_map_max_entries=2)
    candidates = [_candidate("acct-a")]

    uuid_key = SessionKey(digest=b"\x01" * 32, kind="uuid")
    router.place_session(session_key=uuid_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")
    clock.advance(1.0)
    content_key = SessionKey(digest=b"\x02" * 32, kind="content_hash")
    router.place_session(session_key=content_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")
    assert router.pin_count() == 2

    clock.advance(1.0)
    third_key = SessionKey(digest=b"\x03" * 32, kind="uuid")
    router.place_session(session_key=third_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

    assert router.pin_count() == 2
    assert router.get_pin(content_key.digest) is None  # LRU content-hash is evicted before any uuid pin
    assert router.get_pin(uuid_key.digest) is not None
    assert router.get_pin(third_key.digest) is not None
    assert router.removed_pin_counts.get("evicted_lru") == 1


def test_lru_eviction_picks_the_least_recently_used_entry_within_a_kind() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock, pin_map_max_entries=2)
    candidates = [_candidate("acct-a")]

    older_key = SessionKey(digest=b"\x01" * 32, kind="content_hash")
    router.place_session(session_key=older_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")
    clock.advance(5.0)
    newer_key = SessionKey(digest=b"\x02" * 32, kind="content_hash")
    router.place_session(session_key=newer_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

    clock.advance(5.0)
    third_key = SessionKey(digest=b"\x03" * 32, kind="content_hash")
    router.place_session(session_key=third_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

    assert router.get_pin(older_key.digest) is None
    assert router.get_pin(newer_key.digest) is not None
    assert router.get_pin(third_key.digest) is not None


def test_migration_reserved_pin_survives_cap_pressure_and_counts_soft_bound_overflow() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock, pin_map_max_entries=1)
    candidates = [_candidate("acct-a")]

    reserved_key = SessionKey(digest=b"\x01" * 32, kind="content_hash")
    router.place_session(session_key=reserved_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")
    router.set_migration_reserved(reserved_key.digest, True)
    assert router.active_migration_count() == 1

    clock.advance(1.0)
    second_key = SessionKey(digest=b"\x02" * 32, kind="content_hash")
    router.place_session(session_key=second_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

    assert router.get_pin(reserved_key.digest) is not None  # never evicted while migration-reserved
    assert router.get_pin(second_key.digest) is not None
    assert router.pin_count() == 2  # exceeded the nominal bound of 1: the soft bound
    assert router.soft_bound_overflow_count == 1


def test_pin_removal_counters_decrement_exactly_once_per_path() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock, pin_ttl_content_hash_seconds=5.0, pin_map_max_entries=2)
    candidates = [_candidate("acct-a")]

    router.place_session(
        session_key=SessionKey(digest=b"\x01" * 32, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=candidates,
        seed=b"seed",
    )
    router.place_session(
        session_key=SessionKey(digest=b"\x02" * 32, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=candidates,
        seed=b"seed",
    )

    clock.advance(10.0)  # expires both pins above (ttl 5s)
    assert router.purge_expired_pins(now=clock()) == 2
    assert router.removed_pin_counts["expired"] == 2
    assert router.total_removed_pins == 2

    fresh_key = SessionKey(digest=b"\x03" * 32, kind="content_hash")
    router.place_session(session_key=fresh_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")
    assert router.remove_pin(fresh_key.digest) is True
    assert router.remove_pin(fresh_key.digest) is False  # already gone: no double decrement
    assert router.removed_pin_counts["removed"] == 1
    assert router.total_removed_pins == 3

    clock.advance(0.1)
    router.place_session(
        session_key=SessionKey(digest=b"\x04" * 32, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=candidates,
        seed=b"seed",
    )
    clock.advance(0.1)
    router.place_session(
        session_key=SessionKey(digest=b"\x05" * 32, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=candidates,
        seed=b"seed",
    )
    clock.advance(0.1)
    router.place_session(
        session_key=SessionKey(digest=b"\x06" * 32, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=candidates,
        seed=b"seed",
    )
    assert router.removed_pin_counts.get("evicted_lru") == 1
    assert router.total_removed_pins == 4


def test_restore_from_store_recomputes_counters_and_remaining_ttl_from_wall_clock_rows(tmp_path: Any) -> None:
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    try:
        epoch_id = store.balanced_epoch_id
        now_utc = 1_700_000_000.0
        digest = b"\x01" * 32
        pending_write = store.upsert_pin(
            session_key_digest=digest,
            key_kind="uuid",
            account_id="acct-a",
            account_incarnation_id="inc-acct-a",
            last_seen_utc=now_utc,
            expires_at_utc=now_utc + 300.0,
            generation=0,
            balanced_epoch_id=epoch_id,
        )
        pending_write.wait(timeout=5.0)

        restore_result = store.restore(RestoreValidationContext(now_utc=now_utc))
        assert digest in restore_result.pins

        router = _make_router(balanced_epoch_id=epoch_id)
        restored_count = router.restore_from_store(restore_result, now=5_000.0, wall_now=now_utc)

        assert restored_count == 1
        assert router.pin_count() == 1
        entry = router.get_pin(digest)
        assert entry is not None
        assert entry.expires_at_monotonic == pytest.approx(5_000.0 + 300.0)
        assert entry.last_seen_monotonic == pytest.approx(5_000.0)
        assert entry.pending_durability is None
        assert entry.account_id == "acct-a"
        assert entry.generation == 0
    finally:
        store.close()


# -- pending_durability barrier: cancellation-safe, resolve-exactly-once -----


def test_pending_durability_barrier_resolve_is_idempotent_and_releases_a_waiter() -> None:
    async def scenario() -> None:
        barrier = PendingDurabilityBarrier()
        barrier.resolve()
        barrier.resolve()  # a second resolve is a no-op
        assert barrier.is_resolved
        await asyncio.wait_for(barrier.wait(), timeout=1.0)

    asyncio.run(scenario())


def test_concurrent_follower_blocks_on_the_pending_durability_barrier_until_the_write_completes() -> None:
    async def scenario() -> None:
        store = _FakeStore()
        router = _make_router(store=store)
        candidates = [_candidate("acct-a")]
        session_key = SessionKey(digest=b"\x01" * 32, kind="content_hash")

        result = router.place_session(
            session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed"
        )
        assert isinstance(result, PlacementResult) and result.created is True
        assert result.durability_barrier is not None and not result.durability_barrier.is_resolved

        durability_task = asyncio.create_task(router.submit_new_pin_durability(session_key.digest))
        follower_progressed = False

        async def follower() -> None:
            nonlocal follower_progressed
            await router.await_pin_durability(session_key.digest)
            follower_progressed = True

        follower_task = asyncio.create_task(follower())

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(follower_task), timeout=0.05)
        assert follower_progressed is False
        assert len(store.pending_writes) == 1

        store.pending_writes[0].resolve()
        await asyncio.wait_for(follower_task, timeout=1.0)
        await asyncio.wait_for(durability_task, timeout=1.0)

        assert follower_progressed is True
        assert result.durability_barrier.is_resolved
        assert router.persistence_degraded is False

    asyncio.run(scenario())


def test_failed_durable_write_marks_persistence_degraded_and_releases_both_requests() -> None:
    async def scenario() -> None:
        store = _FakeStore()
        router = _make_router(store=store)
        candidates = [_candidate("acct-a")]
        session_key = SessionKey(digest=b"\x02" * 32, kind="content_hash")

        router.place_session(session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

        durability_task = asyncio.create_task(router.submit_new_pin_durability(session_key.digest))
        creator_progressed = False
        follower_progressed = False

        async def creator() -> None:
            nonlocal creator_progressed
            await router.await_pin_durability(session_key.digest)
            creator_progressed = True

        async def follower() -> None:
            nonlocal follower_progressed
            await router.await_pin_durability(session_key.digest)
            follower_progressed = True

        creator_task = asyncio.create_task(creator())
        follower_task = asyncio.create_task(follower())

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(creator_task), timeout=0.05)
        assert creator_progressed is False and follower_progressed is False

        store.pending_writes[0].resolve(error=RuntimeError("disk full"))
        await asyncio.wait_for(creator_task, timeout=1.0)
        await asyncio.wait_for(follower_task, timeout=1.0)
        await asyncio.wait_for(durability_task, timeout=1.0)

        assert creator_progressed is True and follower_progressed is True
        assert router.persistence_degraded is True
        assert router.get_pin(session_key.digest) is not None  # the in-memory pin is retained

    asyncio.run(scenario())


def test_a_cancelled_waiter_does_not_cancel_the_shielded_barrier() -> None:
    async def scenario() -> None:
        store = _FakeStore()
        router = _make_router(store=store)
        candidates = [_candidate("acct-a")]
        session_key = SessionKey(digest=b"\x03" * 32, kind="content_hash")

        result = router.place_session(
            session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed"
        )
        barrier = result.durability_barrier
        assert barrier is not None

        resolve_calls: list[None] = []
        original_resolve = barrier.resolve

        def counting_resolve() -> None:
            resolve_calls.append(None)
            original_resolve()

        barrier.resolve = counting_resolve  # type: ignore[method-assign]

        durability_task = asyncio.create_task(router.submit_new_pin_durability(session_key.digest))
        waiter_task = asyncio.create_task(router.await_pin_durability(session_key.digest))

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiter_task), timeout=0.05)

        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        # resolving the underlying write now must still succeed: cancelling one shielded
        # waiter never cancels (or otherwise disturbs) the barrier itself.
        store.pending_writes[0].resolve()
        await asyncio.wait_for(durability_task, timeout=1.0)

        assert barrier.is_resolved
        assert len(resolve_calls) == 1  # resolved exactly once, despite the cancelled waiter

        # a fresh waiter sees the already-resolved barrier and returns immediately.
        await asyncio.wait_for(router.await_pin_durability(session_key.digest), timeout=1.0)

    asyncio.run(scenario())


def test_refresh_pin_durable_last_seen_delegates_to_the_store() -> None:
    async def scenario() -> None:
        store = _FakeStore()
        router = _make_router(store=store)
        candidates = [_candidate("acct-a")]
        session_key = SessionKey(digest=b"\x07" * 32, kind="content_hash")
        router.place_session(session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

        refresh_task = asyncio.create_task(router.refresh_pin_durable_last_seen(session_key.digest))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(refresh_task), timeout=0.05)
        assert len(store.pending_writes) == 1

        store.pending_writes[0].resolve()
        await asyncio.wait_for(refresh_task, timeout=1.0)

    asyncio.run(scenario())
