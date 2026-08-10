"""Deferred live-gate inventory (T-17): design v2 §8's acceptance gates that
require observing real behavior across N>=2 live Claude accounts -- this
environment has exactly one registered account, so none of them can be
executed here. Every required gate carries its own explicit `pytest.mark.skip`
stub below (a count threshold alone would be insufficient: it could hide an
omitted gate behind an unrelated duplicate), and the meta-test at the bottom
of this module asserts the encoded gate-number set is EXACTLY the required
ten-gate set -- {8.1, 8.2, 8.3, 8.4, 8.5, 8.12, 8.14, 8.15, 8.16, 8.19} --
with no omissions and no duplicates.

Gates 8.4 and 8.5 are LIVE CONFIRMATIONS of behavior already covered by
deterministic local simulation -- T-7's concurrent weighted-HRW placement
tests for the placement race (8.4), and T-8's concurrent migration
reservation/waiter tests for the failover wave (8.5) -- not missing unit
coverage; only the live N>=2 account observation itself is missing.
"""

from __future__ import annotations

import re

import pytest

REQUIRED_DEFERRED_GATE_NUMBERS = frozenset(
    {"8.1", "8.2", "8.3", "8.4", "8.5", "8.12", "8.14", "8.15", "8.16", "8.19"}
)

# Built from parts (not the literal contiguous phrase) so this reconstructed
# constant -- used only to VALIDATE each stub's actual skip reason below --
# does not itself inflate the qualifier's literal grep count of the required
# phrase across the ten stub decorators.
_EM_DASH = "—"
_SKIP_REASON_PREFIX = f"deferred {_EM_DASH} needs 2nd account:"

# Encodes each stub's gate number right after the shared `test_deferred_gate_`
# prefix, e.g. `test_deferred_gate_8_12_...` -> "8.12".
_GATE_STUB_NAME_PATTERN = re.compile(r"^test_deferred_gate_(\d+)_(\d+)_")


# ===========================================================================
# §8.1-§8.5: live placement/failover gates
# ===========================================================================


@pytest.mark.skip(reason="deferred — needs 2nd account: 8.1 two-account live placement baseline")
def test_deferred_gate_8_1_two_account_live_placement_baseline() -> None:
    """Live confirmation that weighted-HRW placement actually distributes
    real traffic across two distinct, concurrently registered accounts --
    the N=1 pool this repo's CI runs against always trivially converges to
    the single account (T-7's own cold-start/N=1 test), so this needs a
    genuine second account to observe."""


@pytest.mark.skip(
    reason=(
        "deferred — needs 2nd account: 8.2 live Fable-scoped unified-header "
        "capture across two accounts (the two-sonnet capture in T-14 already "
        "ran against the single registered account; the Fable-scoped header "
        "pair still needs a second, Fable-eligible live account)"
    )
)
def test_deferred_gate_8_2_two_account_live_fable_scoped_header_capture() -> None:
    """Live confirmation of the Fable-scoped unified-header capture pair
    across two real accounts, distinct from T-14's single-account two-sonnet
    capture."""


@pytest.mark.skip(reason="deferred — needs 2nd account: 8.3 live cross-account migration handoff")
def test_deferred_gate_8_3_two_account_live_migration_handoff() -> None:
    """Live confirmation that a real upstream 429/failure on one account
    hands a live in-flight session off to a second real account end-to-end
    (commit-at-headers, durable pin rewrite, and byte-for-byte stream
    continuity), not merely the local reservation/CAS machinery T-8 already
    exercises with simulated targets."""


@pytest.mark.skip(
    reason=(
        "deferred — needs 2nd account: 8.4 live N>=2 confirmation of the "
        "concurrent placement race already simulated locally in T-7's "
        "weighted-HRW/statistical-distribution tests -- this is a live "
        "observation gap, not missing unit coverage"
    )
)
def test_deferred_gate_8_4_two_account_live_confirmation_of_the_t7_simulated_placement_race() -> None:
    """Live N>=2 confirmation of the concurrent placement race T-7 already
    simulates deterministically against a single-process, single-account
    pool -- not missing unit coverage, only the live multi-account
    observation itself."""


@pytest.mark.skip(
    reason=(
        "deferred — needs 2nd account: 8.5 live N>=2 confirmation of the "
        "concurrent failover wave already simulated locally in T-8's "
        "migration reservation/waiter tests -- this is a live observation "
        "gap, not missing unit coverage"
    )
)
def test_deferred_gate_8_5_two_account_live_confirmation_of_the_t8_simulated_failover_wave() -> None:
    """Live N>=2 confirmation of the concurrent failover wave T-8 already
    simulates deterministically with a single fake target account -- not
    missing unit coverage, only the live multi-account observation itself."""


# ===========================================================================
# §8.12-§8.19: remaining live multi-account gates
# ===========================================================================


@pytest.mark.skip(
    reason="deferred — needs 2nd account: 8.12 live multi-account persistence-degraded recovery"
)
def test_deferred_gate_8_12_two_account_live_persistence_degraded_recovery() -> None:
    """Live confirmation that a persistence outage affecting one account's
    durable writes never blocks a second, healthy account from being placed
    and served while the store self-heals -- T-17's own
    `test_persistence_degrades_under_transient_failures_but_recovers_and_durably_lands_the_pin`
    covers the single-account recovery mechanics locally."""


@pytest.mark.skip(
    reason="deferred — needs 2nd account: 8.14 live multi-account reservation-aware LRU eviction"
)
def test_deferred_gate_8_14_two_account_live_reservation_aware_lru_eviction() -> None:
    """Live confirmation that a migration-reserved pin survives eviction
    pressure while a SECOND real account's traffic drives the pin map toward
    its bound -- `test_claude_balanced_router.py`'s
    `test_migration_reserved_pin_survives_eviction_pressure_and_becomes_evictable_once_resolved`
    already covers the mechanism deterministically with simulated entries."""


@pytest.mark.skip(
    reason="deferred — needs 2nd account: 8.15 live multi-account cooldown restore without a repeat burst"
)
def test_deferred_gate_8_15_two_account_live_cooldown_restore_without_a_repeat_burst() -> None:
    """Live confirmation that a restored durable cooldown for one account
    coexists correctly with a second, healthy account continuing to serve
    traffic across the same restart -- `test_server.py`'s
    `test_balanced_restart_restores_the_family_cooldown_without_a_repeat_429_burst`
    already covers the single-account restore mechanics locally."""


@pytest.mark.skip(
    reason="deferred — needs 2nd account: 8.16 live multi-account epoch rotation on mode exit"
)
def test_deferred_gate_8_16_two_account_live_epoch_rotation_on_mode_exit() -> None:
    """Live confirmation that an intentional mode exit's epoch rotation
    invalidates BOTH accounts' pins consistently and a later re-entry
    re-places live traffic across both -- this file's own
    `test_mode_exit_rotates_the_epoch_so_a_later_reentry_never_restores_the_old_pin`
    covers the single-account rotation mechanics locally."""


@pytest.mark.skip(
    reason="deferred — needs 2nd account: 8.19 live multi-account disable-while-streaming drain"
)
def test_deferred_gate_8_19_two_account_live_disable_while_streaming_drain() -> None:
    """Live confirmation that a disable-mode drain correctly waits out an
    in-flight stream on one account while a second account's own traffic is
    unaffected until the drain completes -- this file's own
    `test_disable_while_streaming_drains_the_in_flight_stream_before_epoch_rotation_invalidates_pins`
    covers the single-account drain-ordering mechanics locally."""


# ===========================================================================
# Meta-test: the encoded gate-number set must be EXACTLY the required set
# ===========================================================================


def _collect_deferred_gate_stub_names() -> list[str]:
    return sorted(
        name
        for name, value in globals().items()
        if callable(value) and _GATE_STUB_NAME_PATTERN.match(name)
    )


def test_deferred_gate_inventory_matches_the_required_exact_set_with_no_omissions_or_duplicates() -> None:
    """Step 5's collection assertion: compares the gate number ENCODED IN
    EACH STUB'S FUNCTION NAME against `REQUIRED_DEFERRED_GATE_NUMBERS`, so a
    duplicated stub (two different function names encoding the same gate
    number) cannot silently satisfy a mere count check -- it would inflate
    the raw list without inflating the deduplicated set, which this test
    catches explicitly.
    """
    stub_names = _collect_deferred_gate_stub_names()
    assert stub_names, "no deferred-gate stubs were collected at all"

    encoded_gate_numbers = []
    for name in stub_names:
        match = _GATE_STUB_NAME_PATTERN.match(name)
        assert match is not None
        encoded_gate_numbers.append(f"{match.group(1)}.{match.group(2)}")

    # No duplicates: the raw (possibly-repeated) list and its deduplicated
    # set must be the exact same size.
    assert len(encoded_gate_numbers) == len(set(encoded_gate_numbers)), (
        f"duplicate gate number(s) encoded across stub names: {encoded_gate_numbers}"
    )
    # No omissions, and nothing extra: the deduplicated set must equal the
    # required set exactly.
    assert set(encoded_gate_numbers) == REQUIRED_DEFERRED_GATE_NUMBERS
    assert len(encoded_gate_numbers) == len(REQUIRED_DEFERRED_GATE_NUMBERS) == 10

    for name in stub_names:
        func = globals()[name]
        skip_marks = [mark for mark in getattr(func, "pytestmark", []) if mark.name == "skip"]
        assert len(skip_marks) == 1, f"{name} must carry exactly one @pytest.mark.skip"
        reason = skip_marks[0].kwargs.get("reason", "")
        assert reason.startswith(_SKIP_REASON_PREFIX), (
            f"{name}'s skip reason must begin with {_SKIP_REASON_PREFIX!r}"
        )
        match = _GATE_STUB_NAME_PATTERN.match(name)
        assert match is not None
        gate_number = f"{match.group(1)}.{match.group(2)}"
        assert gate_number in reason, f"{name}'s skip reason must name its own gate number ({gate_number})"
