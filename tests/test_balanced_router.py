"""Tests for domain-separated session-key derivation and HRW routing math."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pytest

from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache
from claudex_gateway.balanced.polling import ClaudeUsagePollCoordinator
from claudex_gateway.balanced.router import (
    CAPABILITY_CLASSIFIER_VERSION,
    CAPABILITY_EVIDENCE_TTL_SECONDS,
    ClaudeBalancedRouter,
    MigrationReservation,
    PendingDurabilityBarrier,
    PlacementResult,
)
from claudex_gateway.balanced.runtime import ClaudeBalancedRuntime
from claudex_gateway.balanced.selection import (
    AccountCandidate,
    FamilyGateOutcome,
    NoEligibleAccountError,
    ObservationView,
    SessionKey,
    account_headroom,
    binding_windows,
    capability_key,
    classify_balanced_cooldown_scope,
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
from claudex_gateway.balanced.state_model import PinRow, RestoreResult, RestoreValidationContext
from claudex_gateway.balanced.state_store import ClaudePoolRuntimeStateStore


_BALANCED_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "claudex_gateway" / "balanced"
_BALANCED_MODULE_NAMES = (
    "selection",
    "router",
    "polling",
    "runtime",
    "state_model",
    "state_store",
)
_ROUTING_RANK = {"selection": 0, "router": 1, "polling": 2, "runtime": 3}
_PERSISTENCE_MODULES = {"state_model", "state_store"}
_MANIFEST_SYMBOLS = {
    "selection": {
        "_SESSION_KEY_DOMAIN",
        "_PIN_KEY_DOMAIN",
        "_HRW_DOMAIN",
        "_STATELESS_REQUEST_DOMAIN",
        "_QUOTA_FAMILIES",
        "SessionKey",
        "_hmac_sha256",
        "_length_prefixed",
        "_uuid_session_key",
        "_first_user_message",
        "_content_hash_session_key",
        "derive_session_key",
        "hrw_unit_interval",
        "derive_stateless_routing_digest",
        "_FRESH_EXACT_SECONDS",
        "_FRESH_PLUS5_SECONDS",
        "_FRESH_PLUS10_SECONDS",
        "_FRESH_PLUS5_PP",
        "_FRESH_PLUS10_PP",
        "_UNKNOWN_FLOOR_MARGIN",
        "_UNKNOWN_FLOOR_CAP",
        "_WARNING_FRESH_SECONDS",
        "_WARNING_HAIRCUT_FACTOR",
        "_IN_FLIGHT_PRESSURE_WEIGHT",
        "_NON_FABLE_WINDOWS",
        "_FABLE_WINDOWS",
        "_PEEK_WINDOW_TO_BINDING",
        "DEFAULT_PIN_TTL_UUID_SECONDS",
        "DEFAULT_PIN_TTL_CONTENT_HASH_SECONDS",
        "DEFAULT_PIN_MAP_MAX_ENTRIES",
        "_is_ascii_alnum",
        "_bounded_token_present",
        "quota_family",
        "_CAPABILITY_KEY_TOKENS",
        "capability_key",
        "binding_windows",
        "freshness_adjusted_pressure",
        "unknown_floor",
        "_wall_to_monotonic",
        "_WindowObservation",
        "_WarningSignal",
        "RealWindowReading",
        "ObservationView",
        "warning_factor",
        "account_headroom",
        "emergency_capacity",
        "emergency_weight",
        "select_weights",
        "resolve_tie_break",
        "pick_weighted_hrw",
        "NoEligibleAccountError",
        "AccountCandidate",
        "is_eligible_candidate",
        "_FAMILY_GATE_MAX_OBSERVATION_AGE_SECONDS",
        "_FAMILY_GATE_FABLE_WEEKLY_MIN_PERCENT",
        "_FAMILY_GATE_FIVE_HOUR_MAX_PERCENT",
        "_FAMILY_GATE_SEVEN_DAY_MAX_PERCENT",
        "FamilyGateOutcome",
        "classify_balanced_cooldown_scope",
    },
    "router": {
        "_COOLDOWN_RESTORE_MIN_SECONDS",
        "_COOLDOWN_RESTORE_MAX_SECONDS",
        "CAPABILITY_CLASSIFIER_VERSION",
        "CAPABILITY_EVIDENCE_TTL_SECONDS",
        "_CooldownEntry",
        "_entry_cooldown_deadline",
        "_CapabilityEvidenceEntry",
        "PendingDurabilityBarrier",
        "PinEntry",
        "PlacementResult",
        "MigrationOutcome",
        "CommitOutcome",
        "MigrationReservation",
        "ClaudeBalancedRouter",
    },
    "polling": {
        "_USAGE_POLL_INTERVAL_SECONDS",
        "_MANUAL_REFRESH_RATE_LIMIT_SECONDS",
        "PollTickOutcome",
        "PollTickResult",
        "UsagePollAccount",
        "_AccountPollDiagnostics",
        "UsagePollDiagnostics",
        "ClaudeUsagePollCoordinator",
    },
    "runtime": {
        "BalancedRuntimeStatus",
        "BalancedPrepareError",
        "ClaudeBalancedRuntime",
    },
}


def _top_level_definitions(tree: ast.Module) -> set[str]:
    definitions: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.add(node.name)
        elif isinstance(node, ast.Assign):
            definitions.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions.add(node.target.id)
    return definitions


def _balanced_module_trees() -> dict[str, ast.Module]:
    return {
        name: ast.parse(
            (_BALANCED_SOURCE_ROOT / f"{name}.py").read_text(encoding="utf-8")
        )
        for name in _BALANCED_MODULE_NAMES
    }


def _assert_no_unused_imports(module_name: str, tree: ast.Module) -> None:
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases = [
                (alias.asname or alias.name.split(".")[0], alias.name)
                for alias in node.names
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            aliases = [(alias.asname or alias.name, alias.name) for alias in node.names]
        else:
            continue
        for local_name, imported_name in aliases:
            assert imported_name != "*", f"{module_name}: star import is not an inventory"
            assert local_name in loaded_names, f"{module_name}: unused import {imported_name}"


def _balanced_sibling_edges(tree: ast.Module) -> set[str]:
    package_name = "claudex_gateway.balanced"
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported_base = node.module or ""
            if imported_base == package_name:
                edges.update(
                    alias.name for alias in node.names if alias.name in _BALANCED_MODULE_NAMES
                )
            imported_modules = [imported_base]
        else:
            continue
        for imported_module in imported_modules:
            prefix = f"{package_name}."
            if not imported_module.startswith(prefix):
                continue
            sibling_name = imported_module.removeprefix(prefix).partition(".")[0]
            if sibling_name in _BALANCED_MODULE_NAMES:
                edges.add(sibling_name)
    return edges


def _imported_module_names(tree: ast.Module) -> set[str]:
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    return imported_modules


def test_balanced_manifest_symbols_have_one_canonical_owner() -> None:
    definitions = {
        module_name: _top_level_definitions(tree)
        for module_name, tree in _balanced_module_trees().items()
    }

    for expected_owner, symbols in _MANIFEST_SYMBOLS.items():
        for symbol in symbols:
            actual_owners = [
                module_name
                for module_name, module_definitions in definitions.items()
                if symbol in module_definitions
            ]
            assert actual_owners == [expected_owner], (
                symbol,
                expected_owner,
                actual_owners,
            )


def test_balanced_import_inventory_and_dependency_directions() -> None:
    trees = _balanced_module_trees()
    edges: dict[str, set[str]] = {}
    forbidden_consumers = {
        "claudex_gateway.relay",
        "claudex_gateway.admin_api",
        "claudex_gateway.server",
    }

    for module_name, tree in trees.items():
        _assert_no_unused_imports(module_name, tree)
        edges[module_name] = _balanced_sibling_edges(tree)
        assert not (_imported_module_names(tree) & forbidden_consumers)

    for source, source_rank in _ROUTING_RANK.items():
        allowed_dependencies = {
            dependency
            for dependency, dependency_rank in _ROUTING_RANK.items()
            if dependency_rank < source_rank
        }
        if source != "selection":
            allowed_dependencies |= _PERSISTENCE_MODULES
        assert edges[source] <= allowed_dependencies, (
            source,
            edges[source],
            allowed_dependencies,
        )

    assert edges["state_model"] == set()
    assert edges["state_store"] == {"state_model"}
    assert not (edges["state_model"] | edges["state_store"]) & set(_ROUTING_RANK)


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


def _reference_pin_key_digest(seed: bytes, logical_digest: bytes, family: bytes) -> bytes:
    framed = b""
    for field in (logical_digest, family):
        framed += len(field).to_bytes(8, "big") + field
    return hmac.new(seed, b"claudex-pin-key-v2" + b"\x00" + framed, hashlib.sha256).digest()


# ---------------------------------------------------------------------------
# Usage poll coordinator (T-18, fix for gap G-1): the public mirror the
# runtime-owned background driver reads to pace its own loop.
# ---------------------------------------------------------------------------


def test_usage_poll_coordinator_exposes_its_configured_poll_interval_seconds() -> None:
    """`poll_interval_seconds` is a read-only public mirror of the private
    constructor argument -- not a new scheduling knob -- so
    `ClaudeBalancedRuntime`'s background driver can pace its own loop
    against the exact same budget the coordinator itself enforces.
    """

    async def fetch(_account_id: str) -> tuple[dict[str, Any], float | None]:
        raise AssertionError("never called")

    cache = ClaudeAccountUsageCache(fetch)
    router = ClaudeBalancedRouter(balanced_epoch_id="epoch-1")
    coordinator = ClaudeUsagePollCoordinator(
        cache=cache, router=router, poll_interval_seconds=12.5
    )

    assert coordinator.poll_interval_seconds == 12.5


def test_uuid_branch_is_case_insensitive_and_yields_the_same_digest() -> None:
    seed = b"seed-case-insensitive"
    lower = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
    upper = lower.upper()
    lower_body = {"metadata": {"user_id": _claude_code_user_id(lower)}}
    upper_body = {"metadata": {"user_id": _claude_code_user_id(upper)}}

    lower_key = derive_session_key(lower_body, seed, "default")
    upper_key = derive_session_key(upper_body, seed, "default")

    assert lower_key is not None and upper_key is not None
    assert lower_key.kind == upper_key.kind == "uuid"
    assert lower_key.digest == upper_key.digest
    assert lower_key.scoring_digest == _reference_session_key_digest(
        seed, b"uuid", lower.encode("utf-8")
    )


def test_non_rfc4122_variant_falls_back_to_content_hash_branch() -> None:
    seed = b"seed-non-rfc4122"
    ncs_variant_uuid = "11111111-2222-4333-0444-555555555555"
    body = {
        "metadata": {"user_id": _claude_code_user_id(ncs_variant_uuid)},
        "messages": [{"role": "user", "content": "hello"}],
    }

    key = derive_session_key(body, seed, "default")

    assert key is not None
    assert key.kind == "content_hash"


def test_whitespace_padded_uuid_is_rejected_and_falls_back_to_content_hash() -> None:
    seed = b"seed-whitespace"
    padded_uuid = " 3fa85f64-5717-4562-b3fc-2c963f66afa6 "
    body = {
        "metadata": {"user_id": _claude_code_user_id(padded_uuid)},
        "messages": [{"role": "user", "content": "hello"}],
    }

    key = derive_session_key(body, seed, "default")

    assert key is not None
    assert key.kind == "content_hash"


def test_content_hash_digest_is_deterministic_and_key_order_independent() -> None:
    seed = b"seed-content-hash"
    message_in_order = {"role": "user", "content": "café ☕ unicode test", "id": "m-1"}
    message_reordered = {"id": "m-1", "content": "café ☕ unicode test", "role": "user"}
    body_a = {"messages": [message_in_order]}
    body_b = {"messages": [message_reordered]}

    key_a = derive_session_key(body_a, seed, "default")
    key_b = derive_session_key(body_b, seed, "default")

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
    assert key_a.scoring_digest == _reference_session_key_digest(
        seed, b"content_hash", expected_canonical_utf8
    )


def test_no_user_message_and_no_metadata_yields_no_session_key() -> None:
    seed = b"seed-no-session-key"
    body = {"messages": [{"role": "assistant", "content": "hi there"}]}

    assert derive_session_key(body, seed, "default") is None
    assert derive_session_key({}, seed, "default") is None


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
    """A controllable stand-in store: records every `upsert_pin`/`touch_pin_last_seen`/
    `upsert_cooldown`/`upsert_capability_evidence`/`upsert_usage_observation`/
    `delete_all_for_incarnation` call.
    """

    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.cooldown_calls: list[dict[str, Any]] = []
        self.capability_calls: list[dict[str, Any]] = []
        self.usage_observation_calls: list[dict[str, Any]] = []
        self.deleted_incarnations: list[str] = []
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

    def upsert_cooldown(self, **kwargs: Any) -> _FakePendingWrite:
        self.cooldown_calls.append(kwargs)
        pending_write = _FakePendingWrite()
        self.pending_writes.append(pending_write)
        return pending_write

    def upsert_capability_evidence(self, **kwargs: Any) -> _FakePendingWrite:
        self.capability_calls.append(kwargs)
        pending_write = _FakePendingWrite()
        self.pending_writes.append(pending_write)
        return pending_write

    def upsert_usage_observation(self, **kwargs: Any) -> _FakePendingWrite:
        self.usage_observation_calls.append(kwargs)
        pending_write = _FakePendingWrite()
        self.pending_writes.append(pending_write)
        return pending_write

    def delete_all_for_incarnation(self, account_incarnation_id: str) -> _FakePendingWrite:
        self.deleted_incarnations.append(account_incarnation_id)
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
    import claudex_gateway.balanced.selection as selection_module

    call_count = 0

    def fixed_hrw_unit_interval(_seed: bytes, _digest: bytes, _account_id: str) -> float:
        nonlocal call_count
        call_count += 1
        return 0.5

    monkeypatch.setattr(selection_module, "hrw_unit_interval", fixed_hrw_unit_interval)
    weights = {"b": 10.0, "a": 10.0, "c": 10.0}

    winner = pick_weighted_hrw(weights=weights, seed=b"seed", session_key_digest=b"digest", serving_account_id="c")
    assert winner == "c"
    again = pick_weighted_hrw(weights=weights, seed=b"seed", session_key_digest=b"digest", serving_account_id="c")
    assert again == winner

    without_serving_tie = pick_weighted_hrw(
        weights=weights, seed=b"seed", session_key_digest=b"digest", serving_account_id="not-in-pool"
    )
    assert without_serving_tie == "a"
    assert call_count == 9


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


# ==========================================================================
# Migration machinery: reservations, waiters, tokens, generation CAS (T-8)
# ==========================================================================


def _place_and_reserve(
    router: ClaudeBalancedRouter,
    digest: bytes,
    *,
    source_account: str = "acct-a",
    target_account: str = "acct-b",
    attempt_id: str = "owner-1",
) -> MigrationReservation:
    """Place a fresh pin at `digest` for `source_account`, then have `attempt_id` acquire
    the migration reservation for it as the owner.
    """
    router.place_session(
        session_key=SessionKey(digest=digest, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=[_candidate(source_account)],
        seed=b"seed",
    )
    reservation, is_owner = router.acquire_migration_reservation(
        digest,
        source_account=source_account,
        source_generation=0,
        target_account=target_account,
        attempt_id=attempt_id,
    )
    assert is_owner is True
    return reservation


def test_second_caller_for_the_same_session_becomes_a_waiter_on_the_existing_reservation() -> None:
    router = _make_router()
    digest = b"\x10" * 32
    owner_reservation = _place_and_reserve(router, digest)

    assert router.active_migration_count() == 1
    assert router.get_pin(digest).migration_reserved is True

    waiter_reservation, is_owner = router.acquire_migration_reservation(
        digest,
        source_account="acct-a",
        source_generation=0,
        target_account="acct-c",  # a concurrent caller's own (irrelevant) intended target
        attempt_id="waiter-1",
    )

    assert is_owner is False
    assert waiter_reservation is owner_reservation  # never an independent reservation/target
    assert router.migration_token_target("waiter-1") is None  # no token minted for a waiter


def test_concurrent_waiters_resume_immediately_after_commit_at_headers_while_the_stream_stays_open() -> None:
    async def scenario() -> None:
        router = _make_router()
        digest = b"\x11" * 32
        reservation = _place_and_reserve(router, digest)

        waiter_reservation_a, _ = router.acquire_migration_reservation(
            digest, source_account="acct-a", source_generation=0, target_account="acct-z", attempt_id="waiter-a"
        )
        waiter_reservation_b, _ = router.acquire_migration_reservation(
            digest, source_account="acct-a", source_generation=0, target_account="acct-z", attempt_id="waiter-b"
        )
        assert waiter_reservation_a is reservation and waiter_reservation_b is reservation

        marks = [False, False]

        async def waiter(index: int) -> None:
            await router.wait_for_migration_reservation(reservation)
            marks[index] = True

        waiter_tasks = [asyncio.create_task(waiter(0)), asyncio.create_task(waiter(1))]
        for task in waiter_tasks:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
        assert marks == [False, False]

        headers_ingested: list[bool] = []
        outcome, pin, barrier = router.commit_at_headers(
            digest,
            attempt_id="owner-1",
            source_account="acct-a",
            source_generation=0,
            target_account="acct-b",
            target_account_incarnation_id="inc-acct-b",
            ingest_headers=lambda: headers_ingested.append(True),
        )
        assert outcome == "committed"
        assert headers_ingested == [True]
        assert router.get_migration_reservation(digest) is None  # reservation cleared
        assert reservation.outcome == "committed"
        assert reservation.resolved_event.is_set()  # event set exactly once, right here

        await asyncio.wait_for(asyncio.gather(*waiter_tasks), timeout=1.0)
        assert marks == [True, True]

        # the migrated stream is still open: M(target) stays incremented until the
        # attempt's own `finally` releases its token.
        assert router.in_flight_count("acct-b") == 1
        assert router.migration_token_target("owner-1") == "acct-b"

        # a waiter re-reading now sees the migrated pin, not the stale target embedded
        # in the (by-now-cleared) reservation it just waited on.
        current_pin = router.get_pin(digest)
        assert current_pin is not None and current_pin.account_id == "acct-b" and current_pin.generation == 1
        assert current_pin.pending_durability is barrier and not barrier.is_resolved

        # only once the upstream stream terminates does the common `finally` release
        # the token; the reservation is already resolved, so this call only does that.
        resolved_now = router.resolve_migration_owner_terminal(
            digest, attempt_id="owner-1", outcome="terminal_failure"
        )
        assert resolved_now is False
        assert router.in_flight_count("acct-b") == 0
        assert router.migration_token_target("owner-1") is None

    asyncio.run(scenario())


def test_owner_cancellation_wakes_every_blocked_waiter_exactly_once_and_clears_shared_state() -> None:
    async def scenario() -> None:
        router = _make_router()
        digest = b"\x12" * 32
        reservation = _place_and_reserve(router, digest)

        never_ready = asyncio.Event()

        async def owner_attempt() -> None:
            try:
                await never_ready.wait()  # blocked "before headers"
            finally:
                # the balanced runner's own `finally` block: synchronous, no intervening
                # await between ownership verification and event signaling.
                router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")

        async def waiter(results: list[bool], index: int) -> None:
            current, _ = router.acquire_migration_reservation(
                digest,
                source_account="acct-a",
                source_generation=0,
                target_account="acct-z",
                attempt_id=f"waiter-{index}",
            )
            await router.wait_for_migration_reservation(current)
            results[index] = True

        owner_task = asyncio.create_task(owner_attempt())
        results = [False, False, False]
        waiter_tasks = [asyncio.create_task(waiter(results, i)) for i in range(3)]
        for task in waiter_tasks:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.05)

        assert all(router.migration_token_target(f"waiter-{i}") is None for i in range(3))

        owner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner_task

        await asyncio.wait_for(asyncio.gather(*waiter_tasks), timeout=1.0)
        assert results == [True, True, True]

        assert router.get_migration_reservation(digest) is None
        assert reservation.outcome == "terminal_failure"
        assert reservation.resolved_event.is_set()
        assert router.migration_token_target("owner-1") is None
        assert router.in_flight_count("acct-b") == 0

        # the reservation-protected pin can be handled normally afterward, and a later
        # request re-places instead of hanging.
        assert router.get_pin(digest).migration_reserved is False
        new_reservation, is_new_owner = router.acquire_migration_reservation(
            digest, source_account="acct-a", source_generation=0, target_account="acct-c", attempt_id="owner-2"
        )
        assert is_new_owner is True
        assert new_reservation is not reservation

    asyncio.run(scenario())


def test_generation_cas_loss_when_a_stale_attempts_commit_arrives_after_a_newer_owner_committed() -> None:
    router = _make_router()
    digest = b"\x13" * 32
    _place_and_reserve(router, digest, attempt_id="stale-owner")

    # the stale owner times out before its (eventually late) headers arrive.
    router.resolve_migration_owner_terminal(digest, attempt_id="stale-owner", outcome="terminal_failure")

    # a fresh owner re-acquires (Step 6: re-enters the acquisition loop) and commits.
    router.acquire_migration_reservation(
        digest, source_account="acct-a", source_generation=0, target_account="acct-c", attempt_id="fresh-owner"
    )
    outcome, pin, _ = router.commit_at_headers(
        digest,
        attempt_id="fresh-owner",
        source_account="acct-a",
        source_generation=0,
        target_account="acct-c",
        target_account_incarnation_id="inc-acct-c",
    )
    assert outcome == "committed"
    assert pin.account_id == "acct-c" and pin.generation == 1

    # the stale owner's late 2xx headers finally arrive: its CAS no longer matches
    # (its reservation is gone) - it must never overwrite the newer pin.
    assert router.migration_cas_lost == 0
    stale_outcome, stale_pin, stale_barrier = router.commit_at_headers(
        digest,
        attempt_id="stale-owner",
        source_account="acct-a",
        source_generation=0,
        target_account="acct-b",
        target_account_incarnation_id="inc-acct-b",
    )
    assert stale_outcome == "cas_lost"
    assert stale_barrier is None
    assert stale_pin is not None and stale_pin.account_id == "acct-c"  # unchanged by the loser
    assert router.migration_cas_lost == 1  # the attempt-level counter fires regardless

    # the stale owner's own reservation was already resolved to `terminal_failure`
    # earlier (immutable from then on) — this late, already-orphaned CAS attempt never
    # gets to (re)resolve it to `cas_lost`, so the reservation-outcome tally reflects
    # what actually happened to each reservation, not every attempt-level CAS check.
    assert router.migration_outcome_counts["terminal_failure"] == 1
    assert router.migration_outcome_counts["committed"] == 1
    assert "cas_lost" not in router.migration_outcome_counts
    assert router.get_pin(digest).account_id == "acct-c"


def test_preheader_failure_lets_a_waiter_become_the_next_owner_not_the_original_owner_alone() -> None:
    router = _make_router()
    digest = b"\x14" * 32
    _place_and_reserve(router, digest)

    # a concurrent same-session request arrives while the migration is pending and
    # becomes a waiter rather than picking an independent target.
    waiter_reservation, is_owner = router.acquire_migration_reservation(
        digest, source_account="acct-a", source_generation=0, target_account="acct-z", attempt_id="waiter-1"
    )
    assert is_owner is False

    # owner-1's target rejects pre-headers (a classified quota/eligibility failure).
    resolved = router.resolve_migration_preheader_failure(
        digest, attempt_id="owner-1", outcome="retryable_preheader_failure"
    )
    assert resolved is True
    assert waiter_reservation.outcome == "retryable_preheader_failure"
    assert waiter_reservation.resolved_event.is_set()
    assert router.migration_token_target("owner-1") is None  # released alongside the resolve

    # the (woken) waiter re-enters the SAME acquisition loop - it never trusts a
    # privately-continued target - so it can become the new owner here.
    next_reservation, is_next_owner = router.acquire_migration_reservation(
        digest, source_account="acct-a", source_generation=0, target_account="acct-c", attempt_id="waiter-1"
    )
    assert is_next_owner is True
    assert next_reservation is not waiter_reservation
    assert next_reservation.target_account == "acct-c"


def test_preheader_terminal_ordering_makes_the_cooldown_visible_before_a_woken_waiter_rereads() -> None:
    async def scenario() -> None:
        router = _make_router()
        digest = b"\x15" * 32
        _place_and_reserve(router, digest)
        reservation, _ = router.acquire_migration_reservation(
            digest, source_account="acct-a", source_generation=0, target_account="acct-z", attempt_id="waiter-1"
        )

        cooldowns: dict[str, float] = {}
        cooldown_seen_by_waiter: dict[str, float] = {}

        async def waiter() -> None:
            await router.wait_for_migration_reservation(reservation)
            cooldown_seen_by_waiter.update(cooldowns)  # re-read AFTER the wait resolves

        waiter_task = asyncio.create_task(waiter())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiter_task), timeout=0.05)

        # design v2 §4.4's terminal ordering: (1) classify (2) install cooldown/evidence
        # (3) fence target - all in memory, BEFORE (4) resolve+wake.
        cooldowns["acct-b"] = 999.0
        router.resolve_migration_preheader_failure(digest, attempt_id="owner-1", outcome="retryable_preheader_failure")

        await asyncio.wait_for(waiter_task, timeout=1.0)
        assert cooldown_seen_by_waiter == {"acct-b": 999.0}

    asyncio.run(scenario())


def test_owner_cleanup_is_idempotent_after_success() -> None:
    router = _make_router()
    digest = b"\x16" * 32
    _place_and_reserve(router, digest)
    outcome, _pin, _barrier = router.commit_at_headers(
        digest,
        attempt_id="owner-1",
        source_account="acct-a",
        source_generation=0,
        target_account="acct-b",
        target_account_incarnation_id="inc-acct-b",
    )
    assert outcome == "committed"

    first = router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")
    second = router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")
    third = router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")

    assert (first, second, third) == (False, False, False)  # already resolved by commit_at_headers
    assert router.in_flight_count("acct-b") == 0  # the token released exactly once, never negative
    assert router.migration_outcome_counts["committed"] == 1
    assert "terminal_failure" not in router.migration_outcome_counts


def test_owner_cleanup_is_idempotent_after_a_preheader_failure() -> None:
    router = _make_router()
    digest = b"\x17" * 32
    _place_and_reserve(router, digest)

    first = router.resolve_migration_preheader_failure(digest, attempt_id="owner-1")
    second = router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")
    third = router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")

    assert first is True
    assert second is False and third is False
    assert router.in_flight_count("acct-b") == 0
    assert router.migration_outcome_counts["retryable_preheader_failure"] == 1


def test_owner_cleanup_is_idempotent_after_cancellation() -> None:
    async def scenario() -> None:
        router = _make_router()
        digest = b"\x18" * 32
        reservation = _place_and_reserve(router, digest)

        never_ready = asyncio.Event()

        async def owner_attempt() -> None:
            try:
                await never_ready.wait()
            finally:
                router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")

        owner_task = asyncio.create_task(owner_attempt())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(owner_task), timeout=0.05)
        owner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner_task

        first_replay = router.resolve_migration_owner_terminal(
            digest, attempt_id="owner-1", outcome="terminal_failure"
        )
        second_replay = router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="mode_stopped")

        assert first_replay is False and second_replay is False  # already resolved in the cancelled owner's finally
        assert reservation.outcome == "terminal_failure"  # immutable once set; the replay never overwrote it
        assert router.in_flight_count("acct-b") == 0
        assert router.migration_outcome_counts["terminal_failure"] == 1
        assert "mode_stopped" not in router.migration_outcome_counts

    asyncio.run(scenario())


def test_migration_reserved_pin_survives_eviction_pressure_and_becomes_evictable_once_resolved() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock, pin_map_max_entries=1)
    candidates = [_candidate("acct-a")]
    digest = b"\x19" * 32
    _place_and_reserve(router, digest)
    assert router.get_pin(digest).migration_reserved is True

    clock.advance(1.0)
    second_key = SessionKey(digest=b"\x1a" * 32, kind="content_hash")
    router.place_session(session_key=second_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

    assert router.get_pin(digest) is not None  # never evicted while migration-reserved
    assert router.soft_bound_overflow_count == 1

    router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="terminal_failure")
    assert router.get_pin(digest).migration_reserved is False

    clock.advance(1.0)
    third_key = SessionKey(digest=b"\x1b" * 32, kind="content_hash")
    router.place_session(session_key=third_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed")

    assert router.get_pin(digest) is None  # now evictable like any ordinary pin


# -- account removal transition matrix, §5.7 cases 1-5 -----------------------


def test_account_removal_case_1_deletes_an_ordinary_pin_with_no_active_migration() -> None:
    router = _make_router()
    digest = b"\x20" * 32
    router.place_session(
        session_key=SessionKey(digest=digest, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=[_candidate("acct-a")],
        seed=b"seed",
    )

    counts = router.remove_account("acct-a", "inc-acct-a")

    assert counts["ordinary_pin_removed"] == 1
    assert router.get_pin(digest) is None


def test_account_removal_case_2_source_removed_leaves_pin_and_reservation_for_the_owner_to_finish() -> None:
    router = _make_router()
    digest = b"\x21" * 32
    _place_and_reserve(router, digest)

    counts = router.remove_account("acct-a", "inc-acct-a")

    assert counts["source_removed_migration_continues"] == 1
    assert router.get_pin(digest) is not None  # pin untouched
    assert router.get_migration_reservation(digest) is not None  # reservation untouched

    # the CAS never requires the source to still be registered.
    outcome, pin, _barrier = router.commit_at_headers(
        digest,
        attempt_id="owner-1",
        source_account="acct-a",
        source_generation=0,
        target_account="acct-b",
        target_account_incarnation_id="inc-acct-b",
    )
    assert outcome == "committed"
    assert pin.account_id == "acct-b"


def test_account_removal_case_2_orphaned_source_pin_is_deleted_if_the_migration_later_fails() -> None:
    router = _make_router()
    digest = b"\x22" * 32
    _place_and_reserve(router, digest)

    router.remove_account("acct-a", "inc-acct-a")
    assert router.get_pin(digest) is not None

    router.resolve_migration_preheader_failure(digest, attempt_id="owner-1")

    assert router.get_pin(digest) is None  # the now-orphaned source pin is cleaned up too


def test_account_removal_case_3_target_removed_before_headers_resolves_and_wakes_waiters() -> None:
    async def scenario() -> None:
        router = _make_router()
        digest = b"\x23" * 32
        reservation = _place_and_reserve(router, digest)

        waiter_task = asyncio.create_task(router.wait_for_migration_reservation(reservation))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiter_task), timeout=0.05)

        counts = router.remove_account("acct-b", "inc-acct-b")  # target removed before upstream 2xx

        assert counts["target_removed_before_headers"] == 1
        await asyncio.wait_for(waiter_task, timeout=1.0)
        assert reservation.outcome == "target_removed"
        assert router.get_migration_reservation(digest) is None  # commit is prohibited
        assert router.get_pin(digest) is not None  # source pin left intact
        assert router.migration_token_target("owner-1") == "acct-b"  # token stays owned by the attempt

    asyncio.run(scenario())


def test_account_removal_case_4_target_removed_race_at_headers_rejects_commit_without_retry() -> None:
    router = _make_router()
    digest = b"\x24" * 32
    _place_and_reserve(router, digest)

    assert router.migration_commit_rejected_target_removed == 0
    outcome, pin, barrier = router.commit_at_headers(
        digest,
        attempt_id="owner-1",
        source_account="acct-a",
        source_generation=0,
        target_account="acct-b",
        target_account_incarnation_id="inc-acct-b",
        target_still_registered=False,  # removed by the time 2xx headers arrived
    )

    assert outcome == "target_removed"
    assert barrier is None
    assert router.migration_commit_rejected_target_removed == 1
    assert pin is not None and pin.account_id == "acct-a"  # never pins the removed target
    assert router.get_migration_reservation(digest) is None


def test_account_removal_case_5_both_source_and_target_removed_aborts_and_removes_the_pin() -> None:
    router = _make_router()
    digest = b"\x25" * 32
    _place_and_reserve(router, digest)

    first_counts = router.remove_account("acct-a", "inc-acct-a")  # case 2 first: source removed alone
    assert first_counts["source_removed_migration_continues"] == 1
    assert router.get_pin(digest) is not None
    assert router.get_migration_reservation(digest) is not None

    second_counts = router.remove_account("acct-b", "inc-acct-b")  # now the target is ALSO removed - a race

    assert second_counts["both_removed_aborted"] == 1
    assert router.get_pin(digest) is None
    assert router.get_migration_reservation(digest) is None


def test_release_migration_token_is_an_idempotent_pop_that_never_underflows_the_in_flight_count() -> None:
    router = _make_router()
    digest = b"\x26" * 32
    _place_and_reserve(router, digest)
    assert router.in_flight_count("acct-b") == 1

    first = router.release_migration_token("owner-1")
    second = router.release_migration_token("owner-1")
    third = router.release_migration_token("never-existed")

    assert first == "acct-b"
    assert second is None and third is None
    assert router.in_flight_count("acct-b") == 0  # never negative


def test_mode_stopped_resolves_the_reservation_like_any_other_terminal_outcome() -> None:
    router = _make_router()
    digest = b"\x27" * 32
    reservation = _place_and_reserve(router, digest)

    resolved = router.resolve_migration_owner_terminal(digest, attempt_id="owner-1", outcome="mode_stopped")

    assert resolved is True
    assert reservation.outcome == "mode_stopped"
    assert reservation.resolved_event.is_set()
    assert router.migration_outcome_counts["mode_stopped"] == 1
    assert router.migration_token_target("owner-1") is None



# ==========================================================================
# Durable cooldowns, Fable family gate, capability evidence, removal cleanup
# (T-12, design v2 §6.4/§5.5/§5.7, adjudication G)
# ==========================================================================

_GATE_NOW = 1_000_000.0


def _gate_satisfying_view(*, five_hour_age_seconds: float = 0.0) -> ObservationView:
    """An `ObservationView` satisfying every non-family/non-status §6.4 gate
    condition: `fable_weekly` >=99%, `five_hour`/`seven_day` <=70%, all fresh
    (<=15 min old unless overridden), with a valid future Fable reset.
    """
    view = ObservationView()
    view.ingest_window(
        "acct-a",
        "fable_weekly",
        used_percent=99.0,
        source="usage_api",
        observed_at=_GATE_NOW,
        reset_at=_GATE_NOW + 3600.0,
        reset_identity="fable-reset-1",
    )
    view.ingest_window(
        "acct-a",
        "five_hour",
        used_percent=40.0,
        source="usage_api",
        observed_at=_GATE_NOW - five_hour_age_seconds,
    )
    view.ingest_window("acct-a", "seven_day", used_percent=40.0, source="usage_api", observed_at=_GATE_NOW)
    return view


def test_family_gate_passes_when_all_six_conditions_hold() -> None:
    view = _gate_satisfying_view()

    outcome = classify_balanced_cooldown_scope(
        view, account_id="acct-a", model="claude-fable-5", upstream_status_code=429, now=_GATE_NOW
    )

    assert outcome == FamilyGateOutcome("family", "fable_family_gate_satisfied", family_deadline=_GATE_NOW + 3600.0)


def test_family_gate_falls_back_to_account_wide_when_request_family_is_not_fable() -> None:
    view = _gate_satisfying_view()

    outcome = classify_balanced_cooldown_scope(
        view, account_id="acct-a", model="claude-sonnet-5", upstream_status_code=429, now=_GATE_NOW
    )

    assert outcome.scope == "account"
    assert outcome.reason == "request_family_not_fable"
    assert outcome.family_deadline is None


def test_family_gate_falls_back_to_account_wide_when_status_is_not_an_upstream_quota_429() -> None:
    view = _gate_satisfying_view()

    outcome = classify_balanced_cooldown_scope(
        view, account_id="acct-a", model="claude-fable-5", upstream_status_code=403, now=_GATE_NOW
    )

    assert outcome.scope == "account"
    assert outcome.reason == "not_upstream_quota_429"


def test_family_gate_falls_back_to_account_wide_when_fable_weekly_is_below_the_saturation_threshold() -> None:
    view = _gate_satisfying_view()
    view.ingest_window(
        "acct-a", "fable_weekly", used_percent=98.9, source="usage_api", observed_at=_GATE_NOW, reset_at=_GATE_NOW + 3600.0
    )

    outcome = classify_balanced_cooldown_scope(
        view, account_id="acct-a", model="claude-fable-5", upstream_status_code=429, now=_GATE_NOW
    )

    assert outcome.scope == "account"
    assert outcome.reason == "fable_weekly_not_saturated"


def test_family_gate_falls_back_to_account_wide_when_five_hour_observation_is_stale() -> None:
    # >99% and <=70% both still hold; only the freshness (<=15 min) condition fails.
    view = _gate_satisfying_view(five_hour_age_seconds=16 * 60)

    outcome = classify_balanced_cooldown_scope(
        view, account_id="acct-a", model="claude-fable-5", upstream_status_code=429, now=_GATE_NOW
    )

    assert outcome.scope == "account"
    assert outcome.reason == "five_hour_not_clear"


def test_family_gate_falls_back_to_account_wide_when_seven_day_exceeds_the_clear_threshold() -> None:
    view = _gate_satisfying_view()
    view.ingest_window("acct-a", "seven_day", used_percent=70.1, source="usage_api", observed_at=_GATE_NOW)

    outcome = classify_balanced_cooldown_scope(
        view, account_id="acct-a", model="claude-fable-5", upstream_status_code=429, now=_GATE_NOW
    )

    assert outcome.scope == "account"
    assert outcome.reason == "seven_day_not_clear"


def test_family_gate_falls_back_to_account_wide_when_the_fable_reset_is_missing() -> None:
    view = ObservationView()
    view.ingest_window("acct-a", "fable_weekly", used_percent=99.0, source="usage_api", observed_at=_GATE_NOW)
    view.ingest_window("acct-a", "five_hour", used_percent=40.0, source="usage_api", observed_at=_GATE_NOW)
    view.ingest_window("acct-a", "seven_day", used_percent=40.0, source="usage_api", observed_at=_GATE_NOW)

    outcome = classify_balanced_cooldown_scope(
        view, account_id="acct-a", model="claude-fable-5", upstream_status_code=429, now=_GATE_NOW
    )

    assert outcome.scope == "account"
    assert outcome.reason == "fable_reset_not_valid"


# -- family-scoped cooldown: eligibility filtering ---------------------------


def test_family_scoped_cooldown_blocks_fable_placements_but_leaves_default_family_eligible() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock)
    now = clock()
    router.install_cooldown(
        account_id="acct-a",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint=None,
        scope="family",
        model_family="fable",
        deadline=now + 3600.0,
        reason="fable_family_gate_satisfied",
    )

    assert router.family_cooldown_deadline("acct-a", "fable", now=now) == pytest.approx(now + 3600.0)
    assert router.family_cooldown_deadline("acct-a", "default", now=now) is None
    assert router.account_cooldown_deadline("acct-a", now=now) is None

    fable_candidate = _candidate(
        "acct-a", family_cooldown_until=router.family_cooldown_deadline("acct-a", "fable", now=now)
    )
    with pytest.raises(NoEligibleAccountError):
        router.place_session(
            session_key=SessionKey(digest=b"\x30" * 32, kind="content_hash"),
            model="claude-fable-5",
            candidates=[fable_candidate],
            seed=b"seed",
            now=now,
        )

    default_candidate = _candidate(
        "acct-a", family_cooldown_until=router.family_cooldown_deadline("acct-a", "default", now=now)
    )
    placement = router.place_session(
        session_key=SessionKey(digest=b"\x31" * 32, kind="content_hash"),
        model="claude-sonnet-5",
        candidates=[default_candidate],
        seed=b"seed",
        now=now,
    )
    assert placement.account_id == "acct-a"


# -- cooldown expiry on either clock (sleep-paused monotonic guard) ----------


def test_wall_deadline_expires_cooldowns_while_monotonic_clock_is_paused() -> None:
    # macOS sleep: the monotonic clock stands still while wall time runs past
    # the real reset. The wall deadline must end the cooldown regardless.
    clock = _FakeClock()
    wall_clock = _FakeClock(2_000_000.0)
    router = _make_router(clock=clock, wall_clock=wall_clock)
    now = clock()
    router.install_cooldown(
        account_id="acct-a",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint=None,
        scope="account",
        deadline=now + 126_000.0,
        reason="fable_weekly_not_saturated",
    )
    router.install_cooldown(
        account_id="acct-a",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint=None,
        scope="family",
        model_family="fable",
        deadline=now + 126_000.0,
        reason="fable_family_gate_satisfied",
    )

    wall_clock.advance(126_001.0)
    clock.advance(3_600.0)  # the machine was awake for only an hour of it

    assert router.account_cooldown_deadline("acct-a") is None
    assert router.family_cooldown_deadline("acct-a", "fable") is None


def test_cooldown_deadline_reflects_the_earlier_of_both_clocks() -> None:
    clock = _FakeClock()
    wall_clock = _FakeClock(2_000_000.0)
    router = _make_router(clock=clock, wall_clock=wall_clock)
    router.install_cooldown(
        account_id="acct-a",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint=None,
        scope="account",
        deadline=clock() + 3_600.0,
        reason="quota",
    )

    clock.advance(100.0)
    wall_clock.advance(200.0)

    # Wall remaining (3400s) is now the smaller: the reported monotonic
    # deadline shrinks to match it.
    assert router.account_cooldown_deadline("acct-a") == pytest.approx(clock() + 3_400.0)

    clock.advance(3_500.0)  # monotonic deadline passes while wall has time left
    wall_clock.advance(100.0)
    assert router.account_cooldown_deadline("acct-a") is None


def test_restored_cooldowns_carry_a_wall_deadline(tmp_path: Any) -> None:
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    try:
        now_utc = 1_700_000_000.0
        pending_write = store.upsert_cooldown(
            account_id="acct-a",
            scope="account",
            model_family="",
            account_incarnation_id="inc-acct-a",
            account_profile_fingerprint="fp-a",
            deadline_utc=now_utc + 3_600.0,
            reason="quota",
            evidence="",
            updated_at_utc=now_utc,
            high_priority=True,
        )
        pending_write.wait(timeout=5.0)
        restore_result = store.restore(RestoreValidationContext(now_utc=now_utc))

        router = _make_router(balanced_epoch_id=store.balanced_epoch_id)
        router.restore_from_store(restore_result, now=5_000.0, wall_now=now_utc)

        assert router.account_cooldown_deadline(
            "acct-a", now=5_010.0, wall_now=now_utc + 10.0
        ) == pytest.approx(5_010.0 + 3_590.0)
        # Sleep drift after restore: wall passes the deadline while the
        # monotonic clock has barely moved.
        assert router.account_cooldown_deadline("acct-a", now=5_100.0, wall_now=now_utc + 3_601.0) is None
    finally:
        store.close()


# -- family-dimension pins: salted identity, correlated HRW ------------------

_FAMILY_TEST_SESSION_ID = "3fa85f64-5717-4562-b3fc-2c963f66afa6"


def _family_test_body() -> dict[str, Any]:
    return {"metadata": {"user_id": _claude_code_user_id(_FAMILY_TEST_SESSION_ID)}}


def test_session_key_salts_the_pin_digest_by_family_and_shares_the_scoring_digest() -> None:
    seed = b"seed-family-salt"
    body = _family_test_body()

    default_key = derive_session_key(body, seed, "default")
    fable_key = derive_session_key(body, seed, "fable")

    assert default_key is not None and fable_key is not None
    assert default_key.digest != fable_key.digest
    assert default_key.scoring_digest == fable_key.scoring_digest
    assert default_key.digest != default_key.scoring_digest
    assert default_key.digest == _reference_pin_key_digest(
        seed, default_key.scoring_digest, b"default"
    )
    assert fable_key.digest == _reference_pin_key_digest(seed, fable_key.scoring_digest, b"fable")
    assert (default_key.family, fable_key.family) == ("default", "fable")
    with pytest.raises(ValueError):
        derive_session_key(body, seed, "sonnet")


def test_place_session_rejects_a_family_mismatched_model() -> None:
    seed = b"seed-family-guard"
    router = _make_router()
    fable_key = derive_session_key(_family_test_body(), seed, "fable")
    default_key = derive_session_key(_family_test_body(), seed, "default")
    assert fable_key is not None and default_key is not None
    candidates = [_candidate("acct-a")]

    with pytest.raises(ValueError):
        router.place_session(
            session_key=fable_key, model="claude-sonnet-5", candidates=candidates, seed=seed
        )
    with pytest.raises(ValueError):
        router.place_session(
            session_key=default_key, model="claude-fable-5", candidates=candidates, seed=seed
        )

    hand_built_key = SessionKey(digest=b"\x40" * 32, kind="content_hash")
    placement = router.place_session(
        session_key=hand_built_key, model="claude-fable-5", candidates=candidates, seed=seed
    )
    assert placement.created


def test_equal_family_weights_co_locate_both_family_pins_of_one_session() -> None:
    seed = b"seed-colocate"
    router = _make_router()
    default_key = derive_session_key(_family_test_body(), seed, "default")
    fable_key = derive_session_key(_family_test_body(), seed, "fable")
    assert default_key is not None and fable_key is not None
    candidates = [_candidate("acct-a"), _candidate("acct-b")]

    default_placement = router.place_session(
        session_key=default_key, model="claude-sonnet-5", candidates=candidates, seed=seed
    )
    fable_placement = router.place_session(
        session_key=fable_key, model="claude-fable-5", candidates=candidates, seed=seed
    )

    assert fable_placement.created
    assert fable_placement.account_id == default_placement.account_id
    assert router.pin_count() == 2


def test_cross_family_migrations_update_only_their_own_pin() -> None:
    seed = b"seed-cross-family-migrations"
    router = _make_router()
    default_key = derive_session_key(_family_test_body(), seed, "default")
    fable_key = derive_session_key(_family_test_body(), seed, "fable")
    assert default_key is not None and fable_key is not None
    candidates = [_candidate(account_id) for account_id in ("acct-a", "acct-b", "acct-c")]
    source_candidates = [candidates[0]]

    router.place_session(
        session_key=default_key,
        model="claude-sonnet-5",
        candidates=source_candidates,
        seed=seed,
    )
    router.place_session(
        session_key=fable_key,
        model="claude-fable-5",
        candidates=source_candidates,
        seed=seed,
    )
    default_reservation, default_is_owner = router.acquire_migration_reservation(
        default_key.digest,
        source_account="acct-a",
        source_generation=0,
        target_account="acct-b",
        attempt_id="default-owner",
    )
    fable_reservation, fable_is_owner = router.acquire_migration_reservation(
        fable_key.digest,
        source_account="acct-a",
        source_generation=0,
        target_account="acct-c",
        attempt_id="fable-owner",
    )
    assert default_is_owner and fable_is_owner

    default_outcome, default_pin, _default_barrier = router.commit_at_headers(
        default_key.digest,
        attempt_id="default-owner",
        source_account="acct-a",
        source_generation=0,
        target_account="acct-b",
        target_account_incarnation_id="inc-acct-b",
    )

    assert default_outcome == "committed"
    assert default_pin is not None
    assert (default_pin.account_id, default_pin.generation) == ("acct-b", 1)
    fable_pin = router.get_pin(fable_key.digest)
    assert fable_pin is not None
    assert (fable_pin.account_id, fable_pin.generation) == ("acct-a", 0)
    assert default_reservation.outcome == "committed"
    assert router.get_migration_reservation(fable_key.digest) is fable_reservation

    fable_outcome, fable_pin, _fable_barrier = router.commit_at_headers(
        fable_key.digest,
        attempt_id="fable-owner",
        source_account="acct-a",
        source_generation=0,
        target_account="acct-c",
        target_account_incarnation_id="inc-acct-c",
    )

    assert fable_outcome == "committed"
    assert fable_pin is not None
    final_default_pin = router.get_pin(default_key.digest)
    assert final_default_pin is not None
    assert (final_default_pin.account_id, final_default_pin.generation) == ("acct-b", 1)
    assert (fable_pin.account_id, fable_pin.generation) == ("acct-c", 1)


def test_account_removal_deletes_both_family_pins_for_one_session() -> None:
    seed = b"seed-cross-family-removal"
    router = _make_router()
    default_key = derive_session_key(_family_test_body(), seed, "default")
    fable_key = derive_session_key(_family_test_body(), seed, "fable")
    assert default_key is not None and fable_key is not None
    candidates = [_candidate("acct-a")]

    router.place_session(
        session_key=default_key, model="claude-sonnet-5", candidates=candidates, seed=seed
    )
    router.place_session(
        session_key=fable_key, model="claude-fable-5", candidates=candidates, seed=seed
    )

    router.remove_account("acct-a", "inc-acct-a")

    assert router.get_pin(default_key.digest) is None
    assert router.get_pin(fable_key.digest) is None


def test_fable_family_cooldown_diverges_only_the_fable_pin() -> None:
    seed = b"seed-diverge"
    clock = _FakeClock()
    router = _make_router(clock=clock)
    now = clock()
    default_key = derive_session_key(_family_test_body(), seed, "default")
    fable_key = derive_session_key(_family_test_body(), seed, "fable")
    assert default_key is not None and fable_key is not None
    account_ids = ["acct-a", "acct-b"]

    default_placement = router.place_session(
        session_key=default_key,
        model="claude-haiku-4-5",
        candidates=[_candidate(account_id) for account_id in account_ids],
        seed=seed,
        now=now,
    )
    cooled = default_placement.account_id
    surviving = next(account_id for account_id in account_ids if account_id != cooled)
    router.install_cooldown(
        account_id=cooled,
        account_incarnation_id=f"inc-{cooled}",
        account_profile_fingerprint=None,
        scope="family",
        model_family="fable",
        deadline=now + 3600.0,
        reason="fable_family_gate_satisfied",
    )

    fable_candidates = [
        _candidate(
            account_id,
            family_cooldown_until=router.family_cooldown_deadline(account_id, "fable", now=now),
        )
        for account_id in account_ids
    ]
    fable_placement = router.place_session(
        session_key=fable_key, model="claude-fable-5", candidates=fable_candidates, seed=seed, now=now
    )

    assert fable_placement.account_id == surviving
    default_pin = router.get_pin(default_key.digest)
    assert default_pin is not None
    assert default_pin.account_id == cooled
    assert default_pin.generation == 0


def test_family_pins_of_one_session_expire_independently() -> None:
    seed = b"seed-ttl"
    clock = _FakeClock()
    router = _make_router(clock=clock, pin_ttl_uuid_seconds=100.0)
    default_key = derive_session_key(_family_test_body(), seed, "default")
    fable_key = derive_session_key(_family_test_body(), seed, "fable")
    assert default_key is not None and fable_key is not None
    candidates = [_candidate("acct-a")]

    router.place_session(session_key=default_key, model="claude-sonnet-5", candidates=candidates, seed=seed)
    clock.advance(60.0)
    router.place_session(session_key=fable_key, model="claude-fable-5", candidates=candidates, seed=seed)

    clock.advance(50.0)  # past the default pin's TTL, within the fable pin's
    router.purge_expired_pins()
    assert router.get_pin(default_key.digest) is None
    assert router.get_pin(fable_key.digest) is not None


def test_account_scope_cooldown_excludes_the_account_for_every_family() -> None:
    seed = b"seed-account-scope"
    clock = _FakeClock()
    router = _make_router(clock=clock)
    now = clock()
    router.install_cooldown(
        account_id="acct-a",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint=None,
        scope="account",
        deadline=now + 3600.0,
        reason="quota",
    )

    for family, model in (("default", "claude-sonnet-5"), ("fable", "claude-fable-5")):
        key = derive_session_key(_family_test_body(), seed, family)
        assert key is not None
        candidate = _candidate(
            "acct-a", account_cooldown_until=router.account_cooldown_deadline("acct-a", now=now)
        )
        with pytest.raises(NoEligibleAccountError):
            router.place_session(
                session_key=key, model=model, candidates=[candidate], seed=seed, now=now
            )


# -- capability evidence: eligible-only, TTL'd, no cross-key inference -------


def test_capability_evidence_ttl_expires_after_one_hour() -> None:
    clock = _FakeClock()
    router = _make_router(clock=clock)
    router.classify_capability_evidence(
        account_id="acct-a",
        capability_key="fable",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint="fp-1",
        status_code=200,
        evidence_source="probe",
    )
    assert router.is_capability_eligible(
        "acct-a", "fable", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )

    clock.advance(CAPABILITY_EVIDENCE_TTL_SECONDS - 1.0)
    assert router.is_capability_eligible(
        "acct-a", "fable", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )

    clock.advance(1.0)  # exactly at the 1h TTL boundary -> expired
    assert not router.is_capability_eligible(
        "acct-a", "fable", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )


def test_capability_evidence_fingerprint_or_incarnation_mismatch_invalidates_it() -> None:
    router = _make_router()
    router.classify_capability_evidence(
        account_id="acct-a",
        capability_key="opus",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint="fp-1",
        status_code=200,
        evidence_source="probe",
    )

    assert router.is_capability_eligible(
        "acct-a", "opus", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )
    # A plan change unmasks (or here, masks) immediately via the fingerprint.
    assert not router.is_capability_eligible(
        "acct-a", "opus", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-2"
    )
    # A reauth-rotated incarnation invalidates it too.
    assert not router.is_capability_eligible(
        "acct-a", "opus", account_incarnation_id="inc-other", account_profile_fingerprint="fp-1"
    )


def test_capability_evidence_never_implies_a_different_capability_key() -> None:
    router = _make_router()
    router.classify_capability_evidence(
        account_id="acct-a",
        capability_key="sonnet",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint="fp-1",
        status_code=200,
        evidence_source="probe",
    )

    assert router.is_capability_eligible(
        "acct-a", "sonnet", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )
    assert not router.is_capability_eligible(
        "acct-a", "opus", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )
    assert not router.is_capability_eligible(
        "acct-a", capability_key("gpt-4"), account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )


def test_capability_classification_records_no_denied_evidence_no_denied() -> None:
    store = _FakeStore()
    router = _make_router(store=store)

    for status_code in (403, 404, 400):
        router.classify_capability_evidence(
            account_id="acct-a",
            capability_key="fable",
            account_incarnation_id="inc-acct-a",
            account_profile_fingerprint="fp-1",
            status_code=status_code,
            evidence_source="probe",
        )
        assert not router.is_capability_eligible(
            "acct-a", "fable", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
        )

    assert store.capability_calls == []  # never a durable write either, denied or otherwise

    # A genuine 2xx, by contrast, DOES record eligible evidence for the exact key.
    router.classify_capability_evidence(
        account_id="acct-a",
        capability_key="fable",
        account_incarnation_id="inc-acct-a",
        account_profile_fingerprint="fp-1",
        status_code=200,
        evidence_source="probe",
    )
    assert router.is_capability_eligible(
        "acct-a", "fable", account_incarnation_id="inc-acct-a", account_profile_fingerprint="fp-1"
    )
    assert len(store.capability_calls) == 1
    assert store.capability_calls[0]["state"] == "eligible"


def test_capability_key_bounded_family_token_or_lowercased_model_id() -> None:
    assert capability_key("claude-fable-5") == "fable"
    assert capability_key("claude-3-Opus-20240229") == "opus"
    assert capability_key("claude-3-5-sonnet-latest") == "sonnet"
    assert capability_key("claude-haiku-4") == "haiku"
    assert capability_key("unfable-model") == "model:unfable-model"
    assert capability_key("gpt-4") == "model:gpt-4"


# -- account removal: durable cleanup of all four row kinds ------------------


def test_remove_account_deletes_all_four_durable_row_kinds_for_only_that_incarnation(tmp_path: Any) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "runtime.sqlite3"
        store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
        try:
            epoch_id = store.balanced_epoch_id
            now_utc = 1_700_000_000.0

            # The incarnation under test: one row of each of the four kinds.
            digest = b"\x40" * 32
            store.upsert_pin(
                session_key_digest=digest,
                key_kind="content_hash",
                account_id="acct-a",
                account_incarnation_id="inc-1",
                last_seen_utc=now_utc,
                expires_at_utc=now_utc + 300.0,
                generation=0,
                balanced_epoch_id=epoch_id,
            ).wait(timeout=5.0)
            store.upsert_cooldown(
                account_id="acct-a",
                scope="account",
                model_family="",
                account_incarnation_id="inc-1",
                account_profile_fingerprint="fp-1",
                deadline_utc=now_utc + 600.0,
                reason="quota_429",
                evidence="",
                updated_at_utc=now_utc,
            ).wait(timeout=5.0)
            store.upsert_usage_observation(
                account_id="acct-a",
                window="five_hour",
                account_incarnation_id="inc-1",
                account_profile_fingerprint="fp-1",
                used_percent=42.0,
                reset_identity="r1",
                reset_at_utc=now_utc + 3600.0,
                observed_at_utc=now_utc,
                source="usage_api",
            ).wait(timeout=5.0)
            store.upsert_capability_evidence(
                account_id="acct-a",
                capability_key="fable",
                account_incarnation_id="inc-1",
                account_profile_fingerprint="fp-1",
                state="eligible",
                evidence_source="probe",
                classifier_version=CAPABILITY_CLASSIFIER_VERSION,
                observed_at_utc=now_utc,
                expires_at_utc=now_utc + 3600.0,
            ).wait(timeout=5.0)

            # A DIFFERENT incarnation under the SAME account_id (a reauth-rotated
            # slot) -- must survive completely untouched.
            other_digest = b"\x41" * 32
            store.upsert_pin(
                session_key_digest=other_digest,
                key_kind="content_hash",
                account_id="acct-a",
                account_incarnation_id="inc-2",
                last_seen_utc=now_utc,
                expires_at_utc=now_utc + 300.0,
                generation=0,
                balanced_epoch_id=epoch_id,
            ).wait(timeout=5.0)
            store.upsert_cooldown(
                account_id="acct-a",
                scope="family",
                model_family="fable",
                account_incarnation_id="inc-2",
                account_profile_fingerprint="fp-2",
                deadline_utc=now_utc + 600.0,
                reason="fable_family_gate_satisfied",
                evidence="",
                updated_at_utc=now_utc,
            ).wait(timeout=5.0)
            store.upsert_usage_observation(
                account_id="acct-a",
                window="seven_day",
                account_incarnation_id="inc-2",
                account_profile_fingerprint="fp-2",
                used_percent=20.0,
                reset_identity="r2",
                reset_at_utc=now_utc + 3600.0,
                observed_at_utc=now_utc,
                source="usage_api",
            ).wait(timeout=5.0)
            store.upsert_capability_evidence(
                account_id="acct-a",
                capability_key="opus",
                account_incarnation_id="inc-2",
                account_profile_fingerprint="fp-2",
                state="eligible",
                evidence_source="probe",
                classifier_version=CAPABILITY_CLASSIFIER_VERSION,
                observed_at_utc=now_utc,
                expires_at_utc=now_utc + 3600.0,
            ).wait(timeout=5.0)

            router = _make_router(balanced_epoch_id=epoch_id, store=store)

            router.remove_account("acct-a", "inc-1")
            await router.await_account_removal_durability("inc-1")

            assert store.get_pin(digest) is None
            assert store.get_cooldown("acct-a", "account", "") is None
            assert store.get_usage_observation("acct-a", "five_hour") is None
            assert store.get_capability_evidence("acct-a", "fable") is None

            assert store.get_pin(other_digest) is not None
            assert store.get_cooldown("acct-a", "family", "fable") is not None
            assert store.get_usage_observation("acct-a", "seven_day") is not None
            assert store.get_capability_evidence("acct-a", "opus") is not None
        finally:
            store.close()

    asyncio.run(scenario())


# ==========================================================================
# `ClaudeBalancedRuntime.exit_mode` cancellation safety (T-22, fix for gap
# G-7(a)): everything before a successful `persist()` is pre-commit -- ANY
# `BaseException`, including `asyncio.CancelledError`, restores "active"
# with the epoch/store untouched; everything after is finalization, run in
# a cancellation-shielded task so it always completes even if the caller
# is cancelled partway through.
# ==========================================================================


def test_exit_mode_cancelled_while_draining_restores_active_with_epoch_and_store_untouched(
    tmp_path: Path,
) -> None:
    """A cancellation delivered while `exit_mode` is still awaiting drain
    completion -- strictly before the pre-commit `persist()` call -- must
    never leave the runtime wedged "draining": it resumes "active" with the
    transition event set and the epoch/store exactly as they were, and a
    subsequent balanced request is admitted and served normally.
    """

    async def scenario() -> None:
        runtime = ClaudeBalancedRuntime()
        await runtime.prepare_and_publish(
            accounts=[],
            accounts_root=tmp_path,
            runtime_db_path=tmp_path / "runtime.sqlite3",
            persist=lambda: None,
            entry="admin_enable",
        )
        assert runtime.status == "active"
        epoch_id_before = runtime.epoch_id
        epoch_seed_before = runtime.epoch_seed
        store_before = runtime._store

        # A slot held open, as if an in-flight balanced dispatch were still
        # relaying bytes -- `exit_mode` blocks draining on it.
        assert runtime.begin_request()

        exit_task = asyncio.create_task(runtime.exit_mode("disabled", publish=lambda: None))
        await asyncio.sleep(0)
        assert runtime.status == "draining"

        exit_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await exit_task

        # Restored to "active", cleanly -- never wedged "draining", and
        # nothing balanced-only was touched.
        assert runtime.status == "active"
        assert runtime._transition_event.is_set()
        assert runtime.epoch_id == epoch_id_before
        assert runtime.epoch_seed == epoch_seed_before
        assert runtime._store is store_before

        # The held-open slot finishes, and a subsequent request is admitted
        # and served balanced exactly as if the cancelled exit never happened.
        runtime.end_request()
        assert runtime.begin_request()
        try:
            assert runtime.router is not None
            placement = runtime.router.place_session(
                session_key=SessionKey(digest=b"\x11" * 32, kind="content_hash"),
                model="claude-sonnet-5",
                candidates=[AccountCandidate(account_id="acct-a", account_incarnation_id="inc-a")],
                seed=runtime.epoch_seed,
            )
            assert placement.created
            assert placement.account_id == "acct-a"
        finally:
            runtime.end_request()

    asyncio.run(scenario())


def test_exit_mode_cancelled_after_persist_still_finalizes_before_the_cancellation_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancellation delivered AFTER `exit_mode`'s `persist()` commit point --
    here, while post-commit finalization is awaiting the epoch rotation --
    must never truncate finalization: `publish()` still runs, the store still
    closes, and `status` still lands on "disabled"; only once all of that has
    actually completed does the cancellation surface back to the caller.
    """
    rotate_gate = asyncio.Event()

    class _GatedPendingWrite:
        async def wait_async(self) -> None:
            await rotate_gate.wait()

    def _gated_rotate_epoch(self: ClaudePoolRuntimeStateStore) -> _GatedPendingWrite:
        return _GatedPendingWrite()

    async def scenario() -> None:
        runtime = ClaudeBalancedRuntime()
        await runtime.prepare_and_publish(
            accounts=[],
            accounts_root=tmp_path,
            runtime_db_path=tmp_path / "runtime.sqlite3",
            persist=lambda: None,
            entry="admin_enable",
        )
        assert runtime.status == "active"

        monkeypatch.setattr(ClaudePoolRuntimeStateStore, "rotate_epoch", _gated_rotate_epoch)
        published: list[str] = []

        exit_task = asyncio.create_task(
            runtime.exit_mode("disabled", publish=lambda: published.append("disabled"))
        )
        # Let exit_mode drain (nothing held open) and persist (a no-op),
        # then reach the gated rotate_epoch await inside its shielded
        # post-commit finalization.
        for _ in range(20):
            await asyncio.sleep(0)
        assert published == []  # finalization hasn't reached publish() yet

        exit_task.cancel()
        for _ in range(20):
            await asyncio.sleep(0)
        # The caller's cancellation was shielded off finalization: it is
        # still in flight, blocked on the gated rotation, not yet done.
        assert not exit_task.done()
        assert published == []
        assert runtime.status == "draining"

        # Let the rotation complete -- finalization can now run to completion.
        rotate_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await exit_task

        assert published == ["disabled"]
        assert runtime.status == "disabled"
        assert runtime._store is None
        assert runtime.router is None

    asyncio.run(scenario())
