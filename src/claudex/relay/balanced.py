"""Balanced-account relay selection, retries, and serving paths."""

from __future__ import annotations

import asyncio
import logging
import math
import secrets
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from claudex import server_support
from claudex.balanced.router import ClaudeBalancedRouter
from claudex.balanced.runtime import ClaudeBalancedRuntime
from claudex.balanced.selection import (
    AccountCandidate,
    NoEligibleAccountError,
    SessionKey,
    binding_windows,
    capability_key,
    derive_session_key,
    derive_stateless_routing_digest,
    is_eligible_candidate,
    pick_weighted_hrw,
    quota_family,
    select_weights,
    warning_factor,
)
from claudex.claude.account_attempts import AccountLegTracker, try_begin_account_leg
from claudex.claude.accounts import (
    AccountRecord,
    AccountRegistryError,
    load_registry,
)
from claudex.claude.ambient_account import AmbientAccountProvider, is_duplicate_identity
from claudex.claude.quota_429 import (
    Quota429Mark,
    enrich_record_degraded,
    enrich_record_with_family_gate,
    finalize_quota_429_record,
)
from claudex.claude.session_fingerprint import (
    extract_session_uuid,
    observability_session_fingerprint,
)
from claudex.config import GatewayConfig
from claudex.relay.registered import (
    _FailedAttempt,
    _attempt_with_account,
    _claude_account_unavailable,
)

logger = logging.getLogger("claudex.server")

# One acquiring or draining transition should settle the request. This limit
# prevents pathological back-to-back transitions from spinning forever.
_BALANCED_TRANSITION_WAIT_LIMIT = 4


async def _install_balanced_quota_cooldown(
    app_state: Any,
    router: ClaudeBalancedRouter,
    *,
    account_id: str,
    account_incarnation_id: str,
    model: str,
    epoch_seed: bytes,
    mark: Quota429Mark,
) -> str:
    """Install one balanced cooldown and return its canonical incident record."""
    gate = None
    family_gate = None
    requested_family = "default"
    monotonic_now: float | None = None
    wall_now: float | None = None
    canonical_record = (
        '{"degradation_reason":"evidence_enrichment_failed",'
        '"record_degraded":true}'
    )
    try:
        canonical_record = finalize_quota_429_record(mark.record)
    except Exception:
        pass
    evidence = ""

    try:
        monotonic_now = time.monotonic()
        wall_now = time.time()
        requested_family = quota_family(model)
        gate = router.classify_cooldown_scope(
            account_id=account_id,
            model=model,
            upstream_status_code=429,
            now=monotonic_now,
        )
        if gate.scope == "family" and gate.family_deadline is None:
            raise ValueError("family cooldown classification omitted its deadline")
        family_deadline_utc = (
            None
            if gate.family_deadline is None
            else datetime.fromtimestamp(
                wall_now + (gate.family_deadline - monotonic_now),
                tz=timezone.utc,
            )
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        family_gate = {
            "scope": gate.scope,
            "reason": gate.reason,
            "family_deadline_utc": family_deadline_utc,
        }
        session_fingerprint = (
            observability_session_fingerprint(
                epoch_seed, mark.session_literals[1]
            )
            if mark.session_literals is not None
            and len(mark.session_literals) == 2
            else None
        )
        enrich_record_with_family_gate(
            mark.record,
            scope=gate.scope,
            reason=gate.reason,
            family_deadline_utc=family_deadline_utc,
            quota_family=requested_family,
            session_fingerprint=session_fingerprint,
        )
        canonical_record = finalize_quota_429_record(mark.record)
        evidence = canonical_record
    except Exception:
        logger.warning("failed to enrich Claude 429 cooldown evidence")
        try:
            if gate is None:
                installed_scope = "account"
                family_gate = None
            else:
                installed_scope = (
                    "family"
                    if gate.scope == "family" and gate.family_deadline is not None
                    else "account"
                )
                if family_gate is None:
                    family_gate = {
                        "scope": gate.scope,
                        "reason": gate.reason,
                        "family_deadline_utc": None,
                    }
            enrich_record_degraded(
                mark.record,
                installed_scope=installed_scope,
                quota_family=requested_family,
                family_gate=family_gate,
            )
            canonical_record = finalize_quota_429_record(mark.record)
        except Exception:
            canonical_record = (
                '{"degradation_reason":"evidence_enrichment_failed",'
                '"record_degraded":true}'
            )
        evidence = ""

    if monotonic_now is None:
        try:
            monotonic_now = time.monotonic()
        except Exception:
            router.persistence_degraded = True
            return canonical_record
    if wall_now is None:
        try:
            wall_now = time.time()
        except Exception:
            router.persistence_degraded = True
            return canonical_record

    if (
        gate is not None
        and gate.scope == "family"
        and gate.family_deadline is not None
    ):
        installed_scope = "family"
        deadline = gate.family_deadline
        reason = gate.reason
        model_family = requested_family
    else:
        installed_scope = "account"
        deadline = monotonic_now + mark.cooldown_seconds
        reason = (
            gate.reason
            if gate is not None
            else "evidence_classification_unavailable"
        )
        model_family = ""

    try:
        fingerprint = server_support._account_profile_fingerprint(
            app_state, account_id
        )
    except Exception:
        router.persistence_degraded = True
        fingerprint = None

    try:
        pending_write = router.install_cooldown(
            account_id=account_id,
            account_incarnation_id=account_incarnation_id,
            account_profile_fingerprint=fingerprint,
            scope=installed_scope,
            model_family=model_family,
            deadline=deadline,
            reason=reason,
            evidence=evidence,
            now=monotonic_now,
            wall_now=wall_now,
        )
    except Exception:
        router.persistence_degraded = True
        return canonical_record
    if pending_write is not None:
        try:
            await pending_write.wait_async()
        except Exception:
            router.persistence_degraded = True
    return canonical_record


async def _record_balanced_capability_evidence(
    app_state: Any,
    router: ClaudeBalancedRouter,
    *,
    account_id: str,
    account_incarnation_id: str,
    model: str,
    upstream_response: httpx.Response,
) -> None:
    """Record balanced-mode capability evidence from a successful 2xx.

    Only `eligible` evidence for the exact capability key is recorded.
    `classify_capability_evidence` is the authoritative gate; this path never
    records `denied` evidence or infers evidence across keys.
    """
    fingerprint = server_support._account_profile_fingerprint(app_state, account_id)
    if fingerprint is None:
        return
    router.classify_capability_evidence(
        account_id=account_id,
        capability_key=capability_key(model),
        account_incarnation_id=account_incarnation_id,
        account_profile_fingerprint=fingerprint,
        status_code=upstream_response.status_code,
        evidence_source="serve_path_2xx",
    )


def _balanced_routing_not_active() -> JSONResponse:
    """Return the reserved 503 for an inconsistent balanced-routing state.

    This means `claude_account.routing` is published as "balanced" with no
    usable runtime outside a controlled transition. Requests arriving during
    a transition wait and dispatch under the resulting mode instead.
    """
    return _claude_account_unavailable("balanced routing is not active")


async def _passthrough_with_claude_balanced(
    request: Request, raw_body: bytes, parsed_body: Any
) -> Response:
    """Dispatch fail-closed and transition-aware through the balanced runtime.

    Only an active `ClaudeBalancedRuntime` ever serves this mode. A request that
    arrives while a controlled enable ("acquiring") or exit ("draining") transition is
    in flight awaits it (`ClaudeBalancedRuntime.wait_for_transition`), then re-reads the
    published `claude_account.routing` mode and dispatches under THAT mode — it is
    never rejected merely because a controlled transition is running. The 503
    "balanced routing is not active" is reserved for the inconsistent state outside a
    controlled transition; balanced traffic never falls through to single-account or
    fallback routing.
    """
    app_state = request.app.state
    runtime: ClaudeBalancedRuntime = app_state.claude_balanced_runtime
    for _ in range(_BALANCED_TRANSITION_WAIT_LIMIT):
        if runtime.status in ("acquiring", "draining"):
            await runtime.wait_for_transition()
            config: GatewayConfig = app_state.config
            if config.claude_account_routing_mode != "balanced":
                return await _passthrough_to_anthropic(request, raw_body, parsed_body)
            continue
        if runtime.begin_request():
            try:
                return await _passthrough_with_balanced_pool(request, raw_body, parsed_body, runtime)
            finally:
                runtime.end_request()
        break
    return _balanced_routing_not_active()


def _balanced_candidates(
    records: Iterable[AccountRecord], router: ClaudeBalancedRouter, *, family: str, now: float
) -> list[AccountCandidate]:
    """One `AccountCandidate` per registered account for a request's whole retry chain.

    `account_cooldown_until` and `family_cooldown_until` are absolute monotonic
    deadlines read from the router's durable cooldown state. Balanced routing
    never consults the fallback pool's account-wide, in-memory
    `AccountCooldownTracker`. `capability_denied` remains false because the
    current classifier records only `eligible` capability evidence.
    """
    candidates = []
    for record in records:
        ready = record.state == "ready"
        candidates.append(
            AccountCandidate(
                account_id=record.id,
                account_incarnation_id=record.account_incarnation_id,
                ready=ready,
                account_cooldown_until=router.account_cooldown_deadline(record.id, now=now) if ready else None,
                family_cooldown_until=(
                    router.family_cooldown_deadline(record.id, family, now=now) if ready else None
                ),
            )
        )
    return candidates


def _balanced_pick_account(
    router: ClaudeBalancedRouter,
    *,
    session_key_digest: bytes,
    model: str,
    candidates: Sequence[AccountCandidate],
    seed: bytes,
    already_attempted: frozenset[str] = frozenset(),
    now: float | None = None,
) -> str:
    """A weighted-HRW pick against the router's live pressure/in-flight state, WITHOUT
    touching the pin map: `place_session`'s own pick logic, reconstructed here from its
    public building blocks, so unpinnable/count_tokens-fallback routing and each
    migration hop's next-target selection never insert a pin-map entry.
    """
    now = time.monotonic() if now is None else now
    eligible = [
        candidate
        for candidate in candidates
        if is_eligible_candidate(candidate, now=now, already_attempted=already_attempted)
    ]
    if not eligible:
        raise NoEligibleAccountError("no eligible account is available for balanced routing")
    family = quota_family(model)
    account_ids = [candidate.account_id for candidate in eligible]
    floor = router.candidate_set_unknown_floor(account_ids, family, now=now)
    pressures = {
        account_id: router.account_pressure(account_id, family, now=now, floor=floor)
        for account_id in account_ids
    }
    windows = binding_windows(family)
    warning_factors = {
        account_id: warning_factor(router.observations, account_id, windows, now=now)
        for account_id in account_ids
    }
    in_flight = {account_id: router.in_flight_count(account_id) for account_id in account_ids}
    weights = select_weights(
        account_ids, pressures=pressures, warning_factors=warning_factors, in_flight=in_flight
    )
    return pick_weighted_hrw(weights=weights, seed=seed, session_key_digest=session_key_digest)


def _balanced_eligible_candidate_set(
    records_by_id: Mapping[str, AccountRecord]
) -> list[AccountRecord]:
    """Return registered, ready, capability-eligible accounts, ignoring cooldowns.

    The current classifier records only `eligible` capability evidence, so
    every registered, ready account qualifies.
    """
    return [record for record in records_by_id.values() if record.state == "ready"]


def _balanced_all_cooling_response(
    records_by_id: Mapping[str, AccountRecord],
    router: ClaudeBalancedRouter,
    *,
    family: str,
    chain_exhausted_429: Response | None,
) -> Response:
    """Respond after a retry chain or initial placement finds no eligible account.

    A chain that exhausted on a real upstream 429 relays that response
    verbatim. Otherwise this synthesizes an Anthropic-compatible 429 with
    `Retry-After` based on the earliest unblock time among registered, ready,
    capability-eligible accounts. A disabled or capability-denied account
    cannot shorten that value. An empty candidate set returns 503.
    """
    if chain_exhausted_429 is not None:
        return chain_exhausted_429
    candidate_set = _balanced_eligible_candidate_set(records_by_id)
    if not candidate_set:
        return JSONResponse(
            server_support._claude_error_body(
                "api_error", "no registered account is eligible for the requested model"
            ),
            status_code=503,
        )
    now = time.monotonic()

    # An account unblocks at the later of its account-wide and family
    # deadlines; Retry-After uses the earliest unblock time across accounts.
    def _unblock_at(record: AccountRecord) -> float:
        account_deadline = router.account_cooldown_deadline(record.id, now=now) or now
        family_deadline = router.family_cooldown_deadline(record.id, family, now=now) or now
        return max(account_deadline, family_deadline)

    min_unblock_at = min(_unblock_at(record) for record in candidate_set)
    retry_after = max(1, math.ceil(min_unblock_at - now))
    return JSONResponse(
        server_support._claude_error_body(
            "rate_limit_error",
            "every eligible claude account is rate-limited; retry after the cooldown",
        ),
        status_code=429,
        headers={"retry-after": str(retry_after)},
    )


async def _passthrough_with_balanced_pool(
    request: Request, raw_body: bytes, parsed_body: Any, runtime: ClaudeBalancedRuntime
) -> Response:
    """Serve one request through an active balanced runtime.

    The registry is read through without a cache, so CLI and dashboard account
    changes take effect immediately. The session key is derived from the parsed
    body before `_rewrite_metadata_account_uuid` mutates it. Token-count requests
    use `_serve_balanced_count_tokens` instead of this placement and migration
    flow.
    """
    assert runtime.router is not None
    try:
        records = load_registry()
    except AccountRegistryError as exc:
        return _claude_account_unavailable(f"cannot read the claude account registry: {exc}")

    records_by_id = {record.id: record for record in records}
    provider: AmbientAccountProvider | None = request.app.state.claude_ambient_accounts
    if provider is not None:
        member = provider.pool_member()
        if member is not None and not is_duplicate_identity(member, records):
            records_by_id[member.record.id] = member.record
            logger.debug(
                "balanced: ambient account %.8s (%s) joined the candidate set",
                member.record.id,
                member.record.email,
            )
        elif member is not None:
            logger.debug(
                "balanced: ambient login %s suppressed, registered account has the same identity",
                member.record.email,
            )
    model = parsed_body.get("model") if isinstance(parsed_body, dict) else None
    model = model if isinstance(model, str) else ""
    # The routing identity is frozen here, once per request: the same model
    # string (and therefore the same quota family) feeds the key derivation
    # and every downstream placement/cooldown decision.
    session_key = (
        derive_session_key(parsed_body, runtime.epoch_seed, quota_family(model))
        if isinstance(parsed_body, dict)
        else None
    )

    if request.url.path.endswith("/count_tokens"):
        return await _serve_balanced_count_tokens(
            request, raw_body, parsed_body, runtime, records_by_id, session_key, model
        )
    if session_key is not None:
        return await _serve_balanced_pinned_message(
            request, raw_body, parsed_body, runtime, records_by_id, session_key, model
        )
    return await _serve_balanced_stateless_message(
        request, raw_body, parsed_body, runtime, records_by_id, model
    )


async def _serve_balanced_stateless_message(
    request: Request,
    raw_body: bytes,
    parsed_body: Any,
    runtime: ClaudeBalancedRuntime,
    records_by_id: Mapping[str, AccountRecord],
    model: str,
) -> Response:
    """Route an unpinnable request using one fresh stateless HRW digest.

    The digest is reused for the request's complete retry chain, never
    persisted, and never inserted into the pin map.
    """
    try:
        extracted_session = extract_session_uuid(
            parsed_body if isinstance(parsed_body, dict) else {}
        )
        session_literals: tuple[str, ...] | None = extracted_session or ()
        leg_tracker: AccountLegTracker | None = AccountLegTracker(
            "balanced_stateless", session_literals=session_literals
        )
    except Exception:
        session_literals = None
        leg_tracker = None
    router = runtime.router
    assert router is not None
    family = quota_family(model)
    digest = derive_stateless_routing_digest(runtime.epoch_seed, secrets.token_bytes(32))
    candidates = _balanced_candidates(records_by_id.values(), router, family=family, now=time.monotonic())

    attempted: set[str] = set()
    chain_429: Response | None = None
    while True:
        try:
            account_id = _balanced_pick_account(
                router,
                session_key_digest=digest,
                model=model,
                candidates=candidates,
                seed=runtime.epoch_seed,
                already_attempted=frozenset(attempted),
            )
        except NoEligibleAccountError:
            return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=chain_429)

        attempted.add(account_id)
        record = records_by_id[account_id]

        async def _on_quota_429(
            mark: Quota429Mark,
            *,
            _account_id: str = record.id,
            _incarnation: str = record.account_incarnation_id,
        ) -> str:
            canonical_record = await _install_balanced_quota_cooldown(
                request.app.state,
                router,
                account_id=_account_id,
                account_incarnation_id=_incarnation,
                model=model,
                epoch_seed=runtime.epoch_seed,
                mark=mark,
            )
            assert runtime.usage_poll_coordinator is not None
            runtime.usage_poll_coordinator.request_manual_refresh(_account_id)
            return canonical_record

        async def _on_response(
            upstream_response: httpx.Response,
            *,
            _account_id: str = record.id,
            _incarnation: str = record.account_incarnation_id,
        ) -> None:
            await _record_balanced_capability_evidence(
                request.app.state,
                router,
                account_id=_account_id,
                account_incarnation_id=_incarnation,
                model=model,
                upstream_response=upstream_response,
            )

        router.begin_attempt(account_id)
        try:
            attempt_context = try_begin_account_leg(leg_tracker,None)
            outcome = await _attempt_with_account(
                request,
                raw_body,
                parsed_body,
                record,
                attempt_context=attempt_context,
                rate_limit_failover=True,
                session_literals=session_literals,
                pin_created=None,
                on_quota_429=_on_quota_429,
                on_response=_on_response,
            )
        finally:
            router.end_attempt(account_id)
        if not isinstance(outcome, _FailedAttempt):
            return outcome
        chain_429 = outcome.response if outcome.rate_limited else None


async def _serve_balanced_pinned_message(
    request: Request,
    raw_body: bytes,
    parsed_body: Any,
    runtime: ClaudeBalancedRuntime,
    records_by_id: Mapping[str, AccountRecord],
    session_key: SessionKey,
    model: str,
) -> Response:
    """Place or follow `session_key`'s pin, then serve its migration chain.

    Every request resolving a pin generation awaits its `pending_durability`
    barrier before an upstream attempt. A quota 429, existing cooldown,
    removed or disabled account, or classified account-specific auth failure
    creates a reservation and migration-attempt token before retrying another
    eligible account. The target's upstream 2xx headers commit the migration
    and await the durable pin write before any downstream byte is forwarded.
    Once a 2xx response is relayed, no cross-account retry occurs.
    """
    try:
        extracted_session = extract_session_uuid(
            parsed_body if isinstance(parsed_body, dict) else {}
        )
        session_literals: tuple[str, ...] | None = extracted_session or ()
        leg_tracker: AccountLegTracker | None = AccountLegTracker(
            "balanced_pinned", session_literals=session_literals
        )
    except Exception:
        session_literals = None
        leg_tracker = None
    router = runtime.router
    assert router is not None
    family = quota_family(model)
    digest = session_key.digest
    now = time.monotonic()
    candidates = _balanced_candidates(records_by_id.values(), router, family=family, now=now)

    try:
        placement = router.place_session(
            session_key=session_key, model=model, candidates=candidates, seed=runtime.epoch_seed, now=now
        )
    except NoEligibleAccountError:
        return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=None)

    normal_pin_created = placement.created
    if placement.created and placement.durability_barrier is not None:
        asyncio.create_task(router.submit_new_pin_durability(digest))
    await router.await_pin_durability(digest)

    pin = router.get_pin(digest)
    router.touch_pin(
        digest,
        is_message_request=True,
        key_is_live=pin is not None,
        account_still_registered=pin is not None and pin.account_id in records_by_id,
    )

    attempted: set[str] = set()
    chain_429: Response | None = None
    owner_attempt_id: str | None = None
    current_target = ""

    while True:
        pin = router.get_pin(digest)
        if pin is None:
            # Extremely rare mid-flight eviction/removal race: re-place fresh.
            try:
                placement = router.place_session(
                    session_key=session_key,
                    model=model,
                    candidates=_balanced_candidates(
                        records_by_id.values(), router, family=family, now=time.monotonic()
                    ),
                    seed=runtime.epoch_seed,
                )
            except NoEligibleAccountError:
                return _balanced_all_cooling_response(
                    records_by_id, router, family=family, chain_exhausted_429=chain_429
                )
            normal_pin_created = placement.created
            if placement.created and placement.durability_barrier is not None:
                asyncio.create_task(router.submit_new_pin_durability(digest))
            await router.await_pin_durability(digest)
            owner_attempt_id = None
            continue

        if owner_attempt_id is None:
            current_target = pin.account_id

        record = records_by_id.get(current_target)
        cooldown_check_now = time.monotonic()
        eligible_now = (
            record is not None
            and record.state == "ready"
            and router.account_cooldown_deadline(current_target, now=cooldown_check_now) is None
            and router.family_cooldown_deadline(current_target, family, now=cooldown_check_now) is None
        )
        attempted.add(current_target)

        if eligible_now:
            assert record is not None
            attempt_id = owner_attempt_id
            source_account = pin.account_id
            source_generation = pin.generation
            target_record = record

            async def _commit_hook(
                upstream_response: httpx.Response,
                *,
                _attempt_id: str | None = attempt_id,
                _source_account: str = source_account,
                _source_generation: int = source_generation,
                _target_record: AccountRecord = target_record,
            ) -> None:
                if _attempt_id is None or upstream_response.status_code // 100 != 2:
                    return
                commit_outcome, _committed_pin, barrier = router.commit_at_headers(
                    digest,
                    attempt_id=_attempt_id,
                    source_account=_source_account,
                    source_generation=_source_generation,
                    target_account=_target_record.id,
                    target_account_incarnation_id=_target_record.account_incarnation_id,
                    target_still_registered=_target_record.id in records_by_id,
                )
                if commit_outcome == "committed" and barrier is not None:
                    asyncio.create_task(router.submit_new_pin_durability(digest))
                    await router.await_pin_durability(digest)

            def _on_relay_complete(*, _attempt_id: str | None = attempt_id) -> None:
                # M(target) stays incremented for the whole streamed response; this
                # only releases the migration token once the stream terminates. A
                # no-op if `commit_at_headers` above never ran (not a migration leg)
                # or already resolved (or failed to resolve, e.g. cas_lost).
                if _attempt_id is not None:
                    router.resolve_migration_owner_terminal(
                        digest, attempt_id=_attempt_id, outcome="terminal_failure"
                    )

            async def _on_quota_429(
                mark: Quota429Mark,
                *,
                _account_id: str = target_record.id,
                _incarnation: str = target_record.account_incarnation_id,
            ) -> str:
                canonical_record = await _install_balanced_quota_cooldown(
                    request.app.state,
                    router,
                    account_id=_account_id,
                    account_incarnation_id=_incarnation,
                    model=model,
                    epoch_seed=runtime.epoch_seed,
                    mark=mark,
                )
                assert runtime.usage_poll_coordinator is not None
                runtime.usage_poll_coordinator.request_manual_refresh(_account_id)
                return canonical_record

            async def _on_response(
                upstream_response: httpx.Response,
                *,
                _account_id: str = target_record.id,
                _incarnation: str = target_record.account_incarnation_id,
            ) -> None:
                await _record_balanced_capability_evidence(
                    request.app.state,
                    router,
                    account_id=_account_id,
                    account_incarnation_id=_incarnation,
                    model=model,
                    upstream_response=upstream_response,
                )

            if owner_attempt_id is not None:
                try:
                    attempt_context = try_begin_account_leg(leg_tracker,False)
                    outcome = await _attempt_with_account(
                        request,
                        raw_body,
                        parsed_body,
                        target_record,
                        attempt_context=attempt_context,
                        rate_limit_failover=True,
                        session_literals=session_literals,
                        pin_created=False,
                        commit_hook=_commit_hook,
                        on_relay_complete=_on_relay_complete,
                        on_quota_429=_on_quota_429,
                        on_response=_on_response,
                    )
                except BaseException:
                    # If this owner fails before constructing a response, the
                    # streaming relay's `on_finished` hook cannot run. Release the
                    # reservation and migration token here instead.
                    router.resolve_migration_owner_terminal(
                        digest, attempt_id=owner_attempt_id, outcome="terminal_failure"
                    )
                    raise
            else:
                router.begin_attempt(current_target)
                try:
                    attempt_context = try_begin_account_leg(leg_tracker,normal_pin_created)
                    outcome = await _attempt_with_account(
                        request,
                        raw_body,
                        parsed_body,
                        target_record,
                        attempt_context=attempt_context,
                        rate_limit_failover=True,
                        session_literals=session_literals,
                        pin_created=normal_pin_created,
                        on_quota_429=_on_quota_429,
                        on_response=_on_response,
                    )
                finally:
                    router.end_attempt(current_target)

            if not isinstance(outcome, _FailedAttempt):
                return outcome
            chain_429 = outcome.response if outcome.rate_limited else None
        else:
            chain_429 = None

        # --- Migration reservation and next-target selection ----------------
        if owner_attempt_id is not None:
            router.resolve_migration_preheader_failure(digest, attempt_id=owner_attempt_id)
            owner_attempt_id = None

        try:
            next_target = _balanced_pick_account(
                router,
                session_key_digest=session_key.scoring_digest_or_default,
                model=model,
                candidates=candidates,
                seed=runtime.epoch_seed,
                already_attempted=frozenset(attempted),
            )
        except NoEligibleAccountError:
            return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=chain_429)

        reservation, is_owner = router.acquire_migration_reservation(
            digest,
            source_account=pin.account_id,
            source_generation=pin.generation,
            target_account=next_target,
            attempt_id=uuid.uuid4().hex,
        )
        if is_owner:
            owner_attempt_id = reservation.owner_attempt_id
            current_target = next_target
        else:
            await router.wait_for_migration_reservation(reservation)
            owner_attempt_id = None


async def _serve_balanced_count_tokens(
    request: Request,
    raw_body: bytes,
    parsed_body: Any,
    runtime: ClaudeBalancedRuntime,
    records_by_id: Mapping[str, AccountRecord],
    session_key: SessionKey | None,
    model: str,
) -> Response:
    """Serve token counting without mutating balanced-routing state.

    Follow an existing pin without refreshing `last_seen`; otherwise use
    stateless-digest placement. This path never creates a pin, reservation,
    cooldown, or capability evidence, and never retries across accounts
    (`rate_limit_failover=False`).
    """
    try:
        extracted_session = extract_session_uuid(
            parsed_body if isinstance(parsed_body, dict) else {}
        )
        session_literals: tuple[str, ...] | None = extracted_session or ()
        leg_tracker: AccountLegTracker | None = AccountLegTracker(
            "balanced_count_tokens", session_literals=session_literals
        )
    except Exception:
        session_literals = None
        leg_tracker = None
    router = runtime.router
    assert router is not None
    if session_key is not None:
        pin = router.get_pin(session_key.digest)
        if pin is not None:
            record = records_by_id.get(pin.account_id)
            if record is not None:
                if pin.pending_durability is not None:
                    await router.await_pin_durability(session_key.digest)
                attempt_context = try_begin_account_leg(leg_tracker,None)
                outcome = await _attempt_with_account(
                    request,
                    raw_body,
                    parsed_body,
                    record,
                    attempt_context=attempt_context,
                    rate_limit_failover=False,
                    session_literals=session_literals,
                    pin_created=None,
                )
                return outcome.response if isinstance(outcome, _FailedAttempt) else outcome

    scoring_digest = (
        session_key.scoring_digest_or_default
        if session_key is not None
        else derive_stateless_routing_digest(runtime.epoch_seed, secrets.token_bytes(32))
    )
    family = quota_family(model)
    candidates = _balanced_candidates(records_by_id.values(), router, family=family, now=time.monotonic())
    try:
        account_id = _balanced_pick_account(
            router,
            session_key_digest=scoring_digest,
            model=model,
            candidates=candidates,
            seed=runtime.epoch_seed,
        )
    except NoEligibleAccountError:
        return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=None)
    record = records_by_id[account_id]
    attempt_context = try_begin_account_leg(leg_tracker,None)
    outcome = await _attempt_with_account(
        request,
        raw_body,
        parsed_body,
        record,
        attempt_context=attempt_context,
        rate_limit_failover=False,
        session_literals=session_literals,
        pin_created=None,
    )
    return outcome.response if isinstance(outcome, _FailedAttempt) else outcome
