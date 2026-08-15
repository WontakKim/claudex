"""Process-lifetime lifecycle for balanced routing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from claudex.claude.account_usage_cache import ClaudeAccountUsageCache
from claudex.balanced.polling import ClaudeUsagePollCoordinator, UsagePollAccount
from claudex.balanced.router import ClaudeBalancedRouter
from claudex.balanced.state_model import RestoreValidationContext
from claudex.balanced.state_store import ClaudePoolRuntimeStateStore
from claudex.claude.account_profile import load_account_profile_fingerprint
from claudex.claude.accounts import AccountRecord, list_accounts

logger = logging.getLogger(__name__)

# ==========================================================================
# Balanced routing runtime lifecycle
# ==========================================================================

BalancedRuntimeStatus = Literal["disabled", "acquiring", "active", "draining"]


class BalancedPrepareError(RuntimeError):
    """Raised by `ClaudeBalancedRuntime.prepare_and_publish` when the runtime cannot be
    safely readied because a ready account has no valid profile fingerprint. The
    caller (server.py's PUT routing handler) reports this to the admin client;
    preparation is always torn down first, so the previous routing mode is left
    untouched.
    """


class ClaudeBalancedRuntime:
    """Owns balanced routing's whole process-lifetime state machine.

    `status` gates every dispatch (`server._passthrough_with_claude_balanced`):
    "disabled" (no runtime — balanced traffic fails closed), "acquiring" (an enable is
    being prepared — the OLD mode still serves, since `claude_account.routing` itself
    hasn't flipped to "balanced" yet), "active" (`store`/`router` are live and
    `begin_request` admits new dispatch slots), "draining" (an intentional exit or
    process shutdown is underway — no new slot is admitted, in-flight ones are
    awaited). A request arriving mid-transition awaits `wait_for_transition`
    before re-reading the published routing mode and dispatching under it. The
    transition event is cleared for the whole
    "acquiring"/"draining" window and only set once the new state is fully published,
    so a woken waiter never observes a stale mode.

    The server layer supplies `persist` and `publish` hooks so this class can
    enforce the ordering its two distinct lifecycle operations require without
    owning `GatewayConfig` or the settings file itself: enabling persists settings
    before the prepared runtime is published (`prepare_and_publish`); exiting
    persists and publishes the target mode before waking transition waiters
    (`exit_mode`).
    Process shutdown (`shutdown_preserving_epoch`) takes no hook at all — it never
    touches persisted settings or epoch metadata, so a restart can restore them.

    `begin_request` and `end_request` maintain this class's request-slot
    counter, entirely independent of `ClaudeBalancedRouter`'s
    per-account `M(a)` attempt counting: it exists purely so the drain step of
    `exit_mode`/`shutdown_preserving_epoch` has something to wait on.
    """

    def __init__(self) -> None:
        self.status: BalancedRuntimeStatus = "disabled"
        self.epoch_id: str | None = None
        self.epoch_seed: bytes = b""
        self.router: ClaudeBalancedRouter | None = None
        # The usage poll coordinator exists only while this runtime is active;
        # it is `None` in "disabled", "acquiring", or "draining",
        # and also `None` in "active" when `prepare_and_publish` was called
        # without a `usage_cache` (every direct, non-HTTP caller keeps this
        # optional; server.py's real routing handler always supplies one).
        self.usage_poll_coordinator: ClaudeUsagePollCoordinator | None = None
        self.ambient_usage_poll_supplier: (
            Callable[[Sequence[AccountRecord]], UsagePollAccount | None] | None
        ) = None

        self._store: ClaudePoolRuntimeStateStore | None = None
        # The background driver calls `usage_poll_coordinator.run_due_poll`
        # while this runtime is "active" -- `None` whenever no coordinator is
        # driving (every status
        # other than "active", or "active" without a `usage_cache`).
        self._usage_poll_driver_task: asyncio.Task[None] | None = None
        self._accounts_root: Path | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._transition_event = asyncio.Event()
        self._transition_event.set()
        self._active_requests = 0
        self._drain_complete = asyncio.Event()
        self._drain_complete.set()

    @property
    def epoch_active(self) -> bool:
        """Return whether this runtime's current epoch is live in memory.

        This is true only while status is "active". The runtime store's
        `epoch_active` metadata has no public setter and is never written by
        this class; this property is computed from `status` alone.
        """
        return self.status == "active"

    async def wait_for_transition(self) -> None:
        """Block while a controlled enable ("acquiring") or exit ("draining")
        transition is in flight; returns immediately once none is (including when none
        ever was).
        """
        await self._transition_event.wait()

    def begin_request(self) -> bool:
        """Admit one balanced dispatch slot iff `status == "active"`. Returns whether
        the slot was admitted; a caller that admits one MUST call `end_request` exactly
        once, however the dispatch ends.
        """
        if self.status != "active":
            return False
        self._active_requests += 1
        self._drain_complete.clear()
        return True

    def end_request(self) -> None:
        """Release one slot `begin_request` admitted. Always safe to call — a defensive
        no-op floor at zero, never below."""
        if self._active_requests > 0:
            self._active_requests -= 1
        if self._active_requests == 0:
            self._drain_complete.set()

    # -- enabling: prepare the complete runtime, publish only once settings commit ----

    async def prepare_and_publish(
        self,
        *,
        accounts: Sequence[AccountRecord],
        accounts_root: Path,
        runtime_db_path: Path,
        persist: Callable[[], None],
        entry: Literal["startup_restore", "admin_enable"],
        usage_cache: ClaudeAccountUsageCache | None = None,
    ) -> None:
        """Prepare and atomically publish balanced routing.

        Opens and validates the runtime store, restores its state, constructs the
        router, and verifies every ready account has a valid profile
        fingerprint while the old mode remains published, so traffic is
        unaffected until every check passes. `persist()` — the coordinator hook that
        persists+swaps `claude_account.routing` to "balanced" — is invoked exactly
        once, after every check passes and strictly before this runtime is published
        (`status` flips to "active"): it is the commit point. A failure anywhere up to
        and including `persist()` itself tears the (partial) preparation down (closing
        any opened store) and leaves `status` "disabled" — the old mode keeps serving,
        exactly as if this call had never happened.

        `entry` distinguishes the two supported reasons for preparing a runtime:
        `"startup_restore"` is the daemon lifespan restoring an already-persisted
        `"balanced"` mode across a process restart, which reuses the runtime DB's
        existing epoch/pins exactly as `shutdown_preserving_epoch` left them.
        `"admin_enable"` is every OTHER re-entry (an admin PUT transitioning into
        balanced) and durably rotates the epoch — wiping every pin — right after
        the store opens, before anything below restores from it: `exit_mode` is
        the only path that ever invalidates an epoch, and its own rotation can
        itself degrade (`persistence_degraded`) and leave the runtime DB holding
        a stale, intentionally-exited epoch and its pins; an administrative
        re-entry must never resurrect that state, so it always mints a fresh one
        regardless of what the store still contains.

        When supplied, `usage_cache` creates a fresh
        `ClaudeUsagePollCoordinator` for this runtime
        against the same router/store this call just prepared -- omitted, that
        attribute stays `None` (every non-server caller of this method).
        """
        async with self._lifecycle_lock:
            if self.status != "disabled":
                raise RuntimeError(
                    f"cannot enable balanced routing from status {self.status!r}"
                )
            self.status = "acquiring"
            self._transition_event.clear()
            store: ClaudePoolRuntimeStateStore | None = None
            try:
                store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
                if entry == "admin_enable":
                    # Administrative enablement must mint a fresh epoch and
                    # wipe its pins durably before anything below reads or
                    # restores from the store, so a degraded exit's
                    # stale epoch/pins can never be resurrected by a later
                    # administrative re-entry.
                    await store.rotate_epoch().wait_async()
                for record in accounts:
                    if record.state != "ready":
                        continue
                    fingerprint = load_account_profile_fingerprint(accounts_root / record.id)
                    if fingerprint is None:
                        raise BalancedPrepareError(
                            f"claude account {record.id} has no valid "
                            "profile_fingerprint (missing or non-UUID accountUuid); it "
                            "cannot participate in balanced routing until it is "
                            "re-authenticated"
                        )

                restore_result = store.restore(RestoreValidationContext(now_utc=time.time()))
                router = ClaudeBalancedRouter(
                    balanced_epoch_id=store.balanced_epoch_id, store=store
                )
                router.restore_from_store(restore_result)

                persist()

                self._store = store
                self.epoch_id = store.balanced_epoch_id
                self.epoch_seed = store.epoch_seed
                self.router = router
                self._accounts_root = accounts_root
                self.usage_poll_coordinator = (
                    ClaudeUsagePollCoordinator(cache=usage_cache, router=router, store=store)
                    if usage_cache is not None
                    else None
                )
                self.status = "active"
                self._start_usage_poll_driver()
            except BaseException:
                if store is not None:
                    store.close()
                self.status = "disabled"
                raise
            finally:
                self._transition_event.set()

    # -- usage poll driver: runs while active -------------------------------

    def _start_usage_poll_driver(self) -> None:
        """Start the background poll driver after the runtime becomes active.

        This is a no-op when `prepare_and_publish` was called without a
        `usage_cache`, so `usage_poll_coordinator` is `None` and nothing needs
        driving. Must only be called from inside the `_lifecycle_lock` critical
        section that just set `status = "active"`.
        """
        if self.usage_poll_coordinator is None:
            return
        self._usage_poll_driver_task = asyncio.create_task(self._run_usage_poll_driver())

    async def _stop_usage_poll_driver(self) -> None:
        """Cancel and await the poll driver if it is running.

        This must finish before the store closes in both `exit_mode` and
        `shutdown_preserving_epoch`, so no poll tick can ever touch a closed store.
        """
        task = self._usage_poll_driver_task
        self._usage_poll_driver_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_usage_poll_driver(self) -> None:
        """Run the poll driver sequentially until it is cancelled.

        The first tick runs immediately to warm the coordinator, then the loop
        sleeps for the coordinator's own scheduling budget before trying again.
        It runs until `_stop_usage_poll_driver` cancels it and re-reads
        `usage_poll_coordinator`
        every iteration rather than capturing it once, so it stops cleanly the
        moment the attribute is cleared instead of racing a stale reference.
        """
        while True:
            coordinator = self.usage_poll_coordinator
            if coordinator is None:
                return
            await self._usage_poll_driver_tick(coordinator)
            await asyncio.sleep(coordinator.poll_interval_seconds)

    async def _usage_poll_driver_tick(self, coordinator: ClaudeUsagePollCoordinator) -> None:
        """One driver iteration: re-reads the registry (read-through, exactly like
        `server._passthrough_with_balanced_pool`, so an account added/removed
        while balanced routing is active takes effect immediately) and calls
        `run_due_poll` with the current ready set. Never lets an exception escape
        -- an unexpected failure here (a bad registry read, a broken fingerprint
        file, ...) must never crash this loop or surface into a request path.
        """
        try:
            records = list_accounts()
            ready_ids = [record.id for record in records if record.state == "ready"]
            accounts: dict[str, UsagePollAccount] = {}
            accounts_root = self._accounts_root
            if accounts_root is not None:
                for record in records:
                    if record.state != "ready":
                        continue
                    accounts[record.id] = UsagePollAccount(
                        account_id=record.id,
                        account_incarnation_id=record.account_incarnation_id,
                        account_profile_fingerprint=load_account_profile_fingerprint(
                            accounts_root / record.id
                        ),
                    )
            if self.ambient_usage_poll_supplier is not None:
                ambient_account = self.ambient_usage_poll_supplier(records)
                if ambient_account is not None:
                    ready_ids.append(ambient_account.account_id)
                    accounts[ambient_account.account_id] = ambient_account
            await coordinator.run_due_poll(ready_ids, accounts=accounts)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("balanced usage poll driver tick failed", exc_info=True)

    # -- intentional exit: drain, persist, invalidate the epoch, publish -------------

    async def exit_mode(
        self,
        target_mode: str,
        *,
        persist: Callable[[], None] | None = None,
        publish: Callable[[], None],
    ) -> None:
        """Exit balanced mode intentionally for fallback or disabled routing.

        Unlike process shutdown (`shutdown_preserving_epoch`), this is the
        ONLY path that rotates the epoch (invalidating every current-epoch pin) and
        marks it inactive, so a later re-entry (`prepare_and_publish`) always starts
        a fresh epoch.

        Sequence: mark draining (blocks new balanced entrants — `begin_request` only
        admits while "active"), drain in-flight attempts, `persist()` — the
        coordinator hook that durably writes `claude_account.routing` to
        `target_mode`, omitted (`None`, the default) by callers with no settings
        layer of their own — THEN rotate the epoch durably (waits for the
        rotation to commit), THEN `publish()` — the coordinator hook that swaps
        the in-memory published mode, so a woken transition waiter re-reads the
        ALREADY-published target mode — then wake every transition waiter and
        close the store, discarding balanced-only state.

        Crash/cancellation contract: `persist()` is the commit point, and
        everything up to and including it is pre-commit. ANY `BaseException` —
        including `asyncio.CancelledError`, which is not an `Exception` — raised
        while draining or persisting aborts the exit entirely: this runtime
        returns to "active" with the transition event set and its epoch, store,
        and durable pins untouched, and the exception propagates to the caller
        (the PUT handler returns 500 with the mode unchanged; a cancelled caller
        sees its cancellation, with the runtime left cleanly "active" rather than
        wedged "draining"). Once `persist()` has succeeded (or was omitted), the
        target mode is authoritative and this runtime is committed to exiting no
        matter what happens next: finalization (rotate the epoch, degrading on
        failure to `persistence_degraded` rather than rolling back to balanced,
        then `publish()`, then stop+await the poll driver, close the store, and
        land on "disabled") runs in a separate task shielded from the caller's
        own cancellation, so it always completes even if the caller is cancelled
        partway through; the caller's cancellation (if any) is re-raised only
        once that finalization has finished.
        """
        if target_mode == "balanced":
            raise ValueError('exit_mode target_mode must not be "balanced"')
        async with self._lifecycle_lock:
            if self.status != "active":
                raise RuntimeError(
                    f"cannot exit balanced routing from status {self.status!r}"
                )
            self.status = "draining"
            self._transition_event.clear()
            try:
                await self._drain_complete.wait()
                if persist is not None:
                    persist()
            except BaseException:
                # Pre-commit: nothing balanced-only has been touched yet (and
                # persist(), if reached, never committed). Resume serving under
                # "active" with the epoch, store, and pins untouched -- this
                # also covers a cancellation delivered while still draining, so
                # the runtime is never left wedged in "draining" with the
                # transition event cleared.
                self.status = "active"
                self._transition_event.set()
                raise

            # Post-commit: persist() succeeded (or was omitted). Finalization
            # must run to completion no matter what happens to the caller from
            # here on, so it runs in its own task, shielded from the caller's
            # cancellation; a cancellation delivered to the caller only
            # surfaces again once finalization has actually finished.
            finalize_task = asyncio.create_task(self._finalize_exit(publish))
            try:
                await asyncio.shield(finalize_task)
            except asyncio.CancelledError:
                await finalize_task
                raise

    async def _finalize_exit(self, publish: Callable[[], None]) -> None:
        """Complete `exit_mode` after its settings commit.

        Rotate the epoch without rolling back on failure, publish the target mode,
        stop+await the poll driver, close the store, and reset to "disabled". Always
        run inside `asyncio.shield` by its only caller, `exit_mode`, so it completes
        exactly once `persist()` has committed, independent of the caller's own
        cancellation.
        """
        try:
            store = self._store
            assert store is not None
            try:
                await store.rotate_epoch().wait_async()
            except Exception:
                # The target mode is already durably persisted -- the commit
                # point already passed -- so a cleanup failure here must never
                # roll back to balanced; it only degrades the epoch cleanup
                # itself.
                logger.warning(
                    "balanced exit: epoch rotation failed after the target "
                    "mode was already persisted; continuing the exit with "
                    "epoch cleanup persistence_degraded",
                    exc_info=True,
                )
            publish()
        finally:
            # Cancel and await the driver before closing the store so no
            # in-flight or newly scheduled poll tick can touch it once closed.
            await self._stop_usage_poll_driver()
            if self._store is not None:
                self._store.close()
            self._store = None
            self.router = None
            self.usage_poll_coordinator = None
            self._accounts_root = None
            self.epoch_id = None
            self.epoch_seed = b""
            self.status = "disabled"
            self._transition_event.set()

    # -- process shutdown: drain and close, preserving every persisted setting --------

    async def shutdown_preserving_epoch(self) -> None:
        """Drain and close the runtime while preserving the current epoch.

        Unlike `exit_mode`, this never rotates the epoch or touches persisted
        settings, and takes no coordinator hook — so a restart's `prepare_and_publish`
        finds the SAME epoch id/seed/pins/observations/cooldowns/capability evidence
        right where this left them. A no-op when balanced routing was never prepared
        this run (`status == "disabled"`).
        """
        async with self._lifecycle_lock:
            if self.status == "disabled":
                return
            self.status = "draining"
            self._transition_event.clear()
            try:
                await self._drain_complete.wait()
            finally:
                # Use the same cancel-before-close ordering as `exit_mode`;
                # process shutdown must not race a poll tick against the store
                # it is about to close either.
                await self._stop_usage_poll_driver()
                if self._store is not None:
                    self._store.close()
                self._store = None
                self.router = None
                self.usage_poll_coordinator = None
                self._accounts_root = None
                self.epoch_id = None
                self.epoch_seed = b""
                self.status = "disabled"
                self._transition_event.set()
