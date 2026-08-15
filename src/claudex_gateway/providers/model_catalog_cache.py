"""In-memory TTL cache for per-model catalog snapshots.

Codex and Grok clients need to answer per-model catalog questions on every
mapped request's pre-stream path, backed by a catalog lookup that is too slow
(and too failure-prone) to perform inline. ``ModelCatalogCache`` wraps an async
``fetch`` coroutine with a success TTL, serves the last-known snapshot when a
refresh fails, backs off from retrying a failing catalog for a while, and
collapses concurrent cold/expired lookups into a single in-flight fetch. The
snapshot value type is caller-defined.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class ModelCatalogCache(Generic[T]):
    """Caches a model-id -> caller-defined catalog value snapshot on demand."""

    def __init__(
        self,
        fetch: Callable[[], Awaitable[dict[str, T]]],
        *,
        expected_errors: tuple[type[BaseException], ...],
        ttl_seconds: float = 900.0,
        failure_backoff_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._expected_errors = expected_errors
        self._ttl_seconds = ttl_seconds
        self._failure_backoff_seconds = failure_backoff_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._snapshot: dict[str, T] | None = None
        self._snapshot_time: float | None = None
        self._failure_time: float | None = None

    async def get(self, key: str) -> T | None:
        """Return a model's catalog value, refreshing the snapshot if needed.

        Serves the cached snapshot without fetching while it is fresh. On a
        cold or expired snapshot, at most one concurrent refresh is performed
        (guarded by ``asyncio.Lock``); a failed refresh serves the last known
        snapshot (or ``None`` when there isn't one) and suppresses further
        refresh attempts until ``failure_backoff_seconds`` have elapsed.
        """
        if self._is_fresh(self._clock()):
            return self._snapshot.get(key) if self._snapshot is not None else None

        async with self._lock:
            now = self._clock()
            if self._is_fresh(now):
                return self._snapshot.get(key) if self._snapshot is not None else None
            if self._failure_time is not None and now - self._failure_time < self._failure_backoff_seconds:
                return self._snapshot.get(key) if self._snapshot is not None else None

            try:
                fetched = await self._fetch()
            except asyncio.CancelledError:
                # Cancellation always propagates, even when the caller's
                # expected_errors tuple would otherwise catch it.
                raise
            except self._expected_errors:
                self._failure_time = self._clock()
                return self._snapshot.get(key) if self._snapshot is not None else None

            self._snapshot = fetched
            self._snapshot_time = self._clock()
            self._failure_time = None
            return self._snapshot.get(key)

    def _is_fresh(self, now: float) -> bool:
        return self._snapshot_time is not None and now - self._snapshot_time < self._ttl_seconds
