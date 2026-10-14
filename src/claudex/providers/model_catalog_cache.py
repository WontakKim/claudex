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
        self._fetched_at: float | None = None
        self._last_error: BaseException | None = None
        self._generation = 0

    def invalidate(self) -> None:
        """Discard snapshot/error state and prevent pending fetch publication."""
        self._generation += 1
        self._snapshot = None
        self._snapshot_time = None
        self._failure_time = None
        self._fetched_at = None
        self._last_error = None

    @property
    def fetched_at(self) -> float | None:
        """Unix timestamp of the last successful fetch, not the monotonic TTL clock."""
        return self._fetched_at

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def get_cached(self, key: str) -> T | None:
        """Read last-known metadata without performing a refresh."""
        return self._snapshot.get(key) if self._snapshot is not None else None

    async def get(self, key: str) -> T | None:
        snapshot = await self.get_snapshot()
        return snapshot.get(key) if snapshot is not None else None

    async def get_snapshot(self, *, force_refresh: bool = False) -> dict[str, T] | None:
        """Return a snapshot copy; explicit refresh bypasses TTL and error backoff.

        Concurrent ordinary lookups share one fetch. A failed refresh keeps
        the last successful snapshot and its timestamp, or returns None
        when no successful snapshot exists.
        """
        if not force_refresh and self._is_fresh(self._clock()):
            return dict(self._snapshot) if self._snapshot is not None else None

        async with self._lock:
            while True:
                now = self._clock()
                if not force_refresh and (
                    self._is_fresh(now)
                    or (self._failure_time is not None
                        and now - self._failure_time < self._failure_backoff_seconds)
                ):
                    return dict(self._snapshot) if self._snapshot is not None else None

                generation = self._generation
                try:
                    fetched = await self._fetch()
                except asyncio.CancelledError:
                    # Cancellation always propagates, even when the caller's
                    # expected_errors tuple would otherwise catch it.
                    raise
                except self._expected_errors as exc:
                    if generation != self._generation:
                        continue
                    self._failure_time = self._clock()
                    self._last_error = exc
                    return dict(self._snapshot) if self._snapshot is not None else None

                if generation != self._generation:
                    continue
                self._snapshot = fetched
                self._snapshot_time = self._clock()
                self._fetched_at = time.time()
                self._failure_time = None
                self._last_error = None
                return dict(self._snapshot)

    def _is_fresh(self, now: float) -> bool:
        return self._snapshot_time is not None and now - self._snapshot_time < self._ttl_seconds
