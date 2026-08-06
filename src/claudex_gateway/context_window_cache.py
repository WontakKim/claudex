"""In-memory TTL cache for per-model context-window sizes.

Both the Codex and Grok clients need to answer "how many tokens fit in this
model's context window" on every mapped request's pre-stream path, backed by
a catalog lookup that is too slow (and too failure-prone) to perform inline.
``ContextWindowCache`` wraps an async ``fetch`` coroutine with a success TTL,
serves the last-known snapshot when a refresh fails, backs off from retrying
a failing catalog for a while, and collapses concurrent cold/expired lookups
into a single in-flight fetch.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class ContextWindowCache:
    """Caches a model-id -> context-window-size snapshot fetched on demand."""

    def __init__(
        self,
        fetch: Callable[[], Awaitable[dict[str, int]]],
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
        self._snapshot: dict[str, int] | None = None
        self._snapshot_time: float | None = None
        self._failure_time: float | None = None

    async def get(self, model: str) -> int | None:
        """Return the model's context window, refreshing the snapshot if needed.

        Serves the cached snapshot without fetching while it is fresh. On a
        cold or expired snapshot, at most one concurrent refresh is performed
        (guarded by ``asyncio.Lock``); a failed refresh serves the last known
        snapshot (or ``None`` when there isn't one) and suppresses further
        refresh attempts until ``failure_backoff_seconds`` have elapsed.
        """
        if self._is_fresh(self._clock()):
            return self._snapshot.get(model) if self._snapshot is not None else None

        async with self._lock:
            now = self._clock()
            if self._is_fresh(now):
                return self._snapshot.get(model) if self._snapshot is not None else None
            if self._failure_time is not None and now - self._failure_time < self._failure_backoff_seconds:
                return self._snapshot.get(model) if self._snapshot is not None else None

            try:
                fetched = await self._fetch()
            except asyncio.CancelledError:
                # Cancellation always propagates, even when the caller's
                # expected_errors tuple would otherwise catch it.
                raise
            except self._expected_errors:
                self._failure_time = self._clock()
                return self._snapshot.get(model) if self._snapshot is not None else None

            self._snapshot = fetched
            self._snapshot_time = self._clock()
            self._failure_time = None
            return self._snapshot.get(model)

    def _is_fresh(self, now: float) -> bool:
        return self._snapshot_time is not None and now - self._snapshot_time < self._ttl_seconds
