"""Tests for the TTL model-catalog cache with stale-on-error backoff."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from claudex.providers.model_catalog_cache import ModelCatalogCache


class FetchError(Exception):
    """A recoverable catalog-fetch failure used by these tests."""


class OtherError(Exception):
    """An exception type outside the cache's expected_errors tuple."""


class FakeClock:
    """A controllable stand-in for ``time.monotonic`` — no real sleeping."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeFetch:
    """Counts calls and replays a queue of successes/failures in order."""

    def __init__(self) -> None:
        self.calls = 0
        self._queue: list[tuple[str, Any]] = []

    def queue_result(self, result: dict[str, int]) -> None:
        self._queue.append(("ok", result))

    def queue_error(self, error: BaseException) -> None:
        self._queue.append(("error", error))

    async def __call__(self) -> dict[str, int]:
        self.calls += 1
        # Yield to the event loop so a concurrent caller can observe this
        # fetch as in-flight, exercising the cache's single-flight lock.
        await asyncio.sleep(0)
        kind, value = self._queue.pop(0)
        if kind == "error":
            raise value
        return value


def test_fresh_hit_skips_fetch() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), clock=clock)

    async def scenario() -> tuple[int | None, int | None]:
        first = await cache.get("model-a")
        second = await cache.get("model-a")
        return first, second

    first, second = asyncio.run(scenario())

    assert first == 128_000
    assert second == 128_000
    assert fetch.calls == 1


def test_expiry_triggers_exactly_one_fetch() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    fetch.queue_result({"model-a": 256_000})
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), ttl_seconds=900.0, clock=clock)

    async def scenario() -> tuple[int | None, int | None]:
        first = await cache.get("model-a")
        clock.advance(901.0)
        second = await cache.get("model-a")
        return first, second

    first, second = asyncio.run(scenario())

    assert first == 128_000
    assert second == 256_000
    assert fetch.calls == 2


def test_stale_value_served_when_refresh_fails() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    fetch.queue_error(FetchError("catalog unavailable"))
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), ttl_seconds=900.0, clock=clock)

    async def scenario() -> tuple[int | None, int | None]:
        first = await cache.get("model-a")
        clock.advance(901.0)
        second = await cache.get("model-a")
        return first, second

    first, second = asyncio.run(scenario())

    assert first == 128_000
    assert second == 128_000
    assert fetch.calls == 2


def test_backoff_window_suppresses_refetch_then_allows_it_after_elapsing() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    fetch.queue_error(FetchError("catalog unavailable"))
    fetch.queue_result({"model-a": 256_000})
    cache = ModelCatalogCache(
        fetch,
        expected_errors=(FetchError,),
        ttl_seconds=900.0,
        failure_backoff_seconds=300.0,
        clock=clock,
    )

    async def scenario() -> tuple[int | None, int | None, int | None, int | None]:
        first = await cache.get("model-a")
        clock.advance(901.0)
        second = await cache.get("model-a")  # refresh fails, enters backoff
        clock.advance(299.0)
        third = await cache.get("model-a")  # still within backoff: no refetch
        clock.advance(2.0)  # 301s since the failure: backoff has elapsed
        fourth = await cache.get("model-a")
        return first, second, third, fourth

    first, second, third, fourth = asyncio.run(scenario())

    assert (first, second, third) == (128_000, 128_000, 128_000)
    assert fourth == 256_000
    assert fetch.calls == 3


def test_cold_cache_failure_returns_none() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_error(FetchError("catalog unavailable"))
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), clock=clock)

    result = asyncio.run(cache.get("model-a"))

    assert result is None
    assert fetch.calls == 1


def test_concurrent_cold_gets_share_one_in_flight_fetch() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), clock=clock)

    async def scenario() -> tuple[int | None, int | None]:
        return await asyncio.gather(cache.get("model-a"), cache.get("model-a"))

    first, second = asyncio.run(scenario())

    assert first == 128_000
    assert second == 128_000
    assert fetch.calls == 1


def test_successful_refresh_replaces_previously_stale_snapshot() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    fetch.queue_error(FetchError("catalog unavailable"))
    fetch.queue_result({"model-a": 256_000, "model-b": 64_000})
    cache = ModelCatalogCache(
        fetch,
        expected_errors=(FetchError,),
        ttl_seconds=900.0,
        failure_backoff_seconds=300.0,
        clock=clock,
    )

    async def scenario() -> tuple[int | None, int | None, int | None]:
        first = await cache.get("model-a")
        clock.advance(901.0)
        stale = await cache.get("model-a")  # refresh fails, serves stale
        clock.advance(301.0)  # past both the ttl and the backoff window
        refreshed = await cache.get("model-a")
        return first, stale, refreshed

    first, stale, refreshed = asyncio.run(scenario())

    assert first == 128_000
    assert stale == 128_000
    assert refreshed == 256_000
    assert fetch.calls == 3


def test_unexpected_exception_propagates_out_of_get() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_error(OtherError("boom"))
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), clock=clock)

    with pytest.raises(OtherError):
        asyncio.run(cache.get("model-a"))


def test_absent_model_returns_none() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), clock=clock)

    result = asyncio.run(cache.get("model-unknown"))

    assert result is None
    assert fetch.calls == 1


def test_cancellation_propagates_even_when_expected_errors_would_catch_it() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_error(asyncio.CancelledError())
    fetch.queue_result({"model-a": 128_000})
    cache = ModelCatalogCache(fetch, expected_errors=(BaseException,), clock=clock)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cache.get("model-a"))

    # Cancellation must not establish refresh backoff: the next lookup
    # fetches immediately instead of being suppressed for the backoff window.
    assert asyncio.run(cache.get("model-a")) == 128_000
    assert fetch.calls == 2


def test_explicit_refresh_replaces_the_same_fresh_snapshot_used_by_get() -> None:
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    fetch.queue_result({"model-a": 256_000, "model-b": 64_000})
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,))

    async def scenario() -> None:
        assert cache.fetched_at is None
        assert await cache.get("model-a") == 128_000
        assert cache.fetched_at is not None
        assert await cache.get_snapshot(force_refresh=True) == {
            "model-a": 256_000, "model-b": 64_000,
        }
        assert await cache.get("model-a") == 256_000
        assert await cache.get("model-b") == 64_000
        assert cache.last_error is None

    asyncio.run(scenario())
    assert fetch.calls == 2


def test_explicit_refresh_bypasses_backoff_and_preserves_same_snapshot_on_error() -> None:
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    fetch.queue_error(FetchError("catalog unavailable"))
    fetch.queue_error(FetchError("catalog still unavailable"))
    fetch.queue_result({"model-a": 256_000})
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,))

    async def scenario() -> None:
        assert await cache.get_snapshot() == {"model-a": 128_000}
        fetched_at = cache.fetched_at
        assert await cache.get_snapshot(force_refresh=True) == {"model-a": 128_000}
        assert str(cache.last_error) == "catalog unavailable"
        assert cache.fetched_at == fetched_at
        assert await cache.get("model-a") == 128_000
        assert fetch.calls == 2
        assert await cache.get_snapshot(force_refresh=True) == {"model-a": 128_000}
        assert str(cache.last_error) == "catalog still unavailable"
        assert cache.fetched_at == fetched_at
        assert await cache.get_snapshot(force_refresh=True) == {"model-a": 256_000}
        assert cache.last_error is None
        assert await cache.get("model-a") == 256_000

    asyncio.run(scenario())
    assert fetch.calls == 4


def test_invalidation_clears_snapshot_and_error_backoff() -> None:
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    fetch.queue_error(FetchError("old identity unavailable"))
    fetch.queue_error(FetchError("new identity unavailable"))
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,))

    async def scenario() -> None:
        assert await cache.get("model-a") == 128_000
        assert await cache.get_snapshot(force_refresh=True) == {"model-a": 128_000}
        cache.invalidate()
        assert cache.fetched_at is None
        assert cache.last_error is None
        assert await cache.get("model-a") is None
        assert str(cache.last_error) == "new identity unavailable"

    asyncio.run(scenario())
    assert fetch.calls == 3


@pytest.mark.parametrize("old_fetch_fails", [False, True])
def test_invalidated_in_flight_fetch_cannot_republish_old_snapshot_or_error(
    old_fetch_fails: bool,
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def fetch() -> dict[str, int]:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
                if old_fetch_fails:
                    raise FetchError("old identity failed")
                return {"model-a": 128_000}
            return {"model-a": 256_000}

        cache = ModelCatalogCache(fetch, expected_errors=(FetchError,))
        old_lookup = asyncio.create_task(cache.get("model-a"))
        await asyncio.wait_for(started.wait(), timeout=1)
        cache.invalidate()
        new_lookup = asyncio.create_task(cache.get("model-a"))
        release.set()
        assert await asyncio.gather(old_lookup, new_lookup) == [256_000, 256_000]
        assert await cache.get("model-a") == 256_000
        assert cache.last_error is None
        assert calls == 2

    asyncio.run(scenario())


def test_get_cached_reads_last_snapshot_without_triggering_refresh() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_result({"model-a": 128_000})
    cache = ModelCatalogCache(fetch, expected_errors=(FetchError,), clock=clock)

    async def scenario() -> None:
        assert cache.get_cached("model-a") is None
        assert await cache.get("model-a") == 128_000
        clock.advance(901)
        assert cache.get_cached("model-a") == 128_000
        cache.invalidate()
        assert cache.get_cached("model-a") is None

    asyncio.run(scenario())
    assert fetch.calls == 1
