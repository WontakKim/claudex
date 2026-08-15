"""Tests for the TTL model-catalog cache with stale-on-error backoff."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from claudex_gateway.providers.model_catalog_cache import ModelCatalogCache


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
