"""Tests for the per-account Claude usage cache with Retry-After cooldown."""

from __future__ import annotations

import asyncio
from typing import Any

from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache


class FakeClock:
    """A controllable stand-in for ``time.monotonic`` — no real sleeping."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _ok(account_id: str) -> dict[str, Any]:
    return {"provider": "claude", "status": "ok", "error": None, "session": {"id": account_id}}


def _ok_with_windows(
    *, session_resets_at: float | None = None, weekly_resets_at: float | None = None
) -> dict[str, Any]:
    return {
        "provider": "claude",
        "status": "ok",
        "error": None,
        "session": {"used_percent": 10.0, "resets_at": session_resets_at},
        "weekly": {"used_percent": 20.0, "resets_at": weekly_resets_at},
        "fable_weekly": None,
    }


def _err(message: str) -> dict[str, Any]:
    return {"provider": "claude", "status": "error", "error": message, "session": None}


class FakeFetch:
    """Records fetch order and replays per-account queued responses."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.responses: dict[str, list[tuple[Any, float | None] | BaseException]] = {}

    def queue(self, account_id: str, result: Any, retry_after: float | None = None) -> None:
        self.responses.setdefault(account_id, []).append((result, retry_after))

    def queue_exception(self, account_id: str, error: BaseException) -> None:
        self.responses.setdefault(account_id, []).append(error)

    async def __call__(self, account_id: str) -> tuple[dict[str, Any], float | None]:
        self.calls.append(account_id)
        await asyncio.sleep(0)
        queued = self.responses[account_id].pop(0)
        if isinstance(queued, BaseException):
            raise queued
        return queued


def _cache(fetch: FakeFetch, clock: FakeClock, **kwargs: Any) -> ClaudeAccountUsageCache:
    return ClaudeAccountUsageCache(fetch, clock=clock, **kwargs)


def test_fresh_entries_are_served_without_fetching() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _ok("a"))
    cache = _cache(fetch, clock)

    first = asyncio.run(cache.get(["a"]))
    clock.advance(60.0)
    second = asyncio.run(cache.get(["a"]))

    assert fetch.calls == ["a"]
    assert first["a"] == second["a"] == _ok("a")


def test_expired_entries_are_refetched_after_the_ttl() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _ok("a"))
    fetch.queue("a", _ok("a"))
    cache = _cache(fetch, clock)

    asyncio.run(cache.get(["a"]))
    clock.advance(121.0)
    asyncio.run(cache.get(["a"]))

    assert fetch.calls == ["a", "a"]


def test_upstream_fetches_are_serialized_in_request_order() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    for account_id in ("a", "b", "c"):
        fetch.queue(account_id, _ok(account_id))
    cache = _cache(fetch, clock)

    results = asyncio.run(cache.get(["a", "b", "c"]))

    assert fetch.calls == ["a", "b", "c"]
    assert set(results) == {"a", "b", "c"}


def test_retry_after_cooldown_blocks_other_accounts_and_serves_stale() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _ok("a"))
    fetch.queue("b", _err("usage API rate-limited (429); try again shortly"), 30.0)
    cache = _cache(fetch, clock)

    asyncio.run(cache.get(["a"]))
    clock.advance(121.0)  # expire a's entry
    fetch.queue("a", _ok("a"))  # would be consumed only after the cooldown

    results = asyncio.run(cache.get(["b", "a"]))

    # b's 429 opened the cooldown; a's refetch was suppressed, serving stale.
    assert fetch.calls == ["a", "b"]
    assert results["a"] == _ok("a")

    # An account with no cached entry gets a synthesized error under cooldown.
    cooldown_results = asyncio.run(cache.get(["c"]))
    assert cooldown_results["c"]["status"] == "error"
    assert "rate-limited" in cooldown_results["c"]["error"]
    assert fetch.calls == ["a", "b"]

    # After the Retry-After window the fetches resume.
    clock.advance(31.0)
    fetch.queue("c", _ok("c"))
    resumed = asyncio.run(cache.get(["c"]))
    assert resumed["c"] == _ok("c")
    assert fetch.calls == ["a", "b", "c"]


def test_headerless_429_uses_the_failure_backoff_as_cooldown() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _err("usage API rate-limited (429); try again shortly"), None)
    cache = _cache(fetch, clock)

    asyncio.run(cache.get(["a"]))
    clock.advance(59.0)
    assert asyncio.run(cache.get(["b"]))["b"]["status"] == "error"
    assert fetch.calls == ["a"]

    clock.advance(2.0)
    fetch.queue("b", _ok("b"))
    assert asyncio.run(cache.get(["b"]))["b"] == _ok("b")


def test_retry_after_is_clamped_to_the_configured_bounds() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _err("usage API rate-limited (429); try again shortly"), 999_999.0)
    cache = _cache(fetch, clock)

    asyncio.run(cache.get(["a"]))
    clock.advance(3599.0)
    assert asyncio.run(cache.get(["b"]))["b"]["status"] == "error"
    assert fetch.calls == ["a"]

    clock.advance(2.0)
    fetch.queue("b", _ok("b"))
    assert asyncio.run(cache.get(["b"]))["b"] == _ok("b")


def test_non_429_failure_backs_off_that_account_only() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _err("usage API returned 500: boom"))
    fetch.queue("b", _ok("b"))
    cache = _cache(fetch, clock)

    results = asyncio.run(cache.get(["a", "b"]))
    assert results["a"]["status"] == "error"
    assert results["b"] == _ok("b")

    # Within the failure backoff the error result is served without refetching.
    clock.advance(30.0)
    again = asyncio.run(cache.get(["a"]))
    assert again["a"]["status"] == "error"
    assert fetch.calls == ["a", "b"]

    # After the backoff the account retries.
    clock.advance(31.0)
    fetch.queue("a", _ok("a"))
    assert asyncio.run(cache.get(["a"]))["a"] == _ok("a")


def test_unexpected_fetch_exception_is_contained() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue_exception("a", RuntimeError("boom"))
    cache = _cache(fetch, clock)

    results = asyncio.run(cache.get(["a"]))

    assert results["a"]["status"] == "error"
    assert "unexpectedly" in results["a"]["error"]

    # The synthesized failure honors the backoff like any other failure.
    clock.advance(30.0)
    assert asyncio.run(cache.get(["a"]))["a"]["status"] == "error"
    assert fetch.calls == ["a"]


def test_peek_returns_cached_entry_without_fetching_and_none_when_absent() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _ok("a"))
    cache = _cache(fetch, clock)

    assert cache.peek("a") is None
    asyncio.run(cache.get(["a"]))

    # Age is irrelevant: a long-expired entry is still peekable, fetch-free.
    clock.advance(10_000.0)
    assert cache.peek("a") == _ok("a")
    assert cache.peek("never-fetched") is None
    assert fetch.calls == ["a"]


def test_peek_with_metadata_reports_age_source_and_reset_and_none_when_absent() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    envelope = _ok_with_windows(session_resets_at=1_500.0, weekly_resets_at=2_500.0)
    fetch.queue("a", envelope)
    cache = _cache(fetch, clock)

    assert cache.peek_with_metadata("a") is None
    asyncio.run(cache.get(["a"]))

    result, metadata = cache.peek_with_metadata("a")
    assert result == envelope
    assert metadata["session"] == {"age_seconds": 0.0, "source": "usage_api", "reset_at": 1_500.0}
    assert metadata["weekly"] == {"age_seconds": 0.0, "source": "usage_api", "reset_at": 2_500.0}

    # Age advances with the clock, independently of peek()'s age-ignoring behavior.
    clock.advance(45.0)
    _, aged_metadata = cache.peek_with_metadata("a")
    assert aged_metadata["session"]["age_seconds"] == 45.0
    assert aged_metadata["weekly"]["age_seconds"] == 45.0
    assert cache.peek("a") == envelope

    assert cache.peek_with_metadata("never-fetched") is None
    assert fetch.calls == ["a"]


def test_peek_with_metadata_omits_windows_absent_from_the_envelope() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    envelope = {
        "provider": "claude",
        "status": "ok",
        "error": None,
        "session": {"used_percent": 10.0, "resets_at": 1_500.0},
        "weekly": None,
        "fable_weekly": None,
    }
    fetch.queue("a", envelope)
    cache = _cache(fetch, clock)

    asyncio.run(cache.get(["a"]))

    _, metadata = cache.peek_with_metadata("a")
    assert set(metadata) == {"session"}
    assert "weekly" not in metadata
    assert "fable_weekly" not in metadata


def test_peek_with_metadata_has_no_windows_on_failed_fetch() -> None:
    clock = FakeClock()
    fetch = FakeFetch()
    fetch.queue("a", _err("usage API returned 500: boom"))
    cache = _cache(fetch, clock)

    results = asyncio.run(cache.get(["a"]))
    assert results["a"]["status"] == "error"

    result, metadata = cache.peek_with_metadata("a")
    assert result["status"] == "error"
    assert metadata == {}
