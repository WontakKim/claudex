"""Shared HTTP client helpers for provider integrations."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from typing import Any

import anyio
import httpx


class _ShieldedCloseStream(httpx.AsyncByteStream):
    """Shield byte-stream closure while keeping upstream reads cancellable."""

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._stream.__aiter__()

    async def aclose(self) -> None:
        # HTTPX marks its response closed before awaiting the byte-stream close,
        # so an interrupted close cannot be resumed by response.aclose().
        with anyio.CancelScope(shield=True):
            await self._stream.aclose()


def _shield_response_cleanup(response: httpx.Response) -> None:
    """Install cancellation-safe cleanup before reading an owned response."""
    if isinstance(response.stream, _ShieldedCloseStream):
        return
    if not isinstance(response.stream, httpx.AsyncByteStream):
        raise RuntimeError("Expected an asynchronous response stream")
    response.stream = _ShieldedCloseStream(response.stream)


def coerce_context_window(value: Any) -> int | None:
    """Apply the catalog's context-window type policy."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if value > 0 else None
    return None


async def stream_sse_events(
    http_client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    make_error: Callable[[int, str], Exception],
) -> AsyncIterator[dict[str, Any]]:
    """POST a payload and yield SSE data events with JSON object payloads."""
    async with http_client.stream(
        "POST", url, json=payload, headers=headers
    ) as response:
        _shield_response_cleanup(response)
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise make_error(response.status_code, body)

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
                if isinstance(event, dict):
                    yield event
            except json.JSONDecodeError:
                continue


async def stream_with_one_retry(
    get_credentials: Callable[..., Awaitable[Any]],
    stream_once: Callable[[Any], AsyncIterator[dict[str, Any]]],
    *,
    upstream_error: type[Exception],
    should_retry: Callable[[Any, Any], bool],
) -> AsyncIterator[dict[str, Any]]:
    """Stream with at most one force-refreshed credential retry."""
    credentials = await get_credentials()
    try:
        async with aclosing(stream_once(credentials)) as events:
            async for event in events:
                yield event
        return
    except upstream_error as exc:
        if not should_retry(exc, credentials):
            raise

    credentials = await get_credentials(force_refresh=True)
    async with aclosing(stream_once(credentials)) as events:
        async for event in events:
            yield event


async def fetch_models_list(
    http_client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    *,
    label: str,
    make_error: Callable[[int, str], Exception],
    params: dict[str, str] | None = None,
    items_key: str = "data",
    require_object_root: bool = False,
    redact: Callable[[str], str] | None = None,
) -> list[Any]:
    """GET a model catalog and return its validated item list."""
    response = await http_client.get(url, params=params, headers=headers)
    if response.status_code != 200:
        body = redact(response.text) if redact else response.text
        raise make_error(response.status_code, body)
    try:
        parsed = response.json()
    except ValueError as exc:
        # ValueError covers JSONDecodeError, UnicodeDecodeError, and the
        # int-conversion digit limit — every decode failure must surface
        # as a structural catalog failure the model-catalog cache can
        # treat as a failed refresh.
        raise make_error(502, f"{label} models response is not valid JSON") from exc
    if require_object_root and not isinstance(parsed, dict):
        raise make_error(502, f"{label} models response is not a JSON object")
    items = parsed.get(items_key) if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise make_error(502, f"{label} models response has no {items_key} list")
    return items
