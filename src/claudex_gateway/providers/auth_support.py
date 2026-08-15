"""Shared credential-file and token-refresh support for providers."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

_State = TypeVar("_State")


def load_json_credentials(
    path: Path,
    *,
    error_cls: type[Exception],
    missing_message: str,
    encoding: str | None,
) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding=encoding)
    except FileNotFoundError as exc:
        raise error_cls(missing_message) from exc
    try:
        credentials = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise error_cls(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(credentials, dict):
        raise error_cls(f"{path} has an unexpected format")
    return credentials


def replace_credentials_file(
    path: Path, temp_file: Path, text: str, encoding: str | None
) -> None:
    temp_file.write_text(text, encoding=encoding)
    os.chmod(temp_file, 0o600)
    os.replace(temp_file, path)


async def refresh_when_still_stale(
    lock: asyncio.Lock,
    *,
    force_refresh: bool,
    stale_token: str,
    reload: Callable[[], _State],
    token_of: Callable[[_State], str],
    is_stale: Callable[[_State], bool],
    refresh: Callable[[_State], Awaitable[_State]],
) -> _State:
    """Remember which token looked stale: concurrent 401 retries all force-refresh,
    and only the first one holding the lock should actually rotate — with rotating
    refresh tokens a second POST for the same generation can invalidate the fresh
    credentials. The re-read also picks up a rotation the CLI itself just wrote.
    """
    async with lock:
        state = reload()
        if (force_refresh and token_of(state) == stale_token) or is_stale(state):
            state = await refresh(state)
        return state
