"""MCP session to conversation thread bindings."""

import time
from collections.abc import Callable

THREAD_BINDING_TTL_SECONDS = 23 * 60 * 60


class ThreadRegistry:
    """Retain MCP session bindings to ChatGPT conversation threads."""

    def __init__(
        self, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock = clock
        self._bindings: dict[str, tuple[str, float]] = {}

    def bind(self, session_id: str, thread_ref: str) -> None:
        expires_at = self._clock() + THREAD_BINDING_TTL_SECONDS
        self._bindings[session_id] = (thread_ref, expires_at)

    def lookup(self, session_id: str) -> str | None:
        binding = self._bindings.get(session_id)
        if binding is None:
            return None
        thread_ref, expires_at = binding
        if expires_at <= self._clock():
            del self._bindings[session_id]
            return None
        return thread_ref
