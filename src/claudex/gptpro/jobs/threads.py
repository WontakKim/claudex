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
        now = self._clock()
        self._remove_expired_bindings(now)
        self._bindings[session_id] = (thread_ref, now + THREAD_BINDING_TTL_SECONDS)

    def lookup(self, session_id: str) -> str | None:
        self._remove_expired_bindings(self._clock())
        binding = self._bindings.get(session_id)
        if binding is None:
            return None
        return binding[0]

    def _remove_expired_bindings(self, now: float) -> None:
        expired_session_ids = [
            session_id
            for session_id, (_, expires_at) in self._bindings.items()
            if expires_at <= now
        ]
        for session_id in expired_session_ids:
            del self._bindings[session_id]
