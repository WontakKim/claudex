"""Measured execution budgets for background gptpro asks."""

import time
from collections import deque
from collections.abc import Callable
from statistics import quantiles

WATCHDOG_SAMPLE_LIMIT = 64
WATCHDOG_MIN_SAMPLES = 5
WATCHDOG_MARGIN_RATIO = 0.5
WATCHDOG_MIN_BUDGET_SECONDS = 60.0


class AnswerWatchdog:
    """Derive execution budgets from recent successful answer durations."""

    def __init__(
        self,
        initial_budget_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._initial_budget_seconds = initial_budget_seconds
        self._clock = clock
        self._samples: deque[float] = deque(maxlen=WATCHDOG_SAMPLE_LIMIT)

    def record(self, duration_seconds: float) -> None:
        if duration_seconds < 0:
            return
        self._samples.append(duration_seconds)

    def execution_budget_seconds(self) -> float:
        if len(self._samples) < WATCHDOG_MIN_SAMPLES:
            return self._initial_budget_seconds

        p95_seconds = quantiles(
            self._samples, n=100, method="inclusive"
        )[94]
        measured_budget_seconds = p95_seconds * (1 + WATCHDOG_MARGIN_RATIO)
        # Durations above the operational default indicate a slower system.
        # Operators must raise that ceiling explicitly through the env override.
        return min(
            self._initial_budget_seconds,
            max(WATCHDOG_MIN_BUDGET_SECONDS, measured_budget_seconds),
        )
