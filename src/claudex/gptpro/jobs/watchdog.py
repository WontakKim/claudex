"""Measured execution budgets for background gptpro asks."""

import os
import time
from collections import deque
from collections.abc import Callable
from statistics import quantiles

WATCHDOG_SAMPLE_LIMIT = 64
WATCHDOG_MIN_SAMPLES = 5
WATCHDOG_MARGIN_RATIO = 0.5
WATCHDOG_MIN_BUDGET_SECONDS = 60.0


def _minimum_execution_budget_seconds() -> float:
    raw_value = os.environ.get("GPTPRO_MIN_EXECUTION_BUDGET_SECONDS")
    if raw_value is None:
        return WATCHDOG_MIN_BUDGET_SECONDS
    try:
        configured_value = float(raw_value)
    except ValueError:
        return WATCHDOG_MIN_BUDGET_SECONDS
    if configured_value <= 0:
        return WATCHDOG_MIN_BUDGET_SECONDS
    return configured_value


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
        # Raise GPTPRO_MIN_EXECUTION_BUDGET_SECONDS to lift a measured budget;
        # GPTPRO_OVERALL_TIMEOUT_SECONDS only sets its ceiling.
        minimum_budget_seconds = min(
            self._initial_budget_seconds,
            _minimum_execution_budget_seconds(),
        )
        return min(
            self._initial_budget_seconds,
            max(minimum_budget_seconds, measured_budget_seconds),
        )
