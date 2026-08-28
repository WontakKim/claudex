"""In-memory lifecycle registry for background ChatGPT Pro asks.

Job and thread registries live only for the process lifetime. A daemon restart
loses pending jobs and session thread bindings, while callers retain thread_ref
values to revisit conversations.
"""

from claudex.gptpro.jobs.models import AskJob, AskJobState, TurnFinished
from claudex.gptpro.jobs.service import (
    ACTIVE_JOB_STATES,
    JOB_RETENTION_SECONDS,
    QUEUE_TTL_SECONDS,
    QUESTION_SPILL_THRESHOLD_BYTES,
    SWEEP_INTERVAL_SECONDS,
    AskJobService,
)
from claudex.gptpro.jobs.threads import (
    THREAD_BINDING_TTL_SECONDS,
    ThreadRegistry,
)
from claudex.gptpro.jobs.watchdog import (
    WATCHDOG_MARGIN_RATIO,
    WATCHDOG_MIN_BUDGET_SECONDS,
    WATCHDOG_MIN_SAMPLES,
    WATCHDOG_SAMPLE_LIMIT,
    AnswerWatchdog,
)
