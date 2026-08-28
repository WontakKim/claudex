"""Job and turn data models for background gptpro asks."""

from dataclasses import dataclass
from typing import Literal

AskJobState = Literal[
    "queued", "running", "detached", "succeeded", "failed"
]


@dataclass(frozen=True)
class AskJob:
    ask_id: str
    state: AskJobState
    answer: str | None
    failure: str | None
    error_message: str | None
    status_message: str | None
    nonce_marker: str | None
    thread_ref: str | None
    created_at: float
    finished_at: float | None


@dataclass(frozen=True)
class TurnFinished:
    ask_id: str
    thread_ref: str | None
    answer: str
