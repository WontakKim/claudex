"""Semantic checks for the ChatGPT composer and send locators."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from claudex.gptpro.selectors import LOCATOR_CHECK_PROBE_JS

LocatorTarget = Literal["composer", "send"]


@dataclass(frozen=True)
class LocatorCheck:
    verdict: Literal["valid", "fault", "wait"]
    matched: int
    eligible: int
    candidates: int
    lang: str
    invalid_selector: bool

    def describe(self) -> str:
        prefix = "invalid selector; " if self.invalid_selector else ""
        return (
            f"{prefix}selector matches={self.matched}, contract matches={self.eligible}, "
            f"page candidates={self.candidates}, lang={self.lang}"
        )


def classify_check(result: object) -> LocatorCheck:
    if not isinstance(result, Mapping):
        return LocatorCheck("wait", 0, 0, 0, "", False)
    matched = result.get("matched", 0)
    eligible = result.get("eligible", 0)
    candidates = result.get("candidates", 0)
    invalid = bool(result.get("invalidSelector", False))
    verdict: Literal["valid", "fault", "wait"]
    if not invalid and matched == 1 and eligible == 1:
        verdict = "valid"
    elif invalid or candidates > 0:
        verdict = "fault"
    else:
        verdict = "wait"
    return LocatorCheck(
        verdict, matched, eligible, candidates, result.get("lang", ""), invalid,
    )


async def check_locator(
    evaluate: Callable[..., Awaitable[Any]],
    target: LocatorTarget,
    selector: str,
    *,
    composer_selector: str,
) -> LocatorCheck:
    result = await evaluate(
        LOCATOR_CHECK_PROBE_JS,
        {"target": target, "selector": selector, "composerSelector": composer_selector},
    )
    return classify_check(result)
