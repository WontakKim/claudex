"""Structure-only locator recovery with first-use proof and per-language breakers."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

from claudex import paths
from claudex.config import ConfigError, GatewayConfig
from claudex.gptpro.selectors import (
    COMPOSER_SELECTOR,
    LOCATOR_CANDIDATES_PROBE_JS,
    LOCATOR_CHECK_PROBE_JS,
    SEND_BUTTON_SELECTOR,
)
from claudex.providers.auth_support import write_private_json_atomic

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

    def count(name: str) -> int:
        value = result.get(name, 0)
        return value if type(value) is int and value >= 0 else 0

    matched = count("matched")
    eligible = count("eligible")
    candidates = count("candidates")
    invalid = bool(result.get("invalidSelector", False))
    verdict: Literal["valid", "fault", "wait"]
    if not invalid and matched == 1 and eligible == 1:
        verdict = "valid"
    elif invalid or candidates > 0:
        verdict = "fault"
    else:
        verdict = "wait"
    return LocatorCheck(
        verdict, matched, eligible, candidates,
        result.get("lang", "") if isinstance(result.get("lang", ""), str) else "", invalid,
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


logger = logging.getLogger(__name__)
SEED_SELECTORS = {"composer": COMPOSER_SELECTOR, "send": SEND_BUTTON_SELECTOR}
FAILURE_THRESHOLD = 3
BREAKER_COOLDOWN_SECONDS = 3600
RETRY_BACKOFF_SECONDS = 60
MINIMUM_REDISCOVERY_SECONDS = 10
HEALER_MODEL = "claude-sonnet-5-5"
_SYSTEM_PROMPT = (
    'Select a control from structure-only candidates. Reply with exactly one JSON object: '
    '{"status":"selected","candidate":<index>}, {"status":"absent"}, or '
    '{"status":"uncertain"}. Use uncertain when two or more candidates are equally plausible.'
)
_TARGET_DESCRIPTIONS = {
    "composer": (
        "the main message input where the user types a new chat message; "
        "not a search box and not an inline editor for an earlier message"
    ),
    "send": (
        "the button that submits the message typed in the main message input; "
        "not a voice, attachment, model-picker, or stop button"
    ),
}


class LocatorHealingError(Exception):
    """A locator could not be recovered before submission."""


class HealingRefused(LocatorHealingError):
    """Recovery is suspended or the remaining budget is insufficient."""


class HealerUnavailable(LocatorHealingError):
    """The optional gateway Messages healer is unavailable."""


class HealFailed(LocatorHealingError):
    """Recovery did not identify a unique contract-satisfying control."""


def environment_key(lang: str) -> str:
    return lang or "unknown"


@dataclass(frozen=True)
class RediscoveredLocator:
    selector: str
    attempt_id: str | None = None
    environment: str = "unknown"


@dataclass
class LocatorBook:
    path: Path
    healer: Any = None
    clock: Callable[[], float] = time.time
    monotonic: Callable[[], float] = time.monotonic
    pending: dict[tuple[str, str], RediscoveredLocator] = field(default_factory=dict, init=False)
    _locks: dict[tuple[str, str], asyncio.Lock] = field(default_factory=dict, init=False)

    def _load(self) -> dict[str, dict[str, dict[str, Any]]]:
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported locator book")
            environments = data.get("environments")
            if not isinstance(environments, dict):
                raise ValueError("invalid locator environments")
            for env, records in environments.items():
                if not isinstance(env, str) or not isinstance(records, dict):
                    raise ValueError("invalid locator environment")
                for target, record in records.items():
                    if target not in SEED_SELECTORS or not isinstance(record, dict):
                        raise ValueError("invalid locator record")
                    if record.get("selector") is not None and not isinstance(
                        record["selector"], str
                    ):
                        raise ValueError("invalid recorded selector")
                    for name in ("revision", "consecutive_failures"):
                        if type(record.get(name)) is not int or record[name] < 0:
                            raise ValueError("invalid locator counter")
                    for name in ("recorded_at", "open_until", "next_attempt_at"):
                        value = record.get(name)
                        if value is not None and (
                            type(value) not in (int, float) or not math.isfinite(value)
                        ):
                            raise ValueError("invalid locator timestamp")
            return environments
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not read optional locator book %s: %s", self.path, exc)
            return {}

    def _record(self, env: str, target: str) -> dict[str, Any]:
        return self._load().get(environment_key(env), {}).get(target, {})

    def _store(self, env: str, target: str, record: dict[str, Any]) -> None:
        environments = self._load()
        environments.setdefault(environment_key(env), {})[target] = record
        write_private_json_atomic(self.path, {"version": 1, "environments": environments})

    def selector_for(self, env: str, target: str) -> str:
        return self._record(env, target).get("selector") or SEED_SELECTORS[target]

    def refusal(self, env: str, target: str) -> str | None:
        record = self._record(env, target)
        now = self.clock()
        if now < (record.get("open_until") or 0):
            return "locator rediscovery is suspended for one hour after repeated failures"
        if now < (record.get("next_attempt_at") or 0):
            return "locator rediscovery is waiting for its 60-second retry backoff"
        return None

    def abandon(self, env: str, target: str, attempt_id: str | None) -> bool:
        key = (environment_key(env), target)
        pending = self.pending.get(key)
        if attempt_id is None or pending is None or pending.attempt_id != attempt_id:
            return False
        del self.pending[key]
        return True

    def record_success(
        self, env: str, target: str, selector: str, attempt_id: str | None = None,
        *, attempt_env: str | None = None,
    ) -> None:
        if attempt_id is not None and not self.abandon(
            env if attempt_env is None else attempt_env, target, attempt_id,
        ):
            return
        prior = self._record(env, target)
        if prior.get("selector") == selector and prior.get("consecutive_failures", 0) == 0:
            return
        self._store(env, target, {
            "selector": selector, "previous_selector": self.selector_for(env, target),
            "revision": prior.get("revision", 0) + 1, "recorded_at": self.clock(),
            "consecutive_failures": 0, "open_until": None, "next_attempt_at": None,
            "last_failure": None,
        })

    def record_failure(
        self, env: str, target: str, reason: str, attempt_id: str | None,
    ) -> None:
        if self.abandon(env, target, attempt_id):
            self._record_failure(env, target, reason)

    def _record_failure(self, env: str, target: str, reason: str) -> None:
        prior = self._record(env, target)
        failures = prior.get("consecutive_failures", 0) + 1
        now = self.clock()
        half_open = prior.get("open_until") is not None and now >= prior["open_until"]
        is_open = half_open or failures >= FAILURE_THRESHOLD
        self._store(env, target, {
            "selector": prior.get("selector"), "previous_selector": prior.get("previous_selector"),
            "revision": prior.get("revision", 0), "recorded_at": prior.get("recorded_at"),
            "consecutive_failures": failures,
            "open_until": now + BREAKER_COOLDOWN_SECONDS if is_open else None,
            "next_attempt_at": None if is_open else now + RETRY_BACKOFF_SECONDS,
            "last_failure": reason,
        })

    async def rediscover(
        self, evaluate: Callable[..., Awaitable[Any]], target: LocatorTarget, env: str,
        *, failed_selector: str, composer_selector: str, deadline: float,
    ) -> RediscoveredLocator:
        def ensure_budget() -> float:
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise TimeoutError("locator rediscovery deadline expired")
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError
            return remaining

        async with asyncio.timeout(ensure_budget()):
            return await self._rediscover(
                evaluate, target, env, failed_selector=failed_selector,
                composer_selector=composer_selector, remaining=ensure_budget,
            )

    async def _rediscover(
        self, evaluate: Callable[..., Awaitable[Any]], target: LocatorTarget, env: str,
        *, failed_selector: str, composer_selector: str, remaining: Callable[[], float],
    ) -> RediscoveredLocator:
        key = (environment_key(env), target)
        async with self._locks.setdefault(key, asyncio.Lock()):
            remaining()
            pending = self.pending.get(key)
            if pending is not None:
                check = await check_locator(
                    evaluate, target, pending.selector, composer_selector=composer_selector,
                )
                remaining()
                if check.verdict == "valid":
                    return pending
                raise HealingRefused("a rediscovered locator is awaiting first-use proof")
            alternatives = [self.selector_for(env, target), SEED_SELECTORS[target]]
            seen = {failed_selector, None}
            for selector in alternatives:
                if selector in seen:
                    continue
                seen.add(selector)
                check = await check_locator(
                    evaluate, target, selector, composer_selector=composer_selector,
                )
                remaining()
                if check.verdict == "valid":
                    return RediscoveredLocator(selector, environment=key[0])
            refusal = self.refusal(env, target)
            if refusal:
                raise HealingRefused(refusal)
            if remaining() < MINIMUM_REDISCOVERY_SECONDS:
                raise HealingRefused("less than 10 seconds remain for locator rediscovery")
            if self.healer is None:
                raise HealerUnavailable("the locator healer is not configured")
            try:
                probe = await evaluate(LOCATOR_CANDIDATES_PROBE_JS, {
                    "target": target, "composerSelector": composer_selector,
                })
                remaining()
                candidates = probe.get("candidates") if isinstance(probe, Mapping) else None
                if not isinstance(candidates, list) or not candidates:
                    raise HealFailed("no contract-satisfying candidates were found")
                reply = await self.healer.complete(
                    _SYSTEM_PROMPT, _TARGET_DESCRIPTIONS[target] + "\n" + json.dumps(candidates),
                    timeout_seconds=min(remaining(), 30),
                )
                remaining()
                try:
                    text = reply.strip()
                    lines = text.splitlines()
                    if len(lines) >= 3 and lines[0] in ("```", "```json") and lines[-1] == "```":
                        text = "\n".join(lines[1:-1]).strip()
                    selection = json.loads(text)
                except (ValueError, TypeError, AttributeError) as exc:
                    raise HealFailed("the healer reply was not a single JSON object") from exc
                if not isinstance(selection, dict):
                    raise HealFailed("the healer reply was not a single JSON object")
                if selection.get("status") != "selected":
                    raise HealFailed("the healer found the control absent or uncertain")
                index = selection.get("candidate")
                if type(index) is not int or not 0 <= index < len(candidates):
                    raise HealFailed("the healer returned an invalid candidate index")
                selector = candidates[index].get("selector")
                if not isinstance(selector, str) or not selector:
                    raise HealFailed("the selected candidate has no unique structural selector")
                check = await check_locator(
                    evaluate, target, selector, composer_selector=composer_selector,
                )
                remaining()
                if check.verdict == "wait":
                    raise HealingRefused("the selected candidate could not be verified")
                if check.verdict != "valid":
                    raise HealFailed("the selected candidate failed the fresh locator contract check")
            except HealFailed as exc:
                remaining()
                try:
                    self._record_failure(env, target, str(exc))
                except OSError as recording_error:
                    logger.warning("Could not record optional locator failure: %s", recording_error)
                raise
            result = RediscoveredLocator(selector, uuid4().hex, key[0])
            self.pending[key] = result
            return result


class GatewayMessagesHealer:
    async def complete(self, system: str, prompt: str, timeout_seconds: float) -> str:
        try:
            config = GatewayConfig.load()
        except ConfigError as exc:
            raise HealerUnavailable(f"gateway configuration is unavailable: {exc}") from exc
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(config.host, config.host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        headers = {"anthropic-version": "2023-06-01"}
        if config.local_token:
            headers["authorization"] = f"Bearer {config.local_token}"
        # The gateway's own /v1/messages uses model_map: a sonnet mapping applies;
        # otherwise this model passes through to Anthropic.
        payload = {"model": HEALER_MODEL, "max_tokens": 200, "system": system,
                   "messages": [{"role": "user", "content": prompt}]}
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=False) as client:
                response = await client.post(
                    f"http://{host}:{config.port}/v1/messages", headers=headers, json=payload,
                )
        except httpx.TransportError as exc:
            raise HealerUnavailable(f"gateway Messages request failed: {exc}") from exc
        if response.status_code != 200:
            raise HealerUnavailable(f"gateway Messages request returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise HealerUnavailable("gateway Messages response is not JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            raise HealerUnavailable("gateway Messages response has no content list")
        return "".join(
            block["text"] for block in data["content"]
            if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )


_default_book: LocatorBook | None = None


def default_book() -> LocatorBook:
    global _default_book
    if _default_book is None:
        _default_book = LocatorBook(paths.gptpro_locators_file(), healer=GatewayMessagesHealer())
    return _default_book
