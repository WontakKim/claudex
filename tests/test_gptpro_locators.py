"""Semantic locator contracts exercised against Chromium DOM fixtures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import async_playwright

from claudex.gptpro import locators, selectors


@pytest.mark.parametrize(
    ("result", "verdict"),
    [
        (None, "wait"),
        ({"matched": 1, "eligible": 1, "candidates": 1}, "valid"),
        ({"matched": 0, "eligible": 0, "candidates": 1}, "fault"),
        ({"matched": 0, "eligible": 0, "candidates": 0}, "wait"),
        ({"invalidSelector": True, "candidates": 0}, "fault"),
        ({"matched": 2, "eligible": 2, "candidates": 2}, "fault"),
    ],
)
def test_classify_check(result: object, verdict: str) -> None:
    assert locators.classify_check(result).verdict == verdict


def test_check_description() -> None:
    check = locators.classify_check({
        "invalidSelector": True, "candidates": 1, "lang": "ko-KR",
    })
    assert check.describe() == (
        "invalid selector; selector matches=0, contract matches=0, "
        "page candidates=1, lang=ko-KR"
    )


_COMPOSER = (
    '<div class="ProseMirror" contenteditable="true" role="textbox" '
    'aria-label="{label}">Text</div>'
)
_FORM = '<form data-chatgpt-composer>{composer}{button}</form>'
_EMPTY_FORM = _FORM.format(composer=_COMPOSER.format(label=""), button="")


@pytest.mark.parametrize(
    ("lang", "composer_label", "send_label"),
    [("ko-KR", "ChatGPT에게 물어보세요", "보내기"),
     ("en-US", "Ask ChatGPT", "Send"), ("", "", "")],
)
def test_localized_composer_and_send(
    lang: str, composer_label: str, send_label: str,
) -> None:
    html = _FORM.format(
        composer=_COMPOSER.format(label=composer_label),
        button=f'<button type="submit" aria-label="{send_label}">Send</button>',
    )
    assert _check(html, lang=lang).verdict == "valid"
    check = _check(html, target="send", lang=lang)
    assert check.verdict == "valid"
    assert check.lang == lang


@pytest.mark.parametrize(
    ("html", "target", "selector", "verdict", "candidates", "matched"),
    [
        (_FORM.format(composer=_COMPOSER.format(label=""),
                      button='<button type="button">Voice</button>'),
         "send", None, "wait", 0, 0),
        (_EMPTY_FORM, "composer", ".missing", "fault", 1, 0),
        (_EMPTY_FORM, "composer", "[", "fault", 1, 0),
        *[(f'<{tag}{attributes}>{_EMPTY_FORM}</{tag}>',
           "composer", None, "wait", 0, 1)
          for tag, attributes in [("div", ' role="dialog"'), ("dialog", " open"),
                                  ("nav", ""), ("aside", "")]],
        (f'<div style="display:none">{_EMPTY_FORM}</div>',
         "composer", None, "wait", 0, 1),
        (_EMPTY_FORM * 2, "composer", None, "fault", 2, 2),
        (_EMPTY_FORM + '<form><button type="submit">Other</button></form>',
         "send", 'button[type="submit"]', "wait", 0, 1),
        (_FORM.format(composer=_COMPOSER.format(label="") * 2,
                      button='<button type="submit">Send</button>'),
         "send", None, "wait", 0, 1),
        ('<form><textarea>Text</textarea></form>',
         "composer", "textarea", "valid", 1, 1),
        (_COMPOSER.format(label=""), "composer", ".ProseMirror", "wait", 0, 1),
    ],
)
def test_locator_contract(
    html: str, target: locators.LocatorTarget, selector: str | None,
    verdict: str, candidates: int, matched: int,
) -> None:
    check = _check(html, target=target, selector=selector)
    assert (check.verdict, check.candidates, check.matched) == (
        verdict, candidates, matched,
    )
    assert check.invalid_selector == (selector == "[")


def _check(
    html: str, *, target: locators.LocatorTarget = "composer",
    selector: str | None = None, lang: str = "ko-KR",
) -> locators.LocatorCheck:
    async def run() -> locators.LocatorCheck:
        async with async_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                candidates = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/"
                    "Contents/MacOS/Google Chrome for Testing"
                ))
                if not candidates:
                    pytest.skip("No local Chromium executable available")
                executable = candidates[-1]
            browser = await playwright.chromium.launch(
                executable_path=str(executable), headless=True,
            )
            try:
                page = await browser.new_page()
                await page.set_content(f'<html lang="{lang}"><body>{html}</body></html>')
                seed = (selectors.COMPOSER_SELECTOR if target == "composer"
                        else selectors.SEND_BUTTON_SELECTOR)
                return await locators.check_locator(
                    page.evaluate, target, selector if selector is not None else seed,
                    composer_selector=selectors.COMPOSER_SELECTOR,
                )
            finally:
                await browser.close()
    return asyncio.run(run())


class _Healer:
    def __init__(self, reply: str | Exception = '{"status":"selected","candidate":0}') -> None:
        self.reply = reply
        self.calls = 0

    async def complete(self, system: str, prompt: str, timeout_seconds: float) -> str:
        self.calls += 1
        await asyncio.sleep(0)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _evaluate(*, candidates=True, valid=True):
    async def evaluate(expression, args):
        if expression == selectors.LOCATOR_CANDIDATES_PROBE_JS:
            return {"lang": "ko", "candidates": ([{"index": 0, "selector": ".recovered"}] if candidates else [])}
        return {"matched": int(valid and args["selector"] == ".recovered"),
                "eligible": int(valid and args["selector"] == ".recovered"), "candidates": 1, "lang": "ko"}
    return evaluate


def _rediscover(book, **kwargs):
    return book.rediscover(kwargs.pop("evaluate", _evaluate()), "composer", "ko",
                          failed_selector=selectors.COMPOSER_SELECTOR,
                          composer_selector=selectors.COMPOSER_SELECTOR,
                          timeout_seconds=kwargs.pop("timeout_seconds", 20), **kwargs)


def test_book_persistence_breaker_and_success(tmp_path: Path) -> None:
    now = [100.0]
    path = tmp_path / "locators.json"
    book = locators.LocatorBook(path, clock=lambda: now[0])
    assert book.selector_for("ko", "composer") == selectors.COMPOSER_SELECTOR
    book.record_failure("ko", "composer", "absent")
    assert book.refusal("ko", "composer")
    now[0] += 59
    assert book.refusal("ko", "composer")
    now[0] += 1
    assert book.refusal("ko", "composer") is None
    book.record_failure("ko", "composer", "absent")
    now[0] += 60
    book.record_failure("ko", "composer", "absent")
    assert book.refusal("ko", "composer")
    now[0] += 3599
    assert book.refusal("ko", "composer")
    now[0] += 1
    assert book.refusal("ko", "composer") is None
    book.record_failure("ko", "composer", "half-open failed")
    assert book.refusal("ko", "composer")
    book.record_success("ko", "composer", ".recovered")
    restored = locators.LocatorBook(path, clock=lambda: now[0])
    assert restored.refusal("ko", "composer") is None
    assert restored.selector_for("ko", "composer") == ".recovered"
    record = json.loads(path.read_text())["environments"]["ko"]["composer"]
    assert record["revision"] == 1
    assert record["previous_selector"] == selectors.COMPOSER_SELECTOR
    assert record["consecutive_failures"] == 0
    assert record["open_until"] is None
    assert record["next_attempt_at"] is None
    assert record["last_failure"] is None
    assert record["recorded_at"] == now[0]
    restored.record_success("ko", "composer", ".next")
    record = json.loads(path.read_text())["environments"]["ko"]["composer"]
    assert record["revision"] == 2 and record["previous_selector"] == ".recovered"


def test_corrupt_book_is_optional(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "locators.json"
    path.write_text("invalid JSON")
    assert locators.LocatorBook(path).selector_for("ko", "composer") == selectors.COMPOSER_SELECTOR
    assert caplog.records


def test_rediscovery_pending_and_concurrency(tmp_path: Path) -> None:
    healer = _Healer()
    book = locators.LocatorBook(tmp_path / "locators.json", healer=healer)
    async def run():
        return await asyncio.gather(_rediscover(book), _rediscover(book))
    assert asyncio.run(run()) == [".recovered", ".recovered"]
    assert healer.calls == 1
    assert book.pending[("ko", "composer")] == ".recovered"
    assert not book.path.exists()


@pytest.mark.parametrize("reply", ['{"status":"absent"}', '{"status":"uncertain"}', 'bad',
                                   '{"status":"selected","candidate":9}',
                                   '{"status":"selected","candidate":true}'])
def test_failed_selection_counts(tmp_path: Path, reply: str) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer(reply))
    with pytest.raises(locators.HealFailed):
        asyncio.run(_rediscover(book))
    assert book.refusal("ko", "composer")


@pytest.mark.parametrize("candidates,valid", [(False, True), (True, False)])
def test_candidates_and_fresh_check_failures_count(
    tmp_path: Path, candidates: bool, valid: bool,
) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer())
    with pytest.raises(locators.HealFailed):
        asyncio.run(_rediscover(book, evaluate=_evaluate(candidates=candidates, valid=valid)))
    assert book.refusal("ko", "composer")


def test_unavailable_and_short_budget_do_not_count(tmp_path: Path) -> None:
    healer = _Healer()
    book = locators.LocatorBook(tmp_path / "locators.json", healer=healer)
    with pytest.raises(locators.HealingRefused):
        asyncio.run(_rediscover(book, timeout_seconds=9))
    assert healer.calls == 0 and not book.path.exists()
    healer.reply = locators.HealerUnavailable("offline")
    with pytest.raises(locators.HealerUnavailable):
        asyncio.run(_rediscover(book))
    assert not book.path.exists()
    book.record_failure("ko", "composer", "bad")
    with pytest.raises(locators.HealingRefused):
        asyncio.run(_rediscover(book))
    assert healer.calls == 1


def test_seed_reuse(tmp_path: Path) -> None:
    healer = _Healer()
    book = locators.LocatorBook(tmp_path / "locators.json", healer=healer)
    book.record_success("ko", "composer", ".broken")
    async def evaluate(expression, args):
        assert expression == selectors.LOCATOR_CHECK_PROBE_JS
        return {"matched": 1, "eligible": 1}
    result = asyncio.run(book.rediscover(evaluate, "composer", "ko", failed_selector=".broken",
                                        composer_selector=".broken", timeout_seconds=20))
    assert result == selectors.COMPOSER_SELECTOR and healer.calls == 0


@pytest.mark.parametrize("result", [{"matched": "1", "eligible": None, "candidates": -1, "lang": 5},
                                     {"matched": True, "eligible": 1.5, "candidates": []}])
def test_malformed_counts(result: dict[str, Any]) -> None:
    check = locators.classify_check(result)
    assert (check.matched, check.eligible, check.candidates, check.lang) == (0, 0, 0, "")


def test_candidates_privacy_and_form_ownership() -> None:
    async def run():
        async with async_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                options = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"))
                if not options:
                    pytest.skip("No local Chromium executable available")
                executable = options[-1]
            browser = await playwright.chromium.launch(executable_path=str(executable), headless=True)
            try:
                page = await browser.new_page()
                await page.set_content('<html lang="ko"><form data-compose><div id="secret-id" contenteditable="true" role="textbox" aria-label="입력">SECRET CONTENT</div><button type="submit">SECRET BUTTON</button><button type="button">Voice</button></form><nav><form><textarea>SECRET NAV</textarea></form></nav><dialog open><form><textarea>SECRET DIALOG</textarea></form></dialog><form><button type="submit">Other</button></form></html>')
                result = await page.evaluate(selectors.LOCATOR_CANDIDATES_PROBE_JS, {"target": "composer", "composerSelector": "missing"})
                assert "SECRET" not in json.dumps(result)
                assert len(result["candidates"]) == 1
                compiled = result["candidates"][0]["selector"]
                assert "aria-label" not in compiled and "id" not in compiled
                assert await page.locator(compiled).count() == 1
                result = await page.evaluate(selectors.LOCATOR_CANDIDATES_PROBE_JS, {"target": "send", "composerSelector": compiled})
                assert len(result["candidates"]) == 1
                assert await page.locator(result["candidates"][0]["selector"]).count() == 1
            finally:
                await browser.close()
    asyncio.run(run())


@pytest.mark.parametrize("host,expected", [("0.0.0.0", "127.0.0.1"), ("::", "127.0.0.1"), ("::1", "[::1]")])
def test_gateway_healer_request(
    monkeypatch: pytest.MonkeyPatch, host: str, expected: str,
) -> None:
    import httpx
    from claudex.config import GatewayConfig
    config = GatewayConfig(host=host, port=1234, local_token="token")
    monkeypatch.setattr(GatewayConfig, "load", lambda: config)
    def respond(request):
        assert str(request.url) == f"http://{expected}:1234/v1/messages"
        assert request.headers["authorization"] == "Bearer token"
        assert request.headers["anthropic-version"] == "2023-06-01"
        payload = json.loads(request.content)
        assert payload["model"] == "claude-sonnet-5-5"
        assert payload["system"] == "system" and payload["max_tokens"] == 200
        return httpx.Response(200, json={"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    assert asyncio.run(locators.GatewayMessagesHealer().complete("system", "prompt", 20)) == "ab"


@pytest.mark.parametrize("kind", ["status", "transport", "config", "json", "content"])
def test_gateway_unavailable(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    import httpx
    from claudex.config import GatewayConfig, ConfigError
    def load():
        if kind == "config":
            raise ConfigError("invalid")
        return GatewayConfig()
    monkeypatch.setattr(GatewayConfig, "load", load)
    def respond(request):
        if kind == "transport":
            raise httpx.ConnectError("offline")
        if kind == "json":
            return httpx.Response(200, text="bad")
        return httpx.Response(403 if kind == "status" else 200, json={})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(locators.HealerUnavailable):
        asyncio.run(locators.GatewayMessagesHealer().complete("s", "p", 10))


def test_null_compiled_selector_counts(tmp_path: Path) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer())
    original = _evaluate()
    async def evaluate(expression, args):
        if expression == selectors.LOCATOR_CANDIDATES_PROBE_JS:
            return {"candidates": [{"index": 0, "selector": None}]}
        return await original(expression, args)
    with pytest.raises(locators.HealFailed):
        asyncio.run(_rediscover(book, evaluate=evaluate))
    assert book.refusal("ko", "composer")


def test_success_same_selector_is_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json")
    book.record_success("ko", "composer", ".saved")
    def unexpected_write(*args):
        pytest.fail("identical proven selectors must not be written again")
    monkeypatch.setattr(locators, "write_private_json_atomic", unexpected_write)
    book.pending[("ko", "composer")] = ".saved"
    book.record_success("ko", "composer", ".saved")
    assert not book.pending


def test_no_healer_is_optional(tmp_path: Path) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json")
    with pytest.raises(locators.HealerUnavailable):
        asyncio.run(_rediscover(book))
    assert not book.path.exists()


@pytest.mark.parametrize("content", ['[]', '{"version":2,"environments":{}}',
                                     '{"version":1,"environments":{"ko":{"composer":{"revision":"bad"}}}}'])
def test_invalid_book_is_optional(tmp_path: Path, content: str) -> None:
    path = tmp_path / "locators.json"
    path.write_text(content)
    book = locators.LocatorBook(path)
    assert book.selector_for("ko", "composer") == selectors.COMPOSER_SELECTOR
    assert book.refusal("ko", "composer") is None


def test_environment_key() -> None:
    assert locators.environment_key("") == "unknown"
    assert locators.environment_key("ko-KR") == "ko-KR"


def test_deleting_book_clears_breaker_and_selector_without_restart(tmp_path: Path) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json", clock=lambda: 100.0)
    book.record_success("ko", "composer", ".saved")
    for _ in range(locators.FAILURE_THRESHOLD):
        book.record_failure("ko", "composer", "not found")
    assert book.refusal("ko", "composer")
    book.path.unlink()
    assert book.refusal("ko", "composer") is None
    assert book.selector_for("ko", "composer") == selectors.COMPOSER_SELECTOR
    book.record_success("en", "send", ".send")
    assert "ko" not in json.loads(book.path.read_text())["environments"]


def test_store_rereads_before_merging(tmp_path: Path) -> None:
    path = tmp_path / "locators.json"
    first = locators.LocatorBook(path)
    second = locators.LocatorBook(path)
    first.record_success("ko", "composer", ".saved")
    second.record_success("en", "send", ".send")
    assert first.selector_for("en", "send") == ".send"
    assert second.selector_for("ko", "composer") == ".saved"
    assert set(json.loads(path.read_text())["environments"]) == {"ko", "en"}


def test_corrupt_file_replaces_prior_state_on_access(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json")
    book.record_success("ko", "composer", ".saved")
    book.record_failure("ko", "composer", "not found")
    book.path.write_text("invalid JSON")
    assert book.refusal("ko", "composer") is None
    assert book.selector_for("ko", "composer") == selectors.COMPOSER_SELECTOR
    assert "Could not read optional locator book" in caplog.text
