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
    composer_selector: str = selectors.COMPOSER_SELECTOR,
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
                    composer_selector=composer_selector,
                )
            finally:
                await browser.close()
    return asyncio.run(run())


@pytest.mark.parametrize(
    "html",
    [
        '<div data-chatgpt-search-unit-key="t1:user"><form data-edit-message>'
        '<textarea>old</textarea><button type="submit">Save edit</button>'
        '</form></div>',
        '<form role="search"><textarea></textarea></form>',
        '<div role="search"><form><div contenteditable="true">Search</div>'
        '</form></div>',
        '<search><form><textarea></textarea></form></search>',
        '<form><textarea type="search"></textarea></form>',
    ],
    ids=["inline-editor", "search-form", "search-area", "search-element", "search-type"],
)
def test_composer_contract_excludes_message_editors_and_search(html: str) -> None:
    check = _check(html, selector='textarea, [contenteditable="true"]')
    assert (check.verdict, check.eligible, check.candidates) == ("wait", 0, 0)
    assert _candidates(html) == []
    check = _check(html, target="send", selector='button[type="submit"]')
    assert (check.verdict, check.eligible, check.candidates) == ("wait", 0, 0)
    assert _candidates(html, target="send") == []


@pytest.mark.parametrize(
    "container_attributes",
    ['data-chatgpt-search-unit-key="t1:user"', 'role="search"', ''],
    ids=["inline-editor", "search-area", "other-form"],
)
def test_send_contract_requires_eligible_composer(container_attributes: str) -> None:
    html = _EMPTY_FORM + (
        f'<div {container_attributes}><form data-edit-message>'
        '<textarea role="textbox">Old message</textarea>'
        '<button type="submit">Save</button></form></div>'
    )
    composer_selector = 'form textarea[role="textbox"]'
    composer_check = _check(html, selector=composer_selector)
    assert (composer_check.verdict, composer_check.eligible) == ("fault", 0)
    check = _check(
        html, target="send", selector='form[data-edit-message] button[type="submit"]',
        composer_selector=composer_selector,
    )
    assert (check.verdict, check.eligible, check.candidates) == ("wait", 0, 0)
    assert _candidates(html, target="send", composer_selector=composer_selector) == []


def test_send_contract_allows_generic_composer_form() -> None:
    html = '<form><textarea role="textbox">Text</textarea><button type="submit">Send</button></form>'
    composer_selector = 'form textarea[role="textbox"]'
    check = _check(
        html, target="send", selector='button[type="submit"]',
        composer_selector=composer_selector,
    )
    assert (check.verdict, check.eligible, check.candidates) == ("valid", 1, 1)
    assert len(_candidates(html, target="send", composer_selector=composer_selector)) == 1


def test_composer_contract_prefers_known_composer_form() -> None:
    html = _EMPTY_FORM + '<form data-other><textarea>Other editor</textarea></form>'
    check = _check(html)
    assert (check.verdict, check.eligible, check.candidates) == ("valid", 1, 1)
    candidates = _candidates(html)
    assert len(candidates) == 1
    assert candidates[0]["formDataAttributes"] == ["data-chatgpt-composer"]
    assert _check(html, selector=candidates[0]["selector"]).verdict == "valid"
    assert _check(html, selector="form[data-other] textarea").eligible == 0


def test_composer_contract_allows_generic_form_after_markup_drift() -> None:
    html = '<form><div contenteditable="true" role="textbox">Text</div></form>'
    check = _check(html, selector='form > div[contenteditable="true"][role="textbox"]')
    assert (check.verdict, check.eligible, check.candidates) == ("valid", 1, 1)
    candidates = _candidates(html)
    assert len(candidates) == 1
    assert _check(html, selector=candidates[0]["selector"]).verdict == "valid"


def _candidates(
    html: str, *, target: locators.LocatorTarget = "composer",
    composer_selector: str = selectors.COMPOSER_SELECTOR,
) -> list[dict[str, Any]]:
    async def run() -> list[dict[str, Any]]:
        async with async_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                options = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
                    "chromium-*/chrome-mac*/Google Chrome for Testing.app/"
                    "Contents/MacOS/Google Chrome for Testing"
                ))
                if not options:
                    pytest.skip("No local Chromium executable available")
                executable = options[-1]
            browser = await playwright.chromium.launch(
                executable_path=str(executable), headless=True,
            )
            try:
                page = await browser.new_page()
                await page.set_content(html)
                result = await page.evaluate(selectors.LOCATOR_CANDIDATES_PROBE_JS, {
                    "target": target, "composerSelector": composer_selector,
                })
                return result["candidates"]
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
                          deadline=book.monotonic() + kwargs.pop("timeout_seconds", 20), **kwargs)


def _record_test_failure(book, reason):
    attempt = locators.RediscoveredLocator(".candidate", "test-attempt")
    book.pending[("ko", "composer")] = attempt
    book.record_failure("ko", "composer", reason, attempt.attempt_id)


def test_book_persistence_breaker_and_success(tmp_path: Path) -> None:
    now = [100.0]
    path = tmp_path / "locators.json"
    book = locators.LocatorBook(path, clock=lambda: now[0])
    assert book.selector_for("ko", "composer") == selectors.COMPOSER_SELECTOR
    _record_test_failure(book, "absent")
    assert book.refusal("ko", "composer")
    now[0] += 59
    assert book.refusal("ko", "composer")
    now[0] += 1
    assert book.refusal("ko", "composer") is None
    _record_test_failure(book, "absent")
    now[0] += 60
    _record_test_failure(book, "absent")
    assert book.refusal("ko", "composer")
    now[0] += 3599
    assert book.refusal("ko", "composer")
    now[0] += 1
    assert book.refusal("ko", "composer") is None
    _record_test_failure(book, "half-open failed")
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
    results = asyncio.run(run())
    assert [result.selector for result in results] == [".recovered", ".recovered"]
    assert results[0].attempt_id == results[1].attempt_id
    assert healer.calls == 1
    assert book.pending[("ko", "composer")] == results[0]
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
    _record_test_failure(book, "bad")
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
                                        composer_selector=".broken", deadline=book.monotonic() + 20))
    assert result.selector == selectors.COMPOSER_SELECTOR and result.attempt_id is None and healer.calls == 0


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


@pytest.mark.parametrize("host,expected", [("0.0.0.0", "127.0.0.1"), ("::", "[::1]"),
                                          ("::1", "[::1]"), ("gateway.test", "gateway.test")])
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
    book.pending[("ko", "composer")] = locators.RediscoveredLocator(".saved", "attempt")
    book.record_success("ko", "composer", ".saved", "attempt")
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


def test_candidate_data_attribute_values_are_private() -> None:
    names = ["data-conversation-id", "data-draft", "data-message-id", "data-value"]
    sentinels = [f"PRIVATE_{index}_SENTINEL_VALUE" for index in range(8)]
    composer_attributes = " ".join(
        f'{name}="{value}"' for name, value in zip(names, sentinels[:4])
    )
    form_attributes = " ".join(
        f'{name}="{value}"' for name, value in zip(names, sentinels[4:])
    )
    payload = _candidates(
        f'<form {form_attributes} data-state="PRIVATE_STATE">'
        f'<textarea {composer_attributes} data-state="PRIVATE_STATE"></textarea></form>'
    )
    assert len(payload) == 1
    serialized = json.dumps(payload)
    for sentinel in sentinels:
        assert sentinel not in serialized and sentinel[:10] not in serialized
    assert "PRIVATE_STATE" not in serialized and "data-state" not in serialized
    assert payload[0]["dataAttributes"] == names
    assert payload[0]["formDataAttributes"] == names


@pytest.mark.parametrize("reply", [
    '{"status":"selected","candidate":0}',
    ' \n {"status":"selected","candidate":0}\t ',
    '```json\n{"status":"selected","candidate":0}\n```',
    ' \n```\n{"status":"selected","candidate":0}\n```\n ',
])
def test_single_object_reply_accepted(tmp_path: Path, reply: str) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer(reply))
    assert asyncio.run(_rediscover(book)).selector == ".recovered"
    assert not book.path.exists()


@pytest.mark.parametrize("reply", [
    '{"status":"selected","candidate":0}\n{"status":"uncertain"}',
    '{"status":"selected","candidate":0} trailing prose',
    'prose {"status":"selected","candidate":0}',
    '[{"status":"selected","candidate":0}]',
    '```json\n{"status":"selected","candidate":0}\n{"status":"uncertain"}\n```',
    '```python\n{"status":"selected","candidate":0}\n```',
    '```json\n```json\n{"status":"selected","candidate":0}\n```\n```',
])
def test_non_single_object_reply_counts(tmp_path: Path, reply: str) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer(reply))
    with pytest.raises(locators.HealFailed, match="the healer reply was not a single JSON object"):
        asyncio.run(_rediscover(book))
    record = json.loads(book.path.read_text())["environments"]["ko"]["composer"]
    assert record["consecutive_failures"] == 1
    assert book.refusal("ko", "composer")
    assert not book.pending


@pytest.mark.parametrize("host,bind_host", [("127.0.0.1", "127.0.0.1"), ("::", "::1")])
def test_gateway_healer_bypasses_environment_proxy(
    monkeypatch: pytest.MonkeyPatch, host: str, bind_host: str,
) -> None:
    from claudex.config import GatewayConfig

    async def run() -> None:
        gateway_requests = []
        proxy_requests = []

        async def respond(reader, writer, requests):
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                requests.append(headers)
                content_length = next(
                    int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                await reader.readexactly(content_length)
                body = b'{"content":[{"type":"text","text":"selected"}]}'
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        try:
            gateway = await asyncio.start_server(
                lambda reader, writer: respond(reader, writer, gateway_requests), bind_host, 0,
            )
        except OSError as exc:
            if bind_host == "::1":
                pytest.skip(f"IPv6 loopback is unavailable: {exc}")
            raise
        async with gateway:
            proxy = await asyncio.start_server(
                lambda reader, writer: respond(reader, writer, proxy_requests), "127.0.0.1", 0,
            )
            async with proxy:
                proxy_url = f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
                for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                             "http_proxy", "https_proxy", "all_proxy"):
                    monkeypatch.setenv(name, proxy_url)
                for name in ("NO_PROXY", "no_proxy"):
                    monkeypatch.setenv(name, "")
                config = GatewayConfig(
                    host=host, port=gateway.sockets[0].getsockname()[1], local_token="test-local-token",
                )
                monkeypatch.setattr(GatewayConfig, "load", lambda: config)
                assert await locators.GatewayMessagesHealer().complete("system", "prompt", 5) == "selected"
                assert proxy_requests == []
                assert len(gateway_requests) == 1
                assert b"authorization: Bearer test-local-token\r\n" in gateway_requests[0]
                assert gateway_requests[0].startswith(b"POST /v1/messages HTTP/1.1\r\n")

    asyncio.run(run())


def test_environment_key() -> None:
    assert locators.environment_key("") == "unknown"
    assert locators.environment_key("ko-KR") == "ko-KR"


def test_deleting_book_clears_breaker_and_selector_without_restart(tmp_path: Path) -> None:
    book = locators.LocatorBook(tmp_path / "locators.json", clock=lambda: 100.0)
    book.record_success("ko", "composer", ".saved")
    for _ in range(locators.FAILURE_THRESHOLD):
        _record_test_failure(book, "not found")
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
    _record_test_failure(book, "not found")
    book.path.write_text("invalid JSON")
    assert book.refusal("ko", "composer") is None
    assert book.selector_for("ko", "composer") == selectors.COMPOSER_SELECTOR
    assert "Could not read optional locator book" in caplog.text


@pytest.mark.parametrize("blocked", ["lock", "probe", "healer", "fresh_check", "late_reply"])
def test_rediscovery_absolute_deadline(tmp_path, monkeypatch, blocked):
    monkeypatch.setattr(locators, "MINIMUM_REDISCOVERY_SECONDS", 0)
    cancelled = []
    class SlowHealer:
        async def complete(self, *args, **kwargs):
            if blocked in ("probe", "fresh_check"):
                return '{"status":"selected","candidate":0}'
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.append(True)
                if blocked == "late_reply":
                    return "malformed late reply"
                raise
    book = locators.LocatorBook(tmp_path / "locators.json", healer=SlowHealer())
    original = _evaluate()
    async def evaluate(expression, args):
        if ((blocked == "probe" and expression == selectors.LOCATOR_CANDIDATES_PROBE_JS)
                or (blocked == "fresh_check" and expression == selectors.LOCATOR_CHECK_PROBE_JS)):
            await asyncio.sleep(1)
        return await original(expression, args)
    async def run():
        lock = book._locks.setdefault(("ko", "composer"), asyncio.Lock())
        if blocked == "lock":
            await lock.acquire()
        start = asyncio.get_running_loop().time()
        try:
            with pytest.raises(TimeoutError):
                await book.rediscover(evaluate, "composer", "ko",
                    failed_selector=selectors.COMPOSER_SELECTOR,
                    composer_selector=selectors.COMPOSER_SELECTOR, deadline=start + .05)
            assert asyncio.get_running_loop().time() - start < .15
        finally:
            if blocked == "lock":
                lock.release()
    asyncio.run(run())
    assert not book.pending and not book.path.exists()
    assert bool(cancelled) == (blocked in ("healer", "late_reply"))


def test_pending_attempt_ownership(tmp_path):
    healer = _Healer()
    book = locators.LocatorBook(tmp_path / "locators.json", healer=healer)
    async def run():
        first = await _rediscover(book)
        with pytest.raises(locators.HealingRefused, match="awaiting first-use proof"):
            await _rediscover(book, evaluate=_evaluate(valid=False))
        joined = await _rediscover(book)
        assert first.attempt_id == joined.attempt_id
        book.record_failure("ko", "composer", "bad", first.attempt_id)
        book.record_failure("ko", "composer", "bad twice", joined.attempt_id)
        assert book._record("ko", "composer")["consecutive_failures"] == 1
        book.record_success("ko", "composer", ".recovered", first.attempt_id)
        assert book.refusal("ko", "composer")
        newer = locators.RediscoveredLocator(".new", "new-attempt")
        book.pending[("ko", "composer")] = newer
        book.abandon("ko", "composer", first.attempt_id)
        book.record_failure("ko", "composer", "late failure", first.attempt_id)
        assert book.pending[("ko", "composer")] == newer
        book.abandon("ko", "composer", newer.attempt_id)
        assert not book.pending
    asyncio.run(run())
    assert healer.calls == 1


def test_deadline_remaining_recomputed_after_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(locators, "MINIMUM_REDISCOVERY_SECONDS", 0)
    timeouts = []
    class Healer:
        async def complete(self, *args, timeout_seconds):
            timeouts.append(timeout_seconds)
            return '{"status":"selected","candidate":0}'
    book = locators.LocatorBook(tmp_path / "locators.json", healer=Healer())
    async def run():
        lock = book._locks.setdefault(("ko", "composer"), asyncio.Lock())
        await lock.acquire()
        asyncio.get_running_loop().call_later(.05, lock.release)
        await book.rediscover(_evaluate(), "composer", "ko",
            failed_selector=selectors.COMPOSER_SELECTOR,
            composer_selector=selectors.COMPOSER_SELECTOR,
            deadline=asyncio.get_running_loop().time() + .15)
    asyncio.run(run())
    assert 0 < timeouts[0] <= .11


def test_half_open_only_one_attempt_awaits_proof(tmp_path):
    now = [100.0]
    healer = _Healer()
    book = locators.LocatorBook(tmp_path / "locators.json", healer=healer, clock=lambda: now[0])
    for _ in range(locators.FAILURE_THRESHOLD):
        _record_test_failure(book, "failed")
    now[0] += locators.BREAKER_COOLDOWN_SECONDS
    async def run():
        results = await asyncio.gather(
            _rediscover(book), _rediscover(book, evaluate=_evaluate(valid=False)),
            return_exceptions=True,
        )
        assert isinstance(results[0], locators.RediscoveredLocator)
        assert isinstance(results[1], locators.HealingRefused)
        assert len(book.pending) == 1
        book.record_failure("ko", "composer", "half-open failed", results[0].attempt_id)
    asyncio.run(run())
    assert healer.calls == 1
    assert book._record("ko", "composer")["open_until"] == now[0] + locators.BREAKER_COOLDOWN_SECONDS


def test_fresh_check_wait_does_not_count(tmp_path):
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer())
    original = _evaluate()
    async def evaluate(expression, args):
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            return {"matched": 0, "eligible": 0, "candidates": 0}
        return await original(expression, args)
    with pytest.raises(locators.HealingRefused, match="could not be verified"):
        asyncio.run(_rediscover(book, evaluate=evaluate))
    assert not book.pending and not book.path.exists()


@pytest.mark.parametrize("suppresses_cancellation", [False, True])
def test_cancelled_healer_does_not_count(tmp_path, suppresses_cancellation):
    started = asyncio.Event()
    class Healer:
        async def complete(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if suppresses_cancellation:
                    return "malformed cancelled reply"
                raise
    book = locators.LocatorBook(tmp_path / "locators.json", healer=Healer())
    async def run():
        task = asyncio.create_task(_rediscover(book))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert not book.pending and not book.path.exists()


def test_deterministic_reuse_failure_and_abandon_do_not_count(tmp_path):
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer())
    book.record_failure("ko", "composer", "seed failed", None)
    book.abandon("ko", "composer", None)
    assert not book.path.exists()
    book.record_success("ko", "composer", ".restored", None)
    assert book.selector_for("ko", "composer") == ".restored"


@pytest.mark.parametrize("prior_failures", [2, 3])
@pytest.mark.parametrize("has_source_selector", [False, True])
@pytest.mark.parametrize("destination_already_stored", [False, True])
def test_transferred_success_closes_source_circuit(
    tmp_path, prior_failures, has_source_selector, destination_already_stored,
):
    now = [100.0]
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer(),
                               clock=lambda: now[0])
    if has_source_selector:
        book.record_success("unknown", "composer", ".old")
        book.record_success("unknown", "composer", ".source")
    if destination_already_stored:
        book.record_success("ko", "composer", ".recovered")
    for _ in range(prior_failures):
        book._record_failure("unknown", "composer", "healer failed")
        now[0] += (locators.BREAKER_COOLDOWN_SECONDS if book._record(
            "unknown", "composer")["open_until"] else locators.RETRY_BACKOFF_SECONDS)
    source = book._record("unknown", "composer")
    assert book.refusal("unknown", "composer") is None

    async def run():
        result = await book.rediscover(_evaluate(), "composer", "unknown",
            failed_selector=selectors.COMPOSER_SELECTOR,
            composer_selector=selectors.COMPOSER_SELECTOR,
            deadline=book.monotonic() + 20)
        book.record_success("ko", "composer", result.selector, result.attempt_id,
                            attempt_env=result.environment)
    asyncio.run(run())

    assert not book.pending
    assert book.selector_for("ko", "composer") == ".recovered"
    assert book._record("unknown", "composer") == {
        **source, "consecutive_failures": 0, "open_until": None,
        "next_attempt_at": None, "last_failure": None,
    }
    book._record_failure("unknown", "composer", "next healer failed")
    record = book._record("unknown", "composer")
    assert record["consecutive_failures"] == 1
    assert record["open_until"] is None
    assert record["next_attempt_at"] == now[0] + 60


@pytest.mark.parametrize("attempt_id", ["stale", "unknown-attempt"])
def test_stale_transferred_success_preserves_both_records(tmp_path, attempt_id):
    book = locators.LocatorBook(tmp_path / "locators.json")
    book.record_success("ko", "composer", ".destination")
    book._record_failure("unknown", "composer", "healer failed")
    attempt = locators.RediscoveredLocator(".recovered", "owned", "unknown")
    book.pending[("unknown", "composer")] = attempt
    before = book.path.read_bytes()
    book.record_success("ko", "composer", attempt.selector, attempt_id,
                        attempt_env=attempt.environment)
    assert book.path.read_bytes() == before
    assert book.pending[("unknown", "composer")] == attempt


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("destination", ["ko", "unknown"])
def test_success_transfer_checks_attempt_ownership(tmp_path, owned, destination):
    book = locators.LocatorBook(tmp_path / "locators.json", healer=_Healer())
    async def run():
        result = await book.rediscover(_evaluate(), "composer", "",
            failed_selector=selectors.COMPOSER_SELECTOR,
            composer_selector=selectors.COMPOSER_SELECTOR,
            deadline=book.monotonic() + 20)
        assert result.environment == "unknown"
        book.record_success(destination, "composer", result.selector,
                            result.attempt_id if owned else "stale",
                            attempt_env=result.environment)
        if owned:
            assert not book.pending
            assert book.selector_for(destination, "composer") == result.selector
            if destination != "unknown":
                assert book._record("unknown", "composer") == {}
        else:
            assert book.pending[("unknown", "composer")] == result
            assert not book.path.exists()
    asyncio.run(run())
