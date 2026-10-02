"""Semantic locator contracts exercised against Chromium DOM fixtures."""

from __future__ import annotations

import asyncio
from pathlib import Path

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
