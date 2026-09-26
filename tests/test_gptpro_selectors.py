"""Tests pinning the gptpro DOM selectors to the redesigned ChatGPT markup.

The suite has no JavaScript engine, so the JS probes are pinned structurally
while the CSS selectors are exercised against a minimal element tree shaped
like the observed ChatGPT DOM: a ``form[data-chatgpt-composer]`` hosting a
ProseMirror contenteditable composer with aria-labeled buttons, and
``data-chatgpt-search-unit-key`` message units grouped in turn containers.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from claudex.gptpro import selectors

_USER_ID = "11111111-1111-4111-8111-111111111111"
_ASSISTANT_IDS = (
    "22222222-2222-4222-8222-222222222222 33333333-3333-4333-8333-333333333333"
)
_SECOND_USER_ID = "44444444-4444-4444-8444-444444444444"
_SECOND_ASSISTANT_IDS = "55555555-5555-4555-8555-555555555555"


class _Element:
    """Minimal DOM node: tag, attributes, and children."""

    def __init__(
        self,
        tag: str,
        attributes: dict[str, str] | None = None,
        *children: _Element,
    ) -> None:
        self.tag = tag
        self.attributes = dict(attributes or {})
        self.parent: _Element | None = None
        self.children = list(children)
        for child in self.children:
            child.parent = self


_CHUNK_RE = re.compile(r"#[^\[\]\s]+|\.[^\[\]\s]+|\[[^\]]*\]")
_ATTR_RE = re.compile(
    r'^\[\s*(?P<name>[^\]~|^$*=\s]+)'
    r'(?:\s*(?P<operator>\$?=)\s*"(?P<value>[^"]*)")?'
    r'\s*\]$'
)


def _split_top_level(selector: str, separator: str) -> list[str]:
    """Split on a separator that sits outside attribute brackets."""
    parts: list[str] = []
    current = ""
    depth = 0
    for character in selector:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        if depth == 0 and character == separator and current:
            parts.append(current)
            current = ""
        else:
            current += character
    return [part.strip() for part in [*parts, current] if part.strip()]


def _matches_compound(node: _Element, compound: str) -> bool:
    tag = re.match(r"[a-zA-Z][a-zA-Z0-9-]*", compound)
    tag_name = tag.group(0) if tag else ""
    if tag_name and node.tag != tag_name:
        return False
    remainder = compound[len(tag_name) :]
    chunks = _CHUNK_RE.findall(remainder)
    if "".join(chunks) != remainder:
        return False
    for chunk in chunks:
        if chunk.startswith("#"):
            if node.attributes.get("id") != chunk[1:]:
                return False
        elif chunk.startswith("."):
            if chunk[1:] not in node.attributes.get("class", "").split():
                return False
        else:
            attribute = _ATTR_RE.match(chunk)
            if attribute is None:
                return False
            actual = node.attributes.get(attribute.group("name"))
            if actual is None:
                return False
            operator = attribute.group("operator")
            value = attribute.group("value")
            if operator == "=" and actual != value:
                return False
            if operator == "$=" and not actual.endswith(value):
                return False
    return True


def _matches_ancestor_chain(node: _Element, compounds: list[str]) -> bool:
    if not _matches_compound(node, compounds[-1]):
        return False
    remaining = compounds[:-1]
    if not remaining:
        return True
    ancestor = node.parent
    while ancestor is not None:
        if _matches_ancestor_chain(ancestor, remaining):
            return True
        ancestor = ancestor.parent
    return False


def _matches(node: _Element, selector: str) -> bool:
    compounds = _split_top_level(selector, " ")
    return bool(compounds) and _matches_ancestor_chain(node, compounds)


def _iter_tree(root: _Element) -> Iterator[_Element]:
    for child in root.children:
        yield child
        yield from _iter_tree(child)


def _select(root: _Element, selector: str) -> list[_Element]:
    return [
        node
        for node in _iter_tree(root)
        if any(_matches(node, part) for part in _split_top_level(selector, ","))
    ]


def _composer_form(*buttons: _Element) -> _Element:
    return _Element(
        "form",
        {"data-chatgpt-composer": ""},
        _Element("button", {"type": "button", "aria-label": "Attach files"}),
        _Element(
            "div",
            {
                "class": "ProseMirror",
                "contenteditable": "true",
                "role": "textbox",
                "aria-label": "Ask ChatGPT",
            },
        ),
        *buttons,
    )


def _send_button() -> _Element:
    return _Element("button", {"type": "submit", "aria-label": "Send"})


def _stop_button() -> _Element:
    return _Element("button", {"aria-label": "Stop"})


def _start_voice_button() -> _Element:
    return _Element("button", {"aria-label": "Start Voice"})


def _turn(user_id: str, assistant_ids: str, *, turn_index: int) -> _Element:
    return _Element(
        "div",
        {"data-turn-key": user_id},
        _Element(
            "div",
            {
                "data-chatgpt-search-unit-key": f"fallback-turn-{turn_index}:0:user",
                "data-chatgpt-search-message-ids": user_id,
            },
        ),
        _Element(
            "div",
            {
                "data-chatgpt-search-unit-key": (
                    f"fallback-turn-{turn_index}:2:assistant"
                ),
                "data-chatgpt-search-message-ids": assistant_ids,
            },
        ),
    )


def _conversation_dom(*composer_buttons: _Element) -> _Element:
    return _Element(
        "div",
        None,
        _composer_form(*composer_buttons),
        _Element(
            "main",
            None,
            _turn(_USER_ID, _ASSISTANT_IDS, turn_index=0),
            _turn(_SECOND_USER_ID, _SECOND_ASSISTANT_IDS, turn_index=1),
        ),
    )


def test_composer_selector_targets_the_contenteditable_composer() -> None:
    dom = _conversation_dom(_send_button())

    matches = _select(dom, selectors.COMPOSER_SELECTOR)

    assert len(matches) == 1
    assert matches[0].tag == "div"
    assert matches[0].attributes["contenteditable"] == "true"
    assert matches[0].attributes["role"] == "textbox"


def test_send_button_selector_targets_the_composer_form_submit_button() -> None:
    submit = _send_button()
    dom = _Element(
        "div",
        None,
        _composer_form(submit),
        # A look-alike outside the composer form (for example in a dialog)
        # must not be targeted.
        _Element(
            "div",
            {"role": "dialog"},
            _Element("button", {"type": "submit", "aria-label": "Send"}),
        ),
    )

    assert _select(dom, selectors.SEND_BUTTON_SELECTOR) == [submit]


def test_stop_button_selector_targets_the_generating_stop_button() -> None:
    stop = _stop_button()
    generating = _conversation_dom(stop)

    assert _select(generating, selectors.STOP_BUTTON_SELECTOR) == [stop]

    # After completion the stop button is replaced by "Start Voice".
    completed = _conversation_dom(_start_voice_button())

    assert _select(completed, selectors.STOP_BUTTON_SELECTOR) == []


def test_message_unit_selectors_target_units_by_key_suffix() -> None:
    dom = _conversation_dom(_send_button())

    user_units = _select(dom, selectors.USER_MESSAGE_SELECTOR)
    assistant_units = _select(dom, selectors.ASSISTANT_MESSAGE_SELECTOR)

    assert [node.attributes["data-chatgpt-search-unit-key"] for node in user_units] == [
        "fallback-turn-0:0:user",
        "fallback-turn-1:0:user",
    ]
    assert [
        node.attributes["data-chatgpt-search-unit-key"] for node in assistant_units
    ] == [
        "fallback-turn-0:2:assistant",
        "fallback-turn-1:2:assistant",
    ]
    # The combined turn-state selector must keep document order across turn
    # containers: user unit, then its assistant unit, per turn.
    combined = _select(
        dom,
        f"{selectors.USER_MESSAGE_SELECTOR}, {selectors.ASSISTANT_MESSAGE_SELECTOR}",
    )
    assert [node.attributes["data-chatgpt-search-unit-key"] for node in combined] == [
        "fallback-turn-0:0:user",
        "fallback-turn-0:2:assistant",
        "fallback-turn-1:0:user",
        "fallback-turn-1:2:assistant",
    ]


def test_user_units_expose_the_user_message_id() -> None:
    dom = _conversation_dom(_send_button())

    user_units = _select(dom, selectors.USER_MESSAGE_SELECTOR)

    assert [
        node.attributes.get(selectors.MESSAGE_ID_ATTRIBUTE) for node in user_units
    ] == [_USER_ID, _SECOND_USER_ID]


def test_top_level_predicate_excludes_nested_unit_key_elements() -> None:
    predicate = selectors.TOP_LEVEL_ROLE_PREDICATE_JS

    assert "data-chatgpt-search-unit-key" in predicate
    assert "data-message-author-role" not in predicate
    assert "parentElement" in predicate
    assert "closest(" in predicate
    for probe in (
        selectors.TOP_LEVEL_USER_IDS_PROBE_JS,
        selectors.USER_ECHO_PROBE_JS,
        selectors.RELOCK_USER_ECHO_PROBE_JS,
        selectors.TURN_STATE_PROBE_JS,
    ):
        assert predicate in probe


def test_legacy_dom_markup_is_not_selected() -> None:
    legacy = _Element(
        "div",
        None,
        _Element("textarea", {"id": "prompt-textarea"}),
        _Element(
            "div",
            {"data-message-author-role": "user", "data-message-id": _USER_ID},
        ),
        _Element(
            "div",
            {
                "data-message-author-role": "assistant",
                "data-message-id": "22222222-2222-4222-8222-222222222222",
            },
        ),
        _Element("button", {"data-testid": "send-button"}),
        _Element("button", {"data-testid": "stop-button"}),
    )

    assert _select(legacy, selectors.COMPOSER_SELECTOR) == []
    assert _select(legacy, selectors.SEND_BUTTON_SELECTOR) == []
    assert _select(legacy, selectors.STOP_BUTTON_SELECTOR) == []
    assert _select(legacy, selectors.USER_MESSAGE_SELECTOR) == []
    assert _select(legacy, selectors.ASSISTANT_MESSAGE_SELECTOR) == []
