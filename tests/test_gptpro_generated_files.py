"""Behavior tests for delivering files ChatGPT generated in a final answer.

The tests drive the detached answer poller, which shares file collection with
the direct ask path, through a fake browser page that serves the observed
ChatGPT download contract: an authenticated interpreter/download lookup that
returns a signed same-origin estuary content URL.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from claudex.gptpro import ask, runtime

_CONVERSATION_ID = "123e4567-e89b-12d3-a456-426614174000"
_MESSAGE_ID = "11111111-2222-4333-8444-555555555555"
_MARKER = "[gptpro-transport-nonce:generated-files]"
_ACCESS_TOKEN = "secret-access-token"
_SIGNATURE = "secret-signature"
_SESSION_URL = "https://chatgpt.com/api/auth/session"
_DOWNLOAD_PATH = (
    f"/backend-api/conversation/{_CONVERSATION_ID}/interpreter/download"
)
_CONTENT_URL = "https://chatgpt.com/backend-api/estuary/content"
_ZIP_BYTES = b"PK\x03\x04\x14\x00\x00\x00\x08\x00\xff\xfe\x00binary\x00tail"
_MARKDOWN_BYTES = "# Report\n\nUnicode: café ✓\n".encode()


def _conversation(
    text: str,
    *,
    end_turn: bool = True,
    message_id: str = _MESSAGE_ID,
    commentary: str | None = None,
) -> dict[str, object]:
    mapping: dict[str, object] = {
        "user": {
            "parent": None,
            "message": {
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [_MARKER]},
            },
        },
    }
    parent = "user"
    if commentary is not None:
        mapping["commentary"] = {
            "parent": "user",
            "message": {
                "id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [commentary]},
                "channel": "commentary",
                "recipient": "all",
            },
        }
        parent = "commentary"
    mapping["assistant"] = {
        "parent": parent,
        "message": {
            "id": message_id,
            "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": [text]},
            "channel": "final",
            "recipient": "all",
            "status": "finished_successfully" if end_turn else "in_progress",
            "end_turn": end_turn,
        },
    }
    return {"current_node": "assistant", "mapping": mapping}


def _response(
    status: int,
    body: bytes = b"",
    *,
    content_type: str = "application/json",
    **overrides: Any,
) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = None
    result: dict[str, Any] = {
        "status": status,
        "headers": {"content-type": content_type},
        "text": text,
        "json": parsed,
        "bodyBase64": base64.b64encode(body).decode("ascii"),
        "byteLength": len(body),
        "tooLarge": False,
        "redirected": False,
        "url": "",
        "fetchError": None,
        "timedOut": False,
    }
    result.update(overrides)
    return result


class _FilePage:
    """Fake chatgpt.com page serving conversations and generated files."""

    def __init__(
        self,
        conversations: list[dict[str, object]],
        files: dict[str, bytes | int] | None = None,
    ) -> None:
        self.url = "https://chatgpt.com/"
        self.conversations = list(conversations)
        # Content values are bytes, or an int size standing in for a body
        # that is only materialized when the caller's byte bound allows it.
        self.files = dict(files or {})
        self.metadata_overrides: dict[str, dict[str, Any]] = {}
        self.content_overrides: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.on_request: Callable[[dict[str, Any]], None] | None = None
        self.closed = False

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        del url, wait_until, timeout

    async def close(self) -> None:
        self.closed = True

    async def evaluate(self, script: str, argument: Any = None) -> Any:
        if "navigator.userAgent" in script:
            return "Mozilla/5.0 Chrome/151.0.0.0"
        assert isinstance(argument, dict)
        self.requests.append(argument)
        if self.on_request is not None:
            self.on_request(argument)
        target = urlsplit(argument["url"])
        if argument["url"] == _SESSION_URL:
            return _response(
                200, json.dumps({"accessToken": _ACCESS_TOKEN}).encode()
            )
        if target.path == _DOWNLOAD_PATH:
            query = parse_qs(target.query)
            sandbox_path = query["sandbox_path"][0]
            if sandbox_path in self.metadata_overrides:
                return self.metadata_overrides[sandbox_path]
            if sandbox_path not in self.files:
                return _response(
                    404, json.dumps({"detail": "File not found"}).encode()
                )
            file_id = hashlib.sha256(sandbox_path.encode()).hexdigest()[:16]
            return _response(
                200,
                json.dumps(
                    {
                        "status": "success",
                        "download_url": (
                            f"{_CONTENT_URL}?id={file_id}&sig={_SIGNATURE}"
                        ),
                        "metadata": {"file_id": f"file_{file_id}"},
                        "file_name": "server-renamed(2).bin",
                        "mime_type": (
                            "application/zip"
                            if sandbox_path.endswith(".zip")
                            else "text/markdown"
                        ),
                        "file_size_bytes": None,
                    }
                ).encode(),
            )
        if f"{target.scheme}://{target.netloc}{target.path}" == _CONTENT_URL:
            file_id = parse_qs(target.query)["id"][0]
            for sandbox_path, content in self.files.items():
                if hashlib.sha256(sandbox_path.encode()).hexdigest()[:16] != file_id:
                    continue
                if sandbox_path in self.content_overrides:
                    return self.content_overrides[sandbox_path]
                size = content if isinstance(content, int) else len(content)
                max_bytes = argument.get("maxBytes")
                if isinstance(max_bytes, int) and size > max_bytes:
                    return _response(
                        200, b"", content_type="application/octet-stream",
                        tooLarge=True, byteLength=max_bytes + 1,
                    )
                body = b"x" * content if isinstance(content, int) else content
                return _response(
                    200, body, content_type="application/octet-stream",
                    url=argument["url"],
                )
            return _response(404)
        if target.path == f"/backend-api/conversation/{_CONVERSATION_ID}":
            conversation = self.conversations[0]
            if len(self.conversations) > 1:
                conversation = self.conversations.pop(0)
            return _response(200, json.dumps(conversation).encode())
        raise AssertionError(f"unexpected request {argument['url']!r}")

    def urls(self) -> list[str]:
        return [request["url"] for request in self.requests]


class _FileContext:
    def __init__(self, page: _FilePage) -> None:
        self.page = page

    async def new_page(self) -> _FilePage:
        return self.page


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _output_root(home: Path) -> Path:
    return home / ".claudex" / "gptpro" / "outputs"


def _saved_files(home: Path) -> list[Path]:
    root = _output_root(home)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _poll(page: _FilePage) -> ask.AskOutcome:
    async def scenario() -> ask.AskOutcome:
        poller = runtime.DetachPoller(
            lambda: asyncio.sleep(0, result=_FileContext(page))
        )
        future = poller.register(
            _CONVERSATION_ID, _MARKER, runtime._monotonic() + 100.0
        )
        try:
            return await future
        finally:
            await poller.aclose()

    return asyncio.run(scenario())


def _descriptors(outcome: ask.AskOutcome) -> tuple[Any, ...]:
    files = getattr(outcome, "files", None)
    assert files is not None, "AskOutcome does not expose generated files"
    return files


def _completeness(outcome: ask.AskOutcome) -> object:
    complete = getattr(outcome, "files_complete", None)
    assert complete is not None, "AskOutcome does not expose files_complete"
    return complete


def _download_requests(page: _FilePage) -> list[str]:
    return [url for url in page.urls() if _DOWNLOAD_PATH in url]


def _content_requests(page: _FilePage) -> list[dict[str, Any]]:
    return [
        request for request in page.requests
        if request["url"].startswith(_CONTENT_URL)
    ]


def test_binary_and_text_files_are_saved_with_descriptors(
    _isolated_home: Path,
) -> None:
    text = (
        "Get [the bundle](sandbox:/mnt/data/report-bundle.zip) and "
        "[the notes](sandbox:/mnt/data/docs/read-me.md)."
    )
    page = _FilePage(
        [_conversation(text)],
        {
            "/mnt/data/report-bundle.zip": _ZIP_BYTES,
            "/mnt/data/docs/read-me.md": _MARKDOWN_BYTES,
        },
    )

    outcome = _poll(page)

    saved = _saved_files(_isolated_home)
    assert [path.name for path in saved] == ["read-me.md", "report-bundle.zip"]
    assert outcome.text == text
    files = _descriptors(outcome)
    assert _completeness(outcome) is True
    assert [
        (item.name, item.sandbox_path, item.message_id, item.status)
        for item in files
    ] == [
        ("report-bundle.zip", "/mnt/data/report-bundle.zip", _MESSAGE_ID, "saved"),
        ("read-me.md", "/mnt/data/docs/read-me.md", _MESSAGE_ID, "saved"),
    ]
    for item, expected in zip(files, (_ZIP_BYTES, _MARKDOWN_BYTES), strict=True):
        local = Path(item.path)
        assert local.is_absolute()
        assert local.read_bytes() == expected
        assert item.size_bytes == len(expected)
        assert item.sha256 == hashlib.sha256(expected).hexdigest()
        assert item.error is None
        assert local.is_relative_to(_output_root(_isolated_home))
        assert local.stat().st_mode & 0o077 == 0
        assert local.parent.stat().st_mode & 0o077 == 0
    assert [item.mime_type for item in files] == ["application/zip", "text/markdown"]
    assert not [path for path in saved if path.name.startswith(".")]


def test_download_uses_bearer_only_for_lookup_and_canonical_query(
    _isolated_home: Path,
) -> None:
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/a%20b.md)")],
        {"/mnt/data/a b.md": b"text"},
    )

    outcome = _poll(page)

    lookups = [
        request for request in page.requests
        if _DOWNLOAD_PATH in request["url"]
    ]
    assert len(lookups) == 1
    query = parse_qs(urlsplit(lookups[0]["url"]).query)
    assert query == {
        "download_intent": ["true"],
        "message_id": [_MESSAGE_ID],
        "sandbox_path": ["/mnt/data/a b.md"],
    }
    assert lookups[0]["headers"] == {"Authorization": f"Bearer {_ACCESS_TOKEN}"}
    contents = _content_requests(page)
    assert len(contents) == 1
    assert "Authorization" not in contents[0].get("headers", {})
    assert [item.name for item in _descriptors(outcome)] == ["a b.md"]


def test_answer_without_files_keeps_legacy_outcome(_isolated_home: Path) -> None:
    page = _FilePage([_conversation("plain answer")])

    outcome = _poll(page)

    assert outcome == ask.AskOutcome(
        text="plain answer", marker=_MARKER, conversation_id=_CONVERSATION_ID
    )
    assert _descriptors(outcome) == ()
    assert _completeness(outcome) is True
    assert _download_requests(page) == []
    assert not _output_root(_isolated_home).exists()


def test_files_are_collected_only_after_the_turn_is_final(
    _isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = "[f](sandbox:/mnt/data/final.md)"
    page = _FilePage(
        [_conversation(link, end_turn=False), _conversation(link)],
        {"/mnt/data/final.md": b"final"},
    )
    observed: list[str] = []

    def on_request(request: dict[str, Any]) -> None:
        if f"/backend-api/conversation/{_CONVERSATION_ID}" == urlsplit(
            request["url"]
        ).path:
            observed.append("conversation")
        elif _DOWNLOAD_PATH in request["url"]:
            observed.append("download")

    page.on_request = on_request

    monkeypatch.setattr(runtime, "_sleep", lambda _seconds: asyncio.sleep(0))

    outcome = _poll(page)

    assert observed == ["conversation", "conversation", "download"]
    assert [item.status for item in _descriptors(outcome)] == ["saved"]


def test_commentary_links_are_never_downloaded(_isolated_home: Path) -> None:
    page = _FilePage(
        [_conversation(
            "final answer without files",
            commentary="[draft](sandbox:/mnt/data/draft.md)",
        )],
        {"/mnt/data/draft.md": b"draft"},
    )

    outcome = _poll(page)

    assert _download_requests(page) == []
    assert _descriptors(outcome) == ()
    assert outcome.text == "final answer without files"


def test_duplicate_references_download_once(_isolated_home: Path) -> None:
    page = _FilePage(
        [_conversation(
            "[a](sandbox:/mnt/data/a.md) and again [a](sandbox:/mnt/data/a.md)"
        )],
        {"/mnt/data/a.md": b"a"},
    )

    outcome = _poll(page)

    assert len(_download_requests(page)) == 1
    assert [item.sandbox_path for item in _descriptors(outcome)] == [
        "/mnt/data/a.md"
    ]


def test_failed_file_keeps_answer_and_marks_collection_incomplete(
    _isolated_home: Path,
) -> None:
    text = (
        "[ok](sandbox:/mnt/data/ok.md) [gone](sandbox:/mnt/data/expired.md) "
        "[denied](sandbox:/mnt/data/denied.md)"
    )
    page = _FilePage(
        [_conversation(text)],
        {"/mnt/data/ok.md": b"ok", "/mnt/data/denied.md": b"secret"},
    )
    page.content_overrides["/mnt/data/denied.md"] = _response(403, b"expired")

    outcome = _poll(page)

    assert [path.name for path in _saved_files(_isolated_home)] == ["ok.md"]
    assert outcome.text == text
    assert _completeness(outcome) is False
    files = {item.sandbox_path: item for item in _descriptors(outcome)}
    assert files["/mnt/data/ok.md"].status == "saved"
    for sandbox_path in ("/mnt/data/expired.md", "/mnt/data/denied.md"):
        failed = files[sandbox_path]
        assert failed.status == "failed"
        assert failed.path is None
        assert failed.sha256 is None
        assert "HTTP" in failed.error
    for item in files.values():
        assert _SIGNATURE not in repr(item)
        assert _ACCESS_TOKEN not in repr(item)


@pytest.mark.parametrize(
    "download_url",
    [
        f"https://evil.example/backend-api/estuary/content?sig={_SIGNATURE}",
        f"http://chatgpt.com/backend-api/estuary/content?sig={_SIGNATURE}",
        f"https://user@chatgpt.com/backend-api/estuary/content?sig={_SIGNATURE}",
        f"https://chatgpt.com/backend-api/other?sig={_SIGNATURE}",
        f"/backend-api/estuary/content?sig={_SIGNATURE}",
    ],
)
def test_untrusted_download_url_is_refused_without_fetching(
    _isolated_home: Path, download_url: str
) -> None:
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/f.md)")], {"/mnt/data/f.md": b"f"}
    )
    page.metadata_overrides["/mnt/data/f.md"] = _response(
        200,
        json.dumps({"status": "success", "download_url": download_url}).encode(),
    )

    outcome = _poll(page)

    assert page.urls()[-1] != download_url
    assert not [url for url in page.urls() if "estuary" in url or "evil" in url]
    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "untrusted" in failed.error
    assert _SIGNATURE not in failed.error
    assert _completeness(outcome) is False


def test_redirected_content_response_is_refused(_isolated_home: Path) -> None:
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/f.md)")], {"/mnt/data/f.md": b"f"}
    )
    page.content_overrides["/mnt/data/f.md"] = _response(
        200, b"f", redirected=True, url=f"https://evil.example/?sig={_SIGNATURE}",
    )

    outcome = _poll(page)

    assert _saved_files(_isolated_home) == []
    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "redirect" in failed.error
    assert _SIGNATURE not in failed.error


def test_network_error_is_reported_without_browser_detail(
    _isolated_home: Path,
) -> None:
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/f.md)")], {"/mnt/data/f.md": b"f"}
    )
    page.content_overrides["/mnt/data/f.md"] = _response(
        0, fetchError=f"TypeError: Failed to fetch {_CONTENT_URL}?sig={_SIGNATURE}",
    )

    outcome = _poll(page)

    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert _SIGNATURE not in failed.error
    assert "estuary" not in failed.error


def test_timed_out_download_is_reported(_isolated_home: Path) -> None:
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/f.md)")], {"/mnt/data/f.md": b"f"}
    )
    page.content_overrides["/mnt/data/f.md"] = _response(0, timedOut=True)

    outcome = _poll(page)

    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "timed out" in failed.error


@pytest.mark.parametrize(
    "link",
    [
        "sandbox:/mnt/data/../../etc/passwd",
        "sandbox:/mnt/data/%2e%2e/secret",
        "sandbox:/etc/passwd",
        "sandbox:/mnt/data/",
        "sandbox:/mnt/data/dir//file.md",
        "sandbox:/mnt/data/bad%00name.md",
        "sandbox:/mnt/data/back%5Cslash.md",
    ],
)
def test_invalid_sandbox_paths_are_rejected_without_requests(
    _isolated_home: Path, link: str
) -> None:
    page = _FilePage([_conversation(f"[x]({link})")])

    outcome = _poll(page)

    assert _download_requests(page) == []
    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "invalid sandbox path" in failed.error
    assert _completeness(outcome) is False


def test_noncanonical_message_id_is_rejected_without_requests(
    _isolated_home: Path,
) -> None:
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/f.md)", message_id="msg&x=1")],
        {"/mnt/data/f.md": b"f"},
    )

    outcome = _poll(page)

    assert _download_requests(page) == []
    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "message ID" in failed.error


def test_unsafe_file_names_are_sanitized_and_never_overwrite(
    _isolated_home: Path,
) -> None:
    text = (
        "[a](sandbox:/mnt/data/one/report.md) "
        "[b](sandbox:/mnt/data/two/report.md) "
        "[c](sandbox:/mnt/data/.hidden) "
        "[d](sandbox:/mnt/data/we%3Aird%2A%7Cname%3F.md)"
    )
    page = _FilePage(
        [_conversation(text)],
        {
            "/mnt/data/one/report.md": b"one",
            "/mnt/data/two/report.md": b"two",
            "/mnt/data/.hidden": b"hidden",
            "/mnt/data/we:ird*|name?.md": b"weird",
        },
    )

    outcome = _poll(page)

    files = _descriptors(outcome)
    assert [item.status for item in files] == ["saved"] * 4
    names = [item.name for item in files]
    assert len(set(names)) == 4
    assert names[0] == "report.md"
    assert names[1] != "report.md" and names[1].endswith(".md")
    assert not names[2].startswith(".")
    assert all(char not in names[3] for char in ':*|?/\\')
    assert [Path(item.path).read_bytes() for item in files] == [
        b"one", b"two", b"hidden", b"weird",
    ]
    assert len({Path(item.path).parent for item in files}) == 1


def test_symlinked_output_root_is_refused(
    _isolated_home: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = _output_root(_isolated_home)
    root.parent.mkdir(parents=True)
    os.symlink(elsewhere, root)
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/f.md)")], {"/mnt/data/f.md": b"f"}
    )

    outcome = _poll(page)

    assert list(elsewhere.iterdir()) == []
    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "output directory" in failed.error


def test_file_count_limit_reports_every_skipped_file(
    _isolated_home: Path,
) -> None:
    count = 21
    text = " ".join(
        f"[f{index}](sandbox:/mnt/data/f{index:02d}.md)" for index in range(count)
    )
    page = _FilePage(
        [_conversation(text)],
        {f"/mnt/data/f{index:02d}.md": b"x" for index in range(count)},
    )

    outcome = _poll(page)

    files = _descriptors(outcome)
    assert len(files) == count
    assert [item.status for item in files[:20]] == ["saved"] * 20
    assert files[20].status == "failed"
    assert "20" in files[20].error
    assert len(_download_requests(page)) == 20
    assert _completeness(outcome) is False


def test_oversized_file_is_refused_without_saving(_isolated_home: Path) -> None:
    page = _FilePage(
        [_conversation("[big](sandbox:/mnt/data/big.bin)")],
        {"/mnt/data/big.bin": 20 * 1024 * 1024 + 1},
    )

    outcome = _poll(page)

    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "20 MiB" in failed.error
    assert _saved_files(_isolated_home) == []
    (content,) = _content_requests(page)
    assert content["maxBytes"] == 20 * 1024 * 1024


def test_oversized_body_is_refused_even_if_browser_bound_is_ignored(
    _isolated_home: Path,
) -> None:
    page = _FilePage(
        [_conversation("[big](sandbox:/mnt/data/big.bin)")],
        {"/mnt/data/big.bin": b"x"},
    )
    page.content_overrides["/mnt/data/big.bin"] = _response(
        200, b"x" * (20 * 1024 * 1024 + 1)
    )

    outcome = _poll(page)

    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert _saved_files(_isolated_home) == []


def test_total_size_limit_marks_remaining_files_failed(
    _isolated_home: Path,
) -> None:
    size = 18 * 1024 * 1024
    text = " ".join(
        f"[f{index}](sandbox:/mnt/data/f{index}.bin)" for index in range(3)
    )
    page = _FilePage(
        [_conversation(text)],
        {f"/mnt/data/f{index}.bin": size for index in range(3)},
    )

    outcome = _poll(page)

    files = _descriptors(outcome)
    assert [item.status for item in files] == ["saved", "saved", "failed"]
    assert "50 MiB" in files[2].error
    assert _content_requests(page)[2]["maxBytes"] == 50 * 1024 * 1024 - 2 * size
    assert _completeness(outcome) is False


def test_collection_time_budget_marks_remaining_files_failed(
    _isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claudex.gptpro import generated_files

    now = 1_000.0
    monkeypatch.setattr(generated_files, "_monotonic", lambda: now)
    text = "[a](sandbox:/mnt/data/a.md) [b](sandbox:/mnt/data/b.md)"
    page = _FilePage(
        [_conversation(text)], {"/mnt/data/a.md": b"a", "/mnt/data/b.md": b"b"}
    )

    def advance(request: dict[str, Any]) -> None:
        nonlocal now
        if request["url"].startswith(_CONTENT_URL):
            now += generated_files.COLLECTION_TIMEOUT_SECONDS + 1

    page.on_request = advance

    outcome = _poll(page)

    files = _descriptors(outcome)
    assert [item.status for item in files] == ["saved", "failed"]
    assert "time budget" in files[1].error
    assert len(_download_requests(page)) == 1


def test_hanging_browser_fetch_is_bounded(
    _isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claudex.gptpro import generated_files

    monkeypatch.setattr(generated_files, "REQUEST_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(generated_files, "REQUEST_TIMEOUT_MARGIN_SECONDS", 0.01)
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/f.md)")], {"/mnt/data/f.md": b"f"}
    )
    original_evaluate = page.evaluate

    async def evaluate(script: str, argument: Any = None) -> Any:
        if isinstance(argument, dict) and argument["url"].startswith(_CONTENT_URL):
            await asyncio.Event().wait()
        return await original_evaluate(script, argument)

    page.evaluate = evaluate  # type: ignore[method-assign]

    outcome = _poll(page)

    (failed,) = _descriptors(outcome)
    assert failed.status == "failed"
    assert "timed out" in failed.error


def test_download_probe_refuses_redirects_and_bounds_reads() -> None:
    from claudex.gptpro import selectors

    probe = getattr(selectors, "FILE_DOWNLOAD_PROBE_JS", "")
    assert "redirect: 'error'" in probe
    assert "location.origin !== args.origin" in probe
    assert "args.maxBytes" in probe
    assert "controller.abort()" in probe


_OTHER_CONVERSATION_ID = "223e4567-e89b-12d3-a456-426614174000"


def _page_with_slow_download(
    other_conversations: list[dict[str, object]],
) -> tuple[_FilePage, asyncio.Event, asyncio.Event]:
    """Serve a second conversation and hold the first file download open."""
    page = _FilePage(
        [_conversation("[f](sandbox:/mnt/data/slow.md)")],
        {"/mnt/data/slow.md": b"slow"},
    )
    download_started = asyncio.Event()
    release_download = asyncio.Event()
    original_evaluate = page.evaluate

    async def evaluate(script: str, argument: Any = None) -> Any:
        if isinstance(argument, dict):
            url = argument["url"]
            path = urlsplit(url).path
            if path == f"/backend-api/conversation/{_OTHER_CONVERSATION_ID}":
                conversation = other_conversations[0]
                if len(other_conversations) > 1:
                    conversation = other_conversations.pop(0)
                return _response(200, json.dumps(conversation).encode())
            if url.startswith(_CONTENT_URL):
                download_started.set()
                await release_download.wait()
        return await original_evaluate(script, argument)

    page.evaluate = evaluate  # type: ignore[method-assign]
    return page, download_started, release_download


def test_slow_file_collection_does_not_stall_other_registrations(
    _isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_sleep", lambda _seconds: asyncio.sleep(0))

    async def scenario() -> None:
        page, download_started, release_download = _page_with_slow_download(
            [
                _conversation("other answer", end_turn=False),
                _conversation("other answer"),
            ]
        )
        poller = runtime.DetachPoller(
            lambda: asyncio.sleep(0, result=_FileContext(page))
        )
        try:
            slow = poller.register(
                _CONVERSATION_ID, _MARKER, runtime._monotonic() + 100.0
            )
            other = poller.register(
                _OTHER_CONVERSATION_ID, _MARKER, runtime._monotonic() + 100.0
            )
            await asyncio.wait_for(download_started.wait(), 1.0)

            done, _pending = await asyncio.wait({other}, timeout=1.0)

            assert other in done, "a slow file download stalled another answer"
            assert other.result().text == "other answer"
            assert not slow.done()
            release_download.set()
            slow_outcome = await asyncio.wait_for(slow, 1.0)
            assert [item.status for item in slow_outcome.files] == ["saved"]
            await asyncio.wait_for(_wait_until(lambda: page.closed), 1.0)
        finally:
            release_download.set()
            await poller.aclose()

    asyncio.run(scenario())


def test_closing_poller_fails_answer_whose_files_are_downloading(
    _isolated_home: Path,
) -> None:
    async def scenario() -> None:
        page, download_started, release_download = _page_with_slow_download([])
        poller = runtime.DetachPoller(
            lambda: asyncio.sleep(0, result=_FileContext(page))
        )
        slow = poller.register(
            _CONVERSATION_ID, _MARKER, runtime._monotonic() + 100.0
        )
        await asyncio.wait_for(download_started.wait(), 1.0)

        await poller.aclose()

        assert slow.done()
        assert isinstance(slow.exception(), ask.GptProAskError)
        assert page.closed
        remaining = [
            task for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        ]
        assert remaining == []
        release_download.set()

    asyncio.run(scenario())


async def _wait_until(condition: Callable[[], bool]) -> None:
    while not condition():
        await asyncio.sleep(0)


def test_deliveries_finishing_together_close_the_idle_page(
    _isolated_home: Path,
) -> None:
    async def scenario() -> None:
        page = _FilePage(
            [_conversation("[f](sandbox:/mnt/data/f.md)")],
            {"/mnt/data/f.md": b"f"},
        )
        gated = 0
        release_downloads = asyncio.Event()
        original_evaluate = page.evaluate

        async def evaluate(script: str, argument: Any = None) -> Any:
            nonlocal gated
            if isinstance(argument, dict) and argument["url"].startswith(
                _CONTENT_URL
            ):
                gated += 1
                await release_downloads.wait()
            return await original_evaluate(script, argument)

        page.evaluate = evaluate  # type: ignore[method-assign]
        poller = runtime.DetachPoller(
            lambda: asyncio.sleep(0, result=_FileContext(page))
        )
        try:
            deadline = runtime._monotonic() + 100.0
            first = poller.register(_CONVERSATION_ID, _MARKER, deadline)
            second = poller.register(_CONVERSATION_ID, _MARKER, deadline)
            await asyncio.wait_for(
                _wait_until(lambda: gated == 2 and poller._task is None), 1.0
            )
            assert not page.closed

            release_downloads.set()
            await asyncio.wait_for(asyncio.gather(first, second), 1.0)
            await asyncio.wait_for(
                _wait_until(lambda: not poller._deliveries), 1.0
            )

            assert page.closed, "the idle resident page was left open"
        finally:
            release_downloads.set()
            await poller.aclose()

    asyncio.run(scenario())


def test_closing_before_a_scheduled_delivery_starts_settles_its_answer(
    _isolated_home: Path,
) -> None:
    async def scenario() -> None:
        page = _FilePage([])
        poller = runtime.DetachPoller(
            lambda: asyncio.sleep(0, result=_FileContext(page))
        )
        future: asyncio.Future[ask.AskOutcome] = (
            asyncio.get_running_loop().create_future()
        )
        registration = runtime._DetachedRegistration(
            conversation_id=_CONVERSATION_ID,
            marker=_MARKER,
            deadline=runtime._monotonic() + 100.0,
            future=future,
        )
        poller._registrations[id(future)] = registration
        poller._page = page

        async def fetch_json(
            target_url: str, *, headers: Any = None
        ) -> tuple[int, dict[str, object]]:
            del headers
            if target_url == _SESSION_URL:
                return 200, {"accessToken": _ACCESS_TOKEN}
            return 200, _conversation("[f](sandbox:/mnt/data/f.md)")

        poller._fetch_json = fetch_json  # type: ignore[method-assign]

        await poller._poll_registration(id(future), registration)
        await poller.aclose()

        assert future.done(), "the answer was left pending after close"
        assert isinstance(future.exception(), ask.GptProAskError)
        assert page.closed

    asyncio.run(scenario())
