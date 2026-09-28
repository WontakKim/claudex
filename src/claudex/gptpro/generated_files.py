"""Retrieve and store files that ChatGPT generated in a final answer.

ChatGPT links generated files as ``sandbox:/mnt/data/...`` Markdown links. A
link is resolved from the authenticated chatgpt.com page in two requests:

1. ``GET /backend-api/conversation/{conversation_id}/interpreter/download``
   with ``download_intent=true``, the linking ``message_id``, and the
   ``sandbox_path``, authorized by the session access token. It answers JSON
   with ``status: "success"`` and a signed ``download_url``.
2. ``GET download_url``, which must be ``https://chatgpt.com`` plus
   ``/backend-api/estuary/content``. It is fetched with the page's cookies
   but without the access token, and redirects are refused.

Saved files land in a fresh private directory under the gateway's gptpro
output directory. A file that cannot be saved is reported with a sanitized
error instead of failing the answer; errors never contain access tokens or
signed URLs.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit

from claudex import paths
from claudex.gptpro.conversation import (
    TRUSTED_ORIGIN,
    SandboxFileReference,
    is_conversation_id,
    is_trusted_origin_url,
)
from claudex.gptpro.selectors import FILE_DOWNLOAD_PROBE_JS

MAX_FILES = 20
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
# Session and lookup responses are small JSON documents.
MAX_LOOKUP_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60.0
# Extra time for the browser round trip beyond the in-page fetch timeout.
REQUEST_TIMEOUT_MARGIN_SECONDS = 5.0
COLLECTION_TIMEOUT_SECONDS = 300.0
# Upper bound on one collection's wall time, for callers waiting on it.
MAX_COLLECTION_SECONDS = COLLECTION_TIMEOUT_SECONDS + REQUEST_TIMEOUT_MARGIN_SECONDS

SANDBOX_ROOT = "/mnt/data/"
CONTENT_PATH = "/backend-api/estuary/content"
_SESSION_URL = f"{TRUSTED_ORIGIN}/api/auth/session"
_MAX_SANDBOX_PATH_LENGTH = 1024
_MAX_FILE_NAME_BYTES = 200
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^\w.()\- ]")
_MIME_TYPE_PATTERN = re.compile(r"^[\w.+-]+/[\w.+-]+$")
_MIB = 1024 * 1024

logger = logging.getLogger(__name__)
_monotonic: Callable[[], float] = time.monotonic

Evaluate = Callable[[str, Any], Awaitable[Any]]


@dataclass(frozen=True)
class GeneratedFile:
    """Outcome of saving one file linked from a final answer.

    ``path`` is an absolute path on the gateway host, set only when
    ``status`` is ``"saved"``; ``error`` is set only when it is ``"failed"``.
    """

    name: str
    sandbox_path: str
    message_id: str | None
    status: Literal["saved", "failed"]
    path: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    sha256: str | None = None
    error: str | None = None


class _FileError(Exception):
    """A caller-safe reason one generated file was not saved."""


def is_valid_sandbox_path(sandbox_path: str) -> bool:
    """Return whether a path names a file below ``/mnt/data/`` without tricks."""
    if (
        not sandbox_path.startswith(SANDBOX_ROOT)
        or len(sandbox_path) > _MAX_SANDBOX_PATH_LENGTH
        or any(
            character == "\\" or ord(character) < 0x20 or ord(character) == 0x7F
            for character in sandbox_path
        )
    ):
        return False
    segments = sandbox_path.removeprefix(SANDBOX_ROOT).split("/")
    return all(segment not in ("", ".", "..") for segment in segments)


def safe_file_name(sandbox_path: str) -> str:
    """Return a local file name derived from the sandbox path's last segment."""
    base_name = sandbox_path.rsplit("/", 1)[-1]
    name = _UNSAFE_NAME_CHARACTERS.sub("_", base_name).strip(". ")
    if not name:
        return "file"
    stem, suffix = os.path.splitext(name)
    if len(suffix.encode()) > 32:
        stem, suffix = name, ""
    while len((stem + suffix).encode()) > _MAX_FILE_NAME_BYTES:
        stem = stem[:-1]
    return (stem + suffix) or "file"


def _failed(reference: SandboxFileReference, error: str) -> GeneratedFile:
    return GeneratedFile(
        name=safe_file_name(reference.sandbox_path),
        sandbox_path=reference.sandbox_path,
        message_id=reference.message_id,
        status="failed",
        error=error,
    )


def _is_trusted_content_url(value: object) -> bool:
    if not isinstance(value, str) or not is_trusted_origin_url(value):
        return False
    parsed = urlsplit(value)
    return (
        parsed.username is None
        and parsed.password is None
        and parsed.path == CONTENT_PATH
        and not parsed.fragment
    )


class _Collection:
    """Download and save the files of one answer through one browser page."""

    def __init__(self, evaluate: Evaluate, conversation_id: str) -> None:
        self.evaluate = evaluate
        self.conversation_id = conversation_id
        self.deadline = _monotonic() + COLLECTION_TIMEOUT_SECONDS
        self.saved_bytes = 0
        self.directory: Path | None = None

    async def _fetch(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
    ) -> Mapping[str, Any]:
        remaining = self.deadline - _monotonic()
        if remaining <= 0:
            raise _FileError(
                "not downloaded: the file collection time budget of "
                f"{COLLECTION_TIMEOUT_SECONDS:.0f}s was exhausted"
            )
        timeout = min(REQUEST_TIMEOUT_SECONDS, remaining)
        try:
            result = await asyncio.wait_for(
                self.evaluate(
                    FILE_DOWNLOAD_PROBE_JS,
                    {
                        "url": url,
                        "origin": TRUSTED_ORIGIN,
                        "headers": dict(headers),
                        "maxBytes": max_bytes,
                        "timeoutMs": max(1, math.ceil(timeout * 1_000)),
                    },
                ),
                timeout + REQUEST_TIMEOUT_MARGIN_SECONDS,
            )
        except TimeoutError as exc:
            raise _FileError("the browser request timed out") from exc
        except Exception as exc:
            # Browser errors can echo the request URL; report only the type.
            raise _FileError(
                f"the browser request failed ({type(exc).__name__})"
            ) from exc
        if not isinstance(result, Mapping):
            raise _FileError("the browser returned an invalid response")
        if result.get("timedOut") is True:
            raise _FileError("the request timed out")
        if result.get("fetchError"):
            raise _FileError(
                "the request failed without a response "
                "(network error or refused redirect)"
            )
        return result

    @staticmethod
    def _require_success(result: Mapping[str, Any], action: str) -> None:
        status = result.get("status")
        if not isinstance(status, int) or not 200 <= status < 300:
            raise _FileError(f"{action} failed with HTTP {status}")

    @staticmethod
    def _body(
        result: Mapping[str, Any], max_bytes: int, limit_error: str
    ) -> bytes:
        if result.get("tooLarge") is True:
            raise _FileError(limit_error)
        encoded = result.get("bodyBase64")
        if not isinstance(encoded, str):
            raise _FileError("the browser returned no response body")
        if len(encoded) > 4 * math.ceil(max_bytes / 3):
            raise _FileError(limit_error)
        try:
            body = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _FileError("the browser returned an invalid response body") from exc
        if len(body) > max_bytes:
            raise _FileError(limit_error)
        return body

    def _json(self, result: Mapping[str, Any], action: str) -> Mapping[str, Any]:
        body = self._body(
            result, MAX_LOOKUP_BYTES, f"{action} response was too large"
        )
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise _FileError(f"{action} did not return JSON") from exc
        if not isinstance(payload, Mapping):
            raise _FileError(f"{action} did not return a JSON object")
        return payload

    async def fetch_access_token(self) -> str:
        result = await self._fetch(
            _SESSION_URL, headers={}, max_bytes=MAX_LOOKUP_BYTES
        )
        self._require_success(result, "the session lookup")
        access_token = self._json(result, "the session lookup").get("accessToken")
        if not isinstance(access_token, str) or not access_token:
            raise _FileError("the ChatGPT session did not provide an access token")
        return access_token

    async def download(
        self, reference: SandboxFileReference, access_token: str
    ) -> GeneratedFile:
        assert reference.message_id is not None
        query = urlencode({
            "download_intent": "true",
            "message_id": reference.message_id.lower(),
            "sandbox_path": reference.sandbox_path,
        })
        lookup = await self._fetch(
            f"{TRUSTED_ORIGIN}/backend-api/conversation/"
            f"{self.conversation_id}/interpreter/download?{query}",
            headers={"Authorization": f"Bearer {access_token}"},
            max_bytes=MAX_LOOKUP_BYTES,
        )
        self._require_success(lookup, "the download lookup")
        metadata = self._json(lookup, "the download lookup")
        if metadata.get("status") != "success":
            raise _FileError("the download lookup did not report success")
        download_url = metadata.get("download_url")
        if not _is_trusted_content_url(download_url):
            raise _FileError("refused an untrusted download URL")

        remaining_total = MAX_TOTAL_BYTES - self.saved_bytes
        total_limit_error = (
            "not downloaded: the answer's files exceed the "
            f"{MAX_TOTAL_BYTES // _MIB} MiB total limit"
        )
        if remaining_total <= 0:
            raise _FileError(total_limit_error)
        max_bytes = min(MAX_FILE_BYTES, remaining_total)
        limit_error = (
            f"the file exceeds the {MAX_FILE_BYTES // _MIB} MiB per-file limit"
            if max_bytes == MAX_FILE_BYTES
            else total_limit_error
        )
        content = await self._fetch(download_url, headers={}, max_bytes=max_bytes)
        response_url = content.get("url")
        if content.get("redirected") is True or (
            isinstance(response_url, str)
            and response_url
            and response_url != download_url
        ):
            raise _FileError("refused a redirected download response")
        self._require_success(content, "the file download")
        body = self._body(content, max_bytes, limit_error)

        name = safe_file_name(reference.sandbox_path)
        saved_path = self._save(name, body)
        self.saved_bytes += len(body)
        mime_type = metadata.get("mime_type")
        if not isinstance(mime_type, str) or not _MIME_TYPE_PATTERN.match(mime_type):
            mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return GeneratedFile(
            name=saved_path.name,
            sandbox_path=reference.sandbox_path,
            message_id=reference.message_id,
            status="saved",
            path=str(saved_path),
            size_bytes=len(body),
            mime_type=mime_type,
            sha256=hashlib.sha256(body).hexdigest(),
        )

    def _output_directory(self) -> Path:
        if self.directory is not None:
            return self.directory
        root = paths.gptpro_output_dir()
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if root.is_symlink() or not root.is_dir():
                raise _FileError(
                    f"refused the output directory {root}: it is not a real directory"
                )
            stamp = time.strftime("%Y%m%dT%H%M%S")
            self.directory = Path(tempfile.mkdtemp(
                prefix=f"{stamp}-{self.conversation_id[:8]}-", dir=root,
            ))
        except OSError as exc:
            raise _FileError(
                f"could not prepare the output directory {root} "
                f"({exc.strerror or type(exc).__name__})"
            ) from exc
        return self.directory

    def _save(self, name: str, body: bytes) -> Path:
        """Write ``body`` completely, then publish it under an unused name."""
        directory = self._output_directory()
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".partial-", dir=directory
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                return self._publish(Path(temporary_name), directory, name)
            finally:
                os.unlink(temporary_name)
        except OSError as exc:
            raise _FileError(
                f"could not save the file ({exc.strerror or type(exc).__name__})"
            ) from exc

    @staticmethod
    def _publish(temporary: Path, directory: Path, name: str) -> Path:
        # A hard link never replaces an existing entry, unlike a rename.
        stem, suffix = os.path.splitext(name)
        for attempt in range(1, MAX_FILES + 2):
            candidate = directory / (
                name if attempt == 1 else f"{stem}-{attempt}{suffix}"
            )
            try:
                os.link(temporary, candidate)
            except FileExistsError:
                continue
            return candidate
        raise _FileError("could not find an unused file name")


async def collect_generated_files(
    evaluate: Evaluate,
    conversation_id: str | None,
    references: Sequence[SandboxFileReference],
) -> tuple[tuple[GeneratedFile, ...], bool]:
    """Download and save the files a finished answer links to.

    ``evaluate`` runs a page probe on an authenticated chatgpt.com page.
    Returns one entry per reference, in order, and whether every file was
    saved. Only a finished answer's references may be passed.
    """
    if not references:
        return (), True
    results: list[GeneratedFile | None] = [None] * len(references)
    pending: list[int] = []
    for index, reference in enumerate(references):
        if index >= MAX_FILES:
            results[index] = _failed(
                reference,
                f"not downloaded: an answer is limited to {MAX_FILES} files",
            )
        elif not is_valid_sandbox_path(reference.sandbox_path):
            results[index] = _failed(reference, "invalid sandbox path")
        elif not is_conversation_id(reference.message_id):
            results[index] = _failed(reference, "invalid source message ID")
        else:
            pending.append(index)

    if pending and not is_conversation_id(conversation_id):
        for index in pending:
            results[index] = _failed(references[index], "unknown conversation ID")
        pending = []
    if pending:
        assert conversation_id is not None
        collection = _Collection(evaluate, conversation_id.lower())
        try:
            access_token = await collection.fetch_access_token()
        except _FileError as exc:
            for index in pending:
                results[index] = _failed(
                    references[index], f"no download authorization: {exc}"
                )
        else:
            for index in pending:
                try:
                    results[index] = await collection.download(
                        references[index], access_token
                    )
                except _FileError as exc:
                    results[index] = _failed(references[index], str(exc))

    files = tuple(result for result in results if result is not None)
    saved = sum(1 for item in files if item.status == "saved")
    logger.info(
        "gptpro generated files collected (thread=%s saved=%d failed=%d)",
        conversation_id, saved, len(files) - saved,
    )
    for index, item in enumerate(files):
        if item.status == "failed":
            logger.warning(
                "gptpro generated file %d not saved (thread=%s): %s",
                index + 1, conversation_id, item.error,
            )
    return files, saved == len(files)
