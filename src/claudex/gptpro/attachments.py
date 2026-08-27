"""Plain-text attachment upload support for ChatGPT Pro asks."""

from __future__ import annotations

import asyncio
import base64
import inspect
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_ATTACHMENTS_PER_ASK = 10
ATTACH_SETTLE_TIMEOUT_SECONDS = 120.0
ATTACH_SETTLE_POLL_SECONDS = 0.25
FILE_CREATE_PATH = "/backend-api/files"
MAX_TOTAL_ATTACHMENT_BYTES = 1_200_000

_PLAIN_TEXT_MIME = "text/plain"

CREATE_ATTACHMENT_FILE_JS = """({ bytesBase64, name, mime }) => {
  const binary = atob(bytesBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new File([bytes], name, { type: mime });
}"""

CREATE_ATTACHMENT_DATA_TRANSFER_JS = """(files) => {
  const dataTransfer = new DataTransfer();
  for (const file of files) dataTransfer.items.add(file);
  return dataTransfer;
}"""

DISPATCH_ATTACHMENT_DROP_JS = """(dataTransfer) => {
  const preferredSelector = '#thread-bottom-container';
  const fallbackSelector = 'main';
  const target =
    document.querySelector(preferredSelector) ??
    document.querySelector(fallbackSelector);
  if (!target) {
    throw new Error(
      `Attachment drop target not found (${preferredSelector} or ${fallbackSelector}).`,
    );
  }

  for (const eventType of ['dragenter', 'dragover', 'drop']) {
    target.dispatchEvent(
      new DragEvent(eventType, {
        bubbles: true,
        cancelable: true,
        composed: true,
        dataTransfer,
      }),
    );
  }
  return target.matches(preferredSelector)
    ? preferredSelector
    : fallbackSelector;
}"""

READ_BODY_INNER_TEXT_JS = "() => document.body?.innerText ?? ''"

_monotonic = time.monotonic
_sleep = asyncio.sleep


class AttachmentSettleTimeoutError(TimeoutError):
    """The upload responses or visible filename chips did not settle."""


def _load_descriptors(attachment_paths: Sequence[str]) -> list[dict[str, str]]:
    if len(attachment_paths) > MAX_ATTACHMENTS_PER_ASK:
        raise ValueError(
            f"At most {MAX_ATTACHMENTS_PER_ASK} attachments may be sent in one "
            f"ask; received {len(attachment_paths)}."
        )

    loaded: list[tuple[str, bytes, str]] = []
    total_bytes = 0
    for attachment_path in attachment_paths:
        data = Path(attachment_path).read_bytes()
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachments total {total_bytes} bytes exceeds the "
                f"{MAX_TOTAL_ATTACHMENT_BYTES}-byte limit per ask."
            )
        loaded.append((attachment_path, data, Path(attachment_path).name))

    descriptors: list[dict[str, str]] = []
    for attachment_path, data, filename in loaded:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Attachment must be UTF-8 plain text: {attachment_path!r} "
                "(invalid UTF-8)."
            ) from exc
        if b"\x00" in data:
            raise ValueError(
                f"Attachment must be UTF-8 plain text: {attachment_path!r} "
                "(contains NUL bytes)."
            )
        descriptors.append(
            {
                "bytesBase64": base64.b64encode(data).decode("ascii"),
                "name": filename,
                "mime": _PLAIN_TEXT_MIME,
            }
        )
    return descriptors


def _is_completed_file_create_response(response: Any) -> bool:
    try:
        request = response.request
        if request.method != "POST" or not 200 <= response.status < 300:
            return False
        pathname = urlsplit(response.url).path.rstrip("/")
        return pathname == FILE_CREATE_PATH
    except (AttributeError, TypeError, ValueError):
        return False


async def _remove_response_listener(page: Any, listener: Any) -> None:
    try:
        remove_listener = getattr(page, "remove_listener", None)
        if not callable(remove_listener):
            remove_listener = getattr(page, "off", None)
        if not callable(remove_listener):
            return
        result = remove_listener("response", listener)
        if inspect.isawaitable(result):
            await result
    except Exception:
        return


async def _dispose_handle(handle: Any) -> None:
    try:
        await handle.dispose()
    except Exception:
        return


async def attach_files(
    page: Any,
    attachment_paths: Sequence[str],
    *,
    timeout_seconds: float | None = None,
) -> None:
    """Upload UTF-8 plain-text files and wait for responses and filename chips."""
    if not attachment_paths:
        return

    descriptors = _load_descriptors(attachment_paths)
    filenames = [descriptor["name"] for descriptor in descriptors]
    file_handles: list[Any] = []
    data_transfer_handle: Any | None = None
    listener_tasks: set[asyncio.Task[None]] = set()
    completed_file_create_responses = 0

    async def record_completed_response(response: Any) -> None:
        nonlocal completed_file_create_responses
        try:
            await response.finished()
        except Exception:
            return
        completed_file_create_responses += 1

    def on_response(response: Any) -> None:
        if not _is_completed_file_create_response(response):
            return
        task = asyncio.create_task(record_completed_response(response))
        listener_tasks.add(task)
        task.add_done_callback(listener_tasks.discard)

    listener_installed = False
    try:
        page.on("response", on_response)
        listener_installed = True
        for descriptor in descriptors:
            file_handles.append(
                await page.evaluate_handle(CREATE_ATTACHMENT_FILE_JS, descriptor)
            )
        data_transfer_handle = await page.evaluate_handle(
            CREATE_ATTACHMENT_DATA_TRANSFER_JS, file_handles
        )
        await page.evaluate(
            DISPATCH_ATTACHMENT_DROP_JS, data_transfer_handle
        )

        settle_timeout = (
            ATTACH_SETTLE_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        deadline = _monotonic() + settle_timeout

        def create_timeout_error() -> AttachmentSettleTimeoutError:
            expected = ", ".join(repr(filename) for filename in filenames)
            return AttachmentSettleTimeoutError(
                f"Attachments did not settle within {settle_timeout:g} seconds: "
                f"{completed_file_create_responses}/{len(filenames)} completed "
                f"POST {FILE_CREATE_PATH} responses; expected filename chips "
                f"for {expected}."
            )

        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            try:
                body_text_result = await asyncio.wait_for(
                    page.evaluate(READ_BODY_INNER_TEXT_JS),
                    timeout=remaining,
                )
            except TimeoutError:
                break
            body_text = (
                body_text_result if isinstance(body_text_result, str) else ""
            )
            await asyncio.sleep(0)
            if (
                completed_file_create_responses >= len(filenames)
                and all(filename in body_text for filename in filenames)
            ):
                return
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            await _sleep(min(ATTACH_SETTLE_POLL_SECONDS, remaining))

        raise create_timeout_error()
    finally:
        if listener_installed:
            await _remove_response_listener(page, on_response)
        tasks = tuple(listener_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if data_transfer_handle is not None:
            await _dispose_handle(data_transfer_handle)
        for file_handle in file_handles:
            await _dispose_handle(file_handle)
