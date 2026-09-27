"""Plain-text attachment upload support for ChatGPT Pro asks."""

from __future__ import annotations

import asyncio
import base64
import inspect
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from claudex.gptpro.conversation import is_trusted_origin_url

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
  const preferredSelector = 'form[data-chatgpt-composer]';
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

READ_COMPOSER_ATTACHMENT_STATE_JS = r"""(filenames) => {
  const form = document.querySelector('form[data-chatgpt-composer]');
  const state = { ready: {}, processing: {}, failed: {} };
  if (!form) return state;
  const expected = new Set(filenames);
  const attributes = ['title', 'aria-label', 'data-filename'];
  const hasFilenameAttribute = (node, filename) =>
    attributes.some((attribute) => node.getAttribute(attribute) === filename);
  const candidates = Array.from(form.querySelectorAll('*'))
    .filter((node) => node.tagName !== 'BUTTON')
    .filter((node) => !node.closest('[contenteditable], [role="textbox"]'))
    .filter((node) => expected.has((node.textContent || '').trim()) ||
      filenames.some((name) => hasFilenameAttribute(node, name)));
  const labels = candidates.filter((node) =>
    !candidates.some((child) => child !== node && node.contains(child))
  );
  const seen = new Set();
  for (const node of labels) {
    const filename = filenames.find((name) =>
      hasFilenameAttribute(node, name) || (node.textContent || '').trim() === name
    );
    const statusFor = (chip) => {
      const statusText = (chip.textContent || '').replace(filename, '').trim();
      if (/^(upload failed|failed|error)\b/i.test(statusText) ||
          [node, chip].some((element) =>
            /^(failed|error)$/i.test(element.getAttribute('data-state') || '') ||
            element.getAttribute('aria-invalid') === 'true')) return 'failed';
      if (/^(uploading|processing|pending)\b/i.test(statusText) ||
          [node, chip].some((element) =>
            element.getAttribute('aria-busy') === 'true' ||
            /^(uploading|processing|pending)$/i.test(element.getAttribute('data-state') || '')) ||
          chip.querySelector(':scope > [role="progressbar"]')) return 'processing';
      return 'ready';
    };
    let chip = node.parentElement === form ? node : node.parentElement;
    const fallback = hasFilenameAttribute(node, filename) ? chip : null;
    let confirmed = false;
    for (let depth = 0; depth < 2 && chip && chip !== form; depth += 1) {
      const parent = chip.parentElement;
      const canInspectParent = depth === 0 && parent && parent !== form &&
        !labels.some((other) => other !== node && parent.contains(other));
      if (canInspectParent && statusFor(parent) !== 'ready') {
        chip = parent;
        confirmed = true;
        break;
      }
      const hasAction = Array.from(chip.children).some((child) =>
        child.tagName === 'BUTTON' &&
        (child.getAttribute('aria-label') === filename ||
         /remove|delete|close/i.test(child.getAttribute('aria-label') || ''))
      );
      if (hasAction || statusFor(chip) !== 'ready' ||
          hasFilenameAttribute(chip, filename)) {
        confirmed = true;
        break;
      }
      if (!canInspectParent) break;
      chip = parent;
    }
    if (!confirmed) chip = fallback;
    if (!chip || seen.has(chip)) continue;
    seen.add(chip);
    const category = statusFor(chip);
    state[category][filename] = (state[category][filename] || 0) + 1;
  }
  return state;
}"""

_monotonic = time.monotonic
_sleep = asyncio.sleep


class AttachmentSettleTimeoutError(TimeoutError):
    """The upload responses or composer attachment chips did not settle."""

    def __init__(
        self, message: str, *, completed_file_create_responses: int,
        ready_attachments: int,
    ) -> None:
        super().__init__(message)
        self.completed_file_create_responses = completed_file_create_responses
        self.ready_attachments = ready_attachments


class AttachmentUploadFailedError(RuntimeError):
    """A newly attached composer chip reports an upload failure."""


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
        if not is_trusted_origin_url(response.url):
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
    """Upload UTF-8 text files and wait for receipts and ready composer chips."""
    if not attachment_paths:
        return

    descriptors = _load_descriptors(attachment_paths)
    filenames = [descriptor["name"] for descriptor in descriptors]
    expected_counts = Counter(filenames)
    file_handles: list[Any] = []
    data_transfer_handle: Any | None = None
    listener_tasks: set[asyncio.Task[None]] = set()
    completed_file_create_responses = 0
    drop_started = False
    ready_attachments = 0

    async def record_completed_response(response: Any) -> None:
        nonlocal completed_file_create_responses
        try:
            await response.finished()
        except Exception:
            return
        completed_file_create_responses += 1

    def on_response(response: Any) -> None:
        if not drop_started or not _is_completed_file_create_response(response):
            return
        task = asyncio.create_task(record_completed_response(response))
        listener_tasks.add(task)
        task.add_done_callback(listener_tasks.discard)

    listener_installed = False
    try:
        initial_state = await page.evaluate(
            READ_COMPOSER_ATTACHMENT_STATE_JS, filenames
        )
        if not isinstance(initial_state, dict) or not all(
            isinstance(initial_state.get(category), dict)
            for category in ("ready", "processing", "failed")
        ):
            raise RuntimeError("Composer attachment probe returned invalid state")
        page.on("response", on_response)
        listener_installed = True
        for descriptor in descriptors:
            file_handles.append(
                await page.evaluate_handle(CREATE_ATTACHMENT_FILE_JS, descriptor)
            )
        data_transfer_handle = await page.evaluate_handle(
            CREATE_ATTACHMENT_DATA_TRANSFER_JS, file_handles
        )
        drop_started = True
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
                f"POST {FILE_CREATE_PATH} responses; {ready_attachments}/"
                f"{len(filenames)} ready composer attachments; expected filename "
                f"chips for {expected}.",
                completed_file_create_responses=completed_file_create_responses,
                ready_attachments=ready_attachments,
            )

        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            try:
                state = await asyncio.wait_for(
                    page.evaluate(READ_COMPOSER_ATTACHMENT_STATE_JS, filenames),
                    timeout=remaining,
                )
            except TimeoutError:
                break
            if not isinstance(state, dict) or not all(
                isinstance(state.get(category), dict)
                for category in ("ready", "processing", "failed")
            ):
                raise RuntimeError("Composer attachment probe returned invalid state")
            for filename in expected_counts:
                if state["failed"].get(filename, 0) > initial_state[
                    "failed"
                ].get(filename, 0):
                    raise AttachmentUploadFailedError(
                        f"Composer attachment upload failed for {filename!r}."
                    )
            ready_attachments = sum(
                min(
                    count,
                    max(
                        0,
                        state["ready"].get(filename, 0)
                        - sum(
                            initial_state[category].get(filename, 0)
                            for category in ("ready", "processing", "failed")
                        ),
                    ),
                )
                for filename, count in expected_counts.items()
            )
            await asyncio.sleep(0)
            if (
                completed_file_create_responses >= len(filenames)
                and ready_attachments == len(filenames)
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
