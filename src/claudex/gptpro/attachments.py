"""Plain-text attachment upload support for ChatGPT Pro asks."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from claudex.gptpro.conversation import is_trusted_origin_url

MAX_ATTACHMENTS_PER_ASK = 10
ATTACH_SETTLE_TIMEOUT_SECONDS = 120.0
ATTACH_SETTLE_POLL_SECONDS = 0.25
FILE_CREATE_PATH = "/backend-api/files"
FILE_PROCESSING_PATH = "/backend-api/files/process_upload_stream"
MAX_PROCESSING_RESPONSE_BYTES = 262_144
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

READ_EXISTING_ATTACHMENT_NAMES_JS = """() => {
  const form = document.querySelector('form[data-chatgpt-composer]');
  if (!form) return [];
  return [...form.querySelectorAll('button[aria-label]')]
    .map(button => button.getAttribute('aria-label'))
    .filter(label => label.startsWith('Remove '))
    .map(label => label.slice('Remove '.length));
}"""

READ_COMPOSER_ATTACHMENT_STATE_JS = r"""(filenames) => {
  const form = document.querySelector('form[data-chatgpt-composer]');
  const state = { ready: {}, processing: {}, failed: {}, unknown: {} };
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
    if (!chip) {
      state.unknown[filename] = (state.unknown[filename] || 0) + 1;
      continue;
    }
    if (seen.has(chip)) continue;
    seen.add(chip);
    const category = statusFor(chip);
    state[category][filename] = (state[category][filename] || 0) + 1;
  }
  return state;
}"""

_monotonic = time.monotonic
_sleep = asyncio.sleep


def _is_valid_composer_attachment_state(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    for category in ("ready", "processing", "failed", "unknown"):
        counts = state.get(category, {} if category == "unknown" else None)
        if not isinstance(counts, dict) or any(
            not isinstance(name, str)
            or type(count) is not int
            or count < 0
            for name, count in counts.items()
        ):
            return False
    return True


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


def _is_file_processing_response(response: Any) -> bool:
    try:
        return (
            response.request.method == "POST"
            and is_trusted_origin_url(response.url)
            and urlsplit(response.url).path.rstrip("/") == FILE_PROCESSING_PATH
        )
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
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, tuple[int, int]]:
    """Upload text files and return per-name (prior, required) ready counts."""
    if not attachment_paths:
        return {}

    descriptors = _load_descriptors(attachment_paths)
    filenames = [descriptor["name"] for descriptor in descriptors]
    expected_counts = Counter(filenames)
    file_handles: list[Any] = []
    data_transfer_handle: Any | None = None
    listener_tasks: set[asyncio.Task[None]] = set()
    completed_file_create_responses = 0
    receipt_file_ids: list[str] = []
    receipt_file_ids_by_name: dict[str, list[str]] = {}
    metadata_failures: list[str] = []
    processing_events: dict[str, dict[str, Any]] = {}
    processing_failure: str | None = None
    processing_pending = 0
    settle_timeout = (
        ATTACH_SETTLE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    deadline = _monotonic() + settle_timeout
    drop_started = False
    ready_attachments = 0

    def report_progress() -> None:
        if on_progress is not None:
            on_progress(completed_file_create_responses, ready_attachments)

    report_progress()

    async def record_completed_response(response: Any) -> None:
        nonlocal completed_file_create_responses, processing_failure
        try:
            await response.finished()
        except Exception:
            return
        completed_file_create_responses += 1
        report_progress()
        response_json = getattr(response, "json", None)
        if not callable(response_json):
            metadata_failures.append("response JSON unavailable")
            return
        try:
            payload = await asyncio.wait_for(response_json(), timeout=2)
        except Exception as exc:
            metadata_failures.append(type(exc).__name__)
            return
        if not isinstance(payload, dict):
            metadata_failures.append("invalid response JSON")
            return
        file_id = payload.get("file_id")
        if not isinstance(file_id, str) or not re.fullmatch(
            r"file_[A-Za-z0-9_-]{1,251}", file_id
        ):
            metadata_failures.append("file ID unavailable")
            return
        if file_id in receipt_file_ids:
            processing_failure = f"duplicate receipt file ID {file_id}"
            return
        receipt_file_ids.append(file_id)
        try:
            request_json = response.request.post_data_json
        except Exception as exc:
            metadata_failures.append(type(exc).__name__)
            return
        filename = request_json.get("file_name") if isinstance(request_json, dict) else None
        if isinstance(filename, str) and filename in expected_counts:
            receipt_file_ids_by_name.setdefault(filename, []).append(file_id)

    async def record_processing_response(response: Any) -> None:
        nonlocal processing_failure, processing_pending
        try:
            await asyncio.wait_for(
                response.finished(), timeout=max(0.001, deadline - _monotonic())
            )
            if not 200 <= response.status < 300:
                processing_failure = "processing response failed"
                return
            body = await asyncio.wait_for(
                response.text(), timeout=max(0.001, deadline - _monotonic())
            )
            if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_PROCESSING_RESPONSE_BYTES:
                processing_failure = "processing response invalid or oversized"
                return
            for line in body.splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    processing_failure = "processing event invalid"
                    return
                if not isinstance(event, dict):
                    processing_failure = "processing event invalid"
                    return
                file_id = event.get("file_id")
                event_name = event.get("event")
                if not isinstance(file_id, str) or not re.fullmatch(
                    r"file_[A-Za-z0-9_-]{1,251}", file_id
                ) or not isinstance(event_name, str):
                    processing_failure = "processing event identity unavailable"
                    return
                details = processing_events.setdefault(file_id, {})
                if event_name == "file.indexing.completed":
                    extra = event.get("extra")
                    display_name = (
                        extra.get("library_file_name")
                        if isinstance(extra, dict) else None
                    )
                    if (
                        not isinstance(display_name, str) or not display_name
                        or len(display_name) > 255
                        or display_name in (".", "..")
                        or any(char in display_name for char in "/\\")
                        or any(ord(char) < 32 or ord(char) == 127 for char in display_name)
                    ):
                        processing_failure = "processing display name invalid"
                        return
                    if details.get("display_name", display_name) != display_name:
                        processing_failure = "processing display name conflicted"
                        return
                    details["display_name"] = display_name
                elif event_name == "file.processing.completed":
                    details["completed"] = True
                elif event_name.endswith((".failed", ".error")):
                    details["failed"] = True
        except Exception as exc:
            processing_failure = f"processing observation failed ({type(exc).__name__})"
        finally:
            processing_pending -= 1

    def on_response(response: Any) -> None:
        nonlocal processing_pending
        if not drop_started:
            return
        if _is_completed_file_create_response(response):
            task = asyncio.create_task(record_completed_response(response))
        elif _is_file_processing_response(response):
            processing_pending += 1
            task = asyncio.create_task(record_processing_response(response))
        else:
            return
        listener_tasks.add(task)
        task.add_done_callback(listener_tasks.discard)

    listener_installed = False
    try:
        existing_names = await page.evaluate(READ_EXISTING_ATTACHMENT_NAMES_JS)
        if not isinstance(existing_names, list) or len(existing_names) > 100 or any(
            not isinstance(name, str) or not name or len(name) > 255
            for name in existing_names
        ):
            raise RuntimeError("Composer existing attachments probe returned invalid names")
        initial_state = await page.evaluate(
            READ_COMPOSER_ATTACHMENT_STATE_JS,
            list(dict.fromkeys([*filenames, *existing_names])),
        )
        if not _is_valid_composer_attachment_state(initial_state):
            raise RuntimeError("Composer attachment probe returned invalid state")
        last_state = initial_state
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

        def initial_count_for(name: str) -> int:
            return sum(
                initial_state.get(category, {}).get(name, 0)
                for category in ("ready", "processing", "failed", "unknown")
            )

        def resolved_names() -> tuple[Counter[str], Counter[str]]:
            names: Counter[str] = Counter()
            completed: Counter[str] = Counter()
            for filename, count in expected_counts.items():
                ids = receipt_file_ids_by_name.get(filename, [])
                for file_id in ids[:count]:
                    details = processing_events.get(file_id)
                    display_name = (
                        details.get("display_name", filename)
                        if details is not None else filename
                    )
                    names[display_name] += 1
                    if details is None or (
                        details.get("completed") and details.get("display_name")
                    ):
                        completed[display_name] += 1
                remainder = count - min(count, len(ids))
                if remainder:
                    names[filename] += remainder
                    completed[filename] += remainder
            return names, completed

        def describe_files() -> str:
            descriptions = []
            for descriptor in descriptors:
                filename = descriptor["name"]
                ids = receipt_file_ids_by_name.get(filename, [])
                display_names = [
                    processing_events.get(file_id, {}).get("display_name", filename)
                    for file_id in ids
                ] or [filename]
                statuses = []
                for display_name in display_names:
                    if last_state["failed"].get(display_name, 0) > initial_state[
                        "failed"
                    ].get(display_name, 0):
                        status = "failed (composer reports an upload error)"
                    elif last_state["processing"].get(display_name, 0) > initial_state[
                        "processing"
                    ].get(display_name, 0):
                        status = "processing (composer reports upload in progress)"
                    elif last_state["ready"].get(display_name, 0) > initial_count_for(
                        display_name
                    ):
                        status = "ready (composer has a file-specific action)"
                    elif last_state.get("unknown", {}).get(display_name, 0):
                        status = "unknown (filename visible without a ready action)"
                    else:
                        status = "unknown (filename not visible in active composer)"
                    if expected_counts[filename] > 1 and 0 < last_state["ready"].get(
                        display_name, 0
                    ) - initial_count_for(display_name) < expected_counts[filename]:
                        status = (
                            f"ambiguous ({last_state['ready'][display_name] - initial_count_for(display_name)}/"
                            f"{expected_counts[filename]} ready)"
                        )
                    statuses.append(status)
                receipt = (
                    ids[0] if len(ids) == 1
                    else "unavailable (not uniquely attributable)"
                )
                display = (
                    f"; composer name(s) {', '.join(repr(name) for name in display_names)}"
                    if display_names != [filename] else ""
                )
                descriptions.append(
                    f"{filename!r}: {statuses[0]}; receipt file ID {receipt}{display}"
                )
            return "; ".join(descriptions)

        def describe_receipts() -> str:
            attributed = {
                file_id for ids in receipt_file_ids_by_name.values()
                for file_id in ids
            }
            unmatched = [file_id for file_id in receipt_file_ids if file_id not in attributed]
            failures = (
                f"; receipt metadata unavailable ({', '.join(sorted(set(metadata_failures)))})"
                if metadata_failures else ""
            )
            return (
                "receipt file IDs (unattributed to filenames): "
                f"{', '.join(unmatched) if unmatched else 'none'}{failures}"
            )

        def create_timeout_error() -> AttachmentSettleTimeoutError:
            expected = ", ".join(repr(filename) for filename in filenames)
            return AttachmentSettleTimeoutError(
                f"Attachments did not settle within {settle_timeout:g} seconds: "
                f"{completed_file_create_responses}/{len(filenames)} completed "
                f"POST {FILE_CREATE_PATH} responses; {ready_attachments}/"
                f"{len(filenames)} ready composer attachments; expected filename "
                f"chips for {expected}. File states: {describe_files()}; "
                f"{describe_receipts()}; no send click attempted by attachment upload.",
                completed_file_create_responses=completed_file_create_responses,
                ready_attachments=ready_attachments,
            )

        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(0)
            required_names, completed_names = resolved_names()
            try:
                state = await asyncio.wait_for(
                    page.evaluate(
                        READ_COMPOSER_ATTACHMENT_STATE_JS, list(required_names)
                    ),
                    timeout=remaining,
                )
            except TimeoutError:
                break
            if not _is_valid_composer_attachment_state(state):
                raise RuntimeError("Composer attachment probe returned invalid state")
            await asyncio.sleep(0)
            last_state = state
            current_names, current_completed = resolved_names()
            if required_names != current_names or completed_names != current_completed:
                continue
            attributed_ids = {
                file_id for ids in receipt_file_ids_by_name.values()
                for file_id in ids
            }
            failed_ids = [
                file_id for file_id in attributed_ids
                if processing_events.get(file_id, {}).get("failed")
            ]
            if processing_failure or failed_ids:
                reason = processing_failure or (
                    "processing failed for " + ", ".join(sorted(failed_ids))
                )
                raise AttachmentUploadFailedError(
                    f"Attachment {reason}. File states: {describe_files()}; "
                    f"{describe_receipts()}; no send click attempted by attachment upload."
                )
            for name in required_names:
                if state["failed"].get(name, 0) > initial_state[
                    "failed"
                ].get(name, 0):
                    raise AttachmentUploadFailedError(
                        f"Composer attachment upload failed for {name!r}. "
                        f"File states: {describe_files()}; {describe_receipts()}; "
                        "no send click attempted by attachment upload."
                    )
            ready_attachments = sum(
                min(count, max(0, state["ready"].get(name, 0) - initial_count_for(name)))
                for name, count in completed_names.items()
            )
            report_progress()
            if (
                completed_file_create_responses >= len(filenames)
                and processing_pending == 0
                and ready_attachments == len(filenames)
            ):
                return {
                    name: (initial_count_for(name), count)
                    for name, count in required_names.items()
                }
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
