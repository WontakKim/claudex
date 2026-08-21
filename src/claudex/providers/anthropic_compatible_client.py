"""HTTP transport for static Anthropic Messages-compatible backends."""

from __future__ import annotations

import contextlib
import json
from typing import Any, NoReturn

import httpx

from claudex.config.schema import AnthropicCompatibleProvider
from claudex.upstream_errors import UpstreamError


class AnthropicCompatibleUpstreamError(UpstreamError):
    """Raised when a custom Messages backend returns a non-success response."""


class AnthropicCompatibleClient:
    """Send Messages requests with one configured static Bearer credential."""

    def __init__(
        self,
        name: str,
        provider: AnthropicCompatibleProvider,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._api_key = provider.api_key
        redacted_name = self._redact_api_key(name)
        self._name = redacted_name.encode("unicode_escape").decode("ascii")
        if "?" in provider.base_url or "#" in provider.base_url:
            suffix = "query" if "?" in provider.base_url else "fragment"
            raise ValueError(
                "Anthropic-compatible base_url must be a versioned API prefix "
                f"without a query or fragment; found {suffix}"
            )
        self._base_url = provider.base_url.rstrip("/")
        self._http_client = http_client

    async def send_messages(
        self, body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        """POST one request and transfer an open successful response to the caller."""
        request_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"authorization", "x-api-key"}
        }
        request_headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            request = self._http_client.build_request(
                "POST",
                f"{self._base_url}/messages",
                content=body,
                headers=request_headers,
            )
            response = await self._http_client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise self._sanitize_http_error(exc) from None

        if 200 <= response.status_code < 300:
            return response

        try:
            upstream_body = (await self._read_without_closing(response)).decode(
                "utf-8", errors="replace"
            )
            primary_error: BaseException = AnthropicCompatibleUpstreamError(
                response.status_code,
                self._redact_api_key(upstream_body),
                self._name,
            )
        except httpx.HTTPError as exc:
            primary_error = self._sanitize_http_error(exc)
        except BaseException as exc:
            primary_error = exc
        await self._close_and_raise(response, primary_error)

    @staticmethod
    async def _read_without_closing(response: httpx.Response) -> bytes:
        """Read a response while reserving its one close attempt for cleanup."""
        close_response = response.aclose

        async def defer_automatic_close() -> None:
            return None

        response.aclose = defer_automatic_close
        try:
            return await response.aread()
        finally:
            response.aclose = close_response

    async def _close_and_raise(
        self, response: httpx.Response, primary_error: BaseException
    ) -> NoReturn:
        cleanup_error: BaseException | None = None
        try:
            await response.aclose()
        except BaseException as exc:
            cleanup_error = self._sanitize_exception(exc)

        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise primary_error from None

    def _redact_api_key(self, text: str) -> str:
        if not self._api_key:
            return text

        try:
            parsed = json.loads(text)
        except (ValueError, RecursionError):
            return self._replace_api_key_forms(text)

        try:
            redacted, was_changed = self._redact_json_value(parsed)
        except RecursionError:
            return self._replace_api_key_forms(text)
        if not was_changed:
            return self._replace_api_key_forms(text)
        try:
            serialized = json.dumps(redacted, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
            return self._replace_api_key_forms(text)
        return self._replace_api_key_forms(serialized)

    def _redact_json_value(self, value: Any) -> tuple[Any, bool]:
        if isinstance(value, str):
            redacted = value.replace(self._api_key, self._redaction_marker())
            return redacted, redacted != value
        if isinstance(value, list):
            result: list[Any] = []
            was_changed = False
            for item in value:
                redacted, item_changed = self._redact_json_value(item)
                result.append(redacted)
                was_changed = was_changed or item_changed
            return result, was_changed
        if isinstance(value, dict):
            result: dict[Any, Any] = {}
            was_changed = False
            for key, item in value.items():
                redacted_key, key_changed = self._redact_json_value(key)
                redacted_item, item_changed = self._redact_json_value(item)
                result[redacted_key] = redacted_item
                was_changed = was_changed or key_changed or item_changed
            return result, was_changed
        return value, False

    def _replace_api_key_forms(self, text: str) -> str:
        marker = self._redaction_marker()
        for candidate in self._api_key_forms():
            text = text.replace(candidate, marker)
        return text

    def _api_key_forms(self) -> tuple[str, ...]:
        unicode_escaped = self._api_key.encode("unicode_escape").decode("ascii")
        utf8_bytes_repr = repr(self._api_key.encode("utf-8"))
        bearer_value = f"Bearer {self._api_key}"
        bearer_bytes_repr = repr(bearer_value.encode("utf-8"))
        forms = {
            self._api_key,
            json.dumps(self._api_key, ensure_ascii=False)[1:-1],
            json.dumps(self._api_key, ensure_ascii=True)[1:-1],
            unicode_escaped,
            repr(self._api_key),
            repr(self._api_key)[1:-1],
            ascii(self._api_key),
            ascii(self._api_key)[1:-1],
            utf8_bytes_repr,
            utf8_bytes_repr[2:-1],
            repr(bearer_value),
            repr(bearer_value)[1:-1],
            bearer_bytes_repr,
            bearer_bytes_repr[2:-1],
        }
        return tuple(sorted((form for form in forms if form), key=len, reverse=True))

    def _redaction_marker(self) -> str:
        marker = "[REDACTED]"
        if self._api_key not in marker:
            return marker
        return "*" if self._api_key != "*" else "?"

    def _sanitize_http_error(self, exc: httpx.HTTPError) -> httpx.HTTPError:
        sanitized = self._sanitize_exception(exc)
        if isinstance(sanitized, httpx.HTTPError):
            return sanitized
        return httpx.HTTPError(self._redaction_marker())

    def _sanitize_exception(self, exc: BaseException) -> BaseException | None:
        try:
            exc.args = tuple(self._sanitize_exception_arg(arg) for arg in exc.args)
            representations = (str(exc), repr(exc))
        except BaseException:
            return None
        if any(self._redact_api_key(value) != value for value in representations):
            return None
        return exc

    def _sanitize_exception_arg(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_api_key(value)
        if isinstance(value, bytes):
            result = value
            marker = self._redaction_marker().encode()
            for candidate in self._api_key_forms():
                with contextlib.suppress(UnicodeEncodeError):
                    result = result.replace(candidate.encode(), marker)
            return result
        return value
