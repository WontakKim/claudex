"""Shared HTTP client for daemon-aware CLI commands."""

from __future__ import annotations

import enum
import ipaddress
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from claudex_gateway.cli import daemon
from claudex_gateway.config import GatewayConfig

_PROBE_TIMEOUT = 2.0
_ADMIN_TIMEOUT = 5.0


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow any redirect: the 3xx surfaces as an HTTPError.

    Following a redirect would re-send the request headers — including the
    local bearer token on admin calls — to whatever host the Location header
    names, and would classify the probe from the redirect target instead of
    the actual port occupant.
    """

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _urlopen_no_redirect(request: urllib.request.Request | str, timeout: float) -> Any:
    """`urlopen` for probe/admin calls; every 3xx is a final response."""
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


class ProbeOutcome(enum.Enum):
    """The four-way result of probing GET /api/hello for a compact command.

    IDENTIFIED means a claudex-gateway answered; NO_LISTENER means nothing is
    listening at all; FOREIGN means something answered but is not
    claudex-gateway (wrong port occupant, or an HTTP error status); AMBIGUOUS
    covers everything else (timeout, DNS failure, connection reset before a
    response) where liveness genuinely cannot be determined.
    """

    IDENTIFIED = "identified"
    NO_LISTENER = "no_listener"
    FOREIGN = "foreign"
    AMBIGUOUS = "ambiguous"


def _bracket_host(host: str) -> str:
    """Bracket an IPv6 literal for URL use; IPv4 addresses/hostnames pass through."""
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return host
    return f"[{host}]" if parsed.version == 6 else host


def _http_url(host: str, port: int, path: str) -> str:
    return f"http://{_bracket_host(host)}:{port}{path}"


def _probe_endpoint(config: GatewayConfig) -> tuple[str, int]:
    """Resolve the host/port to probe for a compact command.

    A structurally valid daemon record wins over the config: it names the
    process the launcher itself started, which the config's host/port need
    not match (e.g. CLAUDEX_HOST changed after the daemon started). Absent a
    valid record, the resolved config host/port keeps a foreground daemon
    that never wrote a record file discoverable.
    """
    record, _ = daemon._read_daemon_record()
    if record is not None:
        return daemon._connect_host(record["host"]), record["port"]
    return daemon._connect_host(config.host), config.port


def _classify_daemon(host: str, port: int) -> ProbeOutcome:
    """Probe GET /api/hello and classify the occupant.

    Read-only: this never signals a process (no os.kill), it only issues one
    GET request with a short timeout.
    """
    try:
        with _urlopen_no_redirect(
            _http_url(host, port, "/api/hello"), timeout=_PROBE_TIMEOUT
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError:
        # A completed HTTP response, but an error or redirect status: not a
        # valid hello (redirects are never followed, so a 3xx lands here).
        return ProbeOutcome.FOREIGN
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return ProbeOutcome.NO_LISTENER
        # Timeout, DNS failure, connection reset before a response, etc.
        return ProbeOutcome.AMBIGUOUS
    except OSError:
        # e.g. a connection reset while reading the body, after the connect
        # itself succeeded (so it never went through URLError above).
        return ProbeOutcome.AMBIGUOUS

    try:
        payload = json.loads(raw)
    except ValueError:
        return ProbeOutcome.FOREIGN
    if isinstance(payload, dict) and payload.get("hello") == "claudex-gateway":
        return ProbeOutcome.IDENTIFIED
    return ProbeOutcome.FOREIGN


class _AdminTransportError(Exception):
    """The admin request never got an HTTP response (DNS, reset, timeout, ...)."""


@dataclass(frozen=True)
class _AdminHttpResponse:
    status: int
    body: Any  # parsed JSON value when the body decodes, else None
    detail: str  # bounded diagnostic text: an error message, or raw body text


def _parse_admin_body(raw: bytes) -> tuple[Any, str]:
    """Decode an admin response body; never raises on non-JSON content."""
    text = raw.decode("utf-8", errors="replace")
    bounded = text[:500]
    try:
        body = json.loads(text)
    except ValueError:
        return None, bounded
    detail = bounded
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            detail = error["message"]
        elif isinstance(body.get("detail"), str):
            detail = body["detail"]
    return body, detail


def _admin_request(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    local_token: str | None,
    json_body: dict[str, Any] | None = None,
) -> _AdminHttpResponse:
    """GET/PUT an admin endpoint; raises only for a transport-level failure.

    An HTTP error status is a completed response, not a transport failure:
    urllib.error.HTTPError is caught here and turned into a normal
    _AdminHttpResponse so the caller can inspect a 4xx/5xx body (JSON or
    not) instead of it propagating as an exception.
    """
    data = None
    headers = {"Accept": "application/json"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if local_token:
        headers["Authorization"] = f"Bearer {local_token}"
    request = urllib.request.Request(
        _http_url(host, port, path), data=data, headers=headers, method=method
    )
    try:
        with _urlopen_no_redirect(request, timeout=_ADMIN_TIMEOUT) as response:
            body, detail = _parse_admin_body(response.read())
            return _AdminHttpResponse(status=response.status, body=body, detail=detail)
    except urllib.error.HTTPError as exc:
        body, detail = _parse_admin_body(exc.read())
        return _AdminHttpResponse(status=exc.code, body=body, detail=detail)
    except (urllib.error.URLError, OSError) as exc:
        raise _AdminTransportError(str(exc)) from exc


class _AdminCallOutcome(enum.Enum):
    SUCCESS = "success"
    OLDER_DAEMON = "older_daemon"  # 404/405: the daemon predates the endpoint
    FAILURE = "failure"


def _run_admin_envelope(
    host: str,
    port: int,
    method: str,
    path: str,
    parse_envelope: Any,
    local_token: str | None,
    json_body: dict[str, Any] | None,
) -> tuple[_AdminCallOutcome, dict[str, Any] | None, str]:
    """Run one GET/PUT admin call and classify the result.

    Returns (outcome, envelope, detail): envelope is the parse_envelope-
    validated body only on SUCCESS; detail is diagnostic text for FAILURE
    (including a malformed 2xx envelope, any non-404/405 error status, or a
    transport failure).
    """
    try:
        response = _admin_request(
            host,
            port,
            method,
            path,
            local_token=local_token,
            json_body=json_body,
        )
    except _AdminTransportError as exc:
        return _AdminCallOutcome.FAILURE, None, f"admin request failed: {exc}"

    if response.status in (404, 405):
        return _AdminCallOutcome.OLDER_DAEMON, None, ""
    if 200 <= response.status < 300:
        envelope = parse_envelope(response.body)
        if envelope is not None:
            return _AdminCallOutcome.SUCCESS, envelope, ""
        return (
            _AdminCallOutcome.FAILURE,
            None,
            f"admin returned a malformed response (status {response.status})",
        )
    return (
        _AdminCallOutcome.FAILURE,
        None,
        f"admin request failed (status {response.status}): {response.detail}",
    )
