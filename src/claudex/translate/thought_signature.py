"""Encode and decode Google call signatures carried through Claude thinking blocks."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import re
from typing import Any

CARRIER_PREFIX = "claudex-google-call-signature-v1:"
MAX_SIGNATURE_BYTES = 16384
MAX_ENVELOPE_BYTES = 20480
MAX_CARRIER_CHARS = 32768

_BASE64URL_BODY_RE = re.compile(r"[A-Za-z0-9_-]+")
_ENVELOPE_KEYS = frozenset({"v", "provider", "call_id", "sig"})


@dataclass(frozen=True)
class CallSignatureCarrier:
    provider: str
    call_id: str
    signature: str


def is_call_signature_carrier(text: object) -> bool:
    return isinstance(text, str) and text.startswith(CARRIER_PREFIX)


def encode_call_signature_carrier(
    provider: object,
    call_id: object,
    signature: object,
) -> str | None:
    values = (provider, call_id, signature)
    if any(not isinstance(value, str) or not value for value in values):
        return None

    try:
        provider.encode("utf-8")
        call_id.encode("utf-8")
        signature_bytes = signature.encode("utf-8")
    except UnicodeEncodeError:
        return None

    if len(signature_bytes) > MAX_SIGNATURE_BYTES:
        return None

    # The provider name is the replay boundary, so renaming a custom provider
    # intentionally invalidates carriers issued under its previous name.
    envelope = {
        "v": 1,
        "provider": provider,
        "call_id": call_id,
        "sig": signature,
    }
    envelope_bytes = json.dumps(
        envelope,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(envelope_bytes) > MAX_ENVELOPE_BYTES:
        return None

    body = base64.urlsafe_b64encode(envelope_bytes).decode("ascii").rstrip("=")
    carrier = CARRIER_PREFIX + body
    if len(carrier) > MAX_CARRIER_CHARS:
        return None
    return carrier


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def decode_call_signature_carrier(text: object) -> CallSignatureCarrier | None:
    if not is_call_signature_carrier(text):
        return None
    if len(text) > MAX_CARRIER_CHARS:
        return None

    body = text[len(CARRIER_PREFIX) :]
    if _BASE64URL_BODY_RE.fullmatch(body) is None:
        return None

    padded_body = body + "=" * (-len(body) % 4)
    try:
        envelope_bytes = base64.b64decode(
            padded_body,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError):
        return None

    canonical_body = base64.urlsafe_b64encode(envelope_bytes).decode("ascii").rstrip("=")
    if canonical_body != body or len(envelope_bytes) > MAX_ENVELOPE_BYTES:
        return None

    try:
        envelope_text = envelope_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    try:
        envelope = json.loads(
            envelope_text,
            object_pairs_hook=_reject_duplicate_members,
        )
    except (ValueError, RecursionError):
        return None

    if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_KEYS:
        return None
    if type(envelope["v"]) is not int or envelope["v"] != 1:
        return None

    provider = envelope["provider"]
    call_id = envelope["call_id"]
    signature = envelope["sig"]
    values = (provider, call_id, signature)
    if any(not isinstance(value, str) or not value for value in values):
        return None

    try:
        provider.encode("utf-8")
        call_id.encode("utf-8")
        signature_bytes = signature.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(signature_bytes) > MAX_SIGNATURE_BYTES:
        return None

    return CallSignatureCarrier(
        provider=provider,
        call_id=call_id,
        signature=signature,
    )
