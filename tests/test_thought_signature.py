"""Tests for the Google call-signature carrier envelope codec."""

import base64
import json
import re

import pytest

from claudex.translate.thought_signature import (
    CARRIER_PREFIX,
    MAX_CARRIER_CHARS,
    MAX_ENVELOPE_BYTES,
    MAX_SIGNATURE_BYTES,
    CallSignatureCarrier,
    decode_call_signature_carrier,
    encode_call_signature_carrier,
    is_call_signature_carrier,
)


def _carrier_from_bytes(payload: bytes) -> str:
    body = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return CARRIER_PREFIX + body


def _carrier_from_value(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return _carrier_from_bytes(payload)


def test_encode_decode_round_trip_is_verbatim() -> None:
    provider = "gemini-한글"
    call_id = "toolu_é"
    signature = " \topaque\x00é\n "

    carrier = encode_call_signature_carrier(provider, call_id, signature)

    assert carrier is not None
    assert is_call_signature_carrier(carrier)
    assert "=" not in carrier
    assert decode_call_signature_carrier(carrier) == CallSignatureCarrier(
        provider=provider,
        call_id=call_id,
        signature=signature,
    )


def test_decode_rejects_padded_base64url() -> None:
    carrier = encode_call_signature_carrier("provider", "call", "signature")

    assert carrier is not None
    assert decode_call_signature_carrier(carrier + "=") is None


def test_decode_rejects_non_alphabet_characters() -> None:
    for body in ("ab+c", "ab/c", "ab.c", "ab c", "ab\nc"):
        assert decode_call_signature_carrier(CARRIER_PREFIX + body) is None


def test_decode_rejects_invalid_utf8_payload() -> None:
    assert decode_call_signature_carrier(_carrier_from_bytes(b"\xff")) is None


@pytest.mark.parametrize("payload", [b"not-json", b"null", b"[]", b'"text"', b"1"])
def test_decode_rejects_non_json_and_non_object_payload(payload: bytes) -> None:
    assert decode_call_signature_carrier(_carrier_from_bytes(payload)) is None


def test_decode_rejects_missing_and_extra_keys() -> None:
    envelope = {"v": 1, "provider": "p", "call_id": "c", "sig": "s"}
    for key in envelope:
        missing = dict(envelope)
        missing.pop(key)
        assert decode_call_signature_carrier(_carrier_from_value(missing)) is None

    extra = dict(envelope)
    extra["extra"] = "x"
    assert decode_call_signature_carrier(_carrier_from_value(extra)) is None


@pytest.mark.parametrize(
    "payload",
    [
        b'{"v":1,"v":1,"provider":"p","call_id":"c","sig":"s"}',
        b'{"v":1,"provider":"p","provider":"p","call_id":"c","sig":"s"}',
        b'{"v":1,"provider":"p","call_id":"c","call_id":"c","sig":"s"}',
        b'{"v":1,"provider":"p","call_id":"c","sig":"s","sig":"s"}',
    ],
)
def test_decode_rejects_duplicate_object_members(payload: bytes) -> None:
    assert decode_call_signature_carrier(_carrier_from_bytes(payload)) is None


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_decode_rejects_wrong_version_values(version: object) -> None:
    envelope = {"v": version, "provider": "p", "call_id": "c", "sig": "s"}
    assert decode_call_signature_carrier(_carrier_from_value(envelope)) is None


def test_decode_rejects_empty_fields() -> None:
    for field in ("provider", "call_id", "sig"):
        envelope = {"v": 1, "provider": "p", "call_id": "c", "sig": "s"}
        envelope[field] = ""
        assert decode_call_signature_carrier(_carrier_from_value(envelope)) is None


def test_bounds_reject_oversized_signature_envelope_and_carrier() -> None:
    maximum_signature = "é" * (MAX_SIGNATURE_BYTES // 2)
    maximum_signature_carrier = encode_call_signature_carrier("p", "c", maximum_signature)
    assert maximum_signature_carrier is not None
    assert decode_call_signature_carrier(maximum_signature_carrier) == CallSignatureCarrier(
        provider="p",
        call_id="c",
        signature=maximum_signature,
    )

    assert encode_call_signature_carrier("p", "c", maximum_signature + "é") is None
    assert encode_call_signature_carrier("p" * MAX_ENVELOPE_BYTES, "c", "s") is None
    assert (
        decode_call_signature_carrier(
            _carrier_from_bytes(b" " * (MAX_ENVELOPE_BYTES + 1))
        )
        is None
    )

    oversized_body = "A" * (MAX_CARRIER_CHARS - len(CARRIER_PREFIX) + 1)
    assert decode_call_signature_carrier(CARRIER_PREFIX + oversized_body) is None


@pytest.mark.parametrize(
    ("provider", "call_id", "signature"),
    [
        ("", "c", "s"),
        ("p", "", "s"),
        ("p", "c", ""),
        (None, "c", "s"),
        ("p", True, "s"),
        ("p", "c", b"s"),
    ],
)
def test_encode_fail_closed_returns_none_on_invalid_input(
    provider: object,
    call_id: object,
    signature: object,
) -> None:
    assert encode_call_signature_carrier(provider, call_id, signature) is None


def test_encode_rejects_unencodable_unicode_without_raising() -> None:
    for arguments in (
        ("\ud800", "c", "s"),
        ("p", "\ud800", "s"),
        ("p", "c", "\ud800"),
    ):
        assert encode_call_signature_carrier(*arguments) is None


def test_decode_rejects_escaped_lone_surrogate_without_raising() -> None:
    for field in ("provider", "call_id", "sig"):
        envelope = {"v": 1, "provider": "p", "call_id": "c", "sig": "s"}
        envelope[field] = "\ud800"
        payload = json.dumps(envelope, ensure_ascii=True, separators=(",", ":")).encode(
            "ascii"
        )
        assert decode_call_signature_carrier(_carrier_from_bytes(payload)) is None


def test_decode_rejects_pathologically_nested_json_without_raising() -> None:
    payload = (
        b'{"v":1,"provider":"p","call_id":"c","sig":'
        + b"[" * 5000
        + b"0"
        + b"]" * 5000
        + b"}"
    )

    assert len(payload) <= MAX_ENVELOPE_BYTES
    assert decode_call_signature_carrier(_carrier_from_bytes(payload)) is None


def test_prefix_does_not_match_gpt_fernet_signature_regex() -> None:
    fernet_signature = re.compile(r"^gAAAA[A-Za-z0-9_=-]+$")
    carrier = encode_call_signature_carrier("p", "c", "s")

    assert carrier is not None
    assert fernet_signature.fullmatch(CARRIER_PREFIX) is None
    assert fernet_signature.fullmatch(carrier) is None
