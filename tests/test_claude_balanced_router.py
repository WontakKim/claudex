"""Tests for domain-separated session-key derivation and HRW routing math."""

from __future__ import annotations

import hashlib
import hmac
import json

from claudex_gateway.claude_balanced_router import (
    derive_session_key,
    derive_stateless_routing_digest,
    hrw_unit_interval,
)


def _claude_code_user_id(session_id: str, account_uuid: str = "client-account-uuid") -> str:
    """Build the JSON string Claude Code sends as `metadata.user_id`."""
    return json.dumps(
        {"device_id": "d" * 64, "account_uuid": account_uuid, "session_id": session_id},
        separators=(",", ":"),
    )


def _reference_session_key_digest(seed: bytes, kind: bytes, canonical_utf8: bytes) -> bytes:
    """Reimplements the task's documented HMAC framing, independent of the module."""
    frame = (
        b"claudex-session-key-v1\x00"
        + kind
        + b"\x00"
        + len(canonical_utf8).to_bytes(8, "big")
        + canonical_utf8
    )
    return hmac.new(seed, frame, hashlib.sha256).digest()


def test_uuid_branch_is_case_insensitive_and_yields_the_same_digest() -> None:
    seed = b"seed-case-insensitive"
    lower = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
    upper = lower.upper()
    lower_body = {"metadata": {"user_id": _claude_code_user_id(lower)}}
    upper_body = {"metadata": {"user_id": _claude_code_user_id(upper)}}

    lower_key = derive_session_key(lower_body, seed)
    upper_key = derive_session_key(upper_body, seed)

    assert lower_key is not None and upper_key is not None
    assert lower_key.kind == upper_key.kind == "uuid"
    assert lower_key.digest == upper_key.digest
    assert lower_key.digest == _reference_session_key_digest(
        seed, b"uuid", lower.encode("utf-8")
    )


def test_non_rfc4122_variant_falls_back_to_content_hash_branch() -> None:
    seed = b"seed-non-rfc4122"
    ncs_variant_uuid = "11111111-2222-4333-0444-555555555555"
    body = {
        "metadata": {"user_id": _claude_code_user_id(ncs_variant_uuid)},
        "messages": [{"role": "user", "content": "hello"}],
    }

    key = derive_session_key(body, seed)

    assert key is not None
    assert key.kind == "content_hash"


def test_whitespace_padded_uuid_is_rejected_and_falls_back_to_content_hash() -> None:
    seed = b"seed-whitespace"
    padded_uuid = " 3fa85f64-5717-4562-b3fc-2c963f66afa6 "
    body = {
        "metadata": {"user_id": _claude_code_user_id(padded_uuid)},
        "messages": [{"role": "user", "content": "hello"}],
    }

    key = derive_session_key(body, seed)

    assert key is not None
    assert key.kind == "content_hash"


def test_content_hash_digest_is_deterministic_and_key_order_independent() -> None:
    seed = b"seed-content-hash"
    message_in_order = {"role": "user", "content": "café ☕ unicode test", "id": "m-1"}
    message_reordered = {"id": "m-1", "content": "café ☕ unicode test", "role": "user"}
    body_a = {"messages": [message_in_order]}
    body_b = {"messages": [message_reordered]}

    key_a = derive_session_key(body_a, seed)
    key_b = derive_session_key(body_b, seed)

    assert key_a is not None and key_b is not None
    assert key_a.kind == key_b.kind == "content_hash"
    assert key_a.digest == key_b.digest
    expected_canonical_utf8 = json.dumps(
        message_in_order,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert key_a.digest == _reference_session_key_digest(
        seed, b"content_hash", expected_canonical_utf8
    )


def test_no_user_message_and_no_metadata_yields_no_session_key() -> None:
    seed = b"seed-no-session-key"
    body = {"messages": [{"role": "assistant", "content": "hi there"}]}

    assert derive_session_key(body, seed) is None
    assert derive_session_key({}, seed) is None


def test_uuid_and_content_hash_digests_differ_for_identical_underlying_string() -> None:
    seed = b"seed-domain-separation"
    canonical_utf8 = b"identical-underlying-bytes"

    uuid_domain_digest = _reference_session_key_digest(seed, b"uuid", canonical_utf8)
    content_hash_domain_digest = _reference_session_key_digest(
        seed, b"content_hash", canonical_utf8
    )

    assert uuid_domain_digest != content_hash_domain_digest


def test_hrw_unit_interval_stays_in_the_open_interval() -> None:
    seed = b"seed-hrw-range"
    digest = hashlib.sha256(b"a-session-key-digest").digest()

    for account_id in ["account-1", "account-2", "account-3", ""]:
        sample = hrw_unit_interval(seed, digest, account_id)
        assert 0.0 < sample < 1.0


def test_hrw_unit_interval_is_deterministic_under_a_fixed_seed() -> None:
    seed = b"seed-hrw-deterministic"
    digest = hashlib.sha256(b"a-session-key-digest").digest()

    first = hrw_unit_interval(seed, digest, "account-1")
    second = hrw_unit_interval(seed, digest, "account-1")

    assert first == second


def test_hrw_unit_interval_changes_under_a_rotated_seed() -> None:
    digest = hashlib.sha256(b"a-session-key-digest").digest()

    with_original_seed = hrw_unit_interval(b"seed-hrw-original", digest, "account-1")
    with_rotated_seed = hrw_unit_interval(b"seed-hrw-rotated", digest, "account-1")

    assert with_original_seed != with_rotated_seed


def test_stateless_digest_is_deterministic_for_a_fixed_seed_and_nonce() -> None:
    seed = b"seed-stateless"
    nonce = b"n" * 32

    assert derive_stateless_routing_digest(seed, nonce) == derive_stateless_routing_digest(
        seed, nonce
    )


def test_stateless_digest_differs_across_nonces() -> None:
    seed = b"seed-stateless"

    first = derive_stateless_routing_digest(seed, b"n" * 32)
    second = derive_stateless_routing_digest(seed, b"m" * 32)

    assert first != second


def test_stateless_digest_uses_a_distinct_domain_from_session_key_digests() -> None:
    seed = b"seed-stateless-domain"
    nonce = b"n" * 32

    stateless_digest = derive_stateless_routing_digest(seed, nonce)
    uuid_domain_digest = _reference_session_key_digest(seed, b"uuid", nonce)
    content_hash_domain_digest = _reference_session_key_digest(seed, b"content_hash", nonce)

    assert stateless_digest != uuid_domain_digest
    assert stateless_digest != content_hash_domain_digest
