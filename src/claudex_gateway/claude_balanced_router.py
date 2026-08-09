"""Domain-separated session-key derivation for balanced (session-affinity) routing.

Design v2 §2.4/§5.1 (adjudications Q3, H): session keys are HMAC-SHA256
digests under the durable per-deployment epoch seed. Raw Claude Code UUIDs
and message content are never stored — only the digest. Two branches decide
what gets hashed, tried in order:

* uuid — Claude Code's `metadata.user_id` is a JSON string embedding a
  `session_id` (the same field `server._rewrite_metadata_account_uuid`
  parses for `account_uuid` rewriting). A candidate is accepted only when it
  round-trips through `uuid.UUID` with no leading/trailing whitespace and
  carries the RFC 4122 variant; the canonical `str(uuid.UUID(...))` form is
  hashed, so equivalent-but-differently-cased input yields the same digest.
* content_hash — used whenever the uuid branch is unavailable (missing
  metadata, non-string/malformed/non-RFC-4122 `session_id`, ...): the
  complete first `messages` element whose `role` is exactly `"user"` (the
  client's own object, before any gateway mutation) is canonicalized to JSON
  and hashed instead.

A request with neither a usable `session_id` nor a first user message has no
session key at all — the caller must treat it as unpinnable and route it
statelessly via `derive_stateless_routing_digest`.

`hrw_unit_interval` (§2.4) is the separate rendezvous-hashing (Highest
Random Weight) sample: a per-(session, account) digest reduced to a
uniformly distributed point in the open interval (0, 1), so the account with
the largest sample can be picked without keeping any pin-map state.
"""

from __future__ import annotations

import hmac
import json
import uuid
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

_SESSION_KEY_DOMAIN = b"claudex-session-key-v1"
_HRW_DOMAIN = b"claudex-balanced-hrw-v1"
_STATELESS_REQUEST_DOMAIN = b"claudex-stateless-request-v1"


@dataclass(frozen=True)
class SessionKey:
    """A domain-separated session-affinity digest and the branch that produced it."""

    digest: bytes
    kind: Literal["uuid", "content_hash"]


def _hmac_sha256(seed: bytes, message: bytes) -> bytes:
    return hmac.new(seed, message, sha256).digest()


def _length_prefixed(*fields: bytes) -> bytes:
    """Concatenate `fields`, each preceded by its 8-byte big-endian length.

    Length-prefixing keeps concatenated variable-length fields unambiguous:
    without it, hashing `("ab", "c")` and `("a", "bc")` would collide.
    """
    framed = bytearray()
    for field in fields:
        framed += len(field).to_bytes(8, "big")
        framed += field
    return bytes(framed)


def _uuid_session_key(body: dict[str, Any], seed: bytes) -> SessionKey | None:
    """Try the uuid branch; None on anything that isn't a clean RFC 4122 session_id."""
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get("user_id")
    if not isinstance(user_id, str):
        return None
    try:
        parsed_user_id = json.loads(user_id)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed_user_id, dict):
        return None
    candidate = parsed_user_id.get("session_id")
    if not isinstance(candidate, str) or candidate != candidate.strip():
        return None
    try:
        parsed = uuid.UUID(candidate)
    except ValueError:
        return None
    if parsed.variant != uuid.RFC_4122:
        return None
    canonical_utf8 = str(parsed).encode("utf-8")
    digest = _hmac_sha256(
        seed, _SESSION_KEY_DOMAIN + b"\x00uuid\x00" + _length_prefixed(canonical_utf8)
    )
    return SessionKey(digest=digest, kind="uuid")


def _first_user_message(body: dict[str, Any]) -> dict[str, Any] | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def _content_hash_session_key(body: dict[str, Any], seed: bytes) -> SessionKey | None:
    """Try the content-hash branch; None when the body has no user message."""
    first_user_message = _first_user_message(body)
    if first_user_message is None:
        return None
    canonical_utf8 = json.dumps(
        first_user_message,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = _hmac_sha256(
        seed,
        _SESSION_KEY_DOMAIN + b"\x00content_hash\x00" + _length_prefixed(canonical_utf8),
    )
    return SessionKey(digest=digest, kind="content_hash")


def derive_session_key(body: dict[str, Any], seed: bytes) -> SessionKey | None:
    """Derive a session-affinity key for `body`, or None when it is unpinnable.

    Tries the uuid branch first (Claude Code's own `session_id`), then falls
    back to hashing the first user message. Returns None only when neither
    branch has anything to hash.
    """
    return _uuid_session_key(body, seed) or _content_hash_session_key(body, seed)


def hrw_unit_interval(seed: bytes, session_key_digest: bytes, account_id: str) -> float:
    """Map (seed, session_key_digest, account_id) onto the open interval (0, 1).

    Highest-Random-Weight sampling (design v2 §2.4) routes a session to the
    account with the largest sample; the mapping is deterministic and needs
    no stored state to keep a session pinned to the same account run after
    run. `mac`'s high 53 bits give the full double-precision mantissa worth
    of entropy, offset by 0.5 so the interval stays strictly open.
    """
    mac = _hmac_sha256(
        seed,
        _HRW_DOMAIN
        + b"\x00"
        + _length_prefixed(session_key_digest, account_id.encode("utf-8")),
    )
    high_53_bits = int.from_bytes(mac, "big") >> (8 * len(mac) - 53)
    return (high_53_bits + 0.5) / 2**53


def derive_stateless_routing_digest(seed: bytes, nonce: bytes) -> bytes:
    """HMAC digest identifying one stateless (unpinnable) request's retry chain.

    The caller supplies one fresh 32-byte random nonce per request; the
    resulting digest is reused across that request's retries, never
    persisted, and never creates a pin-map entry.
    """
    return _hmac_sha256(seed, _STATELESS_REQUEST_DOMAIN + b"\x00" + _length_prefixed(nonce))
