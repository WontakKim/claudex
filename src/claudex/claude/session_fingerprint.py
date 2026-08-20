"""Shared session UUID extraction and observability fingerprint persistence."""

from __future__ import annotations

import hmac
import json
import logging
import os
import tempfile
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSION_FINGERPRINT_DOMAIN = b"claudex-session-fingerprint-v1"

_SEED_FILENAME = "session-fingerprint-seed"
_SEED_BYTE_LENGTH = 32
_SEED_HEX_LENGTH = _SEED_BYTE_LENGTH * 2
_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")


def extract_session_uuid(body: dict[str, Any]) -> tuple[str, str] | None:
    """Return the raw and canonical RFC 4122 session UUID accepted for routing."""
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
    return candidate, str(parsed)


def observability_session_fingerprint(seed: bytes, canonical_uuid: str) -> str:
    """Return a domain-separated HMAC-SHA256 fingerprint for a canonical UUID."""
    message = SESSION_FINGERPRINT_DOMAIN + b"\x00" + canonical_uuid.encode("utf-8")
    return hmac.new(seed, message, sha256).hexdigest()


def _read_fingerprint_seed(seed_path: Path) -> bytes:
    payload = seed_path.read_text(encoding="ascii")
    if (
        len(payload) != _SEED_HEX_LENGTH
        or any(character not in _LOWERCASE_HEX_DIGITS for character in payload)
    ):
        raise ValueError("invalid session fingerprint seed")
    return bytes.fromhex(payload)


def _warn_seed_unavailable(seed_path: Path, reason: str) -> None:
    logger.warning(
        "Session fingerprint seed at %s %s; session fingerprints are disabled",
        seed_path,
        reason,
    )


def _discard_temporary_seed(temp_path: Path) -> bool:
    try:
        temp_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _load_winning_seed(seed_path: Path) -> bytes | None:
    try:
        return _read_fingerprint_seed(seed_path)
    except (OSError, UnicodeError, ValueError):
        _warn_seed_unavailable(seed_path, "could not be read or validated")
        return None


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_fingerprint_seed(pool_dir: Path, seed_path: Path) -> bytes | None:
    try:
        pool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        seed = os.urandom(_SEED_BYTE_LENGTH)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{_SEED_FILENAME}.tmp-", dir=pool_dir
        )
    except OSError:
        _warn_seed_unavailable(seed_path, "could not be created")
        return None

    temp_path = Path(temp_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as temp_file:
            file_descriptor = -1
            temp_file.write(seed.hex().encode("ascii"))
            temp_file.flush()
            os.fsync(temp_file.fileno())
    except OSError:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        _discard_temporary_seed(temp_path)
        _warn_seed_unavailable(seed_path, "could not be created")
        return None

    publication_lost = False
    try:
        os.link(temp_path, seed_path)
    except FileExistsError:
        publication_lost = True
    except OSError:
        _discard_temporary_seed(temp_path)
        _warn_seed_unavailable(seed_path, "could not be published")
        return None

    # fsync the directory so the linked seed entry survives a crash; the
    # temporary entry's durability does not matter. The losing publisher
    # cannot assume the winner already synchronized, so both paths sync.
    try:
        _fsync_directory(pool_dir)
    except OSError:
        _discard_temporary_seed(temp_path)
        _warn_seed_unavailable(seed_path, "could not be made durable")
        return None

    if not _discard_temporary_seed(temp_path):
        _warn_seed_unavailable(seed_path, "left an unreadable temporary file")
        return None
    if publication_lost:
        return _load_winning_seed(seed_path)
    return seed


def load_or_create_fingerprint_seed(pool_dir: Path) -> bytes | None:
    """Load the fallback fingerprint seed, atomically creating it when absent."""
    seed_path = pool_dir / _SEED_FILENAME
    try:
        return _read_fingerprint_seed(seed_path)
    except FileNotFoundError:
        return _create_fingerprint_seed(pool_dir, seed_path)
    except (OSError, UnicodeError, ValueError):
        _warn_seed_unavailable(seed_path, "could not be read or validated")
        return None
