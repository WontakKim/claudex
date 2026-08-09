"""Versioned account profile fingerprint for registered Claude accounts.

`account_profile_fingerprint` is the one stable identity shared by cooldown
bookkeeping, usage observations, and capability evidence: a digest over the
subset of `oauth-account.json` fields that describe *what plan an account
is* (organization, tier, seat, billing, role, trial state) rather than *who
it belongs to* or *when it was captured*. Only a fixed, explicitly
recognized set of keys is read — anything else in the file, including
free-form profile text and capture timestamps, never reaches the digest.
The file holds no secrets; this module never reads a credentials file.

Every recognized key is present in the normalized object even when the
source file omits it, so an absent key and an explicit JSON `null` produce
the same fingerprint. Nested objects and top-level keys are sorted
recursively by `json.dumps(..., sort_keys=True)`; array order is preserved
and string values are used exactly as stored (no trimming or Unicode
normalization) except for the two UUID fields, which are canonicalized to
lowercase hyphenated form so that e.g. an uppercase-cased id from a
different capture still hashes identically.

`accountUuid` is the one field that gates fingerprinting at all: a missing
or non-UUID value yields no fingerprint (`None`), since without it there is
no stable key to attach cooldowns or usage history to. Callers that need a
fingerprint to activate an account (e.g. balanced-pool activation) must
handle `None` as "this account cannot participate yet".
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

_OAUTH_ACCOUNT_FILENAME = "oauth-account.json"

# Domain-separates this digest from any other sha256 use in the gateway so a
# collision in one context can never be replayed as a valid value in another.
_DOMAIN_TAG = b"claudex-account-profile-v1\x00"

# The exact recognized subset of oauth-account.json. Every other key present
# in the source file — including organization/email/display names and
# capture timestamps — is excluded and never read by this module.
_RECOGNIZED_KEYS: tuple[str, ...] = (
    "accountUuid",
    "organizationUuid",
    "organizationType",
    "organizationRateLimitTier",
    "userRateLimitTier",
    "seatTier",
    "billingType",
    "hasExtraUsageEnabled",
    "organizationRole",
    "workspaceRole",
    "subscriptionCreatedAt",
    "ccOnboardingFlags",
    "claudeCodeTrialDurationDays",
    "claudeCodeTrialEndsAt",
)


def compute_account_profile_fingerprint(oauth_account: dict) -> str | None:
    """Digest the recognized profile fields of a parsed `oauth-account.json`.

    Returns `None` when `accountUuid` is missing or not a UUID — every other
    recognized field is optional and normalizes to `null` when absent.
    """
    canonical_account_uuid = _canonical_uuid_or_none(oauth_account.get("accountUuid"))
    if canonical_account_uuid is None:
        return None

    normalized: dict[str, Any] = {key: oauth_account.get(key) for key in _RECOGNIZED_KEYS}
    normalized["accountUuid"] = canonical_account_uuid
    normalized["organizationUuid"] = _canonicalize_uuid_field(oauth_account.get("organizationUuid"))

    canonical_bytes = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(_DOMAIN_TAG + canonical_bytes).hexdigest()
    return f"v1:{digest}"


def load_account_profile_fingerprint(account_dir: Path) -> str | None:
    """Compute the fingerprint from `account_dir/oauth-account.json`.

    Tolerant like the plan-metadata and account-uuid readers elsewhere in
    the gateway: a missing file, unreadable bytes, or malformed JSON all
    degrade to `None` rather than raising — a stale or absent profile file
    never blocks the caller. Never reads a credentials file.
    """
    oauth_account_file = account_dir / _OAUTH_ACCOUNT_FILENAME
    try:
        parsed = json.loads(oauth_account_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return compute_account_profile_fingerprint(parsed)


def _canonical_uuid_or_none(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _canonicalize_uuid_field(value: object) -> object:
    """Canonicalize a UUID-shaped value; anything else (including `None`,
    which already represents an absent or null field) passes through
    unchanged."""
    canonical = _canonical_uuid_or_none(value)
    return canonical if canonical is not None else value
