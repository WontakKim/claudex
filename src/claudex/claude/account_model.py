"""Canonical Claude account registry models and validation."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STATE_READY = "ready"
_STATE_NEEDS_REAUTH = "needs-reauth"
_VALID_STATES = (_STATE_READY, _STATE_NEEDS_REAUTH)

_ROW_KEYS = (
    "id",
    "email",
    "organizationUuid",
    "organizationName",
    "createdAt",
    "updatedAt",
    "lastAuthenticatedAt",
    "state",
    "accountIncarnationId",
    "upstreamAccountUuid",
)
_ROW_KEYS_SET = frozenset(_ROW_KEYS)

_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


class AccountRegistryError(Exception):
    """Raised for registry-level failures: malformed files, I/O, or a broken
    on-disk invariant. Messages never contain credential values or raw file
    content — only paths, ids, and a description of what went wrong."""


class DuplicateAccountError(AccountRegistryError):
    """Raised by `add_account` when `(email, organizationUuid)` is already
    registered."""


class AccountNotFoundError(AccountRegistryError):
    """Raised by `remove_account` when the id is not a canonical UUID, or is
    canonical but not currently registered."""


@dataclass(frozen=True)
class AccountRecord:
    id: str
    email: str
    organization_uuid: str | None
    organization_name: str | None
    created_at: int
    updated_at: int
    last_authenticated_at: int
    state: str
    # A random id assigned once per distinct login and never reused; see
    # `_resolve_reauth_incarnation` for the exact survive-vs-rotate rule across
    # reauthentication.
    account_incarnation_id: str
    # The canonical (lowercase, hyphenated) Anthropic account uuid captured
    # from `oauth-account.json`'s `accountUuid`, or `None` when it has never
    # been established. Once known, never erased by a later capture that
    # fails to establish it again.
    upstream_account_uuid: str | None

    def to_row(self) -> dict[str, Any]:
        """The exact camelCase JSON shape persisted in `registry.json`."""
        return {
            "id": self.id,
            "email": self.email,
            "organizationUuid": self.organization_uuid,
            "organizationName": self.organization_name,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "lastAuthenticatedAt": self.last_authenticated_at,
            "state": self.state,
            "accountIncarnationId": self.account_incarnation_id,
            "upstreamAccountUuid": self.upstream_account_uuid,
        }


def _validate_account_inputs(
    email: str,
    organization_uuid: str | None,
    organization_name: str | None,
    credentials_json: dict[str, Any],
    oauth_account_json: dict[str, Any] | None,
) -> tuple[str, str | None, str | None, dict[str, Any]]:
    """Normalize identity fields and validate payload shapes, strictly before
    any filesystem write. Returns `(email, organizationUuid,
    organizationName, oauth_payload)` with `oauth_payload` defaulted to `{}`
    when `oauth_account_json` is `None`."""
    try:
        normalized_email = _normalize_email(email)
        normalized_organization_uuid = _normalize_optional_text(
            organization_uuid, field="organizationUuid"
        )
        normalized_organization_name = _normalize_optional_text(
            organization_name, field="organizationName"
        )
    except ValueError as exc:
        raise AccountRegistryError(str(exc)) from exc
    if not isinstance(credentials_json, dict):
        raise AccountRegistryError("credentials_json must be a JSON object")
    if oauth_account_json is not None and not isinstance(oauth_account_json, dict):
        raise AccountRegistryError("oauth_account_json must be a JSON object or null")
    oauth_payload: dict[str, Any] = oauth_account_json if oauth_account_json is not None else {}
    return (
        normalized_email,
        normalized_organization_uuid,
        normalized_organization_name,
        oauth_payload,
    )


def _register_unique_or_raise(
    record_id: str,
    identity: tuple[str, str | None],
    seen_ids: set[str],
    seen_identities: set[tuple[str, str | None]],
    *,
    path: Path,
    index: int,
) -> None:
    if record_id in seen_ids:
        raise AccountRegistryError(f"registry {path}: duplicate id at row {index}")
    seen_ids.add(record_id)
    if identity in seen_identities:
        raise AccountRegistryError(
            f"registry {path}: duplicate account identity at row {index}"
        )
    seen_identities.add(identity)


def _parse_current_rows(parsed: list[object], *, path: Path) -> list[AccountRecord]:
    """Strictly parse rows using the exact current schema."""
    records: list[AccountRecord] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str | None]] = set()
    for index, row in enumerate(parsed):
        record = _parse_row(row, path=path, index=index)
        _register_unique_or_raise(
            record.id,
            (record.email, record.organization_uuid),
            seen_ids,
            seen_identities,
            path=path,
            index=index,
        )
        records.append(record)
    return records


def _check_row_keys(
    row: object, expected_keys: frozenset[str], *, path: Path, index: int
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise AccountRegistryError(f"registry {path}: row {index} must be a JSON object")
    row_keys = set(row)
    unknown_keys = row_keys - expected_keys
    if unknown_keys:
        raise AccountRegistryError(
            f"registry {path}: row {index} has unknown keys: {', '.join(sorted(unknown_keys))}"
        )
    missing_keys = expected_keys - row_keys
    if missing_keys:
        raise AccountRegistryError(
            f"registry {path}: row {index} is missing keys: {', '.join(sorted(missing_keys))}"
        )
    return row


def _validate_shared_fields(
    row: dict[str, Any], *, path: Path, index: int
) -> tuple[str, str, str | None, str | None, int, int, int, str]:
    """Validate the registry fields other than incarnation identity.

    Returns `(id, email, organizationUuid, organizationName, createdAt,
    updatedAt, lastAuthenticatedAt, state)`.
    """
    try:
        record_id = _validate_id(row["id"])
        email = _normalize_email(row["email"])
        organization_uuid = _normalize_optional_text(
            row["organizationUuid"], field="organizationUuid"
        )
        organization_name = _normalize_optional_text(
            row["organizationName"], field="organizationName"
        )
        created_at = _validate_timestamp(row["createdAt"], field="createdAt")
        updated_at = _validate_timestamp(row["updatedAt"], field="updatedAt")
        last_authenticated_at = _validate_timestamp(
            row["lastAuthenticatedAt"], field="lastAuthenticatedAt"
        )
        state = _validate_state(row["state"])
    except ValueError as exc:
        raise AccountRegistryError(f"registry {path}: row {index} is invalid: {exc}") from exc
    return (
        record_id,
        email,
        organization_uuid,
        organization_name,
        created_at,
        updated_at,
        last_authenticated_at,
        state,
    )


def _parse_row(row: object, *, path: Path, index: int) -> AccountRecord:
    """Strictly parse one exact current-schema (10-key) row."""
    checked = _check_row_keys(row, _ROW_KEYS_SET, path=path, index=index)
    (
        record_id,
        email,
        organization_uuid,
        organization_name,
        created_at,
        updated_at,
        last_authenticated_at,
        state,
    ) = _validate_shared_fields(checked, path=path, index=index)
    try:
        account_incarnation_id = _validate_canonical_uuid_field(
            checked["accountIncarnationId"], field="accountIncarnationId"
        )
        upstream_account_uuid = _validate_optional_canonical_uuid_field(
            checked["upstreamAccountUuid"], field="upstreamAccountUuid"
        )
    except ValueError as exc:
        raise AccountRegistryError(f"registry {path}: row {index} is invalid: {exc}") from exc

    return AccountRecord(
        id=record_id,
        email=email,
        organization_uuid=organization_uuid,
        organization_name=organization_name,
        created_at=created_at,
        updated_at=updated_at,
        last_authenticated_at=last_authenticated_at,
        state=state,
        account_incarnation_id=account_incarnation_id,
        upstream_account_uuid=upstream_account_uuid,
    )


def _reject_control_characters(value: str, *, field: str) -> None:
    if _CONTROL_CHARACTER_PATTERN.search(value):
        raise ValueError(f"{field} contains control characters")


def _normalize_email(raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError("email must be a string")
    _reject_control_characters(raw, field="email")
    trimmed = raw.strip()
    if not trimmed:
        raise ValueError("email is required")
    return trimmed.lower()


def _normalize_optional_text(raw: object, *, field: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string or null")
    _reject_control_characters(raw, field=field)
    trimmed = raw.strip()
    return trimmed or None


def _validate_timestamp(raw: object, *, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer epoch-millisecond timestamp")
    if raw < 0:
        raise ValueError(f"{field} must be nonnegative")
    return raw


def _validate_state(raw: object) -> str:
    if raw not in _VALID_STATES:
        raise ValueError(f"state must be one of {_VALID_STATES!r}")
    return raw


def _canonical_uuid_or_none(value: str) -> str | None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    canonical = str(parsed)
    return canonical if canonical == value else None


def _validate_canonical_uuid_field(raw: object, *, field: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string")
    canonical = _canonical_uuid_or_none(raw)
    if canonical is None:
        raise ValueError(f"{field} must be a canonical UUID")
    return canonical


def _validate_optional_canonical_uuid_field(raw: object, *, field: str) -> str | None:
    if raw is None:
        return None
    return _validate_canonical_uuid_field(raw, field=field)


def _validate_id(raw: object) -> str:
    return _validate_canonical_uuid_field(raw, field="id")


def _is_canonical_uuid(value: str) -> bool:
    return _canonical_uuid_or_none(value) is not None


def _canonicalize_account_id(account_id: object) -> str:
    if not isinstance(account_id, str):
        raise AccountNotFoundError("account id must be a string")
    canonical = _canonical_uuid_or_none(account_id)
    if canonical is None:
        raise AccountNotFoundError(f"{account_id!r} is not a valid account id")
    return canonical


def _canonicalize_upstream_uuid(raw: object) -> str | None:
    """Derive the canonical (lowercase, hyphenated) form of a captured
    `accountUuid` value, or `None` when it is missing, not a string, or not
    a valid UUID. Never raises: a malformed or absent upstream identity must
    never block account registration or reauthentication, it only means the
    upstream uuid could not be established this time."""
    if not isinstance(raw, str):
        return None
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError, TypeError):
        return None
    return str(parsed)


def _derive_upstream_account_uuid(oauth_payload: dict[str, Any]) -> str | None:
    """The canonical upstream account uuid captured in-memory at
    `add_account`/`update_account_credentials` time, from the same
    `oauth_payload` dict that gets persisted to `oauth-account.json`."""
    return _canonicalize_upstream_uuid(oauth_payload.get("accountUuid"))


def _resolve_reauth_incarnation(
    existing: AccountRecord, newly_captured_upstream_account_uuid: str | None
) -> tuple[str, str | None]:
    """Apply the reauthentication transition table for
    `account_incarnation_id`/`upstream_account_uuid`. Returns the
    `(account_incarnation_id, upstream_account_uuid)` pair the updated row
    should carry.

    Failure to capture a valid upstream uuid this time (`None`) never
    erases a previously known one nor rotates the incarnation — the
    existing values simply carry forward (covers both "known upstream uuid
    plus missing/malformed metadata" and "both null"). A previously unknown
    upstream uuid that is now known keeps the incarnation and records the
    newly captured uuid. The same valid uuid captured again keeps the
    incarnation. Only a newly captured valid uuid that *differs* from a
    previously known valid uuid rotates the incarnation.
    """
    if newly_captured_upstream_account_uuid is None:
        return existing.account_incarnation_id, existing.upstream_account_uuid
    if existing.upstream_account_uuid is None:
        return existing.account_incarnation_id, newly_captured_upstream_account_uuid
    if existing.upstream_account_uuid == newly_captured_upstream_account_uuid:
        return existing.account_incarnation_id, newly_captured_upstream_account_uuid
    return str(uuid.uuid4()), newly_captured_upstream_account_uuid
