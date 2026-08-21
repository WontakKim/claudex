"""Shared reasoning-effort policies for provider integrations."""

from __future__ import annotations

from collections.abc import Collection

from claudex.config.schema import VALID_REASONING_EFFORTS


def fold_reasoning_effort(requested: str, supported: Collection[str]) -> str:
    """Fold a canonical effort into a provider's supported effort set.

    The result is the highest supported effort that does not exceed the
    requested effort. If every supported effort is higher, the lowest
    supported effort is returned.
    """
    if requested not in VALID_REASONING_EFFORTS:
        raise ValueError(f"requested reasoning effort is not canonical: {requested!r}")

    supported_efforts = frozenset(supported)
    if not supported_efforts:
        raise ValueError("supported reasoning efforts must not be empty")

    invalid_efforts = supported_efforts.difference(VALID_REASONING_EFFORTS)
    if invalid_efforts:
        formatted_efforts = ", ".join(sorted(map(repr, invalid_efforts)))
        raise ValueError(
            f"supported reasoning efforts are not canonical: {formatted_efforts}"
        )

    requested_index = VALID_REASONING_EFFORTS.index(requested)
    for effort in reversed(VALID_REASONING_EFFORTS[: requested_index + 1]):
        if effort in supported_efforts:
            return effort

    return next(effort for effort in VALID_REASONING_EFFORTS if effort in supported_efforts)
