"""Tests for shared provider reasoning-effort policies."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from claudex.providers.reasoning import fold_reasoning_effort


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "high"),
    ],
)
def test_folds_canonical_efforts_into_low_medium_high(
    requested: str, expected: str
) -> None:
    assert fold_reasoning_effort(requested, {"low", "medium", "high"}) == expected


def test_uses_highest_supported_effort_not_above_request() -> None:
    assert fold_reasoning_effort("xhigh", {"minimal", "medium", "max"}) == "medium"


def test_uses_lowest_supported_effort_when_all_exceed_request() -> None:
    assert fold_reasoning_effort("minimal", {"high", "medium"}) == "medium"


@pytest.mark.parametrize(
    "supported",
    [
        ["high", "low", "medium"],
        ("medium", "high", "low"),
        {"low", "medium", "high"},
        frozenset({"high", "medium", "low"}),
    ],
)
def test_result_is_independent_of_collection_type_and_order(
    supported: Collection[str],
) -> None:
    assert fold_reasoning_effort("xhigh", supported) == "high"


def test_rejects_unknown_requested_effort() -> None:
    with pytest.raises(ValueError, match="requested reasoning effort is not canonical"):
        fold_reasoning_effort("future", {"low", "medium", "high"})


def test_rejects_empty_supported_efforts() -> None:
    with pytest.raises(ValueError, match="supported reasoning efforts must not be empty"):
        fold_reasoning_effort("medium", set())


def test_rejects_noncanonical_supported_effort() -> None:
    with pytest.raises(ValueError, match="supported reasoning efforts are not canonical"):
        fold_reasoning_effort("medium", {"low", "future"})
