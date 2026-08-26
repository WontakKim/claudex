"""Tests for pure gptpro backend retry classification."""

from __future__ import annotations

import pytest

from claudex.gptpro import retry


def test_ok_and_session_expiry_classification() -> None:
    assert retry.classify_backend_response(200, {}).action == "ok"
    expired = retry.classify_backend_response(401, {})
    assert expired.action == "session_expired"
    assert expired.retry_after_ms is None


@pytest.mark.parametrize(
    ("value", "expected_action"),
    [("block", "blocked"), (" Challenge ", "challenge")],
)
def test_cf_mitigated_takes_priority_and_normalizes_header_case(
    value: str, expected_action: str
) -> None:
    action = retry.classify_backend_response(
        200, {"CF-MitIGated": value, "Retry-After": "30"}
    )

    assert action.action == expected_action
    assert action.retry_after_ms is None


@pytest.mark.parametrize("marker", ["cf-chl", "challenge", "captcha", "__cf_bm"])
def test_403_with_cf_ray_and_challenge_marker_is_challenge(marker: str) -> None:
    action = retry.classify_backend_response(
        403, {"CF-Ray": "edge-id"}, f"<html>{marker.upper()}</html>"
    )

    assert action.action == "challenge"


@pytest.mark.parametrize(
    ("headers", "body"),
    [({}, "challenge"), ({"cf-ray": "edge-id"}, '{"error":"forbidden"}')],
)
def test_403_without_both_challenge_signals_is_entitlement(
    headers: dict[str, str], body: str
) -> None:
    assert retry.classify_backend_response(403, headers, body).action == "entitlement"


@pytest.mark.parametrize(
    ("headers", "expected_delay"),
    [
        ({"Retry-After": "7"}, 7_000),
        ({}, 1_000),
        ({"retry-after": "1.5"}, 1_000),
        ({"retry-after": "later"}, 1_000),
        ({"retry-after": "0"}, 1_000),
    ],
)
def test_429_parses_only_integer_retry_after_without_jitter(
    headers: dict[str, str], expected_delay: int
) -> None:
    action = retry.classify_backend_response(429, headers)

    assert action.action == "rate_limit"
    assert action.retry_after_ms == expected_delay


def test_5xx_distinguishes_edge_challenge_from_origin_retry() -> None:
    edge = retry.classify_backend_response(
        503, {"cf-ray": "edge-id"}, "CAPTCHA challenge"
    )
    origin_without_marker = retry.classify_backend_response(
        503, {"cf-ray": "edge-id"}, "upstream unavailable"
    )
    origin_without_ray = retry.classify_backend_response(500, {}, "cf-chl")

    assert edge.action == "challenge"
    assert origin_without_marker.action == "origin_retry"
    assert origin_without_ray.action == "origin_retry"


def test_unexpected_status_is_fatal() -> None:
    action = retry.classify_backend_response(418, {}, "teapot")

    assert action.action == "fatal"
    assert action.retry_after_ms is None


def test_rate_limit_delay_has_minimum_and_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retry.random, "randint", lambda lower, upper: upper)

    assert retry.rate_limit_delay_ms(None) == 3_000
    assert retry.rate_limit_delay_ms(500) == 3_000
    assert retry.rate_limit_delay_ms(10_000) == 12_000


def test_rate_limit_delay_caps_the_final_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retry.random, "randint", lambda lower, upper: upper)

    assert retry.rate_limit_delay_ms(59_000) == retry.MAX_RETRY_AFTER_MS
    assert retry.rate_limit_delay_ms(120_000) == retry.MAX_RETRY_AFTER_MS


def test_rate_limit_delay_can_use_zero_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retry.random, "randint", lambda lower, upper: lower)

    assert retry.rate_limit_delay_ms(None) == retry.MIN_RETRY_AFTER_MS
    assert retry.rate_limit_delay_ms(5_000) == 5_000
