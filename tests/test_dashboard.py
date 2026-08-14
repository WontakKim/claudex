"""Regression tests for dashboard route parsing and product language."""

import re
from pathlib import Path


DASHBOARD_PATH = (
    Path(__file__).parents[1] / "src" / "claudex_gateway" / "dashboard.html"
)
DASHBOARD = DASHBOARD_PATH.read_text(encoding="utf-8")
DEVELOPER_COMMENT_PATTERN = re.compile(
    r"<!--.*?-->|/\*.*?\*/|//[^\n]*",
    re.DOTALL,
)
HANGUL_PATTERN = re.compile(r"[ㄱ-ㅎㅏ-ㅣ가-힣]")


def route_of_body() -> str:
    _, function_start, remainder = DASHBOARD.partition("function routeOf(value){")
    assert function_start, "routeOf function is missing"
    body, function_end, _ = remainder.partition("\n}")
    assert function_end, "routeOf function is not closed"
    return body


def test_route_of_preserves_provider_and_complete_model_suffix() -> None:
    body = route_of_body()

    assert 'String(value).indexOf(":")' in body
    assert "var prefix=value.slice(0,at);" in body
    assert "return{provider:prefix,model:value.slice(at+1)};" in body


def test_route_of_has_no_codex_or_provider_membership_fallback() -> None:
    body = route_of_body()

    assert "codex" not in body.lower()
    assert "ROUTE_PROVIDERS" not in body


def test_developer_comments_are_english() -> None:
    comments = DEVELOPER_COMMENT_PATTERN.findall(DASHBOARD)

    assert comments
    assert HANGUL_PATTERN.search("\n".join(comments)) is None


def test_korean_product_language_remains() -> None:
    assert '<html lang="ko">' in DASHBOARD
    assert "<h2>Claude 계정</h2>" in DASHBOARD
    assert '>계정 추가</button>' in DASHBOARD
    assert '<button id="comp-apply">적용</button>' in DASHBOARD
