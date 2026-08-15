"""Regression tests for dashboard assets, route parsing, and product language."""

import re
from pathlib import Path


PACKAGE_PATH = Path(__file__).parents[1] / "src" / "claudex_gateway"
DASHBOARD_HTML = (PACKAGE_PATH / "dashboard" / "dashboard.html").read_text(encoding="utf-8")
DASHBOARD_CSS = (PACKAGE_PATH / "dashboard" / "dashboard.css").read_text(encoding="utf-8")
DASHBOARD_JAVASCRIPT = (PACKAGE_PATH / "dashboard" / "dashboard.js").read_text(encoding="utf-8")
DEVELOPER_COMMENT_PATTERN = re.compile(
    r"<!--.*?-->|/\*.*?\*/|//[^\n]*",
    re.DOTALL,
)
HANGUL_PATTERN = re.compile(r"[ㄱ-ㅎㅏ-ㅣ가-힣]")


def route_of_body() -> str:
    _, function_start, remainder = DASHBOARD_JAVASCRIPT.partition(
        "function routeOf(value){"
    )
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


def test_dashboard_html_references_only_external_assets_at_document_end() -> None:
    assert 'rel="stylesheet" href="/dashboard.css"' in DASHBOARD_HTML
    assert re.search(r"<style\b", DASHBOARD_HTML, re.IGNORECASE) is None

    scripts = list(
        re.finditer(
            r"<script\b([^>]*)>(.*?)</script\s*>",
            DASHBOARD_HTML,
            re.IGNORECASE | re.DOTALL,
        )
    )
    assert len(scripts) == 1
    attributes, inline_body = scripts[0].groups()
    assert re.search(
        r'\bsrc\s*=\s*(["\'])/dashboard\.js\1', attributes, re.IGNORECASE
    )
    assert re.search(r"\bdefer(?:\s|=|$)", attributes, re.IGNORECASE) is None
    assert inline_body.strip() == ""
    assert re.fullmatch(
        r"\s*</body>\s*</html>\s*",
        DASHBOARD_HTML[scripts[0].end() :],
        re.IGNORECASE | re.DOTALL,
    )


def test_dashboard_assets_exclude_document_wrappers() -> None:
    for asset in (DASHBOARD_CSS, DASHBOARD_JAVASCRIPT):
        normalized_asset = asset.lower()
        for forbidden in (
            "<style",
            "</style",
            "<script",
            "</script",
            "</body>",
            "</html>",
        ):
            assert forbidden not in normalized_asset


def test_developer_comments_are_english() -> None:
    for asset in (DASHBOARD_HTML, DASHBOARD_CSS, DASHBOARD_JAVASCRIPT):
        comments = DEVELOPER_COMMENT_PATTERN.findall(asset)
        assert HANGUL_PATTERN.search("\n".join(comments)) is None


def test_korean_product_language_remains_in_owning_assets() -> None:
    assert '<html lang="ko">' in DASHBOARD_HTML
    assert "<h2>Claude 계정</h2>" in DASHBOARD_HTML
    assert '>계정 추가</button>' in DASHBOARD_HTML
    assert '<button id="comp-apply">적용</button>' in DASHBOARD_HTML
    assert "이 게이트웨이는 CLAUDEX_LOCAL_TOKEN 인증이 필요합니다." in DASHBOARD_JAVASCRIPT
