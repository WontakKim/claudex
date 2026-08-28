"""Regression tests for dashboard assets, route parsing, and product language."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


PACKAGE_PATH = Path(__file__).parents[1] / "src" / "claudex"
DASHBOARD_HTML = (PACKAGE_PATH / "dashboard" / "dashboard.html").read_text(encoding="utf-8")
DASHBOARD_CSS = (PACKAGE_PATH / "dashboard" / "dashboard.css").read_text(encoding="utf-8")
DASHBOARD_JAVASCRIPT_PATH = PACKAGE_PATH / "dashboard" / "dashboard.js"
DASHBOARD_HTML_PATH = PACKAGE_PATH / "dashboard" / "dashboard.html"
DASHBOARD_JAVASCRIPT = DASHBOARD_JAVASCRIPT_PATH.read_text(encoding="utf-8")
DASHBOARD_RUNTIME_HARNESS = Path(__file__).parent / "dashboard-runtime-harness.js"
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


@pytest.fixture(scope="module")
def dashboard_runtime_result() -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for executable dashboard behavior tests")

    completed = subprocess.run(
        [
            node,
            str(DASHBOARD_RUNTIME_HARNESS),
            str(DASHBOARD_JAVASCRIPT_PATH),
            str(DASHBOARD_HTML_PATH),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        "dashboard runtime harness failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    result = json.loads(completed.stdout)
    assert result.get("credentialLeak") is False, (
        "dashboard runtime exposed the synthetic credential marker"
    )
    return result


def expected_dashboard_jfetch_requests(
    names: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "url": f'/admin/providers/custom/{names["connectedName"]}/models',
            "options": None,
        },
        {
            "url": f'/admin/providers/custom/{names["unusedName"]}/models',
            "options": None,
        },
        {"url": "/admin/usage", "options": None},
    ]


def assert_dashboard_jfetch_requests(
    requests: list[dict[str, Any]], names: dict[str, str]
) -> None:
    expected = expected_dashboard_jfetch_requests(names)

    assert all(set(request) == {"url", "options"} for request in requests)
    assert all(request["options"] is None for request in requests)

    usage_or_billing_requests = [
        request
        for request in requests
        if "usage" in request["url"].lower()
        or "billing" in request["url"].lower()
    ]
    assert usage_or_billing_requests == [
        {"url": "/admin/usage", "options": None}
    ]
    aggregate_usage_request = json.dumps(usage_or_billing_requests[0], sort_keys=True)
    assert all(name not in aggregate_usage_request for name in names.values())

    assert requests == expected


def test_custom_provider_runtime_uses_wire_labels_and_catalog_capabilities(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    result = dashboard_runtime_result
    names = result["names"]

    assert_dashboard_jfetch_requests(result["jfetchRequests"], names)

    assert "Anthropic Messages" in result["cards"]["configured"]
    assert "Responses API" not in result["cards"]["configured"]
    assert "Responses API" in result["cards"]["connected"]
    assert "Anthropic Messages" not in result["cards"]["connected"]


def test_custom_provider_runtime_renders_distinct_statuses(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    statuses = dashboard_runtime_result["statuses"]

    assert statuses["configured"]["className"] == "stat okv"
    assert statuses["configured"]["text"] == (
        "● CONFIGURED Anthropic Messages configured · "
        "remote connection not verified"
    )
    assert statuses["connected"]["className"] == "stat okv"
    assert statuses["connected"]["text"] == (
        "● CONNECTED Responses API remote catalog verified"
    )
    assert statuses["unused"]["className"] == "stat"
    assert statuses["unused"]["text"].startswith("● UNUSED ")
    assert statuses["error"]["className"] == "stat err"
    assert statuses["error"]["text"] == (
        "● ERROR Anthropic Messages binding unavailable"
    )


def test_catalogless_provider_runtime_accepts_manual_model(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    assert dashboard_runtime_result["manual"] == {
        "providerButtonPresent": True,
        "modelInputAvailable": True,
        "targetAccepted": True,
        "inputCleared": True,
    }


def test_gptpro_session_card_renders_states_and_polls_only_on_mcp(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["gptProSessionStates"]
    card_index = DASHBOARD_HTML.index('id="gptpro-session-card"')
    status_start = DASHBOARD_HTML.index('id="tab-status"')
    status_end = DASHBOARD_HTML.index("</section>", status_start)
    mcp_start = DASHBOARD_HTML.index('id="tab-mcp"')
    mcp_end = DASHBOARD_HTML.index("</section>", mcp_start)

    assert mcp_start < card_index < mcp_end
    assert not status_start < card_index < status_end
    assert states["valid"] == {
        "className": "stat okv",
        "text": "● VALID Expires in 8d 0h",
        "title": "",
    }
    assert states["expiring"] == {
        "className": "stat warn",
        "text": "● EXPIRING SOON Expires in 7d 0h",
        "title": "",
    }
    assert states["expired"]["className"] == "stat err"
    assert "run claudex-gateway gptpro login" in states["expired"]["text"]
    assert states["missing"]["className"] == "stat"
    assert states["missing"]["text"].startswith("● NOT CONFIGURED ")
    assert 'jfetch("/admin/gptpro/session")' in DASHBOARD_JAVASCRIPT
    assert "setInterval(fetchGptProSession,60000)" in DASHBOARD_JAVASCRIPT
    assert 'document.body.dataset.tab==="mcp"' in DASHBOARD_JAVASCRIPT


def test_mcp_tab_assets_wire_admin_operations_and_polling() -> None:
    assert '<a href="#mcp" data-t="mcp">MCP</a>' in DASHBOARD_HTML
    assert (
        'const TAB_NAMES=["settings","status","mcp","map","log"]'
        in DASHBOARD_JAVASCRIPT
    )
    assert 'body[data-tab="mcp"] #tab-mcp' in DASHBOARD_CSS
    assert 'jfetch("/admin/gptpro/mcp")' in DASHBOARD_JAVASCRIPT
    assert 'id="mcp-connect-btn"' in DASHBOARD_HTML
    assert 'id="mcp-connect-copy"' in DASHBOARD_HTML
    assert 'id="mcp-connect-result"' in DASHBOARD_HTML
    assert "function connectClaudeCode(){" in DASHBOARD_JAVASCRIPT
    assert 'jfetch("/admin/gptpro/connect"' in DASHBOARD_JAVASCRIPT
    assert 'jfetch("/admin/gptpro/login"' in DASHBOARD_JAVASCRIPT
    assert "setInterval(pollGptProLogin,2000)" in DASHBOARD_JAVASCRIPT
    assert "claude mcp add --transport http" in DASHBOARD_JAVASCRIPT


def test_mcp_runtime_renders_connection_login_and_doctor(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    result = dashboard_runtime_result

    assert result["mcpInfoStates"] == {
        "open": {
            "command": (
                "claude mcp add --transport http -s user claudex-gptpro "
                "http://127.0.0.1:8787/mcp"
            ),
            "endpoint": "http://127.0.0.1:8787/mcp",
            "authHintHidden": True,
        },
        "authenticated": {
            "command": (
                "claude mcp add --transport http -s user claudex-gptpro "
                "http://127.0.0.1:9000/mcp --header "
                '"Authorization: Bearer <CLAUDEX_LOCAL_TOKEN>"'
            ),
            "endpoint": "http://127.0.0.1:9000/mcp",
            "authHintHidden": False,
        },
    }
    assert result["mcpConnectStates"] == {
        "passed": {
            "className": "codeblock okv",
            "text": (
                "Claude Code MCP registered successfully. "
                "Please restart Claude Code sessions to load it."
            ),
            "hidden": False,
        },
        "failed": {
            "className": "codeblock err",
            "text": "registration failed\n",
            "hidden": False,
        },
    }
    assert result["gptProLoginStates"] == {
        "idle": {
            "buttonText": "Sign in to ChatGPT",
            "buttonDisabled": False,
            "detail": "",
            "polling": False,
        },
        "running": {
            "buttonText": "Cancel login",
            "buttonDisabled": False,
            "detail": (
                "sign in to ChatGPT in the opened browser "
                "gptpro asks are unavailable while signing in."
            ),
            "polling": True,
        },
        "terminal": {
            "buttonText": "Sign in to ChatGPT",
            "buttonDisabled": False,
            "detail": "saved and verified the gptpro session\n",
            "polling": False,
        },
        "sessionRefreshes": 1,
    }
    assert result["gptProDoctorStates"] == {
        "passed": {
            "className": "codeblock okv",
            "text": "doctor passed\n",
            "hidden": False,
        },
        "failed": {
            "className": "codeblock err",
            "text": "doctor failed\n",
            "hidden": False,
        },
    }


def test_custom_provider_runtime_excludes_usage_and_credentials(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    result = dashboard_runtime_result

    assert result["rawFetchRequests"] == []
    assert result["storageWrites"] == {"local": [], "session": []}
    assert result["consoleCalls"] == []
    assert result["credentialLeak"] is False


def test_request_assertion_rejects_early_custom_provider_usage(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    result = dashboard_runtime_result
    names = result["names"]
    assert_dashboard_jfetch_requests(result["jfetchRequests"], names)
    injected_request = {
        "url": f'/admin/usage?provider={names["configuredName"]}',
        "options": None,
    }

    with pytest.raises(AssertionError):
        assert_dashboard_jfetch_requests(
            [injected_request, *result["jfetchRequests"]], names
        )


def javascript_section(start: str, end: str) -> str:
    _, marker, remainder = DASHBOARD_JAVASCRIPT.partition(start)
    assert marker, f"dashboard JavaScript is missing {start!r}"
    section, marker, _ = remainder.partition(end)
    assert marker, f"dashboard JavaScript is missing {end!r} after {start!r}"
    return section


def test_custom_provider_dashboard_has_no_family_or_name_branches() -> None:
    custom_section = javascript_section(
        "var CUSTOM_PROVIDER_WIRE_LABELS=", "/* Purely cosmetic gating"
    )

    assert "openai_compatible" not in custom_section
    assert "anthropic_compatible" not in custom_section
    assert re.search(r"provider\.name\s*={2,3}", custom_section) is None


def test_custom_provider_dashboard_does_not_handle_or_persist_api_keys() -> None:
    custom_section = javascript_section(
        "var CUSTOM_PROVIDER_WIRE_LABELS=", "/* Purely cosmetic gating"
    )

    assert "api_key" not in custom_section
    assert "apiKey" not in custom_section
    assert "localStorage" not in DASHBOARD_JAVASCRIPT
    assert "sessionStorage" not in DASHBOARD_JAVASCRIPT
    assert re.search(r"\bconsole\.", DASHBOARD_JAVASCRIPT) is None
