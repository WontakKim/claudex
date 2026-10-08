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
            "--unhandled-rejections=strict",
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
    # The third custom-models request re-registers the connected provider
    # after the tab-leave check: its late catalog reply must refresh nothing
    # once the chooser has closed.
    return [
        {
            "url": f'/admin/providers/custom/{names["connectedName"]}/models',
            "options": None,
        },
        {
            "url": f'/admin/providers/custom/{names["unusedName"]}/models',
            "options": None,
        },
        {
            "url": f'/admin/providers/custom/{names["connectedName"]}/models',
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
        "providerOptionPresent": True,
        "modelQueryAvailable": True,
        "targetAccepted": True,
        "stateLineShown": True,
    }


def test_picker_runtime_guards_locked_controls_and_forced_events(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    picker = dashboard_runtime_result["picker"]

    assert picker["lockedControls"] == {
        "addButtonDisabled": True,
        "providerSelectDisabled": True,
        "modelQueryDisabled": True,
        "pickerCloseDisabled": True,
    }
    assert picker["nativeLockGuards"] == {"pickerHidden": True, "targetsEmpty": True}
    assert picker["forcedLockGuards"] == {
        "dblclickHidden": True,
        "contextmenuHidden": True,
        "shortcutHidden": True,
        "targetsEmpty": True,
    }
    assert picker["unlockedControls"] == {
        "addButtonEnabled": True,
        "providerSelectEnabled": True,
        "modelQueryEnabled": True,
        "pickerCloseEnabled": True,
    }


def test_picker_runtime_opens_focused_and_switches_provider_cleanly(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    picker = dashboard_runtime_result["picker"]
    names = dashboard_runtime_result["names"]

    assert picker["opened"] == {
        "pickerHidden": False,
        "entry": "button",
        "queryEmpty": True,
        "queryFocused": True,
        "stateHidden": True,
    }
    assert picker["providerSwitch"] == {
        "selectedProvider": names["connectedName"],
        "queryCleared": True,
        "queryFocused": True,
        "suggestionsBefore": [],
    }
    assert picker["catalogAfterResolution"] == {
        "selectedProvider": names["connectedName"],
        "suggestions": ["late-model", "late:model:variant"],
        "stateHidden": True,
    }
    assert picker["cataloglessState"]["stateHidden"] is False
    assert "카탈로그" in picker["cataloglessState"]["stateText"]


def test_picker_runtime_enter_and_options_stage_exact_clean_targets(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    staging = dashboard_runtime_result["picker"]["stagingState"]
    names = dashboard_runtime_result["names"]
    colon_target = f'{names["configuredName"]}:manual:model:alpha'
    enter_target = f'{names["configuredName"]}:manual-enter-beta'

    assert staging["selectedProvider"] == names["connectedName"]
    assert staging["blankIgnored"] is True
    assert staging["afterManualEnter"] == [colon_target]
    assert staging["closedAfterCommit"] is True
    assert staging["afterEnter"] == [colon_target, enter_target]
    assert staging["afterDuplicate"] == [colon_target, enter_target]
    assert staging["duplicateStaysOpen"] is True
    assert staging["mapping"] == {"sonnet": "codex:existing-model"}
    assert staging["counts"] == {"wired": 0, "rewired": 0, "unwired": 0}
    assert staging["isDirty"] is False


def test_picker_runtime_stages_contextual_targets_at_transformed_y(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    contextual = dashboard_runtime_result["picker"]["contextual"]
    names = dashboard_runtime_result["names"]

    # clientY 320 with pan.y 20 and zoom 2 lands at graph y 150; the live
    # target keeps the default stacking position 28.
    assert contextual["contextmenuEntry"] == "contextmenu"
    assert contextual["first"] == [
        {"id": "codex:existing-model", "y": 28},
        {"id": f'{names["connectedName"]}:manual-at-y', "y": 150},
    ]
    # A second node at graph y 155 collides with 150 and is nudged one node
    # height clear of it; default stacking never moves.
    assert contextual["dblclickEntry"] == "dblclick"
    assert contextual["secondId"] == f'{names["connectedName"]}:manual-second-y'
    assert contextual["secondY"] == 203


def test_picker_runtime_excludes_nodes_and_right_button_pan(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    assert dashboard_runtime_result["picker"]["guards"] == {
        "nodeDblclickHidden": True,
        "nodeContextmenuNative": True,
        "blankContextmenuPrevented": True,
        "rightButtonNoPan": True,
        "leftButtonPan": True,
    }


def test_picker_runtime_blocks_ime_and_tab_commits(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    assert dashboard_runtime_result["picker"]["ime"] == {
        "composingBlocked": True,
        "isComposingBlocked": True,
        "tabNeverAdds": True,
    }


def test_picker_runtime_arrows_commit_the_active_suggestion(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    names = dashboard_runtime_result["names"]

    assert dashboard_runtime_result["picker"]["arrows"] == {
        "activeDescendant": "add-opt-1",
        "commitTarget": f'{names["connectedName"]}:late:model:variant',
    }


def test_picker_runtime_closes_on_tab_leave_and_late_callbacks(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    assert dashboard_runtime_result["picker"]["tabLeave"] == {
        "closedOnTabLeave": True,
        "lateCallbackNoReopen": True,
    }


def test_picker_runtime_discard_restores_live_map_and_drops_staging(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    picker = dashboard_runtime_result["picker"]
    names = dashboard_runtime_result["names"]

    assert picker["dirtyBeforeDiscard"] == {
        "mapping": {
            "sonnet": f'{names["configuredName"]}:manual:model:alpha'
        },
        "counts": {"wired": 0, "rewired": 1, "unwired": 0},
        "isDirty": True,
    }
    assert picker["discardedState"] == {
        "mapping": {"sonnet": "codex:existing-model"},
        "addedTargets": [],
        "targets": [{"id": "codex:existing-model", "y": 28}],
        "counts": {"wired": 0, "rewired": 0, "unwired": 0},
        "isDirty": False,
    }


def test_quick_add_gestures_target_blank_board_only() -> None:
    # Creation gestures live on the board itself: double-click and
    # right-click only on blank canvas (board or wire backdrop), and the
    # keyboard shortcut only while the board itself is focused.
    assert 'board.addEventListener("dblclick"' in DASHBOARD_JAVASCRIPT
    assert 'board.addEventListener("contextmenu"' in DASHBOARD_JAVASCRIPT
    assert 'board.addEventListener("keydown"' in DASHBOARD_JAVASCRIPT
    dblclick_section = javascript_section(
        'board.addEventListener("dblclick"', 'board.addEventListener("contextmenu"'
    )
    contextmenu_section = javascript_section(
        'board.addEventListener("contextmenu"', 'board.addEventListener("keydown"'
    )
    shortcut_section = javascript_section(
        'board.addEventListener("keydown"', "function graphY("
    )
    for section in (dblclick_section, contextmenu_section):
        assert "blankBoard(ev)" in section
    assert "ev.preventDefault()" in contextmenu_section
    assert "ev.target!==board" in shortcut_section
    assert 'ev.key!=="A"&&ev.key!=="a"' in shortcut_section
    assert "ev.keyCode===229" in shortcut_section
    # Non-primary pointer starts never pan, drag a wire, or move a node:
    # the right button is reserved for the blank-canvas context menu.
    pointerdown_section = javascript_section(
        'board.addEventListener("pointerdown"', 'document.addEventListener("pointermove"'
    )
    # The guard sits at the very top of the handler, before any pan/drag branch.
    assert pointerdown_section.startswith(',function(ev){')
    assert "if(ev.button!==0)return;" in pointerdown_section[:200]


def test_quick_add_geometry_keeps_distinct_eight_pixel_values() -> None:
    # The chooser anchors to the current add-button rect: right edges align,
    # its bottom sits 8px ABOVE the button (external gap), and the footer's
    # own bottom padding is a separate 8px.
    assert "#node-add{position:absolute;right:10px;bottom:10px;" in DASHBOARD_CSS
    assert ".addpick{position:fixed;" in DASHBOARD_CSS
    # .addpick sets author display:flex, which outranks the UA [hidden] rule;
    # without this companion rule the chooser renders on load (browser-verified).
    assert ".addpick[hidden]{display:none}" in DASHBOARD_CSS
    assert ".addpick-foot{margin:0;padding:7px 10px 8px;" in DASHBOARD_CSS
    assert "var PICKER_GAP=8" in DASHBOARD_JAVASCRIPT
    assert "var PICKER_INSET=8" in DASHBOARD_JAVASCRIPT
    place_section = javascript_section("function placePicker(){", "function openPicker(")
    assert "btn.top-PICKER_GAP" in place_section
    assert "btn.right" in place_section


def test_quick_add_gap_anchors_on_the_actual_layout_box(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    # Real layout paints fractional box heights while offsetHeight reports a
    # rounded integer (248.5 -> 249). Anchoring the chooser bottom on the
    # rounded integer compounds rounding into an 8.5px gap; placement must
    # use the actual layout box so the external gap stays exactly PICKER_GAP
    # (browser-measured at 760px, light and dark).
    geometry = dashboard_runtime_result["picker"]["fractionalGeometry"]
    assert geometry["gapExactlyEight"] is True
    assert geometry["rightEdgeDelta"] == 0


def test_quick_add_state_line_hidden_including_its_space_when_ready() -> None:
    # The picker markup ships the status line hidden by default; renderPicker
    # only reveals it for states that carry information (e.g. a provider with
    # no catalog endpoint), never for ordinary ready catalogs or no matches.
    assert '<div id="picker-state" role="status" aria-live="polite" hidden>' in DASHBOARD_HTML
    render_section = javascript_section("function renderPicker(){", "function syncPickerActive(")
    assert "pickerState.hidden=true" in render_section
    assert "CATALOGLESS[addProvider]" in render_section
    # Suggestions are DOM text, never untrusted provider/model HTML.
    assert "textContent=o.id" in render_section
    assert "replaceChildren" in render_section


def test_quick_add_staging_never_wires_or_dirties_the_map() -> None:
    stage_section = javascript_section("function stageTarget(", "function freeLaneY(")
    assert "addedTargets.push" in stage_section
    assert "DIR.mapping" not in stage_section
    # Contextual placement preserves existing positions and the target-side
    # default x; the y hint seeds DIR.targets, which rebuildColumn keeps.
    lane_section = javascript_section("function freeLaneY(", "function renderProviderOptions(")
    assert "DIR.targets" in lane_section
    assert ".x=" not in lane_section


def test_quick_add_leaves_router_tab_closed_and_impossible_to_reopen_late() -> None:
    set_tab_section = javascript_section("function setTab(", "const TAB_NAMES=")
    assert 'if(t!=="map"' in set_tab_section
    refresh_section = javascript_section(
        "function refreshOpenPicker(", "function graphY("
    )
    assert "modelPicker.hidden" in refresh_section


def test_boot_reports_codex_catalog_failure_without_inventing_suggestions(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    boot = dashboard_runtime_result["codexCatalogBoot"]
    assert boot["className"] == "toast err"
    assert "GET /admin/providers/codex/models" in boot["html"]
    assert "&lt;broken Codex version>" in boot["html"]
    assert "<broken Codex version>" not in boot["html"]
    assert boot["suggestions"] == []


def test_quick_add_keeps_credentials_and_catalogs_on_existing_channels() -> None:
    # The picker reuses the live CATALOG/jfetch closure: no new endpoints and
    # no hardcoded model snapshot may ride along with the new surface.
    assert "CATALOG={codex:[],kimi:[],grok:[]}" in DASHBOARD_JAVASCRIPT
    assert "gpt-5" not in DASHBOARD_JAVASCRIPT
    picker_section = javascript_section(
        'document.getElementById("node-add").addEventListener', "var toastTimer=null"
    )
    assert "/admin/" not in picker_section


def test_mcp_tab_leads_with_connection_and_combines_gptpro_tools() -> None:
    mcp_start = DASHBOARD_HTML.index('id="tab-mcp"')
    mcp_end = DASHBOARD_HTML.index("</section>", mcp_start)
    mcp_markup = DASHBOARD_HTML[mcp_start:mcp_end]

    assert re.findall(r'<div class="card" id="([^"]+)"', mcp_markup) == [
        "mcp-connect-card",
        "gptpro-session-card",
    ]
    gptpro_start = mcp_markup.index('<div class="card" id="gptpro-session-card">')
    gptpro_markup = mcp_markup[gptpro_start:]
    assert "<h2>GPT Pro</h2>" in gptpro_markup
    assert 'id="gptpro-doctor-card"' not in gptpro_markup
    assert 'id="gptpro-doctor-btn"' in gptpro_markup
    assert 'id="gptpro-doctor-output"' in gptpro_markup


def test_gptpro_concurrency_card_wires_native_autosave_and_env_lock() -> None:
    start = DASHBOARD_HTML.index("id=\"gptpro-session-card\"")
    section = DASHBOARD_HTML[start:DASHBOARD_HTML.index("</section>", start)]
    assert (
        section.index("id=\"gptpro-login-detail\"")
        < section.index("id=\"gptpro-concurrency-card\"")
        < section.index("Diagnostics")
    )
    select = re.search(
        r"<select id=\"gptpro-concurrency\"([^>]*)>(.*?)</select>",
        section,
        re.DOTALL,
    )
    assert select is not None
    assert re.search(r"\bdisabled\b", select.group(1))
    assert re.findall(
        r"<option value=\"(\d+)\">\d+</option>", select.group(2)
    ) == [str(value) for value in range(1, 11)]
    placeholder = re.search(r"<option value=\"\"([^>]*)>", select.group(2))
    assert placeholder is not None
    assert re.search(r"\bdisabled\b", placeholder.group(1))
    assert "gptpro-concurrency-apply" not in DASHBOARD_HTML
    assert "gptpro-concurrency-apply" not in DASHBOARD_JAVASCRIPT
    assert "Limits how many GPT Pro ask tabs can run at once." in section
    assert "id=\"gptpro-concurrency-description\"" in section
    assert "id=\"gptpro-concurrency-status\" role=\"status\" aria-live=\"polite\"" in section
    assert "id=\"gptpro-concurrency-rules\"" in section
    assert "aria-describedby=\"gptpro-concurrency-description gptpro-concurrency-status gptpro-concurrency-rules\"" in select.group(1)
    assert section.index("<label for=\"gptpro-concurrency\">") < section.index("<select id=\"gptpro-concurrency\"")
    assert ".gptpro-concurrency-row{display:flex;" in DASHBOARD_CSS
    assert ".gptpro-concurrency-copy{flex:1;min-width:0}" in DASHBOARD_CSS
    assert "#gptpro-concurrency{flex:none;width:64px;" in DASHBOARD_CSS
    assert "class=\"complock\"" in section
    assert "<code id=\"gptpro-concurrency-lock-env\"></code> takes precedence." in section
    assert "GPTPRO_MAX_CONCURRENT_ASKS" in DASHBOARD_JAVASCRIPT
    save_fn = javascript_section(
        "function saveGptProConcurrency(nextLimit){",
        "document.getElementById(\"gptpro-concurrency\").addEventListener",
    )
    assert "applyLockableSetting" not in save_fn
    for source in (
        "jfetch(\"/admin/settings/gptpro\",{", "method:\"PUT\"",
        "headers:JSON_HEADERS", "JSON.stringify({max_concurrent_asks:nextLimit})",
    ):
        assert source in save_fn
    assert (
        "if(t===\"mcp\"){fetchGptProSession();fetchMcpInfo();fetchGptProLogin();"
        "fetchGptProConcurrency()}" in DASHBOARD_JAVASCRIPT
    )
    boot = javascript_section("function boot(){", "\nboot();")
    assert "/admin/settings/gptpro" not in boot
    assert "#gptpro-concurrency-card.locked .complock" in DASHBOARD_CSS
    assert "#gptpro-concurrency-card.locked" in DASHBOARD_CSS


def assert_concurrency_ready(state: dict[str, Any], limit: int) -> None:
    assert state["value"] == str(limit)
    assert state["live"] == state["draft"] == limit
    assert state["hasLoaded"] is True
    assert state["isSaving"] is False
    assert state["selectDisabled"] is False
    assert state["locked"] is False
    assert state["env"] == "GPTPRO_MAX_CONCURRENT_ASKS"


def assert_concurrency_ignored(guard: dict[str, Any]) -> None:
    assert guard["requestsAdded"] == 0
    assert all(state == guard["before"] for state in guard["states"])


def test_gptpro_concurrency_runtime_starts_disabled_and_loads_on_mcp(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["gptProConcurrencyStates"]
    assert states["noApply"] is True
    for state in (states["initial"], states["loading"]):
        assert state["value"] == ""
        assert state["live"] is None
        assert state["draft"] is None
        assert state["hasLoaded"] is False
        assert state["isSaving"] is False
        assert state["selectDisabled"] is True
        assert state["locked"] is False
        assert state["env"] == "GPTPRO_MAX_CONCURRENT_ASKS"
    unavailable = states["startupUnavailable"]
    assert unavailable["hasLoaded"] is False
    assert unavailable["selectDisabled"] is True
    assert unavailable["value"] == ""
    assert unavailable["status"] == "Could not load settings. Editing unavailable."
    assert_concurrency_ignored(states["startupUnavailableGuard"])
    assert_concurrency_ready(states["loaded"], 3)
    assert_concurrency_ignored(states["unloadedGuard"])


def test_gptpro_concurrency_runtime_saves_immediately_and_adopts_server_envelope(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["gptProConcurrencyStates"]
    assert states["immediateRequests"] == [{
        "url": "/admin/settings/gptpro",
        "options": {
            "method": "PUT",
            "headers": {"Content-Type": "application/json"},
            "body": "{\"max_concurrent_asks\":5}",
        },
    }]
    pending = states["pending"]
    assert pending["value"] == "5"
    assert pending["draft"] == 5
    assert pending["live"] == 3
    assert pending["isSaving"] is True
    assert pending["selectDisabled"] is True
    assert pending["status"] == "Saving…"
    assert_concurrency_ready(states["saved"], 4)
    assert states["saved"]["status"] == "Applied · 4 tabs"
    assert states["saved"]["messageKind"] == "success"
    locked = states["savedLocked"]
    assert locked["value"] == "2"
    assert locked["draft"] == locked["live"] == 2
    assert locked["isSaving"] is False
    assert locked["locked"] is True
    assert locked["selectDisabled"] is True
    assert_concurrency_ignored(states["busyGuard"])
    assert states["suppressedGetRequests"] == 0


def test_gptpro_concurrency_runtime_validates_and_blocks_locked_or_invalid_events(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["gptProConcurrencyStates"]
    assert states["validation"] == [True, True, *([False] * 8)]
    assert_concurrency_ignored(states["invalidGuard"])
    assert_concurrency_ignored(states["lockedGuard"])
    locked = states["envLocked"]
    assert locked["value"] == "3"
    assert locked["draft"] == locked["live"] == 3
    assert locked["locked"] is True
    assert locked["selectDisabled"] is True
    assert locked["isSaving"] is False


def test_gptpro_concurrency_runtime_rolls_back_failed_put_and_allows_retry(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["gptProConcurrencyStates"]["saveErrors"]
    assert len(scenarios) == 10
    for scenario in scenarios:
        failed = scenario["failed"]
        assert_concurrency_ready(failed, 3)
        assert failed["status"] == "Could not save settings. Previous value restored."
        assert failed["messageKind"] == "error"
        assert_concurrency_ready(scenario["retried"], 5)
        assert scenario["retried"]["messageKind"] == "success"


def test_gptpro_concurrency_runtime_get_failure_disables_until_valid_recovery(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["gptProConcurrencyStates"]["getErrors"]
    assert len(scenarios) == 10
    for scenario in scenarios:
        unavailable = scenario["unavailable"]
        assert unavailable["hasLoaded"] is False
        assert unavailable["selectDisabled"] is True
        assert unavailable["isSaving"] is False
        assert unavailable["value"] == ""
        assert unavailable["draft"] is None
        assert unavailable["status"] == "Could not load settings. Editing unavailable."
        assert unavailable["messageKind"] == "error"
        assert "ERROR" in scenario["toast"]
        assert "GET /admin/settings/gptpro" in scenario["toast"]
        assert_concurrency_ignored(scenario["guard"])
        assert_concurrency_ready(scenario["recovered"], 2)
        assert scenario["recovered"]["status"] == ""


def test_gptpro_concurrency_runtime_ignores_stale_get_success_and_errors(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["gptProConcurrencyStates"]["staleGets"]
    assert len(scenarios) == 6
    assert {(scenario["outcome"], scenario["settleDuringSave"]) for scenario in scenarios} == {
        (outcome, during) for outcome in ("success", "error", "network")
        for during in (True, False)
    }
    for scenario in scenarios:
        assert scenario["generationAfterSave"] > scenario["generationBeforeSave"]
        assert scenario["before"] == scenario["after"]
        assert scenario["toastBefore"] == scenario["toastAfter"]
        assert_concurrency_ready(scenario["saved"], 4)


def test_gptpro_concurrency_runtime_conflict_waits_for_refresh_and_fails_closed(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["gptProConcurrencyStates"]["conflicts"]
    assert len(scenarios) == 5
    for scenario in scenarios:
        pending = scenario["pendingRefresh"]
        assert pending["locked"] is True
        assert pending["selectDisabled"] is True
        assert pending["isSaving"] is True
        assert pending["live"] == 3  # Never adopt the 409 error body's value 9.
        assert scenario["settledBeforeRefresh"] is False
        assert_concurrency_ignored(scenario["guard"])
        assert [request["options"] and request["options"]["method"] for request in scenario["requestsBeforeRefresh"]] == ["PUT", None]
        assert all(request["url"] == "/admin/settings/gptpro" for request in scenario["requestsBeforeRefresh"])
        refreshed = scenario["refreshed"]
        assert refreshed["isSaving"] is False
        assert scenario["saveSettled"] is True
        if scenario["outcome"] == "unlocked":
            assert_concurrency_ready(refreshed, 2)
        elif scenario["outcome"] == "locked":
            assert refreshed["locked"] is True
            assert refreshed["selectDisabled"] is True
            assert refreshed["live"] == refreshed["draft"] == 2
            assert refreshed["value"] == "2"
        else:
            assert refreshed["locked"] is True
            assert refreshed["selectDisabled"] is True
            assert refreshed["live"] == refreshed["draft"] == 3
            assert refreshed["value"] == "3"
            assert refreshed["status"] == "Could not load current settings. Editing remains locked."
            assert refreshed["messageKind"] == "error"
        assert_concurrency_ready(scenario["recovered"], 7)


def test_gptpro_concurrency_runtime_uses_authenticated_json_channel(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    assert dashboard_runtime_result["gptProConcurrencyStates"]["authRequests"] == [{
        "url": "/admin/settings/gptpro",
        "options": {
            "method": "PUT",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer concurrency-test-token",
            },
            "body": "{\"max_concurrent_asks\":6}",
        },
    }]


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
        "installing": {
            "buttonText": "Cancel login",
            "buttonDisabled": False,
            "detail": (
                "no compatible browser found; installing Playwright Chromium "
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


def test_local_login_hero_describes_effective_balanced_participation(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["localHeroStates"]
    assert "밸런스 서빙 참여 설정" in states["balanced-true"]
    assert "게이트웨이 서빙 설정 미확인" in states["unknown"]
    assert "유효한 토큰 필요" in states["balanced-true"]
    assert "중복 등록 제외" in states["balanced-true"]
    assert "Local organization" in states["balanced-true"]
    assert "게이트웨이 서빙과 무관" not in states["balanced-true"]
    for mode in ["disabled", "fallback", "balanced"]:
        for included in ["true", "false"]:
            if mode == "balanced" and included == "true":
                continue
            assert "게이트웨이 서빙에 사용하지 않음" in states[f"{mode}-{included}"]
    assert "밸런스 서빙 참여 설정" in states["unappliedDraft"]
    assert "로컬 Claude Code 로그인이 없습니다" in states["noLoginIncluded"]
    assert "밸런스 서빙 참여 설정" in states["noLoginIncluded"]
    assert "로컬 로그인 필요" in states["noLoginIncluded"]
    assert "게이트웨이 서빙에 사용하지 않음" in states["noLoginExcluded"]


def test_routing_envelope_requires_read_only_local_login_flag(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    assert dashboard_runtime_result["routingEnvelopeValidation"] == {
        "included": True,
        "excluded": True,
        "missing": False,
        "invalid": False,
    }


def test_routing_first_preserves_local_hero_until_login_fetch_settles(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["localHeroLoadOrder"]
    assert "로컬 Claude Code 로그인이 없습니다" not in states["routingFirst"]
    assert states["routingFirst"] == states["initial"]
    assert states["pendingLocal"] == states["initial"]
    assert "loaded@example.com" in states["loaded"]
    assert "밸런스 서빙 참여 설정" in states["loaded"]
    assert "로컬 Claude Code 로그인이 없습니다" not in states["loaded"]
    assert "게이트웨이 서빙에 사용하지 않음" in states["repaintAfterSuccess"]
    assert "밸런스 서빙 참여 설정" in states["repaintAfterLocalError"]


def test_dashboard_health_boot_sends_token_and_renders_identities_after_admin_retry(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["healthAuthenticationBoot"]
    assert len(scenarios) == 3
    for scenario in scenarios:
        assert scenario["helloHasToken"] is False
        assert scenario["promptCount"] == (1 if scenario["firstTokenWasValid"] else 2)
        admin_requests = scenario["adminRequests"]
        assert len(admin_requests) == (8 if scenario["firstTokenWasValid"] else 16)
        assert all(
            request["hasValidToken"] is scenario["firstTokenWasValid"]
            for request in admin_requests[:8]
        )
        assert all(request["hasValidToken"] for request in admin_requests[8:])
        assert scenario["healthRequests"] == [{"url": "/health", "hasValidToken": True}]
        assert "codex-identity@example.com" in scenario["identities"]["codex"]
        assert "kimi-account-identity" in scenario["identities"]["kimi"]
        assert "grok-identity@example.com" in scenario["identities"]["grok"]
