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


@pytest.mark.parametrize("raw,provider,model", [
    ("codex:model", "codex", "model"),
    ("codex: model", "codex", "model"),
    ("codex :model", "codex", "model"),
    (" \tcodex \t: \tmodel:variant:tail \n", "codex", "model:variant:tail"),
    ("openai-by-name:Manual:Model:alpha", "openai-by-name", "Manual:Model:alpha"),
    (" openai-by-name : Manual:Model:alpha ", "openai-by-name", "Manual:Model:alpha"),
    ("UnlistedCase:UnknownCase:next", "UnlistedCase", "UnknownCase:next"),
    (" UnlistedCase : UnknownCase:next ", "UnlistedCase", "UnknownCase:next"),
])
def test_route_of_preserves_provider_and_complete_model_suffix(
    dashboard_runtime_result: dict[str, Any], raw: str, provider: str, model: str,
) -> None:
    row = next(row for row in dashboard_runtime_result["routeParsing"] if row["raw"] == raw)
    assert row["raw"] == raw
    assert row["parsed"] == {"provider": provider, "model": model}


@pytest.mark.parametrize("raw", [
    "codex:model:variant:tail", "codex: model:variant:tail", "codex :model:variant:tail",
    " \tcodex \t: \tmodel:variant:tail \n",
])
@pytest.mark.parametrize("is_locked", [False, True])
@pytest.mark.parametrize("support", ["supported", "unsupported", "missing"])
def test_applied_route_capabilities_match_backend_parts_without_rewriting_raw_ids(
    dashboard_runtime_result: dict[str, Any], raw: str, is_locked: bool, support: str,
) -> None:
    row = next(row for row in dashboard_runtime_result["appliedRouteStates"]
               if (row["raw"], row["isLocked"], row["support"]) == (raw, is_locked, support))
    assert row["parsed"] == {"provider": "codex", "model": "model:variant:tail"}
    assert row["customParsed"] == {"provider": "openai-by-name", "model": "Custom:Model"}
    assert "configured Fast" in row["header"] and "applied mappings only" in row["header"]
    assert row["before"] == row["after"]
    first = row["after"].split("</li>", 1)[0]
    assert "<b>model:variant:tail</b>" in first
    if support == "supported":
        assert "Supported speeds: Standard, Fast" in first
        assert "Supported · Applicable: Fast · Outbound wire: priority" in first
        assert "No capability entry" not in first
    elif support == "unsupported":
        assert "Unsupported · Applicable: Standard · Outbound wire: omitted (Standard)" in first
        assert "Synthetic catalog does not advertise Fast" in first
    else:
        assert "Unknown · Applicable: Unknown · Outbound wire: Unknown" in first
        assert "No capability entry for this applied model" in first
    assert "canonical-extra" in row["after"]
    assert "Custom:Model" not in row["after"]
    assert "draft-only" not in row["after"] and "staged-only" not in row["after"]
    snapshot = row["snapshot"]
    assert snapshot["rawLive"] == {"opus": raw, "haiku": " openai-by-name : Custom:Model ", "sonnet": "codex:canonical-extra"}
    assert snapshot["rawDraft"]["opus"] == "codex:draft-only"
    assert raw in snapshot["targetIds"] and raw in snapshot["renderedTargetIds"]
    assert " openai-by-name : Custom:Model " in snapshot["targetIds"]
    assert snapshot["isLocked"] is snapshot["isLockVisible"] is snapshot["selectDisabled"] is is_locked
    assert row["requests"] == [{"url": "/admin/providers/codex/capabilities/refresh", "options": {
        "method": "POST", "headers": {"Content-Type": "application/json"}, "body": "{}",
    }}]


@pytest.mark.parametrize("raw", [
    "codex:model:variant:tail", "codex: model:variant:tail", "codex :model:variant:tail",
    " \tcodex \t: \tmodel:variant:tail \n", " openai-by-name : Custom:Model ",
])
def test_selected_speed_consumer_accepts_backend_normalized_codex_prefix_only(
    dashboard_runtime_result: dict[str, Any], raw: str,
) -> None:
    row = next(row for row in dashboard_runtime_result["normalizedSpeedTargets"] if row["raw"] == raw)
    assert row["before"] == []
    assert row["savedTier"] == row["draftTier"] == "fast"
    if raw.strip().startswith("openai-by-name"):
        assert row["requests"] == []
        assert "Enter a codex:model target" in row["result"]
    else:
        assert row["requests"] == [{"url": "/admin/test", "options": {
            "method": "POST", "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"target": raw.strip(), "service_tier": "fast"}, separators=(",", ":")),
        }}]
        assert "Requested UI speed: Fast" in row["result"]
        assert "Final outbound wire tier: priority" in row["result"]
        assert "model:variant:tail" in row["result"]
        assert "Speed confirmation: Confirmed" in row["result"]


@pytest.mark.parametrize("raw,model", [
    ("codex:\u0085model:tail", "model:tail"), ("codex\u0085:model:tail", "model:tail"),
    ("codex:\ufeffmodel:tail", "\ufeffmodel:tail"), ("codex:model:tail\ufeff", "model:tail\ufeff"),
])
def test_route_boundaries_match_python_strip_and_preserve_opaque_model_data(
    dashboard_runtime_result: dict[str, Any], raw: str, model: str,
) -> None:
    row = next(row for row in dashboard_runtime_result["routeBoundaryParity"] if row["raw"] == raw)
    assert row["expectedModel"] == model
    assert row["parsed"] == {"provider": "codex", "model": model}
    assert f"<b>{model}</b>" in row["models"]
    assert "Supported · Applicable: Fast · Outbound wire: priority" in row["models"]
    assert "No capability entry" not in row["models"]
    assert row["live"] == {"opus": raw}
    assert row["targetIds"] == [raw]


@pytest.mark.parametrize("raw,model", [
    ("codex:\u0085model:tail", "model:tail"), ("codex\u0085:model:tail", "model:tail"),
    ("codex:\ufeffmodel:tail", "\ufeffmodel:tail"), ("codex:model:tail\ufeff", "model:tail\ufeff"),
])
def test_explicit_target_keeps_backend_opaque_model_characters(
    dashboard_runtime_result: dict[str, Any], raw: str, model: str,
) -> None:
    row = next(row for row in dashboard_runtime_result["routeBoundaryParity"] if row["raw"] == raw)
    assert len(row["requests"]) == 1
    assert row["requests"][0]["url"] == "/admin/test"
    assert json.loads(row["requests"][0]["options"]["body"]) == {"target": raw, "service_tier": "fast"}
    assert f"response.model: {model}" in row["diagnostic"]
    assert "Speed confirmation: Confirmed" in row["diagnostic"]


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


def test_dashboard_interface_copy_is_english() -> None:
    assert '<html lang="en">' in DASHBOARD_HTML
    assert "<h2>Claude accounts</h2>" in DASHBOARD_HTML
    assert '>Add account</button>' in DASHBOARD_HTML
    assert '<button id="comp-apply">Apply</button>' in DASHBOARD_HTML
    assert "This gateway requires CLAUDEX_LOCAL_TOKEN authentication." in DASHBOARD_JAVASCRIPT
    for asset in (DASHBOARD_HTML, DASHBOARD_JAVASCRIPT, DASHBOARD_CSS):
        assert HANGUL_PATTERN.search(asset) is None


def test_account_toolbar_wraps_on_narrow_screens() -> None:
    start = DASHBOARD_CSS.index('@media (max-width:760px){')
    end = DASHBOARD_CSS.index('@media (prefers-reduced-motion:reduce){', start)
    assert '#scard-accounts .tbl-cap{flex-wrap:wrap}' in DASHBOARD_CSS[start:end]


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
    assert "No catalog" in picker["cataloglessState"]["stateText"]


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
    assert unavailable["status"] == "Could not load settings. Editing unavailable. Reopen MCP to retry."
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
        assert failed["status"] == "Could not confirm the save. Showing the last confirmed value. Reopen MCP to check the current setting."
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
        assert unavailable["status"] == "Could not load settings. Editing unavailable. Reopen MCP to retry."
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
            assert refreshed["status"] == "Could not load current settings. Editing remains locked. Reopen MCP to retry."
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
        "text": "● VALID Authentication cookie expires in 8d 0h",
        "title": "",
    }
    assert states["expiring"] == {
        "className": "stat warn",
        "text": "● EXPIRING SOON Authentication cookie expires in 7d 0h",
        "title": "",
    }
    assert states["expired"]["className"] == "stat err"
    assert "claudex-gateway gptpro login" in states["expired"]["text"]
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
                "MCP registered for the gateway user on the gateway machine. "
                "Restart Claude Code sessions there to load it."
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
    assert "Configured for Balanced routing" in states["balanced-true"]
    assert "Gateway serving policy not yet loaded" in states["unknown"]
    assert "valid token required" in states["balanced-true"]
    assert "duplicate registered identities excluded" in states["balanced-true"]
    assert "Local organization" in states["balanced-true"]
    assert "Unrelated to gateway serving" not in states["balanced-true"]
    for mode in ["disabled", "fallback", "balanced"]:
        for included in ["true", "false"]:
            if mode == "balanced" and included == "true":
                continue
            assert "Not included as a local-login member of the gateway pool" in states[f"{mode}-{included}"]
    assert "Configured for Balanced routing" in states["unappliedDraft"]
    assert "No local Claude Code login found" in states["noLoginIncluded"]
    assert "Configured for Balanced routing" in states["noLoginIncluded"]
    assert "local sign-in required" in states["noLoginIncluded"]
    assert "Not included as a local-login member of the gateway pool" in states["noLoginExcluded"]


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
    assert "No local Claude Code login found" not in states["routingFirst"]
    assert states["routingFirst"] == states["initial"]
    assert states["pendingLocal"] == states["initial"]
    assert "loaded@example.com" in states["loaded"]
    assert "Configured for Balanced routing" in states["loaded"]
    assert "No local Claude Code login found" not in states["loaded"]
    assert "Not included as a local-login member of the gateway pool" in states["repaintAfterSuccess"]
    assert "Configured for Balanced routing" in states["repaintAfterLocalError"]


def test_dashboard_health_boot_sends_token_and_renders_identities_after_admin_retry(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["healthAuthenticationBoot"]
    assert len(scenarios) == 3
    for scenario in scenarios:
        assert scenario["helloHasToken"] is False
        assert scenario["promptCount"] == (1 if scenario["firstTokenWasValid"] else 2)
        admin_requests = scenario["adminRequests"]
        assert len(admin_requests) == (9 if scenario["firstTokenWasValid"] else 18)
        assert all(
            request["hasValidToken"] is scenario["firstTokenWasValid"]
            for request in admin_requests[:9]
        )
        assert all(request["hasValidToken"] for request in admin_requests[9:])
        assert sum(request["url"] == "/admin/providers/codex/capabilities" for request in admin_requests) == (
            1 if scenario["firstTokenWasValid"] else 2
        )
        assert scenario["healthRequests"] == [{"url": "/health", "hasValidToken": True}]
        assert "codex-identity@example.com" in scenario["identities"]["codex"]
        assert "kimi-account-identity" in scenario["identities"]["kimi"]
        assert "grok-identity@example.com" in scenario["identities"]["grok"]


def test_operator_guidance_discloses_scope_and_prerequisites() -> None:
    assert "on the gateway machine, across all projects" in DASHBOARD_HTML
    assert "registration stores the configured local token" in DASHBOARD_HTML
    assert "with a graphical desktop available" in DASHBOARD_HTML
    assert "No graphical browser available?" not in DASHBOARD_HTML
    assert "local cookie check, not a live check" in DASHBOARD_HTML
    assert "provider-prefixed target (provider:model)" in DASHBOARD_HTML
    assert "models whose catalog does not advertise Fast stay standard" in DASHBOARD_HTML
    assert "Speed and usage costs depend on the provider and workload" in DASHBOARD_HTML
    assert "~1.5x" not in DASHBOARD_HTML
    assert "~2–2.5x" not in DASHBOARD_HTML
    assert "Clearing a serving pin does not remove an account from the Balanced pool" in DASHBOARD_HTML
    assert 'var GPTPRO_LOGIN_COMMAND="claudex-gateway gptpro login";' in DASHBOARD_JAVASCRIPT
    assert "run claudex-gateway gptpro login" not in DASHBOARD_JAVASCRIPT
    apply_copy = javascript_section("function apply(){", 'document.getElementById("applybtn")')
    assert "Router mapping" in apply_copy
    assert "Claude → Codex" not in apply_copy


def test_toasts_expose_error_and_success_announcements(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    feedback = dashboard_runtime_result["operatorFeedback"]
    assert feedback["alertToast"] == {
        "role": "alert", "live": "assertive", "atomic": "true", "html": "<b>First failure</b>"
    }
    assert feedback["statusToast"] == {
        "role": "status", "live": "polite", "atomic": "true", "html": "Saved"
    }
    assert feedback["rejectedCode"]["toast"]["role"] == "alert"
    assert "&lt;invalid code>" in feedback["rejectedCode"]["toast"]["html"]
    assert "<invalid code>" not in feedback["rejectedCode"]["toast"]["html"]


def test_login_code_errors_remain_in_dialog_without_changing_attempt_requests(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    feedback = dashboard_runtime_result["operatorFeedback"]
    rejected = feedback["rejectedCode"]
    assert rejected["hidden"] is False
    assert "<invalid code>" in rejected["text"]
    assert "Check the code and sign-in status before retrying" in rejected["text"]
    assert feedback["retainedCodeError"] == rejected["text"]
    assert feedback["disconnectedCode"]["hidden"] is False
    assert "Gateway unreachable" in feedback["disconnectedCode"]["text"]
    assert feedback["acceptedCode"] == {"text": "", "hidden": True}
    assert feedback["staleCodeIgnored"] is True
    assert feedback["codeRequests"] == [{
        "url": "/admin/providers/claude/login/code",
        "options": {
            "method": "POST",
            "headers": {"Content-Type": "application/json", "X-Login-Attempt": "feedback-attempt"},
            "body": '{"code":"synthetic-sign-in-code"}',
        },
    }] * 3
    assert 'id="login-code-error" role="alert" aria-live="assertive" aria-atomic="true" hidden' in DASHBOARD_JAVASCRIPT
    assert 'aria-describedby="login-code-error"' in DASHBOARD_JAVASCRIPT


def test_save_response_loss_copy_distinguishes_displayed_and_persisted_values(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    outcome = dashboard_runtime_result["operatorFeedback"]["responseLoss"]
    assert outcome["persistedLimit"] == 5
    assert outcome["displayedLimit"] == "3"
    assert outcome["message"] == (
        "Could not confirm the save. Showing the last confirmed value. "
        "Reopen MCP to check the current setting."
    )
    assert outcome["requests"] == [{
        "url": "/admin/settings/gptpro",
        "options": {
            "method": "PUT", "headers": {"Content-Type": "application/json"},
            "body": '{"max_concurrent_asks":5}',
        },
    }]


def test_account_pin_and_usage_labels_do_not_claim_exclusive_serving_or_duration(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    feedback = dashboard_runtime_result["operatorFeedback"]
    assert "Serving pin set" in feedback["servingCopy"]["pinned"]
    assert "Clear serving pin" in feedback["servingCopy"]["pinned"]
    assert "Set serving pin" in feedback["servingCopy"]["unpinned"]
    assert "Serving pin set" not in feedback["servingCopy"]["unpinned"]
    assert ">Remove</button>" in feedback["servingCopy"]["unpinned"]
    assert "Session window" in feedback["unknownDurationUsage"]
    assert "5-hour" not in feedback["unknownDurationUsage"]


@pytest.mark.parametrize(
    "plan_type,expected_html",
    [
        ("<future-plan>", "<b>&lt;future-plan></b> plan"),
        ("PRO", "<b>PRO</b> plan"),
        ("constructor", "<b>constructor</b> plan"),
        ("toString", "<b>toString</b> plan"),
        ("__proto__", "<b>__proto__</b> plan"),
        ("", ""),
        (None, ""),
    ],
)
def test_usage_plan_preserves_unknown_codex_values(
    dashboard_runtime_result: dict[str, Any], plan_type: str | None, expected_html: str,
) -> None:
    rendered = next(
        row for row in dashboard_runtime_result["usageRendering"]["plans"]
        if row["provider"] == "codex" and row["planType"] == plan_type
    )
    assert rendered["html"] == expected_html
    assert rendered["rawPlanType"] == plan_type


@pytest.mark.parametrize("provider", ["claude", "kimi", "grok"])
@pytest.mark.parametrize("plan_type", ["pro", "prolite", "promax", "plus", "<future-plan>"])
def test_usage_plan_keeps_other_providers_raw(
    dashboard_runtime_result: dict[str, Any], provider: str, plan_type: str,
) -> None:
    rendered = next(
        row for row in dashboard_runtime_result["usageRendering"]["plans"]
        if row["provider"] == provider and row["planType"] == plan_type
    )
    assert rendered["html"] == f"<b>{plan_type.replace('<', '&lt;')}</b> plan"
    assert rendered["rawPlanType"] == plan_type


def test_codex_usage_windows_preserve_weekly_only(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    body = dashboard_runtime_result["usageRendering"]["windows"]["weeklyOnly"]
    assert "Weekly window" in body
    assert "37%" in body
    assert body.count('class="uwin"') == 1
    assert "Session window" not in body
    assert "0%" not in body
    assert "unlimited" not in body.lower()
    assert "Reset credits" not in body


def test_codex_usage_windows_do_not_invent_unreported_limits(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    body = dashboard_runtime_result["usageRendering"]["windows"]["noWindows"]
    assert "No usage windows reported" in body
    assert 'class="uwin"' not in body
    assert "Session window" not in body
    assert "Weekly window" not in body
    assert "0%" not in body
    assert "unlimited" not in body.lower()
    assert "Reset credits" not in body


@pytest.mark.parametrize(
    "plan_type,expected_label",
    [
        ("pro", "Pro 200"),
        ("prolite", "Pro 100"),
        ("promax", "Pro 500"),
        ("plus", "Plus"),
    ],
)
def test_codex_usage_plan_labels(
    dashboard_runtime_result: dict[str, Any], plan_type: str, expected_label: str,
) -> None:
    rendered = next(
        row for row in dashboard_runtime_result["usageRendering"]["plans"]
        if row["provider"] == "codex" and row["planType"] == plan_type
    )
    assert rendered["html"] == f"<b>{expected_label}</b> plan"
    assert rendered["rawPlanType"] == plan_type


def test_codex_usage_copy_discloses_shared_provider_usage_and_credits() -> None:
    match = re.search(
        r'<h2>Codex Status</h2>.*?<div class="sub">(.*?)</div>',
        DASHBOARD_HTML,
        re.DOTALL,
    )
    assert match is not None
    copy = match.group(1)
    assert "ChatGPT Work and Codex share provider-reported usage and credits" in copy
    assert "including activity outside this gateway" in copy


def test_codex_speed_selector_supports_three_tiers_and_reloads(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexSpeedStates"]
    assert states["selectPresent"] is True
    assert states["validEnvelopes"] == [True, True, True]
    assert states["invalidEnvelopes"] == [False, False, False, False]
    assert states["initial"] == {
        "value": "standard", "live": None, "draft": None, "applyDisabled": True,
    }
    assert [row["request"] for row in states["selections"]] == [
        {"service_tier": "fast"}, {"service_tier": "ultrafast"}, {"service_tier": None},
    ]
    for row, tier, value in zip(
        states["selections"], ["fast", "ultrafast", None], ["fast", "ultrafast", "standard"],
    ):
        assert row["draft"]["draft"] == tier
        assert row["draft"]["applyDisabled"] is False
        assert row["saved"] == {"value": value, "live": tier, "draft": tier, "applyDisabled": True}
    assert states["reloaded"] == {
        "value": "ultrafast", "live": "ultrafast", "draft": "ultrafast", "applyDisabled": True,
    }
    assert states["nonSettingsRequests"] == [
        {"url": "/admin/providers/codex/capabilities", "options": None},
    ] * 3


def test_codex_speed_control_and_consumption_notice_live_in_router() -> None:
    router = DASHBOARD_HTML.split('id="tab-map"', 1)[1].split('id="tab-status"', 1)[0]
    assert 'id="codex-card"' in router
    assert 'id="codex-fast"' not in DASHBOARD_HTML
    assert '<option value="standard">Standard</option>' in router
    assert '<option value="fast">Fast</option>' in router
    assert '<option value="ultrafast">Ultrafast</option>' in router
    assert router.index('id="codex-consumption"') < router.index('id="codex-speed"')
    notice = router.split('id="codex-consumption"', 1)[1].split('</div>', 1)[0]
    assert "included allowance" in notice
    assert "8x Standard" in notice
    assert "purchased credits" in notice
    assert "6x Standard" in notice
    assert "not a speed guarantee" in notice
    assert 'aria-describedby="codex-consumption codex-speed-hint"' in router


def test_codex_settings_start_disabled_and_reject_locked_or_invalid_events(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexSettingsStates"]
    initial = states["initial"]
    assert initial["hasLoaded"] is False
    assert initial["value"] == ""
    assert initial["selectDisabled"] is True
    assert initial["applyDisabled"] is True
    for row in states["locks"]:
        assert row["state"]["locked"] is True
        assert row["state"]["value"] == (row["tier"] or "standard")
        assert row["state"]["draft"] == row["tier"]
        assert row["state"]["selectDisabled"] is True
        assert row["state"]["applyDisabled"] is True
        assert row["requests"] == []
    assert states["invalidChange"]["state"]["value"] == "fast"
    assert states["invalidChange"]["state"]["draft"] == "fast"
    assert states["invalidChange"]["requests"] == []


def test_codex_settings_block_events_while_save_is_pending(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenario = dashboard_runtime_result["codexSettingsStates"]["pendingSave"]
    for state in (scenario["pendingState"], scenario["afterForcedEvents"]):
        assert state["isSaving"] is True
        assert state["selectDisabled"] is True
        assert state["applyDisabled"] is True
        assert state["draft"] == "ultrafast"
    assert len(scenario["requests"]) == 1
    assert json.loads(scenario["requests"][0]["options"]["body"]) == {"service_tier": "ultrafast"}
    assert scenario["saved"]["live"] == "ultrafast"
    assert scenario["saved"]["isSaving"] is False


def test_codex_settings_keep_draft_on_save_errors_and_reload_confirmed_state(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["codexSettingsStates"]["saveErrors"]
    assert len(scenarios) == 4
    for row in scenarios:
        failed = row["failed"]
        assert failed["live"] == "fast"
        assert failed["draft"] == "ultrafast"
        assert failed["value"] == "ultrafast"
        assert failed["applyDisabled"] is False
        assert failed["isSaving"] is False
        assert "Could not confirm" in failed["status"]
        assert "Reload settings" in failed["status"]
        assert "APPLIED" not in failed["toast"]
        recovered = row["recovered"]
        assert recovered["value"] == "ultrafast"
        assert recovered["live"] == recovered["draft"] == "ultrafast"
        assert recovered["hasLoaded"] is True
        assert recovered["applyDisabled"] is True
    error = next(row for row in scenarios if row["outcome"] == "error")
    assert "&lt;save failed>" in error["failed"]["toast"]


def test_codex_settings_load_errors_disable_editing_until_reload(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["codexSettingsStates"]["loadErrors"]
    assert len(scenarios) == 3
    for row in scenarios:
        assert row["failed"]["hasLoaded"] is False
        assert row["failed"]["selectDisabled"] is True
        assert row["failed"]["applyDisabled"] is True
        assert "Reload settings" in row["failed"]["status"]
        assert row["guardRequests"] == []
        assert row["recovered"]["hasLoaded"] is True
        assert row["recovered"]["value"] == "ultrafast"
        assert row["recovered"]["selectDisabled"] is False


def test_codex_settings_conflict_uses_only_valid_refresh_and_keeps_failed_lock(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenarios = dashboard_runtime_result["codexSettingsStates"]["conflicts"]
    assert len(scenarios) == 5
    for row in scenarios:
        pending = row["pendingRefresh"]
        assert pending["locked"] is True
        assert pending["live"] == "fast"
        assert pending["draft"] == "ultrafast"
        assert pending["selectDisabled"] is True
        assert pending["isSaving"] is True
        assert [request["url"] for request in row["beforeRefreshRequests"]] == [
            "/admin/settings/codex", "/admin/settings/codex",
        ]
        refreshed = row["refreshed"]
        assert refreshed["isSaving"] is False
        if row["outcome"] in ("unlocked", "locked"):
            assert refreshed["live"] is None
            assert refreshed["draft"] is None
            assert refreshed["value"] == "standard"
            assert refreshed["locked"] is (row["outcome"] == "locked")
        else:
            assert refreshed["locked"] is True
            assert refreshed["live"] == "fast"
            assert "Editing remains locked" in refreshed["status"]
        assert row["recovered"]["locked"] is False
        assert row["recovered"]["value"] == "ultrafast"


def test_codex_capabilities_render_mixed_applied_models_without_entitlement_inference(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexCapabilitiesStates"]
    mixed = states["mixed"]
    assert "Current" in mixed["header"]
    assert "configured Ultrafast" in mixed["header"]
    assert "applied mappings only" in mixed["header"]
    assert states["expectedFetchedTime"] in mixed["time"]
    models = mixed["models"]
    assert "Supported speeds: Standard, Fast, Ultrafast" in models
    assert "Supported speeds: Standard, Fast" in models
    assert "Supported speeds: Unknown" in models
    assert "Applicable: Ultrafast" in models
    assert "Applicable: Standard" in models
    assert "Outbound wire: ultrafast" in models
    assert "Outbound wire: omitted (Standard)" in models
    assert "Unsupported" in models
    assert "Unknown" in models
    assert "Catalog does not advertise ultrafast" in models
    assert "not listed in the authenticated catalog" in models
    assert "tier metadata is incomplete" in models
    assert "Advertised tier IDs: priority, ultrafast, future-tier" in models
    assert "manual:future&lt;script>" in models
    assert "manual:future<script>" not in models
    assert "draft-only" not in models
    assert "staged-only" not in models
    assert "unrelated" not in models
    assert "kimi-model" not in models
    assert models.count("<b>both</b>") == 1
    assert "No applied Codex mappings" in states["noMappings"]["models"]


def test_codex_capabilities_remain_configured_not_draft_selection(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexCapabilitiesStates"]
    draft = states["draftSelection"]
    assert draft["header"] == states["mixed"]["header"]
    assert draft["models"] == states["mixed"]["models"]
    assert "Configured speed: Ultrafast" in draft["settings"]
    assert "Draft speed: Fast (not applied)" in draft["settings"]


def test_codex_capabilities_preserve_top_level_stale_unknown_and_fetch_time(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexCapabilitiesStates"]
    stale = states["stale"]
    assert "Stale" in stale["header"]
    assert "Catalog refresh failed: <busy>" in stale["header"]
    assert stale["time"] == states["mixed"]["time"]
    assert "Supported" in stale["models"]
    unknown = states["unknown"]
    assert "Unknown" in unknown["header"]
    assert "Authentication unavailable" in unknown["header"]
    assert "Last fetched: Never" == unknown["time"]
    assert "Supported speeds: Unknown" in unknown["models"]
    assert "Current account metadata unavailable" in unknown["models"]
    assert "Unsupported" not in unknown["models"]


def test_codex_metadata_boot_and_setting_or_mapping_saves_read_serving_capabilities_only(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexMetadataStates"]
    assert "configured Ultrafast" in states["boot"]["header"]
    assert sum(row["url"] == "/admin/providers/codex/capabilities" for row in states["bootRequests"]) == 1
    assert states["draft"]["requests"] == []
    assert states["draft"]["state"]["header"] == states["boot"]["header"]
    assert [row["url"] for row in states["saved"]["requests"]] == [
        "/admin/settings/codex", "/admin/providers/codex/capabilities",
    ]
    assert "configured Fast" in states["saved"]["state"]["header"]
    assert "Outbound wire: priority" in states["saved"]["state"]["models"]
    assert [row["url"] for row in states["mappingSaved"]["requests"]] == [
        "/admin/settings/mapping", "/admin/providers/codex/capabilities",
    ]
    assert "<b>unrelated</b>" in states["mappingSaved"]["state"]["models"]
    assert all(row["url"] != "/admin/test" for row in states["allRequests"])
    assert all(
        row["options"] is None
        for row in states["allRequests"] if row["url"] == "/admin/providers/codex/capabilities"
    )


def test_codex_metadata_refresh_is_explicit_post_and_updates_same_account_support(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexMetadataStates"]
    refreshed = states["refreshed"]
    assert refreshed["requests"] == [{
        "url": "/admin/providers/codex/capabilities/refresh",
        "options": {"method": "POST", "headers": {"Content-Type": "application/json"}, "body": "{}"},
    }]
    assert "configured Ultrafast" in refreshed["state"]["header"]
    assert "Unsupported" in states["boot"]["models"]
    assert "Unsupported" not in refreshed["state"]["models"]
    assert "<b>fast-only</b>" in refreshed["state"]["models"]
    assert refreshed["state"]["time"] != states["boot"]["time"]
    assert "Refresh capabilities" in DASHBOARD_HTML
    assert "no model request" in DASHBOARD_HTML
    assert len([row for row in states["allRequests"] if row["url"].endswith("/capabilities/refresh")]) == 1


def test_codex_metadata_ignores_obsolete_reply_and_uses_authenticated_json_channel(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexMetadataStates"]
    assert states["race"] is not None
    assert "new current snapshot" in states["race"]["state"]["header"]
    assert "obsolete snapshot" not in states["race"]["state"]["header"]
    assert states["race"]["state"]["value"] == "ultrafast"
    assert states["race"]["locked"] is False
    assert states["authRequests"] == [{
        "url": "/admin/providers/codex/capabilities/refresh",
        "options": {"method": "POST", "headers": {
            "Content-Type": "application/json", "Authorization": "Bearer metadata-test-token",
        }, "body": "{}"},
    }]


def test_codex_metadata_failures_and_malformed_responses_never_claim_support(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexMetadataFailureStates"]
    assert len(states["scenarios"]) == 18
    for row in states["scenarios"]:
        failed = row["failed"]
        assert "Unknown" in failed["header"]
        assert "Refresh capabilities" in failed["header"]
        assert "Supported speeds: Standard" not in failed["models"]
        assert "Applicable: Ultrafast" not in failed["models"]
        assert "Supported speeds: Unknown" in failed["models"]
        assert failed["time"] == "Last fetched: Unavailable"
        assert row["editingEnabled"] is True
        assert row["refreshEnabled"] is True
        if row["outcome"] not in ("http", "network"):
            assert "Invalid capability response" in failed["header"]
            assert "Gateway unreachable" not in failed["header"]
        assert "Current" in row["recovered"]["header"]
        assert "Applicable: Ultrafast" in row["recovered"]["models"]
        assert [request["url"] for request in row["requests"]] == [
            "/admin/providers/codex/capabilities/refresh",
            "/admin/providers/codex/capabilities/refresh",
        ]
    http = next(row for row in states["scenarios"] if row["outcome"] == "http")
    assert "<catalog unavailable>" in http["failed"]["header"]
    assert "No capability entry for this applied model" in states["missingModel"]["models"]


def test_codex_metadata_pending_refresh_and_setting_change_keep_scope_honest(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexMetadataFailureStates"]
    pending = states["pending"]
    assert pending is not None
    assert pending["during"]["disabled"] is True
    assert pending["during"]["busy"] == "true"
    assert "Checking metadata" in pending["during"]["state"]["header"]
    assert len(pending["requests"]) == 1
    assert "Stale" in pending["after"]["header"]
    assert "Same-account fetch failed" in pending["after"]["header"]
    changed = states["changedSetting"]
    assert changed["settings"] == "Configured speed: Fast."
    assert "configured Ultrafast" not in changed["header"]
    assert "Unknown" in changed["header"]
    assert "Applicable: Ultrafast" not in changed["models"]
    assert "configured Fast" in states["changedSettingResolved"]["header"]
    assert "Applicable: Fast" in states["changedSettingResolved"]["models"]


def test_codex_selected_speed_test_is_explicit_and_retains_ordinary_untiered_test(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexSpeedTestStates"]
    assert len(states["rows"]) == 7
    for row in states["rows"]:
        assert row["selectionRequests"] == []
        assert len(row["requests"]) == 1
        assert row["requests"][0]["url"] == "/admin/test"
        assert row["requests"][0]["options"]["method"] == "POST"
        assert row["requests"][0]["options"]["headers"] == {"Content-Type": "application/json"}
        assert json.loads(row["requests"][0]["options"]["body"]) == {
            "target": "codex:manual:model", "service_tier": row["requested"],
        }
        assert row["result"]["live"] is None
        assert row["result"]["draft"] == row["requested"]
        assert row["result"]["hidden"] is False
    assert len(states["ordinary"]["requests"]) == 1
    ordinary = states["ordinary"]["requests"][0]
    assert ordinary["url"] == "/admin/test"
    assert json.loads(ordinary["options"]["body"]) == {"target": "codex:manual:model"}
    assert "OK" in states["ordinary"]["toast"]
    assert "ordinary-model" in states["ordinary"]["toast"]
    assert DASHBOARD_HTML.index('id="codex-consumption"') < DASHBOARD_HTML.index('id="codex-speed-test"')
    assert 'aria-describedby="codex-speed-warning"' in DASHBOARD_HTML
    assert "can consume allowance or purchased credits" in DASHBOARD_HTML
    assert "selector's draft" in DASHBOARD_HTML
    assert "without saving" in DASHBOARD_HTML


def test_codex_selected_speed_test_separates_requested_wire_report_and_confirmation(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    for row in dashboard_runtime_result["codexSpeedTestStates"]["rows"]:
        html = row["result"]["html"]
        requested_label = {None: "Standard", "fast": "Fast", "ultrafast": "Ultrafast"}[row["requested"]]
        assert "Connection: Responded" in html
        assert f"Requested UI speed: {requested_label}" in html
        assert f"Final outbound wire tier: {row['sent'] or 'omitted (Standard)'}" in html
        assert f"Backend-reported tier: {row['reported'] or 'Not reported (unconfirmed)'}" in html
        confirmation_label = {
            "confirmed": "Confirmed", "different": "Different tier", "reported": "Reported (no accelerated tier requested)",
            "unconfirmed": "Unconfirmed",
        }[row["confirmation"]]
        assert f"Speed confirmation: {confirmation_label}" in html
        assert "served-model" in html
        if row["reported"] is None:
            assert "Speed confirmation: Confirmed" not in html
        if row["reason"]:
            assert row["reason"].replace("<", "&lt;") in html
        assert "<provider chose another tier>" not in html
        assert "12ms" in html


def test_codex_selected_speed_test_guards_targets_and_preserves_pending_request_choice(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexSpeedTestStates"]
    for row in states["guards"]:
        assert row["requests"] == []
        assert "Enter a codex:model target" in row["result"]["html"]
        assert row["result"]["role"] == "alert"
    pending = states["pending"]
    assert pending["before"]["disabled"] is True
    assert "Testing" in pending["before"]["html"]
    assert len(pending["requests"]) == 1
    assert pending["after"]["disabled"] is False
    assert pending["after"]["draft"] == "fast"
    assert "Requested UI speed: Ultrafast" in pending["after"]["html"]
    assert "Requested UI speed: Fast" not in pending["after"]["html"]
    assert states["authRequests"] == [{
        "url": "/admin/test", "options": {"method": "POST", "headers": {
            "Content-Type": "application/json", "Authorization": "Bearer speed-test-token",
        }, "body": '{"target":"codex:manual:model","service_tier":"fast"}'},
    }]


@pytest.mark.parametrize("outcome", [
    "legacy", "missing-body", "http", "failed-response", "network", "null-diagnostic", "array-diagnostic",
    "invalid-requested", "echo-mismatch", "missing-wire", "invalid-wire", "invalid-report", "invalid-confirmation",
    "false-confirmed", "false-reported", "invalid-capability", "invalid-capability-reason", "invalid-confirmation-reason",
])
def test_codex_selected_speed_missing_or_malformed_diagnostics_are_unconfirmed(
    dashboard_runtime_result: dict[str, Any], outcome: str,
) -> None:
    rows = dashboard_runtime_result["codexSpeedFailureStates"]["rows"]
    assert len(rows) == 18
    row = next(row for row in rows if row["outcome"] == outcome)
    result = row["result"]
    html = result["html"]
    assert len(row["requests"]) == 1
    assert "Requested UI speed: Ultrafast" in html
    assert "Speed confirmation: Unconfirmed" in html
    assert "Speed confirmation: Confirmed" not in html
    assert result["disabled"] is False
    assert result["hidden"] is False
    if outcome == "legacy":
        assert "Connection: Responded" in html
        assert "did not return tier diagnostics" in html
        assert "Final outbound wire tier: Not available" in html
    elif outcome == "missing-body":
        assert "Invalid speed-test response" in html
    elif outcome == "http":
        assert "&lt;auth failed>" in html
    elif outcome == "failed-response":
        assert "Connection: Failed (502)" in html
        assert "&lt;stream incomplete>" in html
        assert "Backend-reported tier: ultrafast" in html
        assert "did not complete successfully" in html
        assert result["role"] == "alert"
    elif outcome == "network":
        assert "Connection: Unreachable" in html
        assert "Gateway unreachable" in html
        assert result["role"] == "alert"
    else:
        assert "Connection: Responded" in html
        assert "Invalid tier diagnostic response" in html
        assert "Final outbound wire tier: Not available" in html
        assert "Backend-reported tier: Not reported (unconfirmed)" in html


def test_codex_selected_speed_blocks_unknown_or_busy_settings_but_allows_locked_selection(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    states = dashboard_runtime_result["codexSpeedFailureStates"]
    assert len(states["blocked"]) == 3
    for row in states["blocked"]:
        assert row["disabled"] is True
        assert row["requests"] == []
    assert len(states["locked"]["requests"]) == 1
    assert json.loads(states["locked"]["requests"][0]["options"]["body"]) == {
        "target": "codex:manual:model", "service_tier": "ultrafast",
    }
    assert "Speed confirmation: Confirmed" in states["locked"]["result"]["html"]


@pytest.mark.parametrize("outcome", ["saved", "conflict", "reload-error"])
def test_codex_boot_does_not_overwrite_newer_reload_save_or_conflict_lock(
    dashboard_runtime_result: dict[str, Any], outcome: str,
) -> None:
    scenario = next(row for row in dashboard_runtime_result["codexBootRaceStates"] if row["outcome"] == outcome)
    newer = scenario["newer"]
    if outcome == "reload-error":
        assert newer["hasLoaded"] is False
        assert newer["disabled"] is True
        assert "Reload settings" in newer["status"]
    else:
        assert newer["live"] == newer["draft"] == "ultrafast"
        assert newer["value"] == "ultrafast"
        assert newer["locked"] is (outcome == "conflict")
    assert scenario["final"] == newer
    assert all(row["url"] != "/admin/test" for row in scenario["requests"])


@pytest.mark.parametrize("outcome", ["http", "malformed", "network"])
def test_codex_boot_settings_failures_disable_unknown_selection_and_can_reload(
    dashboard_runtime_result: dict[str, Any], outcome: str,
) -> None:
    scenario = next(row for row in dashboard_runtime_result["codexBootFailureStates"] if row["outcome"] == outcome)
    assert scenario["failed"]["hasLoaded"] is False
    assert scenario["failed"]["value"] == ""
    assert scenario["failed"]["disabled"] is True
    assert scenario["failed"]["testDisabled"] is True
    assert "Could not load settings" in scenario["failed"]["status"]
    assert "Reload settings" in scenario["failed"]["status"]
    assert scenario["recovered"] == {"hasLoaded": True, "value": "ultrafast", "disabled": False}
    assert all(row["url"] != "/admin/test" for row in scenario["requests"])


def test_codex_long_action_labels_do_not_inherit_fixed_connection_button_width() -> None:
    selectors = "#codex-reload,#codex-capabilities-refresh,#codex-speed-test"
    rule = re.search(re.escape(selectors) + r"\{([^}]+)\}", DASHBOARD_CSS)
    assert rule is not None
    assert "width:auto" in rule.group(1)
    assert "padding-inline:14px" in rule.group(1)
    assert "max-width:100%" in rule.group(1)
    assert ".ctrow button{width:112px;flex:none;padding-inline:0;text-align:center}" in DASHBOARD_CSS


@pytest.mark.parametrize("provider", ["kimi", "grok"])
@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("authentication", ["none", "valid", "retry"])
@pytest.mark.parametrize("outcome", ["network", "http", "malformed"])
def test_optional_catalog_boot_preserves_core_state_and_authentication(
    dashboard_runtime_result: dict[str, Any], provider: str, locked: bool,
    authentication: str, outcome: str,
) -> None:
    scenario = next(row for row in dashboard_runtime_result["optionalCatalogBoot"]
                    if (row["provider"], row["locked"], row["authentication"], row["outcome"])
                    == (provider, locked, authentication, outcome))
    mapping = {"sonnet": "codex:saved-main", "haiku": "kimi:saved-secondary"}
    assert scenario["initial"] == {"live": {}, "draft": {}, "locked": False, "codexLoaded": False}
    assert set(scenario["decodedBeforeFailure"]) == {
        "/admin/settings/mapping", "/admin/settings/codex", "/admin/settings/log-level",
        "/admin/settings/compaction", "/admin/providers/claude/pool/routing",
    }
    assert scenario["live"] == scenario["draft"] == mapping
    assert scenario["isLocked"] is locked
    assert scenario["isLockVisible"] is locked
    assert {"sonnet", "haiku"}.issubset(scenario["sources"])
    assert set(scenario["targets"]) == set(mapping.values())
    assert scenario["codex"] == {"live": "fast", "draft": "fast", "hasLoaded": True, "value": "fast", "disabled": False}
    assert scenario["compaction"] is None
    assert scenario["routing"] == "disabled"
    assert "Gateway OK" in scenario["health"]
    assert "configured Fast" in scenario["capabilities"]
    assert scenario["catalogs"]["codex"] == ["saved-main"]
    assert scenario["catalogs"][provider] == []
    assert scenario["catalogs"]["grok" if provider == "kimi" else "kimi"] == ["peer-suggestion"]
    requests = scenario["requests"]
    assert all(row["method"] == "GET" and row["url"] != "/admin/test" for row in requests)
    assert len(requests) == (20 if authentication == "retry" else 11)
    assert scenario["prompts"] == {"none": 0, "valid": 1, "retry": 2}[authentication]
    health = [row for row in requests if row["url"] == "/health"]
    assert health == [{"url": "/health", "method": "GET", "hasValidToken": authentication != "none"}]
    optional = [row for row in requests if row["url"] == f"/admin/providers/{provider}/models"]
    assert [row["hasValidToken"] for row in optional] == (
        [False, True] if authentication == "retry" else [authentication == "valid"]
    )
    assert scenario["warnings"] == []
    if outcome == "network":
        message = scenario["toast"]["html"]
        assert scenario["toast"]["role"] == "alert"
        assert "SUGGESTIONS UNAVAILABLE" in message
        assert f"GET /admin/providers/{provider}/models" in message
        assert "model suggestions" in message
        assert "manually" in message
    else:
        assert scenario["toast"] == {"html": "", "role": ""}


@pytest.mark.parametrize("name", ["provider-auth-cancel", "provider-upstream-cancel", "provider-auth-reenter", "unclassified-401"])
def test_provider_401_retains_local_token_without_prompt_or_retry(
    dashboard_runtime_result: dict[str, Any], name: str,
) -> None:
    scenario = next(row for row in dashboard_runtime_result["authErrorScopeStates"]["wrappers"] if row["name"] == name)
    assert scenario["status"] == 401
    assert scenario["prompts"] == 0
    assert scenario["requestsAtResponse"] == [{"url": "/admin/providers/codex/models", "hasValidToken": True}]
    assert scenario["tokenRetained"] is True
    assert scenario["tokenCleared"] is False
    assert scenario["subsequentStatus"] == 200
    assert scenario["subsequentRequests"] == [
        {"url": "/admin/settings/codex", "hasValidToken": True}, {"url": "/health", "hasValidToken": True},
    ]
    if name != "unclassified-401":
        assert scenario["body"]["error"]["type"] == "authentication_error"
        assert scenario["body"]["error"]["code"] is None
        assert "Synthetic" in scenario["body"]["error"]["message"]


@pytest.mark.parametrize("name", ["provider-auth", "provider-upstream"])
def test_provider_401_boot_surfaces_provider_failure_and_keeps_authenticated_core(
    dashboard_runtime_result: dict[str, Any], name: str,
) -> None:
    scenario = next(row for row in dashboard_runtime_result["authErrorScopeStates"]["boots"] if row["name"] == name)
    assert scenario["promptsAfterBoot"] == scenario["prompts"] == 1
    assert scenario["tokenRetained"] is True
    assert scenario["subsequentStatus"] == 200
    assert scenario["mapping"] == {"sonnet": "codex:saved-model"}
    assert scenario["tier"] == "fast"
    assert scenario["hasLoaded"] is True
    assert "fixture-identity@example.invalid" in scenario["identity"]
    assert "GET /admin/providers/codex/models" in scenario["feedback"]
    assert "401" in scenario["feedback"]
    assert "&lt;Synthetic" in scenario["feedback"]
    assert len([row for row in scenario["requests"] if row["url"] == "/admin/providers/codex/models"]) == 1
    assert all(row["hasValidToken"] for row in scenario["requests"] if row["url"] != "/api/hello")
    assert all(row["url"] != "/admin/test" for row in scenario["requests"])


@pytest.mark.parametrize("name", ["no-local-auth", "non-admin", "non-401", "local-retry", "local-cancel", "local-repeated-rejection"])
def test_local_token_recovery_remains_bounded_and_respects_existing_guards(
    dashboard_runtime_result: dict[str, Any], name: str,
) -> None:
    scenario = next(row for row in dashboard_runtime_result["authErrorScopeStates"]["wrappers"] if row["name"] == name)
    is_local = name.startswith("local-")
    assert scenario["prompts"] == (1 if is_local else 0)
    assert len(scenario["requestsAtResponse"]) == (2 if name in ("local-retry", "local-repeated-rejection") else 1)
    assert scenario["status"] == (200 if name == "local-retry" else 403 if name == "non-401" else 401)
    assert scenario["tokenCleared"] is (name == "local-cancel")
    assert scenario["tokenRetained"] is (name != "local-cancel")
    assert scenario["subsequentStatus"] == (401 if name == "local-cancel" else 200)
    assert all(row["hasValidToken"] is (name != "local-cancel") for row in scenario["subsequentRequests"])
    if name == "non-admin":
        assert scenario["requestsAtResponse"] == [{"url": "/public/catalog", "hasValidToken": False}]
    if name in ("local-retry", "local-repeated-rejection"):
        assert [row["hasValidToken"] for row in scenario["requestsAtResponse"]] == [False, True]


def test_concurrent_stale_local_rejection_does_not_discard_the_corrected_token(
    dashboard_runtime_result: dict[str, Any],
) -> None:
    scenario = dashboard_runtime_result["authErrorScopeStates"]["concurrent"]
    assert scenario["prompts"] == 1
    assert scenario["correctedStatus"] == scenario["staleStatus"] == 200
    assert scenario["tokenRetained"] is True
    assert scenario["requests"] == [
        {"url": "/admin/settings/log-level", "hasValidToken": False},
        {"url": "/admin/settings/codex", "hasValidToken": False},
        {"url": "/admin/settings/codex", "hasValidToken": True},
        {"url": "/admin/settings/log-level", "hasValidToken": True},
    ]
