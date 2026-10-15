# Dashboard

The gateway serves a runtime dashboard at `GET /` for editing the model map,
selecting Codex request speed, managing Claude accounts, checking provider
readiness, and testing model connections before wiring them.

Opening `http://127.0.0.1:8787/` uses the same guarded admin API as the CLI:

- **Settings** contains the [compaction reroute](compaction.md#compaction-reroute),
  [Codex speed](#codex-speed-and-serving-capabilities), Claude account routing,
  and registered-account management. The speed selector lives in **General**.
  See [Claude accounts](#claude-accounts) below.
- **Status** shows built-in login and subscription usage plus custom-provider
  readiness. Kimi and Grok cards remain cosmetically hidden until a login or
  mapped route makes them relevant; this never affects routing or settings.
  Custom providers do not receive usage or billing cards. Codex plan labels
  show **Pro 200** for `pro`, **Pro 100** for `prolite`, **Pro 500** for
  `promax`, and **Plus** for `plus`. Unknown plan names remain escaped raw
  provider values; the usage API retains its original `plan_type`.
  ChatGPT Work and Codex share provider-reported usage and credits, including
  activity outside this gateway. Only reported usage windows are shown; a
  missing session window does not imply zero usage or an unlimited allowance.
  When the Codex account has reset credits, the Codex card can spend one to
  reset its usage windows immediately (`POST /admin/providers/codex/reset-credit`).
  Spending is irreversible, so the button only opens a confirmation dialog, and
  confirming there is the only action that sends the request.
- **MCP** keeps gateway-wide Claude Code connection setup above per-backend
  sections, with GPT Pro session, ask concurrency, and diagnostics controls
  in the first backend card. See [GPT Pro MCP](#gpt-pro-mcp) below.
- **Log** reads `GET /admin/logs` and changes the persisted runtime log level
  through `PUT /admin/settings/log-level`.
- **Router** contains the **Router mapping** canvas and a shared
  [model connection test](#model-connection-test) for provider-prefixed targets.
  Targets are added through an in-canvas quick-add chooser.

Open the chooser with the **Add node** button pinned at the board's
bottom-right corner, by double-clicking or right-clicking empty canvas, or by
focusing the board (click empty canvas, then Tab) and pressing Shift+A. The
last two place the new node at the clicked lane; the button and shortcut use
the default stacking below the existing nodes. Nodes, ports, wires, and the
zoom controls never open it — right-clicking them keeps the native menu — and
the shortcut only fires while the board itself is focused, so typing in any
field cannot trigger it.

The chooser lists the providers the dashboard currently shows. Switching
providers clears the search and refocuses it. Catalog suggestions appear as
you type; a model ID no catalog suggests — including differently-cased IDs
with extra colons — can still be entered verbatim and committed with Enter,
in every catalog state (loading, failed, or not configured). The chooser
closes on Escape, the **Close** button, an outside click, or leaving the Router
tab. The chooser opens right-aligned above the add button with an 8px gap
between its bottom edge and the button's top edge, clamped inside the visible
canvas and viewport; the footer hint's own bottom padding is a separate 8px.

Committing stages the target as an unwired node; adding alone never wires,
marks the draft dirty, saves, or contacts a provider. Wire it to a source to
change the draft, use Discard to clear staged changes, and use Apply to send
the complete draft to `PUT /admin/settings/mapping`. Duplicate targets are
shown only once, and a duplicate commit keeps the chooser open. Contextually
placed nodes land at the clicked height, nudged down past any node already
occupying that lane; every other node keeps its position.

The Router board becomes view-only when `CLAUDEX_MODEL_MAP` overrides the
persisted map. The add button, the chooser, and every creation gesture are
disabled while this lock is active, and an open chooser closes when the lock
engages. This lock does not lock the independent Codex speed setting or tests.

## Codex speed and serving capabilities

The **Codex speed** row lives in **Settings > General**. Enter the local-token
prompt when authentication is required.

Before applying Ultrafast, read its contextual consumption notice: under the
current provider policy, **Ultrafast consumes included allowance at 8x Standard
and purchased credits at 6x Standard**. These are consumption ratios, not latency
or output-speed multipliers. A Pro 500 label does not establish entitlement, and
the dashboard never enables Ultrafast automatically. The notice stays visible
while Ultrafast is selected or still configured, even with a different draft.

1. Choose **Standard**, **Fast**, or **Ultrafast**. The selection is a draft until
   you press its **Apply** button; a changed draft shows the configured and
   not-yet-applied speeds. Standard saves `null` and omits the outbound tier;
   Fast uses `priority`; Ultrafast uses `ultrafast`, subject to each model's
   advertised support. Unknown or unsupported acceleration uses Standard.
2. Press the speed row's **Apply** to persist and apply the speed live. The
   loaded row returns to its compact selector, Apply, and contextual hint.
   Selecting or saving a speed does not run a model request.
3. **Reload settings** appears after a settings error. It reads the actual
   server selection and replaces the selector draft. A failed or invalid save
   keeps the draft and does not claim success: the server may already have
   persisted a change if its response was lost, so reload before retrying.

Failed settings reads disable the selector until a valid reload. With
`CLAUDEX_CODEX_SERVICE_TIER` set, the selector and Apply are read-only behind a
LOCKED band. A `409` conflict locks editing immediately and adopts state only
from a fresh settings GET. A failed conflict refresh stays locked until a valid
reload. The independent Router connection test remains available.

The speed setting does not establish model support or confirm the tier used by
a provider. To inspect or refresh serving metadata, including after an account
upgrade, use the guarded [capability status and refresh API](providers.md#capability-status-and-refresh).
The dashboard does not request this capability list on load or after settings
or mapping saves. The backend API remains separate from the model-suggestion
catalog used by the Router chooser; manual model IDs remain usable.

### Model connection test

In **Router**, enter a provider-prefixed target (`provider:model`) in **Model
connection test** below the board, then click **Test** or press Enter. The shared
action accepts built-in and configured custom-provider targets, including manual
model IDs and model suffixes with additional colons. It sends `POST /admin/test`
with only `target`, never `service_tier`, regardless of the configured Codex speed
or an unsaved General draft.

Test sends one real model request; provider usage limits apply. A small prompt
does not bound Codex output length; see [Codex limitations](providers.md#limitations).
The button is disabled while a request is pending. Success shows latency and
`response.model`; errors show the failure and allow a retry. A local-token
rejection uses the dashboard's normal authentication prompt and retry path.
Connection success and latency do not confirm accelerated service or completed
inference; see the [ordinary connection-check semantics](providers.md#explicit-tier-diagnostics).

Explicit Codex tier probes remain available through the guarded
[diagnostic API](providers.md#explicit-tier-diagnostics), not through a separate
dashboard action. These opt-in requests run inference and can consume allowance.

## GPT Pro MCP

The MCP tab leads with gateway-wide connection setup, followed by one card per
MCP backend. GPT Pro is the first backend section:

- **Connect Claude Code** runs `claude mcp add` on the gateway host through
  **Add to Claude Code**, so the `claude` CLI must be on that process's `PATH`.
  Registration changes the user-scope MCP settings of the operating-system
  user running the gateway, not those of a user viewing the dashboard from
  another machine. When local authentication is enabled, registration stores
  the configured local token as a bearer header in that user's Claude Code MCP
  settings. A copyable user-scope command remains available as a manual
  fallback; replace its `<CLAUDEX_LOCAL_TOKEN>` placeholder when present and
  run it on the machine and as the user whose Claude Code should connect.
  The endpoint must be reachable from that machine. Restart Claude Code
  sessions after registration to load the tools.
- **GPT Pro** shows the saved ChatGPT Pro session and refreshes it once per
  minute while the tab is visible. It can start, monitor, and cancel an
  interactive ChatGPT sign-in. Its Diagnostics subsection runs the server-side
  doctor and displays the output. Sign-in opens a visible browser window on
  the gateway host, as `claudex-gateway gptpro login` does from a terminal,
  so either path needs a graphical session on that host.
- **Ask concurrency** limits how many GPT Pro ask tabs can run at once (1–10,
  default: 2). Selecting a value saves and applies it to the running daemon
  immediately; there is no Apply button. Editing pauses while saving or while
  settings are unavailable. A failed save restores the displayed last confirmed
  value; this does not roll back a change already accepted by the server. If
  the response is lost or invalid, leave and reopen the MCP tab to read the
  actual setting before retrying. The control is read-only with a LOCKED band
  while `GPTPRO_MAX_CONCURRENT_ASKS` is set in the gateway environment. Raising
  concurrency does not lift ChatGPT-side rate limits.

The tab uses these guarded admin API operations:

- `GET /admin/settings/gptpro` reads the concurrency limit and environment lock.
- `PUT /admin/settings/gptpro` persists and applies the concurrency limit live.
- `GET /admin/gptpro/session` reads the saved session state.
- `GET /admin/gptpro/login` reads the current login state.
- `POST /admin/gptpro/login` starts a login.
- `DELETE /admin/gptpro/login` cancels an active login or clears a completed one.
- `POST /admin/gptpro/doctor` runs the diagnostic.
- `GET /admin/gptpro/mcp` returns the endpoint and authentication requirement.
- `POST /admin/gptpro/connect` runs the user-scope Claude Code MCP registration.

A login temporarily owns the shared browser profile. GPT Pro asks are
unavailable while sign-in is starting, running, or being cancelled.

## Claude accounts

The Settings tab's **Claude accounts** category manages the accounts described in
[Claude accounts](claude-accounts.md). The routing mode selector lives in the
**General** category and is read-only while `CLAUDEX_CLAUDE_ACCOUNT_ROUTING` is
set.

Compaction reroute and account routing each allow one pending save. Apply
shows `…` and stays disabled until the save, including any `409` refresh,
settles; changing a draft does not submit another request. On success,
the controls adopt the server-returned value. If a lock refresh fails,
the setting stays locked. Apply a later draft only after the pending
operation finishes and the setting is unlocked.

- The local CLI login card shows this machine's own Claude Code login and its
  usage. The card is informational: the login serves traffic only when
  balanced routing includes it (see
  [Balanced routing](claude-accounts.md#balanced-routing-across-the-pool)).
- Each registered account shows its plan, usage windows, and routing state:
  ready, cooling down until a time, or unavailable because it needs a new
  login.
- **Add account** and an account's **Sign in again** run
  `claude auth login --claudeai` on the gateway host, so the `claude` CLI must
  be installed there. The dialog shows the authorization URL to open in a
  browser and accepts the pasted code. If the login matches an already
  registered identity, the dialog asks before it replaces the stored
  credentials.
- **Set serving pin** and **Clear serving pin** set and clear the serving
  account. **Serving pin set** identifies the selected account. These controls
  are read-only while `CLAUDEX_CLAUDE_ACCOUNT_ID` is set.
- **Remove** needs a second click to confirm and is disabled for the serving
  account. Clear the serving account first.
- While balanced routing is active, a pool usage freshness badge and a
  **Refresh usage** button appear. The button queues a rate-limited usage poll
  instead of fetching inline.

The panel uses these guarded admin API operations:

- `GET /admin/providers/claude/accounts` lists registered accounts without
  secrets.
- `GET /admin/providers/claude/local` reads the local CLI login's identity,
  and `GET /admin/usage?provider=claude` reads its usage.
- `DELETE /admin/providers/claude/accounts/{account_id}` removes an account and
  returns `409` for the serving account.
- `GET`, `PUT`, and `DELETE /admin/providers/claude/pool/serving` read, set,
  and clear the serving account.
- `GET` and `PUT /admin/providers/claude/pool/routing` read and set the routing
  mode.
- `GET /admin/providers/claude/pool/status` reads each account's routing state.
- `GET /admin/providers/claude/pool/usage` reads per-account usage. Add
  `?refresh` to queue a poll in balanced mode.
- `GET`, `POST`, and `DELETE /admin/providers/claude/login`, plus
  `POST /admin/providers/claude/login/code` and
  `POST /admin/providers/claude/login/replace`, drive the browser login.

Only one dashboard login runs at a time, and it cannot run while an
interactive CLI `account add` holds the same machine-wide login lock. In either case, the
start request returns `409`.

## Custom-provider metadata and status

`GET /admin/settings/mapping` supplies custom-provider metadata without API
keys. The dashboard uses metadata rather than provider names to choose behavior:

| `wire_kind` | Dashboard label |
| --- | --- |
| `responses` | Responses API |
| `anthropic_messages` | Anthropic Messages |

`catalog_available` controls catalog loading. When it is `true`, the dashboard
may call `GET /admin/providers/custom/{name}/models` for autocomplete. When it
is `false`, the dashboard never calls that endpoint. A missing catalog is a
capability boundary, not a connection failure.

The custom-provider status terms are intentionally different:

- **Connected** means the Responses-family remote catalog verification used by
  health succeeded.
- **Configured** means the provider definition and route binding are ready, but
  no catalog or remote connection was inferred. Static Anthropic-compatible
  providers use this state because their health check performs no remote I/O.
- **Unused** means a Responses-family provider's check failed while no
  `<name>:` target is mapped, so the failure does not affect readiness.
- **Error** means the applicable local binding or remote catalog check failed.

Neither configured nor general gateway health proves remote model entitlement
or full Messages compatibility. Enter a complete `provider:model` target in the
connection test, or call `POST /admin/test`, to issue one minimal remote request.
For an Anthropic-compatible provider this is the explicit remote Messages
verification path.

## Model catalogs and manual entry

Built-in quick-add suggestions use:

- `GET /admin/providers/codex/models`
- `GET /admin/providers/kimi/models`
- `GET /admin/providers/grok/models`

Catalog-capable custom providers use
`GET /admin/providers/custom/{name}/models`. Catalogs supply suggestions, not a
model allowlist. A browser transport failure for an optional Kimi or Grok catalog
shows a **Suggestions unavailable** notice without stopping successful saved
mapping, settings, and health responses from loading. Enter a model ID manually
when suggestions are unavailable. The chooser's manual entry remains
authoritative: a typed model ID can be staged and mapped even when it did not
come from a catalog.
Catalog-less Anthropic-compatible providers therefore require manual model
IDs; the chooser shows a short notice for them instead of an empty list.

Custom provider names can be arbitrary valid configured names, including names
that sound like another API family. Names never determine wire labels, catalog
behavior, or status semantics.

## Local dashboard authentication

When `CLAUDEX_LOCAL_TOKEN` is set, the dashboard prompts for it once per page
load and retains it in an in-memory closure for that page only. It is attached
as a bearer header to admin requests and is never written to the DOM, a URL,
`localStorage`, `sessionStorage`, console output, or gateway logs. When the
local gateway guard rejects an admin request with `401` and
`error.code: invalid_api_key`, the dashboard prompts once more and retries that
request once. Cancelling leaves the local token unset; a concurrent stale
rejection cannot discard a token already corrected by another request.
Provider authentication failures can also return `401`, but remain provider
errors: the dashboard reports them without clearing the valid local token,
asking for it again, or retrying the provider request. Check the failing
provider's login rather than re-entering `CLAUDEX_LOCAL_TOKEN`.
This page-only token handling is separate from MCP registration,
which persists the gateway's configured token in Claude Code's MCP settings as
described above.
