# Dashboard

The gateway serves a runtime dashboard at `GET /` for editing the model map,
selecting Codex request speed, managing Claude accounts, checking provider
readiness, and testing model connections before wiring them.

Opening `http://127.0.0.1:8787/` uses the same guarded admin API as the CLI:

- **Settings** contains the [compaction reroute](compaction.md#compaction-reroute),
  Claude account routing, and registered-account management.
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
- **Router** contains [Codex speed and serving capabilities](#codex-speed-and-serving-capabilities),
  the **Router mapping** canvas, and model connection tests. Targets are added
  through an in-canvas quick-add chooser. The ordinary test checks connections;
  a separate selected-speed test opts into a Codex tier diagnostic.

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

Open **Router** and enter the local-token prompt when authentication is required.
Before choosing a speed, read the consumption notice: under the current provider
policy, **Ultrafast consumes included allowance at 8x Standard and purchased
credits at 6x Standard**. These are consumption ratios, not latency or output-speed
multipliers. A Pro 500 label does not establish entitlement, and the dashboard
never enables Ultrafast automatically.

1. Inspect the applied Codex mappings in the **Codex speed** card. Each model
   shows known supported speeds, advertised wire tier IDs, and the applicable
   speed and outbound wire tier for the **configured** selection. The status
   and timestamp above the list describe the serving metadata snapshot. Mixed
   mappings can apply acceleration to one model while another uses Standard.
   Provider/model boundary whitespace is ignored for capability lookups and
   explicit Codex target checks, matching backend parsing. This uses Python's
   whitespace boundaries, not broader character normalization: opaque model
   characters such as U+FEFF and later colons remain part of the ID. Stored
   mapping strings and board-node identities are not rewritten.
2. Use **Refresh capabilities** after a login or subscription change, or to retry
   an unavailable catalog. It refreshes the same snapshot used for serving,
   including same-account upgrades with an unchanged token. It sends metadata
   requests only, not model inference.
3. Choose **Standard**, **Fast**, or **Ultrafast**. The selection is a draft until
   you press its **Apply** button. Standard saves `null` and omits the outbound
   tier; Fast uses `priority`; Ultrafast uses `ultrafast`, subject to each model's
   support. The card continues to label applicability for the configured speed,
   not for an unsaved selection. Router draft changes and staged nodes are also
   excluded from the list until the mapping is applied.
4. Press **Apply** to persist and apply the speed live. Successful speed or mapping
   saves read capability status again; they do not force a catalog refresh or
   run a model request. **Reload settings** reads the actual server selection
   and replaces the selector draft. It also reloads capability status.

Capability states answer different questions:

- **Current** means the serving snapshot is current; it does not mean every
  mapped model supports acceleration.
- **Stale** retains usable last-known metadata and the original last-fetched
  time. The top-level reason remains visible even when individual models show
  support. Refresh to retry the failed fetch.
- **Unknown** means support could not be established, such as an unavailable
  catalog, an unlisted model, or incomplete tier metadata. It is not proven
  unsupported. Per-model reasons explain the fallback.
- **Unsupported** means the model's usable catalog does not advertise the
  configured accelerated tier. Its applicable tier remains Standard.

A partial catalog can show known Fast support while Ultrafast is unknown.
Standard remains applicable without accelerated metadata. The list is not an
allowlist: manual model IDs remain usable in the Router. `Last fetched: Never`
means the backend has no successful snapshot for the current identity;
`Unavailable` means the dashboard could not read a valid snapshot. Failed or
malformed capability responses show Unknown and a retry action, without
blocking editing of successfully loaded settings.

A failed or invalid save keeps the draft and does not claim success. The server
may already have persisted a change if its response was lost, so use **Reload
settings** before retrying. Failed settings reads disable the selector and the
selected-speed test until a valid reload; the ordinary connection test remains
available. With `CLAUDEX_CODEX_SERVICE_TIER` set, the selector and Apply are
read-only behind a LOCKED band. A `409` conflict locks editing immediately and
adopts state only from a fresh settings GET. A failed conflict refresh stays
locked until a valid reload. Capability refresh and testing the loaded locked
selection remain available.

### Explicit selected-speed test

Enter a `codex:<model-id>` target in **Model connection test** below the board.
The ordinary **Test** button (also Enter in that field) sends a minimal,
**untiered** connection request, even if the configured Codex speed is Ultrafast.
Provider usage limits still apply to that real request.

Read the warning before using **Test selected Codex speed**: it runs one real
Codex request using the selector's draft, **without saving it**, and can consume
allowance or purchased credits. The small prompt does not bound Codex output
length; see [Codex limitations](providers.md#limitations). No inference runs on
page load, speed selection, saves, or capability refresh. Non-Codex targets are
rejected locally by the selected-speed action.

The result keeps connection outcome separate from three tier observations:

| Observation | What to read |
| --- | --- |
| Requested UI speed | The selection submitted for this test, even if the selector changes while it runs |
| Final outbound wire tier | The backend's final sent tier: `priority`, `ultrafast`, or omitted/Standard |
| Backend-reported tier | The exact provider response value; a missing report is unconfirmed, not presumed Standard |

**Confirmed** requires a successful model response and provider evidence matching
an accelerated request. **Different tier** means the provider reported another
tier. **Reported (no accelerated tier requested)** preserves a provider report
for a Standard test. **Unconfirmed** covers missing evidence, missing or invalid
diagnostics, and failed responses. Connection success, `response.model`, a plan
label, or latency alone does not confirm speed. Diagnostic latency includes
metadata and stream processing; it is not a speed guarantee or billing estimate.

Unknown or unsupported acceleration can send Standard. The result includes the
backend reason, including dispatch fallback when authentication or tier support
changed after preflight. Capability-list wire tiers are preflight applicability,
not provider confirmation; use the explicit test to inspect what actually went
out and what the provider reported.

The guarded operations are `GET`/`PUT /admin/settings/codex`,
`GET /admin/providers/codex/capabilities`, and
`POST /admin/providers/codex/capabilities/refresh` with `{}`. The ordinary
`POST /admin/test` omits `service_tier`; the selected-speed action includes
`null`, `fast`, or `ultrafast`. See [capability status and explicit diagnostics](providers.md#capability-status-and-refresh)
for the full backend contract.

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
