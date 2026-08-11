# claudex-gateway

A lightweight local gateway that runs mapped Claude Code models on the OpenAI
Codex or Grok backend and relays everything else to Anthropic untouched.

```text
Claude Code ── "codex:" mapped ───▶ claudex-gateway ── Codex Responses API ─▶ Codex
Claude Code ── "grok:" mapped ────▶ claudex-gateway ── Grok Responses API ──▶ Grok
Claude Code ── unmapped model ────▶ claudex-gateway ── verbatim relay ──────▶ Anthropic API
```

Mapped models run on Codex or Grok; everything else goes to Anthropic
untouched.

## What it does

- Serves `POST /v1/messages` in Anthropic Messages format: models with a
  `CLAUDEX_MODEL_MAP` entry are translated to the Codex Responses backend,
  everything else is forwarded byte-for-byte to the real Anthropic API with
  the client's own credentials.
- Supports streaming and non-streaming responses.
- Translates text, images, PDF documents (base64 `application/pdf` blocks in
  user messages — other document forms are rejected with a clear error rather
  than silently dropped), thinking/reasoning blocks, function calls and
  results (with 64-char-safe names for long MCP tool namespaces), usage,
  stop reasons, and native web search; mid-conversation `system` messages
  keep operator authority as Responses `developer` messages.
- Routes models mapped with a `grok:` prefix to the Grok Responses backend
  (`cli-chat-proxy.grok.com`) through the same translation layer as Codex,
  minus the payload fields Grok rejects and with reasoning effort clamped to
  the model's supported levels.
- Answers as the Claude model the client requested — the Codex or Grok
  target model never appears on the Anthropic wire, so Claude Code heuristics
  keyed on model names keep working.
- Serves `GET /health` with the readiness state of the Codex and Grok
  upstreams.
- Serves a runtime dashboard at `GET /` for editing the model map, checking
  provider health, and testing model connections before wiring them.
- Reuses each provider's CLI login — no gateway-side auth: the Codex CLI's
  `~/.codex/auth.json` and the Grok CLI's `~/.grok/auth.json`, each
  refreshed in place like the CLI itself does.

Two Anthropic contract points cannot be preserved on the Codex path and are
explicit choices, not bugs:

- `max_tokens` is validated but not enforced: the Codex backend rejects the
  Responses `max_output_tokens` parameter, so mapped requests cannot cap
  output length upstream and the gateway does not truncate locally.
- `POST /v1/messages/count_tokens` for Codex-mapped models returns a
  characters/4 estimate. Mapped prompts are never sent to Anthropic just to
  be counted and no Codex tokenizer is available, so treat the number as a
  rough gauge for context-usage display, not an exact count for billing or
  hard limits. Unmapped models pass through to Anthropic's real counter.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- For Codex targets: a logged-in [Codex CLI](https://github.com/openai/codex)
  (`codex login`)
- For Grok targets: a logged-in [Grok CLI](https://github.com/xai-org/grok-build)
  (`grok login`)
- For `claudex-gateway account add` (interactive): the `claude` CLI on
  `PATH` and an interactive terminal — it launches `claude auth login
  --claudeai` to capture a fresh login.
  `account add --from <dir>` does not launch `claude`; it imports an
  already-authenticated config directory instead.
- On macOS, capture reads Claude's credentials from a Keychain item scoped
  to the exact config-directory string, so `--from <dir>` must name the
  exact string used at that login — no `~` expansion, trailing-slash
  cleanup, or other path canonicalization.
- Interactive capture supports Claude Code builds that use scoped Keychain
  credential storage (2.1+). A build that stores credentials elsewhere fails
  the capture cleanly; if such a build also replaced the machine-level
  `claude` sign-in during the login, `account add` prints an explicit
  warning with recovery guidance.
- Interactive capture (`account add` without `--from`) is POSIX-only in
  this version; on Windows, use `account add --from <dir>` instead.

The server starts even when a login is missing; `/health` reports the
credential state per provider.

## Usage

### Start the gateway

```sh
uv run claudex-gateway               # background; logs to ~/.claudex/gateway.log
uv run claudex-gateway --foreground  # attached to the terminal
uv run claudex-gateway stop
```

Starting is idempotent: when the port already answers with the same gateway
version, the command reports the running instance and exits. A daemon left
running across a package update is stopped and replaced automatically when
its identity can be verified (any 0.4.0+ daemon), and a port occupied by
something else fails loudly. The background log file is rewritten on every
start.

`stop` (and the automatic stale-daemon replacement) verifies the daemon's
identity before sending any signal: the pid, host, port, and per-start nonce
recorded in `~/.claudex/gateway.pid` must match what the running gateway
reports on `GET /api/hello`. A record that cannot be verified — the bare-pid
file of a pre-0.4 install, a corrupt record, or a pid that no longer answers
as the recorded gateway — is never signaled; the command explains what to
stop manually instead (a one-time step when upgrading from 0.3.x). This
makes pid reuse harmless: a recycled pid can no longer be terminated by
mistake.

### Claude Code → Codex

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

A logged-in Claude Code needs no token setup: it attaches its own credentials,
which the Codex path never uses and the passthrough path forwards to Anthropic
untouched.

A shell alias covers the launch part:

```sh
alias claudex='ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude'
```

To run only some Claude models on Codex and keep the rest on the real
Anthropic API, see
[Mixing Claude and Codex models](#mixing-claude-and-codex-models).

### Claude Code → Grok

The gateway reuses the Grok CLI login — no gateway-side login step. With the
CLI logged in (`grok login`, tokens at `~/.grok/auth.json`, or
`grok login --api-key` for a plain Grok API key), route models to Grok with a
`grok:` prefix in the map:

```json
{
  "model_map": {"opus": "grok:grok-4.5", "haiku": "codex:gpt-5.6-luna"}
}
```

Every value names its provider (`codex:` or `grok:`); a bare model name is
rejected at boot and on `PUT`, so an entry always says which backend serves
it. Grok speaks the same Responses API family as the Codex backend, so requests
reuse the full Claude → Responses translation (streaming and non-streaming,
thinking, tool use); only the wire quirks differ. On the way out the gateway
drops the fields Grok rejects (`previous_response_id`, `stream_options`,
`stop`, …) and adapts reasoning: models with thinking levels
(`grok-4.5`, `grok-4.3`, `grok-3-mini`, `grok-3-mini-fast`,
`grok-4.20-multi-agent-0309`) keep the effort, clamped to Grok's
`low`/`medium`/`high` vocabulary, while every other model runs without a
reasoning config — sending one to a non-thinking model fails upstream.
A newly released thinking model simply runs at its default effort until the
gateway's list catches up.

### Claude accounts

`claudex-gateway account add` launches the `claude` CLI (`claude auth login
--claudeai`) in a temporary config directory and captures the resulting
login into the local account registry. `account add --from <dir>` does not
launch `claude`: it instead imports an already-completed login from `<dir>`,
which must be the exact config-directory string used at that login (on
macOS this selects a Keychain item scoped to that exact string — no `~`
expansion, trailing-slash cleanup, or other path canonicalization).
Interactive capture supports Claude Code builds that use scoped Keychain
credential storage (2.1+) and fails cleanly otherwise. It is POSIX-only in
this version — on Windows, always use `--from <dir>` instead.

Adding an account whose `(email, organization)` identity is already
registered replaces that account's stored credentials in place after a
confirmation prompt — the account keeps its id, so a `account use`
selection keeps working. This doubles as the re-auth flow for an account
whose refresh token has gone stale. Non-interactive runs (piped stdin with
`--from`) require `--yes` to confirm the replacement.

```sh
uv run claudex-gateway account add                 # interactive `claude` login
uv run claudex-gateway account add --from <dir>    # import an existing login
uv run claudex-gateway account list
uv run claudex-gateway account remove <id>         # prompts for confirmation
uv run claudex-gateway account remove <id> --yes   # skips the confirmation prompt
uv run claudex-gateway account use <id|email>      # serve passthrough with this account
uv run claudex-gateway account use off             # back to forwarding client credentials
uv run claudex-gateway account use                 # show the current selection
```

Captured credentials are stored under `~/.claudex/accounts/claude/`: one
directory per account (mode `0700`, POSIX file permissions) holding
`credentials.json` and `oauth-account.json` (mode `0600` each), plus a
shared `registry.json` (also mode `0600`) that lists accounts without
secrets. `claudex-gateway` never prints captured credential payloads.

Duplicate detection is keyed on normalized email + `organizationUuid`,
where missing matches missing: an account with no `organizationUuid`
collides only with another account that also has none, never with one
that has an `organizationUuid` set. `account remove <id>` deletes that
account's local copy only — it does not revoke the OAuth grant at
Anthropic, which stays valid until revoked from Anthropic's own account
settings.

### Serving with a registered account (`account use`)

`claudex-gateway account use <id|email>` selects one registered account to
serve all Anthropic passthrough traffic (`/v1/messages` for unmapped models
and `count_tokens`). With a selection active, the gateway consumes the
client's `Authorization`/`x-api-key` headers and serves upstream with the
selected account's OAuth token instead — Claude Code no longer needs a real
Anthropic login of its own (`ANTHROPIC_AUTH_TOKEN` set to the gateway local
token, or any placeholder when no local token is configured, is enough).
The gateway owns the token lifecycle: access tokens are refreshed ~5
minutes before expiry against the Claude Code token endpoint, rotated
refresh tokens are persisted atomically before use, and a 401 triggers one
refresh-and-retry before the error surfaces. `metadata.user_id`'s
`account_uuid` is rewritten to the serving account so a request never names
a different account than the one serving it.

The selection is the flat `claude_account.id` settings key (env override:
`CLAUDEX_CLAUDE_ACCOUNT_ID`). `account use` manages it through the same
channel decision table as `compact`: a confirmed running daemon is updated
live through the `/admin/providers/claude/pool/serving` endpoint (no
restart needed), a settings-file write is used only when no live daemon can
be confirmed, and an ambiguous probe refuses to apply changes.
`account use off` clears the pin via `DELETE` and returns to today's
default: client credentials forwarded untouched.

```sh
curl http://127.0.0.1:8787/admin/providers/claude/pool/serving
curl -X PUT http://127.0.0.1:8787/admin/providers/claude/pool/serving \
  -H 'Content-Type: application/json' \
  -d '{"account_id": "<registered-account-id>"}'
curl -X DELETE http://127.0.0.1:8787/admin/providers/claude/pool/serving
```

The endpoint honors the same `CLAUDEX_LOCAL_TOKEN` and Host guard as the
other admin routes; a `PUT` requires a registered id (clearing is `DELETE`,
never a null `PUT`), persists the change to `settings.json`, and both
writes are refused with `409` when `CLAUDEX_CLAUDE_ACCOUNT_ID` is set in
the environment.

Caveats to accept consciously:

- Quota and billing land on the selected account, not the client's own
  subscription, and Anthropic's response rate-limit headers (and the CLI's
  usage display) reflect the serving account.
- If the selected account is removed or its refresh token becomes invalid,
  passthrough fails with a clear gateway 503 — there is never a silent
  fallback to client credentials. Re-add the account or run
  `account use off`. (With the `fallback` routing mode enabled — see the
  next section — the remaining ready accounts serve instead.)
- The [compaction reroute](#compaction-reroute) still uses the credentials
  the client itself sent: with a credential-less client it records
  `skipped_no_credentials` and falls back to the mapped model as usual.
- Subscription OAuth tokens are licensed for the holder's own Claude Code
  use; serving other clients with them is a gray zone.

### Ordered fallback across registered accounts

Multi-account routing is an explicit opt-in, off by default: with the
routing mode `disabled`, only the pinned serving account is used and a
`429` relays to the client verbatim. Selecting the `fallback` mode turns
the pin into the head of a fallback chain: every **ready** account is a
pool member, ordered serving-account-first and then by registration time.
When the account being served with answers `429`, the gateway puts it on
an in-memory cooldown, transparently retries the same request with the
next account in the chain, and fails back automatically once the cooldown
expires — the client never has to handle the rate limit itself as long as
any account has quota left.

The mode is the `claude_account.routing` settings key — a policy document
like `{"mode": "fallback"}`, leaving room for future modes (a usage-
balancing `balanced` is reserved and refused until implemented) — managed
at runtime through `/admin/providers/claude/pool/routing`:

```sh
curl http://127.0.0.1:8787/admin/providers/claude/pool/routing
curl -X PUT http://127.0.0.1:8787/admin/providers/claude/pool/routing \
  -H 'Content-Type: application/json' \
  -d '{"mode": "fallback"}'
```

The env override `CLAUDEX_CLAUDE_ACCOUNT_ROUTING` holds the same document
JSON-encoded (empty string = disabled) and locks the endpoint with `409`
while set.

How long a rate-limited account sits out is taken from the best signal the
429 offers: a `Retry-After` header or a reset timestamp when present,
otherwise the account's cached usage window resets (as shown in the
dashboard), otherwise a 60-second default — in practice Anthropic's OAuth
quota rejections carry no machine-readable reset, so the cached usage data
is what turns a blind minute into an accurate multi-hour cooldown. Each
account's routing state is visible at
`GET /admin/providers/claude/pool/status` (`ready`, `cooldown` with a
`cooldown_until` epoch-ms deadline, or `unavailable`) and lives in daemon
memory only: a restart clears it, at worst costing one extra upstream
probe.

Boundaries to know:

- Only `429` triggers failover, and only in `fallback` mode. Auth failures
  durably mark the account `needs-reauth` (excluded from the chain until a
  re-login); other upstream errors and network failures are relayed as
  before — retrying a different account would not help them.
- When every account is rate-limited, the client sees a real `429`: the last
  upstream rejection while probing, or a synthesized one with `Retry-After`
  once everything is already cooling (upstream is then not contacted at all).
- Failover only happens before any response byte is relayed; a stream that
  dies midway is reported in-band, as before.
- Rate limits are per account **per model tier** upstream, but the pool's
  cooldown is per account: a Fable-scoped weekly limit cools the whole
  account even for requests other models could still serve. Per-model
  eligibility is a known follow-up.
- The CLI's own usage display remains unreliable under pooling: its usage
  query authenticates with the placeholder token and fails, and response
  rate-limit headers reflect whichever account served. Use the dashboard's
  per-account usage view instead.

### Compact command

Manage the opt-in compaction reroute (see [Compaction
reroute](#compaction-reroute) below) from the command line:

```sh
uv run claudex-gateway compact                       # show current state
uv run claudex-gateway compact set claude:<model-id>  # enable, target <model-id>
uv run claudex-gateway compact off                    # disable
```

When a compatible daemon is identified, the command uses its authenticated
`GET`/`PUT /admin/settings/compaction` API, so successful changes take effect
immediately with no restart. An identified older daemon returning `404` or
`405` falls back to the settings file and warns that a restart is
required. With no listener, or a confirmed foreign process on the port,
the command reads or writes `settings.json` directly (picked up the next
time the gateway starts). Ambiguous reachability fails closed for `set`
and `off` and shows settings-file state with an unreachable warning for
the read-only form. Any other failure from an identified daemon — an
error status such as the environment-lock `409`, a malformed response, or
a transport failure — exits non-zero without touching the settings file.

## Configuration

Every variable can be set in `~/.claudex/settings.json` or as an
environment variable; the environment wins when both are set (even when set
to an empty string), so the file holds your durable setup and the environment
stays available for one-off overrides.

| Variable | Default | Description |
| --- | --- | --- |
| `CLAUDEX_HOST` | `127.0.0.1` | Bind address |
| `CLAUDEX_PORT` | `8787` | Bind port |
| `CLAUDEX_MODEL_MAP` | empty | JSON mapping of Claude names, exact or substring, to provider-prefixed target models — `codex:`-prefixed values run on Codex, `grok:`-prefixed values on Grok; unmapped models are relayed verbatim to Anthropic |
| `CLAUDEX_REASONING_EFFORT` | derived | Force `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` on Codex requests |
| `CODEX_HOME` | `~/.codex` | Directory containing Codex `auth.json` |
| `GROK_HOME` | `~/.grok` | Directory containing the Grok CLI's `auth.json` |
| `CLAUDEX_LOG_LEVEL` | `info` | Process log verbosity: `debug`, `info`, `warning`, or `error`; editable at runtime from the dashboard |
| `CLAUDEX_LOCAL_TOKEN` | unset | Bearer token required by the model request routes and the admin/dashboard routes when set; mandatory for non-loopback binds. See [the passthrough interaction](#mixing-claude-and-codex-models) |
| `CLAUDEX_COMPACTION_MODEL` | unset | `compaction.model` setting: opt-in `claude:<model-id>` reroute target for oversized Claude Code compaction requests; unset (default) disables the reroute entirely. See [Compaction reroute](#compaction-reroute) |
| `CLAUDEX_CLAUDE_ACCOUNT_ID` | unset | `claude_account.id` setting: id of the registered Claude account that serves Anthropic passthrough traffic; unset (default) forwards client credentials untouched. See [Serving with a registered account](#serving-with-a-registered-account-account-use) |
| `CLAUDEX_CLAUDE_ACCOUNT_ROUTING` | unset | `claude_account.routing` setting as a JSON-encoded policy document, e.g. `{"mode": "fallback"}`; unset or empty (default) keeps multi-account routing disabled. See [Ordered fallback across registered accounts](#ordered-fallback-across-registered-accounts) |

### settings.json

The settings key for each variable is its environment name minus the
`CLAUDEX_` prefix, lowercased (the CLI-home variables — `CODEX_HOME` and
`GROK_HOME` — having no prefix, keep their full names as
`codex_home` / `grok_home`). Values use native JSON types,
so the model map is a plain object instead of JSON-in-a-string:

```json
{
  "model_map": {"opus": "codex:gpt-5.6-sol", "haiku": "codex:gpt-5.6-luna"}
}
```

The file is optional and validated at startup: unknown keys and wrongly typed
values abort the boot so a typo cannot be silently ignored.

### Runtime mapping API

The model map can be changed while the gateway is running — no restart of
the gateway or its clients:

```sh
curl http://127.0.0.1:8787/admin/settings/mapping
curl -X PUT http://127.0.0.1:8787/admin/settings/mapping \
  -H 'Content-Type: application/json' \
  -d '{"model_map": {"opus": "codex:gpt-5.6-sol"}}'
```

A `PUT` accepts `model_map`, replaces it for all subsequent requests, and
persists it to `settings.json` (other keys in the file are preserved). When
`CLAUDEX_MODEL_MAP` is set the request is refused with `409` — the
environment would silently win again on restart. The endpoint honors
`CLAUDEX_LOCAL_TOKEN` and only answers requests whose `Host` is the gateway
itself, so foreign web pages cannot drive it from a browser.

### Compaction reroute

The compaction reroute is opt-in and disabled by default (unset
`compaction.model`); once configured to a `claude:<model-id>` target, it
triggers only for a detected Claude Code compaction request whose estimated
size exceeds the mapped backend's context window, and makes a single
Anthropic attempt using the client's own credentials, billed by Anthropic
separately from the mapped backend. Its failure semantics: missing eligible
Anthropic credentials skip the reroute entirely; connection failures and
non-2xx responses before a streaming HTTP 2xx commit fall back silently to
the mapped backend; failures after that commit surface as in-band SSE
errors without fallback; non-streaming requests fall back on any failure
before a complete, valid JSON response is obtained. The literals used to
detect a compaction request are a versioned contract with Claude Code
`2.1.223` — re-verified against the client binary before ever being
changed, never guessed.

```sh
curl http://127.0.0.1:8787/admin/settings/compaction
curl -X PUT http://127.0.0.1:8787/admin/settings/compaction \
  -H 'Content-Type: application/json' \
  -d '{"model": "claude:claude-opus-5"}'
curl -X PUT http://127.0.0.1:8787/admin/settings/compaction \
  -H 'Content-Type: application/json' \
  -d '{"model": null}'
```

`GET`/`PUT /admin/settings/compaction` read and change the same `compaction.model`
setting on a running gateway, honoring the same `CLAUDEX_LOCAL_TOKEN` and
Host guard as the other admin routes; a `PUT` persists the change to
`settings.json` and, like the mapping API, is refused with `409` when
`CLAUDEX_COMPACTION_MODEL` is set in the environment. The [`compact`
command](#compact-command) is a CLI front end for this same API.

### Dashboard

Opening `http://127.0.0.1:8787/` in a browser serves a dashboard on top of
the same admin API: the Settings tab holds gateway settings behind a
category rail (currently a single General category with the
[compaction reroute](#compaction-reroute) target), the Status tab shows each
provider's login state and subscription usage — the Grok card (and
its Router add-node button) appears only when a local login is detected or
the model map already routes to it; hiding is cosmetic and never affects
routing or `settings.json` — the Log tab tails the gateway's recent
log lines (`GET /admin/logs`) and holds the runtime log-level control
(`PUT /admin/settings/log-level`, applied immediately and persisted), and the Router
tab is a canvas editor — drag a port to wire a model, Apply to `PUT` the
map, and use the connection test box (`POST /admin/test`) to send one
minimal request through the gateway before wiring a new model id. The board
turns view-only when `CLAUDEX_MODEL_MAP` overrides the map, and the add-node
suggestions are loaded from the live model catalogs
(`GET /admin/providers/codex/models`, `GET /admin/providers/grok/models`).

When `CLAUDEX_LOCAL_TOKEN` is set, the dashboard prompts for the token once
per page load and keeps it in memory for the lifetime of that page only — it
is attached as a bearer header to admin requests and never written to the
URL, browser storage, or logs. A wrong token triggers exactly one re-prompt.

### Model mapping examples

Route Claude Code model names to Codex and Grok models:

```sh
CLAUDEX_MODEL_MAP='{"fable":"grok:grok-4.5","sonnet":"codex:gpt-5.6-terra","haiku":"codex:gpt-5.6-luna"}' \
  uv run claudex-gateway
```

Keys match exactly first, then as substrings, where the longest matching key
wins (`claude-haiku` beats a catch-all `claude`). A request with no match is
relayed verbatim to Anthropic. Values with an unknown provider prefix (or an
empty model after the prefix) are rejected at startup and by the mapping API,
so a typo like `"grk:grok-4.5"` fails loudly instead of surfacing as a
baffling upstream error.

`GET /health` returns `200` with `status: "ok"` when the Codex credential is
usable and — only if the map routes to Grok — the Grok credential too;
otherwise it returns `503` with `status: "error"`. Each provider's state is
always listed under `providers`.

### Mixing Claude and Codex models

The map decides per request: mapped models are translated to Codex, unmapped
models are forwarded byte-for-byte (headers, betas, credentials) to the real
Anthropic API. Model names stay real Claude names on both paths, so every
Claude Code heuristic keyed on the model name keeps working.

```sh
CLAUDEX_MODEL_MAP='{"opus":"codex:gpt-5.6-sol","haiku":"codex:gpt-5.6-luna"}' \
uv run claudex-gateway
```

Here `/model opus` and background haiku tasks run on Codex while fable and
sonnet stay on Anthropic. By default, passthrough requires a real Claude
login: Claude Code attaches its own OAuth token, and the gateway forwards it
to Anthropic for unmapped models. `ANTHROPIC_AUTH_TOKEN=dummy` must not be
set — the placeholder would be forwarded and rejected by Anthropic.

`CLAUDEX_LOCAL_TOKEN` shares that `Authorization` header: when the token is
set, clients authenticate to the gateway with it, and the same header is what
Anthropic receives for unmapped models — so passthrough traffic fails with an
auth error unless the token happens to be a credential Anthropic accepts. To
run with a local token, add a catch-all map entry (a substring key every
Claude model name contains, e.g. `{"claude": "codex:gpt-5.5"}`) so no request ever
reaches the passthrough path.

Both constraints disappear when a [registered account is
selected](#serving-with-a-registered-account-account-use): the gateway then
consumes client credentials instead of forwarding them, so a dummy token —
or the local token itself — is exactly what the client should send.

## Development

```sh
uv run pytest
```

