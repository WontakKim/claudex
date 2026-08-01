# claudex-gateway

A lightweight local gateway that runs mapped Claude Code models on the OpenAI
Codex or Kimi backend and relays everything else to Anthropic untouched.

```text
Claude Code ── mapped model ──────▶ claudex-gateway ── Codex Responses API ─▶ Codex
Claude Code ── "kimi:" mapped ────▶ claudex-gateway ── near-verbatim relay ─▶ Kimi coding API
Claude Code ── unmapped model ────▶ claudex-gateway ── verbatim relay ──────▶ Anthropic API
```

It ports the Claude → Responses translation layer and the Claude → Kimi relay
of [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) to Python
without the multi-provider registry: mapped models to Codex or Kimi,
everything else to Anthropic.

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
- Relays models mapped with a `kimi:` prefix to Kimi's coding endpoint
  (`api.kimi.com/coding`), which speaks the Anthropic Messages API natively —
  no schema translation, only the model name is rewritten out and restored,
  and the client's credentials are replaced with the gateway's Kimi OAuth
  token (`claudex-gateway login kimi`).
- Answers as the Claude model the client requested — the Codex or Kimi target
  model never appears on the Anthropic wire, so Claude Code heuristics keyed
  on model names keep working.
- Serves `GET /health` with the readiness state of the Codex and Kimi
  upstreams.
- Serves a runtime dashboard at `GET /` for editing the model map, checking
  provider health, and testing model connections before wiring them.
- Reuses the Codex CLI login at `~/.codex/auth.json`; the Kimi login is built
  in and stores its tokens at `~/.claudex/kimi-auth.json`.

Two Anthropic contract points cannot be preserved on the Codex path and are
explicit choices, not bugs:

- `max_tokens` is validated but not enforced: the Codex backend rejects the
  Responses `max_output_tokens` parameter, so mapped requests cannot cap
  output length upstream and the gateway does not truncate locally.
- `POST /v1/messages/count_tokens` for Codex-mapped models returns a
  characters/4 estimate. Mapped prompts are never sent to Anthropic just to
  be counted and no Codex tokenizer is available, so treat the number as a
  rough gauge for context-usage display, not an exact count for billing or
  hard limits. Kimi-mapped models use Kimi's native counter (falling back to
  the same estimate when it is unavailable), and unmapped models pass through
  to Anthropic's real counter.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- For Codex targets: a logged-in [Codex CLI](https://github.com/openai/codex)
  (`codex login`)
- For Kimi targets: a Kimi subscription logged in via
  `claudex-gateway login kimi`

The server starts even when a login is missing; `/health` reports the
credential state per provider.

## Install (in-house, zxcv)

On the company network the gateway ships as a self-contained CLI through the
[zxcv](https://github.wrtn.club/wrtn-tech/zxcv) Homebrew channel — no Python
or uv required:

```sh
zxcv install claudex
```

Then use `claudex-gateway` wherever the examples below say
`uv run claudex-gateway`. The install also puts a ready-made `claudex`
command on PATH that starts the gateway when it is not already running and
launches Claude Code through it; `claudex settings` opens the gateway
dashboard in the browser instead, and every other argument is passed through
to `claude` untouched (override the address with `CLAUDEX_HOST` /
`CLAUDEX_PORT` if you changed the bind). The rest of this README covers
running from a checkout with uv.

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
untouched. Only when there is no Claude login at all, satisfy Claude Code's
startup check with a placeholder token:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 ANTHROPIC_AUTH_TOKEN=dummy claude
```

The placeholder is forwarded to Anthropic for unmapped models and rejected
there, so this setup needs a catch-all map entry — a substring key that every
Claude model name contains, e.g. `{"claude": "gpt-5.5"}` — to keep every
request on Codex.

zxcv installs ship this as the `claudex` command (which also starts the
gateway first when needed); from a checkout, an alias covers the launch part:

```sh
alias claudex='ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude'
```

To run only some Claude models on Codex and keep the rest on the real
Anthropic API, see
[Mixing Claude and Codex models](#mixing-claude-and-codex-models).

### Claude Code → Kimi

Log in once with the built-in OAuth device flow — it prints a verification
URL and a code, waits for approval in the browser, and stores the tokens at
`~/.claudex/kimi-auth.json` (auto-refreshed from then on, no daemon restart
needed):

```sh
uv run claudex-gateway login kimi
```

Then route models to Kimi with a `kimi:` prefix in the map:

```json
{
  "model_map": {"opus": "kimi:k3", "haiku": "gpt-5.6-luna"}
}
```

A bare value stays a Codex model (`codex:` is also accepted explicitly), so
existing maps keep working unchanged. Kimi's coding endpoint speaks the
Anthropic Messages API natively, so requests and responses — streaming and
non-streaming, thinking, tool use — are relayed as-is; only the model name
and credentials are swapped.

The model ID after `kimi:` bypasses the gateway untouched: it is sent to Kimi
exactly as written and never validated against a model list, so a newly
released model works the moment Kimi ships it — no gateway update needed. The
authoritative list of valid IDs is Kimi's own live catalog, which the gateway
exposes for map authoring (and as the preset source for the future dashboard):

```sh
curl http://127.0.0.1:8787/admin/kimi/models
```

The endpoint requires a completed `login kimi` and honors the same
`CLAUDEX_LOCAL_TOKEN` and Host guard as the other admin routes; the response
is Kimi's catalog verbatim, unshaped by the gateway. Copy the `id` exactly —
the catalog mixes naming styles (e.g. `kimi-for-coding` next to `k3`), which
is precisely why the gateway refuses to normalize them.

## Configuration

Every variable can be set in `~/.claudex/settings.json` or as an
environment variable; the environment wins when both are set (even when set
to an empty string), so the file holds your durable setup and the environment
stays available for one-off overrides.

| Variable | Default | Description |
| --- | --- | --- |
| `CLAUDEX_HOST` | `127.0.0.1` | Bind address |
| `CLAUDEX_PORT` | `8787` | Bind port |
| `CLAUDEX_MODEL_MAP` | empty | JSON mapping of Claude names, exact or substring, to target models — bare or `codex:`-prefixed values run on Codex, `kimi:`-prefixed values on Kimi; unmapped models are relayed verbatim to Anthropic |
| `CLAUDEX_REASONING_EFFORT` | derived | Force `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` on Codex requests |
| `CODEX_HOME` | `~/.codex` | Directory containing Codex `auth.json` |
| `CLAUDEX_LOG_LEVEL` | `info` | Process log verbosity: `debug`, `info`, `warning`, or `error`; editable at runtime from the dashboard |
| `CLAUDEX_LOCAL_TOKEN` | unset | Bearer token required by the model request routes and the admin/dashboard routes when set; mandatory for non-loopback binds. See [the passthrough interaction](#mixing-claude-and-codex-models) |

### settings.json

The settings key for each variable is its environment name minus the
`CLAUDEX_` prefix, lowercased (`CODEX_HOME`, having no prefix, keeps its full
name as `codex_home`). Values use native JSON types, so the model map is a
plain object instead of JSON-in-a-string:

```json
{
  "model_map": {"opus": "gpt-5.6-sol", "haiku": "gpt-5.6-luna"}
}
```

The file is optional and validated at startup: unknown keys and wrongly typed
values abort the boot so a typo cannot be silently ignored.

### Runtime mapping API

The model map can be changed while the gateway is running — no restart of
the gateway or its clients:

```sh
curl http://127.0.0.1:8787/admin/mapping
curl -X PUT http://127.0.0.1:8787/admin/mapping \
  -H 'Content-Type: application/json' \
  -d '{"model_map": {"opus": "gpt-5.6-sol"}}'
```

A `PUT` accepts `model_map`, replaces it for all subsequent requests, and
persists it to `settings.json` (other keys in the file are preserved). When
`CLAUDEX_MODEL_MAP` is set the request is refused with `409` — the
environment would silently win again on restart. The endpoint honors
`CLAUDEX_LOCAL_TOKEN` and only answers requests whose `Host` is the gateway
itself, so foreign web pages cannot drive it from a browser.

### Dashboard

Opening `http://127.0.0.1:8787/` in a browser (`claudex settings` on zxcv
installs) serves a dashboard on top of the same admin API: the General tab
shows the Codex upstream's health, the Log tab tails the gateway's recent
log lines (`GET /admin/logs`) and holds the runtime log-level control
(`PUT /admin/log-level`, applied immediately and persisted), and the Router
tab is a canvas editor — drag a port to wire a model, Apply to `PUT` the
map, and use the connection test box (`POST /admin/test`) to send one
minimal request through the gateway before wiring a new model id. The board
turns view-only when `CLAUDEX_MODEL_MAP` overrides the map, and the Codex
column is loaded from the live Codex model catalog
(`GET /admin/codex/models`). The dashboard predates the Kimi direction — its
model column is Codex-only and `kimi:`-prefixed targets show up as plain
values, though the connection test understands the prefix and probes the
right backend; a redesign is planned separately and will draw its Kimi
presets from `GET /admin/kimi/models`.

When `CLAUDEX_LOCAL_TOKEN` is set, the dashboard prompts for the token once
per page load and keeps it in memory for the lifetime of that page only — it
is attached as a bearer header to admin requests and never written to the
URL, browser storage, or logs. A wrong token triggers exactly one re-prompt.

### Model mapping examples

Route Claude Code model names to Codex and Kimi models:

```sh
CLAUDEX_MODEL_MAP='{"fable":"gpt-5.6-sol","opus":"kimi:k3","sonnet":"gpt-5.6-terra","haiku":"gpt-5.6-luna"}' \
  uv run claudex-gateway
```

Keys match exactly first, then as substrings, where the longest matching key
wins (`claude-haiku` beats a catch-all `claude`). A request with no match is
relayed verbatim to Anthropic. Values with an unknown provider prefix (or an
empty model after the prefix) are rejected at startup and by the mapping API,
so a typo like `"kim:k2.5"` fails loudly instead of surfacing as a baffling
upstream error.

`GET /health` returns `200` with `status: "ok"` when the Codex credential is
usable and — only if the map routes to Kimi — the Kimi credential too;
otherwise it returns `503` with `status: "error"`. Each provider's state is
always listed under `providers`.

### Mixing Claude and Codex models

The map decides per request: mapped models are translated to Codex, unmapped
models are forwarded byte-for-byte (headers, betas, credentials) to the real
Anthropic API. Model names stay real Claude names on both paths, so every
Claude Code heuristic keyed on the model name keeps working.

```sh
CLAUDEX_MODEL_MAP='{"opus":"gpt-5.6-sol","haiku":"gpt-5.6-luna"}' \
uv run claudex-gateway
```

Here `/model opus` and background haiku tasks run on Codex while fable and
sonnet stay on Anthropic. Passthrough requires a real Claude login: Claude
Code attaches its own OAuth token, and the gateway forwards it to Anthropic
for unmapped models. `ANTHROPIC_AUTH_TOKEN=dummy` must not be set — the
placeholder would be forwarded and rejected by Anthropic.

`CLAUDEX_LOCAL_TOKEN` shares that `Authorization` header: when the token is
set, clients authenticate to the gateway with it, and the same header is what
Anthropic receives for unmapped models — so passthrough traffic fails with an
auth error unless the token happens to be a credential Anthropic accepts. To
run with a local token, add a catch-all map entry (a substring key every
Claude model name contains, e.g. `{"claude": "gpt-5.5"}`) so no request ever
reaches the passthrough path.

## Development

```sh
uv run pytest
```

### Releasing

Bump `version` in `pyproject.toml` and merge to `main`. The release workflow
builds the self-contained darwin-arm64 tarball with
`scripts/build-darwin-asset.sh` and attaches it to a GHE release
`v<version>`; users get it once the central zxcv repo's
`packages/claudex.json` points at that release (repo/tag/asset).

To inspect the asset locally, run the same build script on a Mac — it packs
`build/claudex-gateway-<version>-darwin-arm64.tar.gz` and smoke-tests the
bundled runtime.
