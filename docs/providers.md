# Providers

The built-in Codex, Kimi, and Grok providers reuse their CLI authentication:
the Codex CLI's `~/.codex/auth.json`, the Kimi Code CLI's `~/.kimi-code`
credential store, and the Grok CLI's `~/.grok/auth.json`, each refreshed in
place like the CLI itself does. Custom providers are different: they use static
credentials from gateway configuration and do not participate in CLI login,
OAuth, or refresh flows.

## Codex

- For Codex targets: a logged-in [Codex CLI](https://github.com/openai/codex)
  (`codex login`)

Launch Claude Code through the gateway:

```sh
ENABLE_TOOL_SEARCH=true ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

A logged-in Claude Code needs no token setup: it attaches its own credentials,
which the Codex path never uses and the passthrough path forwards to Anthropic
untouched.

A shell alias covers the launch part:

```sh
alias claudex='ENABLE_TOOL_SEARCH=true ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude'
```

To run only some Claude models on Codex and keep the rest on the real
Anthropic API, see
[Mixing Claude and Codex models](model-mapping.md#mixing-claude-and-codex-models).

### Behavior

Supports streaming and non-streaming responses.

Translates text, images, PDF documents (base64 `application/pdf` blocks in
user messages — other document forms are rejected with a clear error rather
than silently dropped), thinking/reasoning blocks, function calls and
results (with 64-char-safe names for long MCP tool namespaces), usage,
stop reasons, and native web search; mid-conversation `system` messages
keep operator authority as Responses `developer` messages.

### MCP tool search and context usage

Claude Code disables tool search by default for a non-Anthropic base URL. With
many MCP tools, that loads their full schemas into the first request even when
the prompt is short. The bundled `claudex` launcher (see
[From the macOS release](getting-started.md#from-the-macos-release)) enables
tool search unless `ENABLE_TOOL_SEARCH` is already set. For a shell alias or a
source installation, include the variable as shown above. Changing a gateway
request cannot enable a tool that the running Claude Code client has disabled.

Claude Code searches its catalog locally and sends the discovered definitions.
Responses routes preserve the selected names in search results, use stable
callable aliases for long names, and replay ordered tool additions, schema
redefinitions, removals, and explicit re-additions. Only the final active
functions are callable. Historical definitions and change records retain their
original order; historical references do not reactivate functions withdrawn by
explicit removal events. Classic clients can retain disconnected tools in their
catalog without a removal event, so live availability and permissions still
belong to the client's executor. Tool errors remain errors in the translated
conversation.

Responses routes validate tool schemas locally. If any tool callable in the
request has an invalid schema, declares an unsupported `$schema` dialect, or
uses a `$ref` that the schema does not embed, both `/v1/messages` and
`/v1/messages/count_tokens` fail with HTTP 400 `invalid_request_error` before
anything is sent upstream. The message names the function (a name that is not a
valid Responses function name appears as its `mcp__gw_` alias). The gateway
never retrieves remote schemas, so fix that MCP server's tool schema or
disconnect the server. With tool search disabled, every connected MCP tool is
callable and therefore checked.

The gateway also validates completed function arguments against the current
schema before emitting an executable call. Text and reasoning still stream,
while function calls wait for validation. If the arguments fail validation,
regex checks during argument validation exceed the 2-second budget for that
call, or the backend stream cannot be matched to one complete call, including a
stream that ends without a terminal event, no tool call is emitted. The gateway returns an API error
instead: an `error` event on a streaming response, or HTTP 502 on a
non-streaming one.

This bridge uses ordinary Responses functions. It does not reproduce Anthropic's
position-specific schema rendering or guarantee the same prompt-cache behavior.
Mapped token counting estimates the prepared prompt, while response usage remains
the backend's reported usage. A Claude-facing model name or `[1m]` suffix does not
establish the actual mapped backend's context capacity.

Unmapped Anthropic requests keep their original tool protocol. Native
Anthropic-compatible providers remain responsible for accepting that protocol;
use `ENABLE_TOOL_SEARCH=false` if a particular upstream does not support it.
Anthropic's hosted `tool_search_tool_*` API tools are a different protocol from
Claude Code's client `ToolSearch`; Responses routes reject them with HTTP 400
instead of converting them into functions.

### Model suggestions

The dashboard's Codex add-node suggestions come from the live catalog when
`codex --version` is available. The gateway uses the newer of the installed
CLI version and its verified bundled client version (0.157.1) for both the
catalog request and its User-Agent. After a 60-second cache expires, the
next suggestion request checks the installed version again, so updating Codex
does not require restarting the gateway. If the CLI is not installed,
suggestions use a short built-in list instead, without fetching the catalog
or requiring Codex credentials. A CLI that cannot report its version or a
failed live catalog request produces a visible dashboard error rather than
falling back to that list. These presets only suggest model IDs; context
windows and Fast availability still come from the live catalog, never from
preset metadata.

### Fast mode

Opt into Codex Fast mode with `"codex": { "service_tier": "fast" }` in
`settings.json` or `CLAUDEX_CODEX_SERVICE_TIER=fast`; the gateway sends
Responses `service_tier: "priority"` only when the live model catalog
advertises Fast, while unknown or unsupported models silently stay standard.
Fast mode burns ChatGPT-plan usage about 2–2.5x faster and speeds responses
about 1.5x.

## Kimi

- For Kimi targets: a logged-in Kimi Code CLI (`kimi login`)

The gateway reuses the Kimi Code CLI login — no gateway-side login step.
With the CLI logged in (`kimi login`, tokens at
`~/.kimi-code/credentials/kimi-code.json`), route models to Kimi with a
`kimi:` prefix in the map:

```json
{
  "model_map": {"opus": "kimi:k3", "haiku": "codex:gpt-5.6-luna"}
}
```

Every value names its provider (`codex:`, `kimi:`, `grok:`, or a configured
custom prefix); a bare model name is rejected at boot and on `PUT`, so an entry
always says which backend serves it. Kimi's coding endpoint speaks the
Anthropic Messages API natively, so requests and responses — streaming and
non-streaming, thinking, tool use — are relayed as-is; only the model name
and credentials are swapped.

The model ID after `kimi:` bypasses the gateway untouched: it is sent to Kimi
exactly as written and never validated against a model list, so a newly
released model works the moment Kimi ships it — no gateway update needed. The
authoritative list of valid IDs is Kimi's own live catalog, which the gateway
exposes for map authoring and the Router's add-node suggestions:

```sh
curl http://127.0.0.1:8787/admin/providers/kimi/models
```

The endpoint requires a logged-in Kimi Code CLI and honors the same
`CLAUDEX_LOCAL_TOKEN` and Host guard as the other admin routes; the response
is Kimi's catalog verbatim, unshaped by the gateway. Copy the `id` exactly —
the catalog mixes naming styles (e.g. `kimi-for-coding` next to `k3`), which
is precisely why the gateway refuses to normalize them.

Kimi and static Anthropic-compatible custom providers share the generic native
Messages relay, including streaming and non-streaming response ownership and
requested-model restoration. Their transport policies remain separate. Kimi
keeps its existing CLI-derived authentication, live model catalog, and native
token counter. A static custom provider strips caller credentials, applies one
configured Bearer credential, has no catalog or remote token counter, and does
not refresh or retry authentication. See [Custom providers](custom-providers.md#anthropic-compatible-schema).

## Grok

- For Grok targets: a logged-in [Grok CLI](https://github.com/xai-org/grok-build)
  (`grok login`)

The gateway reuses the Grok CLI login — no gateway-side login step. With the
CLI logged in (`grok login`, tokens at `~/.grok/auth.json`, or
`grok login --api-key` for a plain Grok API key), route models to Grok with a
`grok:` prefix in the map:

```json
{
  "model_map": {"opus": "grok:grok-4.5", "haiku": "codex:gpt-5.6-luna"}
}
```

Grok speaks the same Responses API family as the Codex backend, so requests
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

The gateway uses the newer of the installed `grok --version` and its verified
bundled client version (0.2.93) for both `x-grok-client-version` and User-Agent
on Responses and model catalog requests. After a 60-second cache expires, the
next request checks the installed version again, so updating Grok does not
require restarting the gateway. If the CLI is absent or cannot report its
version, the gateway falls back to the bundled version.

## Custom providers

Named custom routes can use either the OpenAI Responses wire family or the
Anthropic Messages wire family. Provider names do not select behavior; the
configured family and its `wire_kind` do. Static Messages providers support
only the configured versioned prefix plus `/messages`, require manually entered
model IDs, and use local approximate token counting. Health confirms their
configuration and route binding without contacting the remote service;
`POST /admin/test` performs the explicit remote Messages verification.

The complete schemas, authentication boundaries, dashboard semantics, and
validation scope are in [Custom providers](custom-providers.md#custom-providers).
No custom provider is added to built-in usage or billing views.

## Limitations

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
