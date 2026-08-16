# Providers

claudex-gateway reuses each provider's CLI login — no gateway-side auth: the
Codex CLI's `~/.codex/auth.json`, the Kimi Code CLI's `~/.kimi-code` credential
store, and the Grok CLI's `~/.grok/auth.json`, each refreshed in place like the
CLI itself does.

## Codex

- For Codex targets: a logged-in [Codex CLI](https://github.com/openai/codex)
  (`codex login`)

Launch Claude Code through the gateway:

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
[Mixing Claude and Codex models](model-mapping.md#mixing-claude-and-codex-models).

### Behavior

Supports streaming and non-streaming responses.

Translates text, images, PDF documents (base64 `application/pdf` blocks in
user messages — other document forms are rejected with a clear error rather
than silently dropped), thinking/reasoning blocks, function calls and
results (with 64-char-safe names for long MCP tool namespaces), usage,
stop reasons, and native web search; mid-conversation `system` messages
keep operator authority as Responses `developer` messages.

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

Every value names its provider (`codex:`, `kimi:`, or `grok:`); a bare model
name is rejected at boot and on `PUT`, so an entry always says which backend
serves it. Kimi's coding endpoint speaks the
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
