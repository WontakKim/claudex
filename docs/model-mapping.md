# Model mapping

## Behavior

- Serves `POST /v1/messages` in Anthropic Messages format: models with a
  `CLAUDEX_MODEL_MAP` entry are routed to the provider-prefixed target
  backend, while everything else is forwarded byte-for-byte to the real
  Anthropic API with the client's own credentials.
- Answers as the Claude model the client requested. The mapped upstream model
  is restored in Responses-family output and in both streaming and non-streaming
  native Messages output, so Claude Code heuristics keyed on model names keep
  working for built-in and custom routes.

## Runtime mapping API

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

## Model mapping examples

Route Claude Code model names to Codex, Kimi, Grok, or a configured custom
provider. This environment-only example uses built-ins:

```sh
CLAUDEX_MODEL_MAP='{"fable":"grok:grok-4.5","opus":"kimi:k3","sonnet":"codex:gpt-5.6-terra","haiku":"codex:gpt-5.6-luna"}' \
  uv run claudex-gateway
```

Keys match exactly first, then as substrings, where the longest matching key
wins (`claude-haiku` beats a catch-all `claude`). A request with no match is
relayed verbatim to Anthropic. Values with an unknown provider prefix (or an
empty model after the prefix) are rejected at startup and by the mapping API,
so a typo like `"kim:k2.5"` fails loudly instead of surfacing as a baffling
upstream error.

`GET /health` returns `200` with `status: "ok"` when required provider
readiness checks pass; otherwise it returns `503` with `status: "error"`.
Catalog-capable custom providers use their remote catalog for that readiness
check. Catalog-less Anthropic-compatible providers perform no remote health I/O:
an `ok` state confirms configuration and binding only, not remote entitlement or
Messages compatibility. Use `POST /admin/test` with a complete mapped target for
explicit remote verification.

## Custom provider targets

Custom provider names become model-map prefixes. The configured family, not the
name, selects the wire behavior. For example, a provider named
`openai-by-name` can still use `anthropic_compatible` and the Anthropic Messages
wire when configured in that family.

OpenAI-compatible entries retain the required `wire_api: "responses"` schema and
live catalog behavior. Anthropic-compatible entries have only `base_url` and
`api_key`; their versioned prefix receives exactly `/messages`, they require
manual model IDs, and token counting uses the local approximate characters/4
fallback. See [Custom providers](custom-providers.md#custom-providers).

## Mixing Claude and Codex models

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
selected](claude-accounts.md#serving-with-a-registered-account-account-use): the gateway then
consumes client credentials instead of forwarding them, so a dummy token —
or the local token itself — is exactly what the client should send.
