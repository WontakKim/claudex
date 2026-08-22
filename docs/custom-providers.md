# Custom providers

Custom providers add named model-map prefixes without adding a built-in vendor.
The gateway supports two families:

- `custom_providers.openai_compatible` for OpenAI Responses-compatible upstreams.
- `custom_providers.anthropic_compatible` for Anthropic Messages-compatible upstreams.

Define either or both families in `~/.claudex/settings.json`:

```json
{
  "model_map": {
    "haiku": "responses-local:upstream-small",
    "opus": "messages-local:upstream-large"
  },
  "custom_providers": {
    "openai_compatible": {
      "responses-local": {
        "wire_api": "responses",
        "base_url": "https://responses.example/api/v1",
        "api_key": "replace-with-static-key"
      }
    },
    "anthropic_compatible": {
      "messages-local": {
        "base_url": "https://messages.example/v1",
        "api_key": "replace-with-static-key"
      }
    }
  }
}
```

Provider names must match `^[a-z][a-z0-9-]{0,31}$`, must be unique across
families, and cannot be `codex`, `kimi`, `grok`, `claude`, or `anthropic`.
The model ID after the provider prefix is supplied by the operator and is sent
upstream without a gateway model allowlist.

Custom providers are edited in `settings.json` only; the dashboard does not
provide provider CRUD. Restart the daemon after changing a provider definition.
`CLAUDEX_CUSTOM_PROVIDERS` accepts the same `custom_providers` object as
JSON-encoded text, and `CLAUDEX_CUSTOM_PROVIDERS=''` disables all custom
providers.

## OpenAI-compatible schema

Each `custom_providers.openai_compatible` entry requires exactly these fields:

- `wire_api`: must be exactly `"responses"`. The backward-compatible
  `wire_api="responses"` contract remains required; Chat Completions upstreams
  are not supported.
- `base_url`: an HTTPS URL, except that plain HTTP is allowed for loopback
  hosts. It may contain a path prefix. The gateway appends `/responses` for
  inference and `/models` for catalog discovery.
- `api_key`: a non-empty static credential. The gateway does not perform OAuth
  or credential refresh for this family.

A mapped request uses the existing Claude-to-Responses translation path. The
translated payload is sent as produced: custom providers do not receive the
built-in Codex or Grok payload sanitizers, `max_output_tokens` injection, or
reasoning-effort clamping. Context-overflow phrase detection may not recognize
an upstream's wording; use an exact provider-prefixed
`context_window_map` override when catalog metadata is unavailable or
insufficient.

An upstream `401` is returned as an upstream error. The gateway does not refresh
or retry a custom OpenAI-compatible credential.

## Anthropic-compatible schema

Each `custom_providers.anthropic_compatible` entry requires exactly these
fields:

- `base_url`: a versioned API prefix, such as an installation's documented
  `/v1` prefix. The gateway strips trailing slashes and appends exactly
  `/messages` for inference. A query string or fragment is invalid.
- `api_key`: a non-empty static credential.

There is no `wire_api` field for this family. The transport does not infer or
probe `/models` or `/messages/count_tokens`, and those operations must not be
added to `base_url`.

For each Messages request, the gateway removes caller `Authorization` and
`X-API-Key` credentials, removes connection-specific headers, removes the
Claude OAuth beta marker, and sends the configured credential as
`Authorization: Bearer <configured value>`. Authentication is always static:
there is no browser or CLI OAuth flow, refresh, or automatic retry after `401`
or `429`.

The generic Messages relay supports streaming and non-streaming requests. It
changes the outgoing top-level model to the mapped upstream model, then restores
the Claude model requested by the caller in a non-streaming JSON response or the
streaming `message_start` event. Other request content remains structurally
unchanged. This preserves tool-use/tool-result continuation and passes native
thinking and signature blocks through without interpreting them. Other
streaming events are relayed unchanged apart from normal response-header
filtering and error handling.

The transport returns an open successful response to the generic relay. The
relay then owns and closes that response on normal completion, cancellation,
or stream failure. This ownership boundary is shared with Kimi, but the
transport capabilities are not: Kimi has its existing CLI-derived
authentication, live catalog, and native token counter, while a static
Anthropic-compatible provider has only the configured Bearer credential and
`/messages` transport.

`POST /v1/messages/count_tokens` for this family never contacts a guessed
upstream token-count endpoint. It returns the gateway's local approximate
`max(characters / 4, 1)` count, which is suitable for a context-usage display,
not billing or hard context-limit decisions.

## Catalog, health, and dashboard semantics

`GET /admin/settings/mapping` exposes safe custom-provider metadata including
`wire_kind` and `catalog_available`; it never includes `api_key`.

- Responses-family metadata uses `wire_kind: "responses"`, which the dashboard
  labels **Responses API**. Its current backend provides a catalog, so
  `catalog_available` is `true`. A successful health catalog lookup is shown as
  **connected**.
- Messages-family metadata uses `wire_kind: "anthropic_messages"`, which the
  dashboard labels **Anthropic Messages**. Static custom providers have no
  catalog, so `catalog_available` is `false`. The dashboard does not call the
  custom catalog endpoint and shows a healthy local binding as **configured**,
  not connected.

A configured state means the definition and route binding are ready. It does
not prove that the remote provider accepts the credential, model, Messages
shape, tools, thinking blocks, or streaming behavior. `GET /health` performs no
remote I/O for a catalog-less static Messages provider, even when that provider
is required by the model map.

Model IDs for catalog-less providers must be entered manually in the Router's
existing model field. Catalogs are autocomplete only for providers that expose
one; manual IDs remain valid for every custom provider.

Use the dashboard connection test or call `POST /admin/test` with a complete
`provider:model` target to perform one remote Messages verification before
wiring a model. The test does not turn health readiness into a persistent remote
entitlement guarantee.

Custom providers do not appear in subscription usage or billing views.

## Compatibility and validation scope

The repository's deterministic tests cover generic dispatch independent of a
provider's name, streaming events outside `message_start`, streaming and
non-streaming model restoration, response ownership, header and credential
policy, and the local token-count fallback. The relay's top-level model-only
request rewrite is the basis for the tool, thinking, and signature preservation
described above. This coverage is not a live external-provider end-to-end
validation.

Z.AI is a possible use case for the generic Anthropic-compatible family only if
the exact provider surface is tested against this contract. It is not a
built-in provider: the gateway has no Z.AI default URL, constant,
vendor-specific branch, or model allowlist. Do not guess a versioned operation
URL. Live compatibility is guaranteed only for the exact provider surfaces and
operations that have been validated live.
