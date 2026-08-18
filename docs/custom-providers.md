# Custom providers

Custom providers register additional OpenAI-compatible Responses upstreams under
their own model-map prefixes. Define them in `~/.claudex/settings.json`:

```json
{
  "model_map": {
    "haiku": "wrtn:gpt-5.5"
  },
  "custom_providers": {
    "openai_compatible": {
      "wrtn": {
        "wire_api": "responses",
        "base_url": "https://model.wrtn.club/api/v1",
        "api_key": "sk-arena-..."
      }
    }
  }
}
```

The `openai_compatible` family contains named provider entries. All three entry
fields are required:

- `wire_api` must be exactly `"responses"`; Chat Completions upstreams are not
  supported.
- `base_url` must use HTTPS. Plain HTTP is allowed only for a loopback host. The
  URL may include a path prefix such as `/api/v1`; the gateway appends
  `/responses` and `/models`.
- `api_key` is a non-empty static literal. Custom providers do not use OAuth or
  refresh their credentials.

Provider names must match `^[a-z][a-z0-9-]{0,31}$`. The names `codex`, `kimi`,
`grok`, `claude`, and `anthropic` are reserved. A map entry such as
`"haiku": "wrtn:gpt-5.5"` sends that Claude model through the existing OpenAI
Responses translation path to the `wrtn` provider.

Custom providers are edited in `settings.json` only — the dashboard has no
provider CRUD — and a daemon restart is required before changes take effect.
`CLAUDEX_CUSTOM_PROVIDERS` can supply the same object stored under
`custom_providers` as JSON-encoded text; `CLAUDEX_CUSTOM_PROVIDERS=''` means no
custom providers.

Behavioral boundaries to know:

- The translated payload is sent exactly as produced: there is no provider
  sanitizer, `max_output_tokens` injection, or reasoning-effort clamping.
- Context-overflow phrase detection may not recognize a translated upstream's
  wording. Compaction rerouting still works from an exact provider-prefixed
  context-window override or the window reported by the provider's model catalog.
- An upstream `401` surfaces as an upstream error; there is no credential refresh
  or retry.
- The dashboard has no usage or billing card for custom providers.
