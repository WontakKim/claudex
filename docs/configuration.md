# Configuration

Every variable can be set in `~/.claudex/settings.json` or as an
environment variable; the environment wins when both are set (even when set
to an empty string), so the file holds your durable setup and the environment
stays available for one-off overrides.

| Variable | Default | Description |
| --- | --- | --- |
| `CLAUDEX_HOST` | `127.0.0.1` | Bind address |
| `CLAUDEX_PORT` | `8787` | Bind port |
| `CLAUDEX_MODEL_MAP` | empty | JSON mapping of Claude names, exact or substring, to provider-prefixed target models — built-in prefixes are `codex:`, `kimi:`, and `grok:`, and each configured custom-provider name adds another prefix; unmapped models are relayed verbatim to Anthropic |
| `CLAUDEX_CONTEXT_WINDOW_MAP` | empty | JSON mapping of exact provider-prefixed model targets to positive integer context-window overrides; an override takes precedence over that provider's catalog value, e.g. `{"codex:gpt-5.6-sol": 872000}` |
| `CLAUDEX_CUSTOM_PROVIDERS` | empty | JSON-encoded document containing `openai_compatible` and/or `anthropic_compatible` named providers; an empty string means no custom providers. See [Custom providers](custom-providers.md#custom-providers) |
| `CLAUDEX_REASONING_EFFORT` | derived | Force `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` on Codex requests |
| `CODEX_HOME` | `~/.codex` | Directory containing Codex `auth.json` |
| `GROK_HOME` | `~/.grok` | Directory containing the Grok CLI's `auth.json` |
| `KIMI_CODE_HOME` | `~/.kimi-code` | Directory containing the Kimi Code CLI's credential store |
| `CLAUDEX_LOG_LEVEL` | `info` | Process log verbosity: `debug`, `info`, `warning`, or `error`; editable at runtime from the dashboard |
| `CLAUDEX_LOCAL_TOKEN` | unset | Bearer token required by the model request routes and the admin/dashboard routes when set; mandatory for non-loopback binds. See [the passthrough interaction](model-mapping.md#mixing-claude-and-codex-models) |
| `CLAUDEX_COMPACTION_MODEL` | unset | `compaction.model` setting: opt-in `claude:<model-id>` reroute target for oversized Claude Code compaction requests; unset (default) disables the reroute entirely. See [Compaction reroute](compaction.md#compaction-reroute) |
| `CLAUDEX_CLAUDE_ACCOUNT_ID` | unset | `claude_account.id` setting: id of the registered Claude account that serves Anthropic passthrough traffic; unset (default) forwards client credentials untouched. See [Serving with a registered account](claude-accounts.md#serving-with-a-registered-account-account-use) |
| `CLAUDEX_CLAUDE_ACCOUNT_ROUTING` | unset | `claude_account.routing` setting as a JSON-encoded policy document, e.g. `{"mode": "fallback"}` or `{"mode": "balanced", "include_local_login": false}`; `include_local_login` defaults to `true` and controls local Claude Code login participation in balanced mode only. Unset or empty keeps multi-account routing disabled. See [Ordered fallback across registered accounts](claude-accounts.md#ordered-fallback-across-registered-accounts) |

## settings.json

The settings key for each variable is its environment name minus the
`CLAUDEX_` prefix, lowercased (the CLI-home variables — `CODEX_HOME`,
`GROK_HOME`, `KIMI_CODE_HOME` — having no prefix, keep their full names as
`codex_home` / `grok_home` / `kimi_code_home`). Values use native JSON types,
so maps are plain objects instead of JSON-in-a-string:

```json
{
  "model_map": {"opus": "codex:gpt-5.6-sol", "haiku": "codex:gpt-5.6-luna"},
  "context_window_map": {"codex:gpt-5.6-sol": 872000}
}
```

The file is optional and validated at startup: unknown keys and wrongly typed
values abort the boot so a typo cannot be silently ignored.

A custom-provider document can contain both supported families:

```json
{
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

The OpenAI-compatible entry keeps the required `wire_api: "responses"` field.
The Anthropic-compatible entry has only `base_url` and `api_key`; its
`base_url` is a versioned API prefix to which the transport appends exactly
`/messages`. Query and fragment suffixes are invalid. See
[Custom providers](custom-providers.md#anthropic-compatible-schema) for the
static authentication, catalog, and token-count boundaries.
