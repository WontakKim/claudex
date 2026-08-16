# Configuration

Every variable can be set in `~/.claudex/settings.json` or as an
environment variable; the environment wins when both are set (even when set
to an empty string), so the file holds your durable setup and the environment
stays available for one-off overrides.

| Variable | Default | Description |
| --- | --- | --- |
| `CLAUDEX_HOST` | `127.0.0.1` | Bind address |
| `CLAUDEX_PORT` | `8787` | Bind port |
| `CLAUDEX_MODEL_MAP` | empty | JSON mapping of Claude names, exact or substring, to provider-prefixed target models — `codex:`-prefixed values run on Codex, `kimi:`-prefixed values on Kimi, `grok:`-prefixed values on Grok; unmapped models are relayed verbatim to Anthropic |
| `CLAUDEX_CUSTOM_PROVIDERS` | empty | JSON-encoded custom-provider document; an empty string means no custom providers. See [Custom providers](custom-providers.md#custom-providers) |
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
so the model map is a plain object instead of JSON-in-a-string:

```json
{
  "model_map": {"opus": "codex:gpt-5.6-sol", "haiku": "codex:gpt-5.6-luna"}
}
```

The file is optional and validated at startup: unknown keys and wrongly typed
values abort the boot so a typo cannot be silently ignored.
