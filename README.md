# claudex-gateway

A lightweight local gateway that runs mapped Claude Code models on Codex,
Kimi, Grok, or configured custom Responses/Messages backends and relays
everything else to Anthropic untouched.

```text
Claude Code ── "codex:" mapped ───▶ claudex-gateway ── Codex Responses API ─▶ Codex
Claude Code ── "kimi:" mapped ────▶ claudex-gateway ── near-verbatim relay ─▶ Kimi coding API
Claude Code ── "grok:" mapped ────▶ claudex-gateway ── Grok Responses API ──▶ Grok
Claude Code ── custom mapped ──────▶ claudex-gateway ── Responses or Messages ▶ configured upstream
Claude Code ── unmapped model ────▶ claudex-gateway ── verbatim relay ──────▶ Anthropic API
```

- Mapped models run on Codex, Kimi, Grok, or a named custom provider, while everything else is relayed to Anthropic untouched.
- Custom providers support the OpenAI Responses and Anthropic Messages wire families; static Messages providers require manual model IDs and explicit remote verification.
- The gateway reuses each built-in provider's CLI login and can serve traffic through registered Claude accounts with fallback routing.
- Configuration, model mapping, compaction, the dashboard, and detailed behavior are documented in docs/.

Built-in provider prerequisites and logins are covered in
[Providers](docs/providers.md). Static custom-provider schemas and capability
boundaries are covered in [Custom providers](docs/custom-providers.md).

## Quickstart

Start the gateway (details in [Getting started](docs/getting-started.md)):

```sh
uv run claudex-gateway               # background; logs to ~/.claudex/gateway.log
uv run claudex-gateway --foreground  # attached to the terminal
uv run claudex-gateway stop
```

Run a mapped Claude model on another backend (details in [Model mapping](docs/model-mapping.md)):

```sh
CLAUDEX_MODEL_MAP='{"opus":"codex:gpt-5.6-sol","haiku":"codex:gpt-5.6-luna"}' \
uv run claudex-gateway
```

Point Claude Code at the gateway:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

## Docs

- [Getting started](docs/getting-started.md)
- [Providers](docs/providers.md)
- [Claude accounts](docs/claude-accounts.md)
- [Custom providers](docs/custom-providers.md)
- [Configuration](docs/configuration.md)
- [Model mapping](docs/model-mapping.md)
- [Compaction](docs/compaction.md)
- [Dashboard](docs/dashboard.md)
