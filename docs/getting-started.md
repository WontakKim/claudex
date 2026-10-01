# Getting started

## Requirements

- A source checkout with Python 3.11+ and [uv](https://docs.astral.sh/uv/), or
  the macOS arm64 release asset, which bundles its own Python
- Claude Code (`claude`) on `PATH`
- A provider login for each backend you map to; see [Providers](providers.md)

The server starts even when a login is missing; `/health` reports the
credential state per provider.

## Start the gateway

From the repository root of a source checkout:

```sh
uv run claudex-gateway               # background; logs to ~/.claudex/gateway.log
uv run claudex-gateway --foreground  # attached to the terminal
uv run claudex-gateway stop
```

A successful background start prints
`claudex-gateway started on http://127.0.0.1:8787` with the daemon pid and log
path. An invalid configuration prints `configuration error: ...` and exits
without starting; a daemon that exits during startup prints the last lines of
its log.

Starting is idempotent: when the port already answers with the same gateway
version, the command reports the running instance and exits. That running
instance keeps its configuration, so stop and start it to apply edited
settings or environment variables (see [Configuration](configuration.md)).
A daemon left running across a package update is stopped and replaced
automatically when its identity can be verified (any 0.4.0+ daemon), and a
port occupied by something else fails loudly. The background log file is
rewritten on every start.

`stop` (and the automatic stale-daemon replacement) verifies the daemon's
identity before sending any signal: the pid, host, port, and per-start nonce
recorded in `~/.claudex/gateway.pid` must match what the running gateway
reports on `GET /api/hello`. A record that cannot be verified — the bare-pid
file of a pre-0.4 install, a corrupt record, or a pid that no longer answers
as the recorded gateway — is never signaled; the command explains what to
stop manually instead (a one-time step when upgrading from 0.3.x). This
makes pid reuse harmless: a recycled pid can no longer be terminated by
mistake.

### From the macOS release

The `claudex-gateway-<version>-darwin-arm64.tar.gz` release asset runs only on
macOS arm64 and needs neither uv nor a system Python. From the extracted
directory, `./bin/claudex-gateway` accepts the same commands as
`uv run claudex-gateway`, and `./bin/claudex` starts the gateway if needed and
then launches Claude Code through it:

```sh
./bin/claudex            # extra arguments are passed to claude
./bin/claudex settings   # open the dashboard instead
```

The launcher sets `ENABLE_TOOL_SEARCH=true` unless the variable is already set.
It builds the gateway URL from `CLAUDEX_HOST` and `CLAUDEX_PORT` in its own
environment, defaulting to `127.0.0.1:8787`; a host or port set only in
`settings.json` is not seen by the launcher, so export the same values.

### Check it is running

```sh
curl -s http://127.0.0.1:8787/health
```

The response lists every built-in and custom provider under `providers`. It
returns `200` with `"status": "ok"` when the Codex credentials load and every
Kimi, Grok, or custom provider the model map routes to is ready; otherwise it
returns `503` with `"status": "error"`, and the failing entry carries a
`detail`. Codex credentials count toward readiness even when nothing is mapped
to Codex; other providers report `"required": false` when unmapped. `/health`
does not require `CLAUDEX_LOCAL_TOKEN`. When that token is configured, requests
without a valid bearer token omit account ids, emails, and account names and
receive generic diagnostics instead of free-form error details; provider names
and readiness stay unchanged. Send `Authorization: Bearer <CLAUDEX_LOCAL_TOKEN>`
to see the full details, as the dashboard does. With no token configured, the
response is unchanged.

## Connect Claude Code

```sh
ENABLE_TOOL_SEARCH=true ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
```

Claude Code disables MCP tool search for a non-Anthropic base URL unless
`ENABLE_TOOL_SEARCH` is set; see
[MCP tool search and context usage](providers.md#mcp-tool-search-and-context-usage).
With an empty model map, every request is relayed to Anthropic, by default
with Claude Code's own credentials. To run Claude models on another backend,
see [Model mapping](model-mapping.md).

## Development

```sh
uv run pytest
```
