# Getting started

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)

The server starts even when a login is missing; `/health` reports the
credential state per provider.

## Start the gateway

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

### Check it is running

`GET /health` reports the readiness state of the Codex, Kimi, and Grok
upstreams.

## Development

```sh
uv run pytest
```
