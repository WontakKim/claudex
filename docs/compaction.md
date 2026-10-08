# Compaction

## Compact command

Manage the opt-in compaction reroute (see [Compaction
reroute](#compaction-reroute) below) from the command line:

```sh
uv run claudex-gateway compact                       # show current state
uv run claudex-gateway compact set claude:<model-id>  # enable, target <model-id>
uv run claudex-gateway compact off                    # disable
```

When a compatible daemon is identified, the command uses its authenticated
`GET`/`PUT /admin/settings/compaction` API, so successful changes take effect
immediately with no restart. An identified older daemon returning `404` or
`405` falls back to the settings file and warns that a restart is
required. With no listener, or a confirmed foreign process on the port,
the command reads or writes `settings.json` directly (picked up the next
time the gateway starts). Ambiguous reachability fails closed for `set`
and `off` and shows settings-file state with an unreachable warning for
the read-only form. Any other failure from an identified daemon — an
error status such as the environment-lock `409`, a malformed response, or
a transport failure — exits non-zero without touching the settings file.

## Compaction reroute

The compaction reroute is opt-in and disabled by default (unset
`compaction.model`); once configured to a `claude:<model-id>` target, it
triggers only for a detected Claude Code compaction request mapped to a
Responses backend (Codex, Grok, or a Responses custom provider) whose
estimated prompt size exceeds that backend's known context window, and
makes a single Anthropic attempt using the client's own credentials, billed
by Anthropic separately from the mapped backend. Its failure semantics: missing eligible
Anthropic credentials skip the reroute entirely; connection failures and
non-2xx responses before a streaming HTTP 2xx commit fall back silently to
the mapped backend; failures after that commit surface as in-band SSE
errors without fallback; non-streaming requests fall back on any failure
before a complete, valid JSON response is obtained.

Registered-account OAuth credentials are not substituted for the client's
headers. The reroute keeps a nonblank `x-api-key` or a nonempty Bearer token,
except a Bearer matching the configured `CLAUDEX_LOCAL_TOKEN`. It records
`skipped_no_credentials` only when no credential header remains. The filter
does not validate upstream credentials: a dummy Bearer can still trigger an
Anthropic attempt, with a non-2xx response recorded as `fallback_mapped`.

The context window is the provider-prefixed `context_window_map` override or,
without one, the backend's catalog value; when neither is known, the reroute
never triggers. For a request that carries tools, the size estimate covers the
prompt as translated for that backend, including active tool schemas.

The literals used to detect a compaction request are a versioned contract
with Claude Code `2.1.223` — re-verified against the client binary before
ever being changed, never guessed. Detection also accepts the compaction
instruction when it follows exact `Tool loaded.` note blocks, as observed
in Claude Code `2.1.286` requests.

```sh
curl http://127.0.0.1:8787/admin/settings/compaction
curl -X PUT http://127.0.0.1:8787/admin/settings/compaction \
  -H 'Content-Type: application/json' \
  -d '{"model": "claude:claude-opus-5"}'
curl -X PUT http://127.0.0.1:8787/admin/settings/compaction \
  -H 'Content-Type: application/json' \
  -d '{"model": null}'
```

`GET`/`PUT /admin/settings/compaction` read and change the same `compaction.model`
setting on a running gateway, honoring the same `CLAUDEX_LOCAL_TOKEN` and
Host guard as the other admin routes; a `PUT` persists the change to
`settings.json` and, like the mapping API, is refused with `409` when
`CLAUDEX_COMPACTION_MODEL` is set in the environment. The [`compact`
command](#compact-command) is a CLI front end for this same API.
