# Compaction

## Compact command

Manage the opt-in compaction reroute (see [Compaction
reroute](#compaction-reroute) below) from the command line:

```sh
uv run claudex-gateway compact                       # show current state
uv run claudex-gateway compact set claude:<model-id>  # native Claude destination
uv run claudex-gateway compact set codex:<model-id>   # built-in provider destination
uv run claudex-gateway compact set messages-local:large-model  # registered custom destination
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
a transport failure — exits non-zero without a local settings-file fallback.
For a write request, a lost or unusable response does not prove the server
rejected the change. Once the daemon is reachable, query its current target
with `uv run claudex-gateway compact` before retrying.

## Compaction reroute

The compaction reroute is opt-in and disabled by default (unset
`compaction.model`). Its **trigger source** is a detected Claude Code compaction
request mapped to a Responses backend (Codex, Grok, or a Responses custom
provider) whose estimated prompt size exceeds that source backend's known
context window. Configuring a destination does not broaden this trigger to
ordinary requests or Messages sources.

The **destination** is explicit and independent of the source mapping:

- `claude:<model-id>` sends one direct Anthropic attempt.
- `codex:<model-id>`, `kimi:<model-id>`, or `grok:<model-id>` uses that built-in
  provider's existing backend and login credentials. Authenticate the intended
  provider first as described in [Providers](providers.md).
- `<registered-custom-provider>:<model-id>` sends one attempt through that
  provider's configured Responses or Messages backend. Register it first as
  described in [Custom providers](custom-providers.md). Provider registrations
  are fixed for the process lifetime; adding or changing one requires a restart.

Unregistered provider names are rejected. An identical source and destination
bypass the reroute.

In the dashboard, choose a provider and a model, then click **Apply**. Provider
choices follow the dashboard's configured/available providers. Built-in and
custom provider catalogs supply model choices when available; choose **Custom**
to enter an unlisted ID or when a catalog is missing or fails. Native Claude
keeps its curated choices and manual entry. Loading catalogs does not replace
the saved target.

Model IDs are not checked against a remote catalog when saving. Non-Claude model
IDs preserve case and may contain colons; native Claude IDs cannot contain
colons. Selecting or saving a target does not send an inference request.

Native Claude destinations use eligible client credentials and are billed by
Anthropic separately from the mapped backend. Registered-account OAuth
credentials are not substituted. The reroute keeps a nonblank `x-api-key` or a
nonempty Bearer token, except a Bearer matching `CLAUDEX_LOCAL_TOKEN`. It records
`skipped_no_credentials` only when no credential header remains, then skips
rerouting and lets the mapped backend handle the request. The filter
does not validate upstream credentials: a dummy Bearer can still trigger an
Anthropic attempt, with a non-2xx response recorded as `fallback_mapped`.

Built-in and custom destinations instead use their existing provider
configuration and authentication, without requiring client Anthropic credentials. Caller
credentials are not forwarded to these providers. If a custom provider is
configured without an API key, the custom attempt is unauthenticated.

Connection failures and non-2xx responses before stream commitment fall back
to the original mapped backend. For native Claude and Messages
destinations, commitment occurs when the upstream HTTP request is accepted
with a 2xx response. Responses destinations commit the downstream stream
after the first decoded event passes basic envelope checks, not full
event-specific validation. An upstream HTTP 200 alone is not enough: a failed
envelope check still falls back, but malformed event-specific data discovered
after commitment may produce an in-band SSE error without fallback.
Non-streaming attempts fall back unless a complete, valid response is obtained.
Responses destinations reuse
the [Responses stream liveness](providers.md#responses-stream-liveness) behavior;
native Claude and Messages destinations do not.

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
# Requires a registered messages-local provider:
curl -X PUT http://127.0.0.1:8787/admin/settings/compaction \
  -H 'Content-Type: application/json' \
  -d '{"model": "messages-local:large-model"}'
curl -X PUT http://127.0.0.1:8787/admin/settings/compaction \
  -H 'Content-Type: application/json' \
  -d '{"model": null}'
```

`GET`/`PUT /admin/settings/compaction` read and change the same `compaction.model`
setting on a running gateway, honoring the same `CLAUDEX_LOCAL_TOKEN` and
Host guard as the other admin routes; a `PUT` persists the change to
`settings.json` and, like the mapping API, is refused with `409` when
`CLAUDEX_COMPACTION_MODEL` is set in the environment. The [`compact`
command](#compact-command) is a CLI front end for this same API. To disable
rerouting, send `PUT {"model": null}`; the API removes the stored
`compaction.model` key rather than persisting JSON `null`. When editing
`settings.json` directly, remove that key: explicit JSON `null` fails startup
validation.

A successful API save persists the target before making it effective for the
next request; in-flight requests keep their original target. A server-rejected
change leaves the active target unchanged. A network error or lost response
leaves the outcome uncertain: read `GET /admin/settings/compaction` or use the
read-only `compact` command before retrying. The response remains
`{"model": ..., "env_locked": ..., "last_reroute": ...}`. The dashboard does not
show reroute diagnostics; use the API or `compact` command. A non-null
`last_reroute` retains its seven fields: `outcome`, `timestamp`, `target_model`,
`mapped_model`, `estimated_prompt_tokens`, `context_window`, and `detail`.
`target_model` is a bare model ID for native Claude and a provider-qualified ID
for non-Claude destinations.

### Correcting invalid legacy targets

A value such as `claude:wrtn:model` is invalid; it is **not** automatically
reinterpreted as a custom destination. Edit the stored `compaction.model` in
`settings.json` explicitly to `wrtn:model` only if `wrtn` is the intended,
registered custom provider. Otherwise choose a valid `claude:<model-id>` or
remove the `compaction.model` key to disable rerouting; do not replace it with
JSON `null`. Correct or unset `CLAUDEX_COMPACTION_MODEL` too if
it contains the invalid target, then restart the gateway with the corrected
configuration. Do not rely on `compact set` or `compact off` to repair invalid
startup configuration: those commands also load and validate configuration
before making a change.
