# Claude accounts

`claudex-gateway account add` launches the `claude` CLI (`claude auth login
--claudeai`) in a temporary config directory and captures the resulting
login into the local account registry. `account add --from <dir>` does not
launch `claude`: it instead imports an already-completed login from `<dir>`,
which must be the exact config-directory string used at that login (on
macOS this selects a Keychain item scoped to that exact string — no `~`
expansion, trailing-slash cleanup, or other path canonicalization).
Interactive capture supports Claude Code builds that use scoped Keychain
credential storage (2.1+) and fails cleanly otherwise. It is POSIX-only in
this version — on Windows, always use `--from <dir>` instead.

`account add --from` copies the source login credentials; future refreshes of
that copy are not coordinated with the source CLI. Avoid independently using
both credential copies. Prefer interactive `account add` for isolated
gateway-owned credentials. To retain an active local Claude Code login, use
balanced routing's [local-login participation](#local-claude-code-login), which
leaves token refresh to the CLI.

The [dashboard](dashboard.md#claude-accounts) offers **Add account**,
**Sign in again**, **Set serving pin**, **Clear serving pin**, and **Remove**.
Its add flow runs the same
`claude auth login --claudeai` capture on the gateway host and takes the
pasted login code in the browser. Only one Claude login, from the CLI or the
dashboard, can run on a machine at a time.

Adding an account whose `(email, organization)` identity is already
registered replaces that account's stored credentials in place after a
confirmation prompt — the account keeps its id, so an `account use`
selection keeps working. This doubles as the re-auth flow for an account
whose refresh token has gone stale. Non-interactive runs (piped stdin with
`--from`) require `--yes` to confirm the replacement.

```sh
uv run claudex-gateway account add                 # interactive `claude` login
uv run claudex-gateway account add --from <dir>    # import an existing login
uv run claudex-gateway account list
uv run claudex-gateway account remove <id>         # prompts for confirmation
uv run claudex-gateway account remove <id> --yes   # skips the confirmation prompt
uv run claudex-gateway account use <id|email>      # serve passthrough with this account
uv run claudex-gateway account use off             # back to forwarding client credentials
uv run claudex-gateway account use                 # show the current selection
```

Captured credentials are stored under `~/.claudex/accounts/claude/`: one
directory per account (mode `0700`, POSIX file permissions) holding
`credentials.json` and `oauth-account.json` (mode `0600` each), plus a
shared `registry.json` (also mode `0600`) that lists accounts without
secrets. `claudex-gateway` never prints captured credential payloads.

Duplicate detection is keyed on normalized email + `organizationUuid`,
where missing matches missing: an account with no `organizationUuid`
collides only with another account that also has none, never with one
that has an `organizationUuid` set. `account remove <id>` deletes that
account's local copy only — it does not revoke the OAuth grant at
Anthropic, which stays valid until revoked from Anthropic's own account
settings. The CLI removes even the account selected by `account use`;
passthrough then fails until you select another account. The dashboard
refuses to remove the selected account (`409`) until the selection is
cleared.

## Serving with a registered account (`account use`)

`claudex-gateway account use <id|email>` sets the serving pin for Anthropic
passthrough traffic (`/v1/messages` for unmapped models and `count_tokens`).
With routing disabled, this registered account serves all passthrough traffic;
`fallback` and `balanced` use the pool policies described below. For a request
served by a registered account, the gateway consumes the
client's `Authorization`/`x-api-key` headers and serves upstream with that
account's OAuth token instead — Claude Code no longer needs a real
Anthropic login of its own (`ANTHROPIC_AUTH_TOKEN` set to the gateway local
token, or any placeholder when no local token is configured, is enough).
The gateway owns the token lifecycle: access tokens are refreshed ~5
minutes before expiry against the Claude Code token endpoint, rotated
refresh tokens are persisted atomically before use, and a 401 triggers one
refresh-and-retry before the error surfaces. `metadata.user_id`'s
`account_uuid` is rewritten to the serving account so a request never names
a different account than the one serving it.

The selection is the flat `claude_account.id` settings key (env override:
`CLAUDEX_CLAUDE_ACCOUNT_ID`). `account use` manages it through the same
channel decision table as `compact`: a confirmed running daemon is updated
live through the `/admin/providers/claude/pool/serving` endpoint (no
restart needed), a settings-file write is used only when no live daemon can
be confirmed, and an ambiguous probe refuses to apply changes.
`account use off` clears the pin via `DELETE` and returns to the default:
client credentials forwarded untouched. Balanced routing is the exception: it
serves from the account pool whether or not an account is selected (see
[Balanced routing](#balanced-routing-across-the-pool)). Clearing the pin does
not unregister the account or exclude it from balanced routing; registered
ready accounts can continue to serve requests.

```sh
curl http://127.0.0.1:8787/admin/providers/claude/pool/serving
curl -X PUT http://127.0.0.1:8787/admin/providers/claude/pool/serving \
  -H 'Content-Type: application/json' \
  -d '{"account_id": "<registered-account-id>"}'
curl -X DELETE http://127.0.0.1:8787/admin/providers/claude/pool/serving
```

The endpoint honors the same `CLAUDEX_LOCAL_TOKEN` and Host guard as the
other admin routes; a `PUT` requires a registered id (clearing is `DELETE`,
never a null `PUT`), persists the change to `settings.json`, and both
writes are refused with `409` when `CLAUDEX_CLAUDE_ACCOUNT_ID` is set in
the environment.

Caveats to accept consciously:

- Quota and billing land on the selected account, not the client's own
  subscription, and Anthropic's response rate-limit headers (and the CLI's
  usage display) reflect the serving account.
- If the selected account is removed or its refresh token becomes invalid,
  passthrough fails with a clear gateway 503 — there is never a silent
  fallback to client credentials. If Anthropic still rejects a freshly
  refreshed token, that request fails with `401` and later requests get the
  503. Re-add the account or run `account use off`. (With the `fallback`
  routing mode enabled — see the next section — the remaining ready accounts
  serve instead.)
- The [compaction reroute](compaction.md#compaction-reroute) uses only eligible
  credential headers sent by the client; registered-account OAuth credentials
  are not substituted. It records `skipped_no_credentials` only when its header
  filter leaves no credential—for example, a Bearer matching the configured
  gateway-local token with no nonblank `x-api-key`. A dummy Bearer when no local
  token is configured can instead trigger one direct Anthropic attempt; an
  upstream non-2xx response records `fallback_mapped`, not
  `skipped_no_credentials`.
- Subscription OAuth tokens are licensed for the holder's own Claude Code
  use; serving other clients with them is a gray zone.

## Ordered fallback across registered accounts

Multi-account routing is an explicit opt-in, off by default: with the
routing mode `disabled`, only the pinned serving account is used and a
`429` relays to the client verbatim. Selecting the `fallback` mode turns
the pin into the head of a fallback chain: every **ready** account is a
pool member, ordered serving-account-first and then by registration time.
Fallback needs a serving account from `account use`; without one,
passthrough keeps forwarding client credentials.
When the account being served with answers `429`, the gateway puts it on
an in-memory cooldown, transparently retries the same request with the
next account in the chain, and fails back automatically once the cooldown
expires — the client never has to handle the rate limit itself as long as
any account has quota left.

The mode is the `claude_account.routing` settings key — a policy document
such as `{"mode": "fallback"}` with `mode` set to `disabled`, `fallback`, or
`balanced` — managed at runtime through
`/admin/providers/claude/pool/routing` or the dashboard's routing selector.
The endpoint accepts only the `mode` key.

```sh
curl http://127.0.0.1:8787/admin/providers/claude/pool/routing
curl -X PUT http://127.0.0.1:8787/admin/providers/claude/pool/routing \
  -H 'Content-Type: application/json' \
  -d '{"mode": "fallback"}'
```

The env override `CLAUDEX_CLAUDE_ACCOUNT_ROUTING` holds the same document
JSON-encoded (empty string = disabled) and locks the endpoint with `409`
while set.

How long a rate-limited account sits out is taken from the best signal the
429 offers: a `Retry-After` header or a reset timestamp when present,
otherwise the account's cached usage window resets (as shown in the
dashboard), otherwise a 60-second default — in practice Anthropic's OAuth
quota rejections carry no machine-readable reset, so the cached usage data
is what turns a blind minute into an accurate multi-hour cooldown. Each
account's routing state is visible at
`GET /admin/providers/claude/pool/status` (`ready`, `cooldown` with a
`cooldown_until` epoch-ms deadline, or `unavailable`) and lives in daemon
memory only: a restart clears it, at worst costing one extra upstream
probe.

Boundaries to know:

- Failover happens only in `fallback` mode, on a `429` or an account-specific
  auth failure. A refresh that needs a new login, or a `401` that persists
  after a forced refresh, also durably marks the account `needs-reauth`,
  which excludes it until a re-login. Other upstream errors and network
  failures are relayed as before — retrying a different account would not
  help them.
- When every account is rate-limited, the client sees a real `429`: the last
  upstream rejection while probing, or a synthesized one with `Retry-After`
  once everything is already cooling (upstream is then not contacted at all).
- Failover only happens before any response byte is relayed; a stream that
  dies midway is reported in-band, as before.
- Rate limits are per account **per model tier** upstream, but the fallback
  pool's cooldown is per account: a Fable-scoped weekly limit cools the whole
  account even for requests other models could still serve. Balanced routing
  can limit such a cooldown to Fable models.
- The CLI's own usage display remains unreliable under pooling: its usage
  query authenticates with the placeholder token and fails, and response
  rate-limit headers reflect whichever account served. Use the dashboard's
  per-account usage view instead.

## Balanced routing across the pool

The `balanced` mode spreads Claude Code sessions across every ready
registered account instead of waiting for a `429`. It does not need a
serving account and never forwards client credentials, even after
`account use off`. If an account is selected, it gets the first session
while no usage readings or session assignments exist yet, and it wins ties.

Each session stays on one account. The session is identified by Claude
Code's session id, or by a hash of the first user message when the request
has none. A new session goes to an account picked at random, weighted
toward accounts with more remaining quota in their latest usage readings.
Usage is polled in the background, with at most one upstream usage call
every five minutes across the pool.

When the session's account answers `429`, is already cooling down, was
removed or needs a new login, or fails authentication, the gateway moves
the session to another eligible account. As in fallback mode, this happens
only before any response byte reaches the client. A Fable-model `429` cools
only the account's Fable models when fresh usage readings show the Fable
weekly window exhausted while the five-hour and seven-day windows still have
room; otherwise the whole account cools. When no account
is eligible, the client gets the upstream `429` or a synthesized `429` with
`Retry-After`. If no account is ready at all, the client gets `503`.
Token counting follows the session's account when one is already assigned,
but never moves a session or retries another account.

Enable balanced routing like the other modes:

```sh
curl -X PUT http://127.0.0.1:8787/admin/providers/claude/pool/routing \
  -H 'Content-Type: application/json' \
  -d '{"mode": "balanced"}'
```

The gateway prepares the pool before switching, and the previous mode keeps
serving until then. Enabling fails with `400` when a ready account's
captured profile has no valid account UUID; re-add that account with
`account add`.

Session assignments, cooldowns, and usage readings are stored in
`~/.claudex/claude-account-pool/claude-account-pool-runtime.sqlite3` and
survive a daemon restart. Switching to `disabled` or `fallback` discards the
session assignments. If a persisted `balanced` mode cannot be restored at
startup, the daemon still starts and logs the error, and passthrough returns
`503` (`balanced routing is not active`). Fix the cause and retry the
`PUT` above to activate the pool without restarting. If preparation still
fails, the endpoint returns `400` for an invalid account profile or `503`
for an unavailable runtime dependency, and balanced passthrough keeps
returning `503`. You can instead switch to `disabled` or `fallback` through
the same endpoint even while the balanced runtime is inactive.

While balanced routing is active, `GET /admin/providers/claude/pool/usage`
answers from cached readings only. Add `?refresh` to queue a rate-limited
poll. `GET /admin/providers/claude/pool/status` adds a pool-wide
`usage_freshness` value: `fresh`, `partial`, or `degraded`.

### Local Claude Code login

In balanced mode, the machine's own Claude Code login also joins the pool.
The gateway reads its current credential without modifying or refreshing it,
leaving the CLI as the sole token refresher and avoiding the single-use
refresh-token race that could log one process out. The login drops out while
its access token is expired, and a registered account with the same identity
takes precedence. It does not appear in `account list` or the routing
status.

To opt out, set `"include_local_login": false` in the policy document, for
example `{"mode": "balanced", "include_local_login": false}`, in
`settings.json` or `CLAUDEX_CLAUDE_ACCOUNT_ROUTING`, then restart the daemon.
The key defaults to `true` and affects only balanced routing. The routing
endpoint and dashboard selector accept only `mode` and preserve this key
when changing modes, including across disable/re-enable. Disabling routing
keeps a policy document with `"mode": "disabled"` when it carries this key,
so the opt-out also survives a restart while routing is disabled.
