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
settings.

## Serving with a registered account (`account use`)

`claudex-gateway account use <id|email>` selects one registered account to
serve all Anthropic passthrough traffic (`/v1/messages` for unmapped models
and `count_tokens`). With a selection active, the gateway consumes the
client's `Authorization`/`x-api-key` headers and serves upstream with the
selected account's OAuth token instead — Claude Code no longer needs a real
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
`account use off` clears the pin via `DELETE` and returns to today's
default: client credentials forwarded untouched.

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
  fallback to client credentials. Re-add the account or run
  `account use off`. (With the `fallback` routing mode enabled — see the
  next section — the remaining ready accounts serve instead.)
- The [compaction reroute](compaction.md#compaction-reroute) still uses the credentials
  the client itself sent: with a credential-less client it records
  `skipped_no_credentials` and falls back to the mapped model as usual.
- Subscription OAuth tokens are licensed for the holder's own Claude Code
  use; serving other clients with them is a gray zone.

## Ordered fallback across registered accounts

Multi-account routing is an explicit opt-in, off by default: with the
routing mode `disabled`, only the pinned serving account is used and a
`429` relays to the client verbatim. Selecting the `fallback` mode turns
the pin into the head of a fallback chain: every **ready** account is a
pool member, ordered serving-account-first and then by registration time.
When the account being served with answers `429`, the gateway puts it on
an in-memory cooldown, transparently retries the same request with the
next account in the chain, and fails back automatically once the cooldown
expires — the client never has to handle the rate limit itself as long as
any account has quota left.

The mode is the `claude_account.routing` settings key — a policy document
like `{"mode": "fallback"}` or `{"mode": "balanced"}` — managed at runtime
through `/admin/providers/claude/pool/routing`. Its optional
`include_local_login` key defaults to `true` and applies only to balanced
routing.

In balanced mode, the machine's own Claude Code login automatically joins
the pool. The gateway reads its current credential without modifying or
refreshing it, leaving the CLI as the sole token refresher and avoiding the
single-use refresh-token race that could log one process out. A matching
registered identity takes precedence; set `"include_local_login": false` in
the policy document to opt out.

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

- Only `429` triggers failover, and only in `fallback` mode. Auth failures
  durably mark the account `needs-reauth` (excluded from the chain until a
  re-login); other upstream errors and network failures are relayed as
  before — retrying a different account would not help them.
- When every account is rate-limited, the client sees a real `429`: the last
  upstream rejection while probing, or a synthesized one with `Retry-After`
  once everything is already cooling (upstream is then not contacted at all).
- Failover only happens before any response byte is relayed; a stream that
  dies midway is reported in-band, as before.
- Rate limits are per account **per model tier** upstream, but the pool's
  cooldown is per account: a Fable-scoped weekly limit cools the whole
  account even for requests other models could still serve. Per-model
  eligibility is a known follow-up.
- The CLI's own usage display remains unreliable under pooling: its usage
  query authenticates with the placeholder token and fails, and response
  rate-limit headers reflect whichever account served. Use the dashboard's
  per-account usage view instead.
