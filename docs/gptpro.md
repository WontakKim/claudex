# GPT Pro

GPT Pro lets Claude Code ask ChatGPT Pro through the gateway's MCP endpoint at
`http://127.0.0.1:8787/mcp` by default. This capability moved from the
standalone Claude Code plugin into the gateway; it is not a model-mapping
provider.

The integration automates a saved ChatGPT web session. It returns background
job handles instead of holding an MCP call open while ChatGPT generates an
answer.

## Setup

The macOS release tarball includes the MCP and Playwright dependencies. From
the extracted release directory, run:

```sh
./bin/claudex-gateway gptpro login
./bin/claudex-gateway gptpro status
./bin/claudex-gateway gptpro doctor
```

In a source checkout, sync the project dependencies first:

```sh
uv sync
uv run claudex-gateway gptpro login
uv run claudex-gateway gptpro status
```

`login` opens a visible browser on the gateway host; complete ChatGPT sign-in
there within five minutes. It then saves the session and verifies it with a
headless ChatGPT check. `status` and `doctor` inspect only local state and do
not contact ChatGPT, so a valid `status` does not prove that ChatGPT still
accepts the session. `claudex-gateway gptpro ask "<question>"` runs one
foreground ask in a new conversation, without attachments or MCP job handles.

The dashboard MCP tab leads with gateway-wide Claude Code connection setup,
including the MCP endpoint and a copyable command. Its GPT Pro backend card
shows the saved session status, starts and monitors interactive ChatGPT sign-in,
and runs the same doctor diagnostic.

When `CLAUDEX_LOCAL_TOKEN` is set, `/mcp` requires it as an
`Authorization: Bearer` header, and like the admin routes it answers only
requests whose `Host` names the gateway. The dashboard's one-click registration
adds that header; its copyable command contains a `<CLAUDEX_LOCAL_TOKEN>`
placeholder to replace.

Run `uv run claudex-gateway gptpro doctor` to diagnose the
saved session, Chrome profile and lock, and Playwright dependency.

Session state, the persistent Chrome profile, and its lock live under
`~/.claudex/gptpro/`. The session file is
`~/.claudex/gptpro/session.json`, and the browser profile is
`~/.claudex/gptpro/chrome-profile/`. Sign-in tries system Chrome first, then the
matching Playwright Chromium. If neither is available, sign-in
downloads the matching Playwright Chromium automatically with the same Python
executable that runs the gateway, then retries once. Source installs therefore
use the project virtual environment, while release installs use the bundled
Python and do not depend on `uv` or a system Python. Playwright stores the
download in its normal per-user cache, honors `PLAYWRIGHT_BROWSERS_PATH`, and
does not write a browser
into the extracted release directory. Generic gateway startup, status, doctor,
and asks never download a browser. If a release installation reports that
Playwright itself is missing, reinstall the latest release tarball; in a source
checkout, rerun `uv sync`.

The ask runtime lazily starts one warm, headless persistent browser context and
reuses it. Login uses the same profile in a visible browser. A profile lock
prevents login and an ask runtime, or two ask runtimes, from using that profile
at the same time, and a CLI ask fails while the gateway's runtime holds the
profile. If login reports that another gptpro ask is using the browser
profile, stop the gateway process that owns the runtime before logging in again.
Starting sign-in from the dashboard instead releases the gateway's idle ask
runtime itself; the dashboard refuses to start sign-in while any GPT Pro ask is
queued, running, or detached.

Cloudflare clearance cookies are bound to the browser that obtained them, so
clearance from the visible login browser does not carry over to the headless
runtime. Each time the ask runtime launches the profile, it clears the
profile's Cloudflare cookies and presents a user agent without the headless
marker, so the headless browser can obtain its own clearance. While waiting for
the ChatGPT composer, an ask tolerates Cloudflare challenge markup for up to 10
seconds, within its deadline, before failing with `challenge`. A challenge or
block reported by a ChatGPT backend response fails the ask immediately.

## Release build hosts

Run `./scripts/build-darwin-asset.sh` to assemble
`build/claudex-gateway-<version>-darwin-arm64.tar.gz`. The build supports macOS
arm64 and cross-build hosts such as Linux arm64. It requires `uv`, `curl`, and
the standard shell/archive tools; cross-builds also use a host CPython matching
the bundled series (currently 3.12), located or downloaded by `uv`. The build
host does not change the target: the extracted release still runs on macOS
arm64, with the same `bin/` and `python/` layout and bundled dependencies.

All builds verify the checksum-pinned runtime and require Mach-O arm64 support
in bundled Python, every `.so`/`.dylib`, and Playwright's `driver/node`.
Universal binaries are accepted when they contain an arm64 slice.

- **macOS arm64:** requires `lipo` from the Xcode command line tools and runs the
  bundled launcher usage check, MCP server construction, and Playwright driver
  startup. These checks do not launch or download a browser.
- **Cross-build hosts:** validate Mach-O headers without macOS tools, then use
  host CPython with the bundled site-packages to check the platform-independent
  CLI usage error. The build explicitly logs that bundled Darwin runtime
  verification was skipped. This does not verify native-library loading, the
  bundled launcher, or MCP/Playwright execution on macOS.

Smoke tests use an isolated home and do not write Python bytecode into the
bundle. A Linux-assembled asset can receive full runtime verification on an
arm64 Mac; successful cross-assembly alone does not provide that guarantee.

Forks may select their own CI runner. Before building, run `uv sync --frozen`
and `uv run --frozen pytest`; MCP and Playwright are default dependencies, so
workflows must not request the removed `gptpro` extra.

## MCP tool contract

The gateway exposes four tools. Each schema rejects keys other than those
listed below.

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `ask_gpt_pro` | `question` (required string), `thread` (optional string), `attachments` (optional array of strings) | Starts a background ask and returns immediately with `{"ask_id": ..., "thread_ref": ...}`. A fresh ask can initially have a null `thread_ref`. |
| `ask_gpt_pro_status` | `ask_id` (required string) | Returns `ask_id`, `state`, nullable `status_message`, `thread_ref`, `nonce_marker`, `failure`, and `error_message`, plus `evidence`, nullable `source_ask_id`, and `recovery_guidance`. Unknown or expired IDs are tool errors. |
| `ask_gpt_pro_result` | `ask_id` (required string) | After `succeeded`, returns `ask_id`, the Markdown `answer`, `thread_ref`, `nonce_marker`, `evidence`, nullable `source_ask_id`, `files`, and `files_complete` (see [Generated files](#generated-files)). After `failed`, returns an MCP tool error with a readable explanation and structured diagnostics. Calling it while `queued`, `running`, or `detached` is an error. |
| `recover_gpt_pro` | Either `ask_id` (a retained failed job) or `thread_ref` (conversation UUID) and `nonce_marker` | Starts a separate, bounded, read-only recovery job and returns its `ask_id`, nullable `source_ask_id`, and `thread_ref`. Poll its status and result as usual. It never sends a prompt or attaches files. |

The normal caller flow is:

1. Call `ask_gpt_pro` once and retain its `ask_id` and any `thread_ref`.
2. Poll `ask_gpt_pro_status` while the state is `queued`, `running`, or
   `detached`.
3. Call `ask_gpt_pro_result` only after the state is `succeeded` or `failed`.
4. If a failed ask might have submitted, retain its `ask_id`, `thread_ref`, and
   `nonce_marker`. Use `recover_gpt_pro` to look for that turn without sending a
   second prompt. A recovery job has its own `ask_id` and does not rewrite the
   failed source job.

Send a self-contained `question` with the code, logs, and context needed for the
answer. If an answer begins with `GPTPRO_CONTEXT_REQUEST_V1`, gather the
requested material and call `ask_gpt_pro` again; omitting `thread` continues the
conversation that just succeeded.

### Thread selection

`thread` has three modes:

- Omit it to continue the conversation from this MCP session's most recent
  successful completion. If the session has no binding, the ask starts a new
  conversation.
- Pass `"new"` to force a fresh conversation.
- Pass a conversation UUID from an earlier `thread_ref` to revisit that
  conversation explicitly.

A `thread_ref` can become visible while a job is in progress, but the MCP
session binding changes only when that job succeeds. Failed jobs do not replace
the session's last successful binding. Separate MCP sessions share a
conversation only when callers explicitly pass the same UUID.

### Attachments and large questions

`attachments` contains file paths on the gateway host. Files must be UTF-8
plain text without NUL bytes. An ask accepts at most 10 files and 1,200,000
bytes in total. A completed file-create response does not confirm composer
readiness: ChatGPT may rename a file during indexing, so a renamed composer
chip counts only when its displayed name comes from the trusted processing
stream for that file's create-receipt ID and processing completes. Any missing,
failed, or unsettled attachment blocks submission, including immediately before
Send; failure details show requested and resolved names, available file IDs, and
observed composer states.

Questions larger than 35,000 UTF-8 bytes are automatically moved into a
temporary text attachment, so callers should send the complete question rather
than truncate it. The generated spill file consumes one attachment slot and
counts toward the total byte limit.

### Generated files

When ChatGPT's final answer links files it generated as
`sandbox:/mnt/data/...` Markdown links, the gateway downloads them after the
answer is final and saves them on the gateway host. Links in commentary,
tool calls, or an unfinished turn are never downloaded, and neither are
Markdown examples inside code blocks or code spans. This applies to
direct, detached, and recovered answers, and to `claudex-gateway gptpro ask`.

`files` has one entry per distinct linked sandbox path, in link order:

| Field | Meaning |
| --- | --- |
| `name` | Local file name, derived from the link's last path segment. |
| `sandbox_path` | The decoded `/mnt/data/...` path from the link. |
| `message_id` | The ChatGPT message that linked the file. |
| `status` | `saved` or `failed`. |
| `path` | Absolute path of the saved file on the gateway host; null when failed. |
| `size_bytes`, `mime_type`, `sha256` | Saved file size, type, and SHA-256 digest; null when failed. |
| `error` | Why a failed file was not saved; null when saved. |

`files_complete` is `true` when every linked file was saved, including when
the answer links no files (`files` is then empty). It is `false` when any
file failed; the `answer` text is still complete, so read the per-file
`error` values.

`path` is a filesystem path on the gateway host, not a URL, and the
`sandbox:` link in the answer is not downloadable by the caller. Read saved
files on the gateway host. A client on another machine needs a shared
filesystem or other out-of-band access to that host; the gateway does not
serve these files over HTTP. The CLI prints the answer to stdout unchanged
and reports saved paths and failures on stderr.

Each answer's files are saved in a new private directory (mode 0700) under
`~/.claudex/gptpro/outputs/`. Files are written completely before they
appear under their final name, never overwrite an existing file, and are
not extracted or executed; archives stay as downloaded. The gateway never
deletes saved files, so remove old output directories manually.

Downloads use the authenticated ChatGPT browser page. The access token is
sent only to the `chatgpt.com` download lookup; the returned download URL
must be a `https://chatgpt.com/backend-api/estuary/content` URL and is
fetched without the token, and redirects are refused. Errors never include
tokens or signed download URLs. Fixed limits bound each answer: at most 20
files, 20 MiB per file, 50 MiB in total, 60 seconds per request, and 300
seconds for all of the answer's files. A file beyond a limit is reported as
failed rather than silently omitted. File saving must also finish within the
job's execution deadline; it does not add time to that deadline. Expiry of the
execution deadline fails the job even if the answer text was already observed.

## Job lifecycle

| State | Meaning | Next states and recovery |
| --- | --- | --- |
| `queued` | The job is waiting for admission, normally behind an in-flight ask on the same conversation. | Becomes `running` (a recovery job becomes `detached`), or `failed` with `expired` if same-conversation admission exceeds the 900-second queue TTL. |
| `running` | The job is admitted; navigation, submission, or answer observation may be in progress. This state alone does not prove submission. | Becomes `detached` during read-only polling, or `succeeded` or `failed`. |
| `detached` | The gateway polls the server for an existing answer; ChatGPT may still be generating, or extraction may have failed. | Remains observable through normal status polling, then becomes `succeeded` or `failed`. |
| `succeeded` | The answer is settled and available from `ask_gpt_pro_result`. | Terminal. The successful `thread_ref` becomes this MCP session's binding. |
| `failed` | The queue or provider execution ended with a classified failure. | Terminal. Fetch the result for the operational error and use any preserved conversation metadata for recovery. |

`status_message` is supplemental progress. In particular, `waiting for the
in-flight answer` identifies same-conversation queueing, while `detached; polling
for the answer` identifies server-side recovery. State remains authoritative.

A post-click `no_raw_turn`, `echo_timeout`, `timeout`, `navigation_failed`,
`submit_failed`, or `error` failure can move a job to `detached` for up to
`GPTPRO_RAW_TURN_RECOVERY_SECONDS` (5400 seconds by default), but only within
the original ask's remaining 90-minute execution window, provided its
conversation ID and nonce are known. Pre-click failures and `session_expired`,
`challenge`, and `rate_limited_timeout` failures do not trigger this polling.
A non-positive window disables it. Only a finished, nonempty, nonce-correlated
raw assistant turn counts as a recovered answer. The job
retains conversation ownership while polling, so follow-up asks for the same
conversation remain queued until it settles. `recover_gpt_pro` uses the same
ownership rule to inspect an existing turn after a failure, with a new execution
window of at most 90 minutes starting after its own admission. It creates a new
recovery job and does not change the source failed job. A shorter configured
polling window retains the bounded file-saving allowance for an answer found
in time, but neither automatic nor manual recovery can save files beyond its
90-minute execution deadline.

`expired` specifically means the 900-second same-conversation queue wait ended
before this job ran. A queued ordinary ask did not submit; a queued recovery
job may still refer to an earlier server turn and retains its thread and nonce
for another read-only lookup. An execution timeout or lost echo, by contrast,
does not establish that submission failed.

### Evidence and failed-turn recovery

Status and result snapshots include independent observations in `evidence`:
`submission` is `not_attempted`, `uncertain` (click attempted without
confirmation), or `confirmed` (a nonce-matched server user echo or recovered
server answer); neither an upload response, click, nor outgoing request alone
confirms server acceptance. `upload_receipts` counts
completed file-create responses, while `ready_attachments` counts attachments
seen ready in the composer; either may be null when not observed.
`conversation_id` is the known thread ID, either explicitly selected or
observed from a correlated ChatGPT event. `generation_observed` and
`answer_seen` do not imply `raw_extracted`: only a completed, correlated raw
answer establishes that. `failure_stage` identifies where execution stopped; `recovery_failure` and
`recovery_detail` separately report why read-only recovery failed without
erasing the original failure. `recovery` is `not_attempted`, `polling`,
`unavailable`, `exhausted`, or `recovered`. Null and false fields mean the
observation was not made, not that
ChatGPT definitely did nothing.

If a failed ask retains a `thread_ref` and `nonce_marker`, call
`recover_gpt_pro` with its `ask_id`. After a gateway restart, use the saved
`thread_ref` and `nonce_marker` instead. Poll the new recovery job; a failure
to recover means no matching finished answer was obtained within the window,
not proof that none exists. If either identifier is missing, this read-only
lookup cannot identify the turn: inspect the ChatGPT conversation before
considering any new submission. Re-run login for `session_expired` or
`challenge`; for `rate_limited_timeout`, wait, then recover if the ask may have
submitted. Do not blindly resubmit a prompt whose submission outcome is
uncertain.

Poll status every 30-60 seconds or longer rather than in a tight loop. Detached
answer recovery uses a separate server-side polling interval and does not
require frequent client polling.

### Composer and send locator recovery

The gateway locates the composer, send button, and stop button by structure
because labels follow the UI language. Only the composer and send button can
be rediscovered. A locator fault persisting for about four seconds while a
contract-satisfying control is present triggers one rediscovery per control
per ask. Candidates are collected by structure only: no conversation text or
composer contents are included.

The gateway asks `claude-sonnet-5-5` through its own `/v1/messages` endpoint.
This follows `model_map` like any client request: a `sonnet` mapping routes the
request to the mapped provider; otherwise it passes through to Anthropic.
The structure-only candidate description, including tags, roles, UI labels,
data attribute names (not values), and positions, is sent to whichever provider handles that
request. The gateway must be running and reachable at its configured host and
port. If it is unreachable, for example during a CLI `claudex-gateway gptpro ask`
while the gateway is stopped, rediscovery is unavailable and a locator fault
ends the ask with `locator_unresolved`.

The selected candidate's structural selector is compiled locally and checked
again against the control contract. It is used first and saved in
`~/.claudex/gptpro/locators.json`, keyed by UI language, only after that use
succeeds: retained prompt text proves the composer; a nonce-matched user echo
proves the send button. While a rediscovered locator awaits its first successful
use, other asks reuse it when it matches their page; otherwise they end with
`locator_unresolved` without another rediscovery. A rediscovery that cannot
finish within the control's wait budget ends the ask with `locator_unresolved`
and does not count as a failed rediscovery. `locator_unresolved` means nothing
was sent: retrying cannot duplicate that prompt, but may fail again.

After a failed rediscovery, a 60-second backoff prevents another rediscovery.
Three consecutive failures suspend rediscovery for one hour. Asks do not wait
for either window to expire: if the recorded and built-in locators still fail
the contract, the ask ends immediately with `locator_unresolved` without
calling the LLM; if either locator matches again, the ask proceeds normally.
After the hour, one ask may rediscover again; a failure restarts the one-hour
suspension. An unreachable gateway or rejected Messages request does not count
as a failed rediscovery. Deleting `locators.json` returns to the built-in
locators and clears the suspension.

## Scheduling behavior

Asks that target the same conversation are serialized with one conversation
owner. A job with an explicit `thread_ref` waits for the current owner. A fresh
job begins owning its conversation as soon as the gateway discovers and
latches the conversation ID, so an explicit follow-up cannot submit before the
fresh turn finishes. Other conversations remain eligible to run in parallel.

Browser submission concurrency is bounded by a tab semaphore. The default is
two active ask tabs. When another submitter is waiting for a tab, an in-flight
ask may detach only after both its submitted user echo and conversation ID are
known. Detaching closes that ask tab, releases admission capacity, and moves
answer recovery to one resident polling tab.

The detached poller checks every 45 seconds. An HTTP 429 doubles the interval,
up to 300 seconds. A successful fetch restores the 45-second interval, and an
idle poller also resets it. While a longer backoff is active, the same delay is
added before each newly admitted ask submits so new submissions do not worsen
provider contention.

Queue and execution limits are separate. Waiting for the current owner of the
same conversation is bounded by the fixed 900-second queue TTL and does not
consume the ask's execution window. Browser-capacity queueing is also excluded.
After browser admission, one monotonic 90-minute maximum covers submission
jitter, page setup, the initial browser interaction, detached polling,
automatic recovery, and generated-file saving. Polling and recovery do not
restart this deadline. A ready result returns immediately; the maximum is not
a mandatory wait.

The watchdog controls a separate initial browser-stage budget, not the total
90-minute window. Before five successful duration samples exist, that initial
budget is the configured ceiling. After that, the gateway uses the p95 of up to
64 recent successful durations plus 50 percent. The measured budget is clamped to the
configured minimum and ceiling; if the minimum exceeds the ceiling, the ceiling
wins. Only successful ask durations update these measurements; failed asks do
not.

## Operations

Ask-tab concurrency is the `gptpro.max_concurrent_asks` setting (integer 1–10,
default 2), read from `settings.json` or `GPTPRO_MAX_CONCURRENT_ASKS`. See
[Configuration](configuration.md) for source precedence and startup validation.

The gptpro scheduler reads these environment variables directly:

| Variable | Default | Behavior |
| --- | --- | --- |
| `GPTPRO_OVERALL_TIMEOUT_SECONDS` | `900` | Positive floating-point initial browser-stage budget ceiling in seconds. It cannot extend the fixed 90-minute total execution limit. Missing, non-numeric, zero, and negative values use the default. This is only a ceiling: increasing it does not raise a lower measured budget (`p95 × 1.5`); use `GPTPRO_MIN_EXECUTION_BUDGET_SECONDS` to raise that floor. Only successful ask durations affect the measurement; failures do not. |
| `GPTPRO_MIN_EXECUTION_BUDGET_SECONDS` | `60` | Positive floating-point floor for the execution budget reduced by watchdog measurements. Missing, non-numeric, zero, and negative values use the default. Values above `GPTPRO_OVERALL_TIMEOUT_SECONDS` are clamped to that ceiling. |
| `GPTPRO_RAW_TURN_RECOVERY_SECONDS` | `5400` | Polling window in seconds, for eligible uncertain or incomplete turns and explicit read-only recovery. Automatic recovery and file saving are capped by the original execution deadline. A shorter polling window retains the existing bounded file-saving allowance within the total limit. Positive values are capped at 5400; zero or negative values disable recovery. Missing, non-numeric, and nonfinite values use the default. |

Set overrides in the environment that starts the gateway. A background daemon
inherits that launch environment, so stop and start it to apply changes. If the
new start omits an override, the new daemon returns to the default; these
variables are not persisted in `~/.claudex/settings.json`.

For conservative operation, keep sustained use at about eight asks per 15
minutes or less. This is an operating recommendation, not a gateway-enforced
quota. Increasing tab concurrency does not remove ChatGPT-side rate limits.

Job records and MCP-session thread bindings are in memory. Terminal job records
are retained for about 24 hours, and successful session bindings expire after
23 hours while the process remains alive. A gateway restart loses ask IDs, job
records, pending recovery work, and implicit session bindings. Save both
`thread_ref` and `nonce_marker` before restarting if an uncertain turn may need
read-only recovery afterward. A saved `thread_ref` alone can select the
conversation as `thread`, but cannot identify which turn to recover.
