# GPT Pro

GPT Pro lets Claude Code ask ChatGPT Pro through the gateway's MCP endpoint at
`http://127.0.0.1:8787/mcp` by default. This capability moved from the
standalone Claude Code plugin into the gateway; it is not a model-mapping
provider.

The integration automates a saved ChatGPT web session. It returns background
job handles instead of holding an MCP call open while ChatGPT generates an
answer.

## Setup

Install the optional MCP and browser dependencies:

```sh
uv sync --extra gptpro
```

Sign in through the interactive browser and save the session:

```sh
uv run claudex-gateway gptpro login
```

Check the saved session without contacting ChatGPT:

```sh
uv run claudex-gateway gptpro status
```

The dashboard MCP tab shows the saved session status, starts and monitors
interactive ChatGPT sign-in, and runs the same doctor diagnostic. It also
provides the MCP endpoint and a copyable Claude Code connection command.

Run `uv run --extra gptpro claudex-gateway gptpro doctor` to diagnose the
saved session, Chrome profile and lock, and Playwright dependency.

Session state, the persistent Chrome profile, and its lock live under
`~/.claudex/gptpro/`. The session file is
`~/.claudex/gptpro/session.json`, and the browser profile is
`~/.claudex/gptpro/chrome-profile/`. Google Chrome or a Playwright Chromium
browser must be available.

The ask runtime lazily starts one warm, headless persistent browser context and
reuses it. Login uses the same profile in a visible browser. A profile lock
prevents login and an ask runtime, or two ask runtimes, from using that profile
at the same time. If login reports that another gptpro ask is using the browser
profile, stop the gateway process that owns the runtime before logging in again.

## MCP tool contract

The gateway exposes three tools. Each schema rejects keys other than those
listed below.

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `ask_gpt_pro` | `question` (required string), `thread` (optional string), `attachments` (optional array of strings) | Starts a background ask and returns immediately with `{"ask_id": ..., "thread_ref": ...}`. A fresh ask can initially have a null `thread_ref`. |
| `ask_gpt_pro_status` | `ask_id` (required string) | Returns `ask_id`, `state`, the latest nullable `status_message`, and the nullable `thread_ref`. Unknown or expired IDs are tool errors. |
| `ask_gpt_pro_result` | `ask_id` (required string) | After `succeeded`, returns `ask_id`, the Markdown `answer`, and `thread_ref`. After `failed`, returns an MCP tool error. Calling it while the job is `queued`, `running`, or `detached` is an error. |

The normal caller flow is:

1. Call `ask_gpt_pro` once and retain its `ask_id` and any `thread_ref`.
2. Poll `ask_gpt_pro_status` while the state is `queued`, `running`, or
   `detached`.
3. Call `ask_gpt_pro_result` only after the state is `succeeded` or `failed`.

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
bytes in total.

Questions larger than 35,000 UTF-8 bytes are automatically moved into a
temporary text attachment, so callers should send the complete question rather
than truncate it. The generated spill file consumes one attachment slot and
counts toward the total byte limit.

## Job lifecycle

| State | Meaning | Next states and recovery |
| --- | --- | --- |
| `queued` | The job is waiting for admission, normally behind an in-flight ask on the same conversation. | Becomes `running`, or `failed` with `expired` if same-conversation admission exceeds the 900-second queue TTL. |
| `running` | The job has been admitted and submission or answer generation is in progress. | Becomes `detached`, `succeeded`, or `failed`. |
| `detached` | The prompt was submitted, its user echo and conversation ID were secured, and ChatGPT continues generating while the gateway polls the conversation. | Remains recoverable through normal status polling, then becomes `succeeded` or `failed`. |
| `succeeded` | The answer is settled and available from `ask_gpt_pro_result`. | Terminal. The successful `thread_ref` becomes this MCP session's binding. |
| `failed` | The queue or provider execution ended with a classified failure. | Terminal. Fetch the result for the operational error and use any preserved conversation metadata for recovery. |

`status_message` is supplemental progress. In particular, `waiting for the
in-flight answer` identifies same-conversation queueing, while `detached; polling
for the answer` identifies server-side recovery. State remains authoritative.

`expired` specifically means the 900-second same-conversation queue wait ended
before admission; it does not mean the ChatGPT execution budget was consumed.
The job retains its `thread_ref` and any nonce marker that exists. Revisit the
conversation by passing the preserved `thread_ref` as `thread`, and use the
marker when available to locate the turn and attempt answer recovery.

Other actionable failures include `session_expired`, `challenge`,
`rate_limited_timeout`, `timeout`, and `echo_timeout`. Re-run login for an
expired session or browser challenge. For rate-limit and timeout failures,
check ChatGPT and the network, wait when appropriate, and retry deliberately.
Other executor failures are also surfaced through the result error.

Poll status every 30-60 seconds or longer rather than in a tight loop. Detached
answer recovery uses a separate server-side polling interval and does not
require frequent client polling.

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
applied to new ask admission so new submissions do not worsen provider
contention.

Queue and execution limits are separate. Waiting for the current owner of the
same conversation is bounded by the fixed 900-second queue TTL and does not
consume the ask's execution budget. On admission, the job receives the current
execution budget.

Before five successful duration samples exist, the execution budget is the
configured ceiling. After that, the gateway uses the p95 of up to 64 recent
successful durations plus 50 percent. The measured budget is clamped toward a
60-second floor but never above the configured ceiling. Failed asks do not
update these measurements.

## Operations

The gptpro scheduler reads these environment variables directly:

| Variable | Default | Behavior |
| --- | --- | --- |
| `GPTPRO_OVERALL_TIMEOUT_SECONDS` | `900` | Positive floating-point execution-budget ceiling in seconds. Missing, non-numeric, zero, and negative values use the default. |
| `GPTPRO_MAX_CONCURRENT_ASKS` | `2` | Integer ask-tab concurrency. Non-integer values use the default; values below 1 are clamped to 1. |

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
records, pending recovery work, and implicit session bindings. A saved
`thread_ref` remains a ChatGPT conversation UUID, so callers can still revisit
that conversation after restart by passing it explicitly as `thread`.
