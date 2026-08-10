# HANDOFF — claudex-gateway balanced mode: PIPELINE COMPLETE (complete_with_warnings)

## Outcome (2026-08-10)

The /gpt-auto pipeline (spec → exec → review with ChatGPT Pro hard gates at
every seam) for the **`balanced` Claude account-pool routing mode** is
**finished**. Workflow `207f814` terminal state:
`complete_with_warnings/complete/complete_with_warnings`,
`review.final_verdict = "complete_with_warnings"`. The final Pro gate
endorsed the verdict verbatim ("NO CHANGES ARE NEEDED").

- main at **`ce01958`** (base `10a32fd`, 42 commits, +14,428/−335, 25 files).
  **NOT pushed** (owner rule). Live 8787 daemon untouched (still old v0.2.0).
- Suite: **1255 passed / 13 designed skips** — and passes WITH the live
  daemon holding `balanced-router.lock` (that was gap G-6).
- Design authority was `.docs/research/balanced-mode-design-v2.md`; all Pro
  gates ran on the same ChatGPT design thread.

## What was built

- `claude_balanced_router.py` (2,762 lines): session keys (domain-separated
  HMAC), weighted-HRW picker with emergency formula, pin map (TTL/LRU/
  reservation-protected), migration reservation/waiter/token + commit-at-2xx-
  headers machinery, `ClaudeBalancedRuntime` lifecycle (prepare_and_publish
  with required `entry` discriminator; cancellation-safe `exit_mode` with
  persist-as-commit-point + shielded finalization; `shutdown_preserving_epoch`),
  usage poll coordinator + runtime-owned background driver.
- `claude_pool_runtime_state.py` (1,108 lines): WAL SQLite store,
  synchronous=FULL, serialized writer with causal high-priority ordering,
  incarnation+fingerprint restore validation, corruption quarantine,
  newer-schema refusal, epoch rotation.
- `claude_account_profile.py`, `claude_unified_headers.py` (recognized-header
  table ships EMPTY — inert until a successful live capture),
  `scripts/capture_unified_headers.py` (probe).
- `server.py`: process-lifetime pool lease (all modes), balanced dispatch
  chain + transition-await, transactional PUT enable/exit, usage_freshness
  (session+weekly binding pair required for fresh), count_tokens balanced
  path, unified-header capture plumbing.
- `claude_accounts.py` registry migration + reauth incarnation transitions;
  config.py un-reserved "balanced"; dashboard Balanced option +
  fresh/partial/degraded pill; account_usage_cache per-window metadata.

## Review history (for the record)

- Loop 1: 5 gaps (G-1..G-5). Corrective T-18 (G-1 driver), T-19 (G-2
  freshness), T-20 (G-3 exit order).
- The round-14 Pro gate found 2 blocking defects in T-20 → **G-7** (exit_mode
  cancellation safety; epoch resurrection after degraded exit) → T-22.
- **G-6** discovered during verification: T-9's pool lease broke ~135
  naive-lifespan tests whenever a live daemon held the lock → T-21 (test-only
  HOME isolation).
- Loop 2: researcher + 13/13 deterministic validator checks + analyst — all
  five criticals CLOSED with line-level evidence; **G-4/G-5 remain open,
  normal** (the "warnings").

## Open warnings (G-4, G-5) — future work if the owner wants

- **G-4** (`server.py:1513-1516`): balanced count_tokens no-pin branch routes
  by the derivable session-key digest instead of a fresh stateless digest.
  Behavioral deviation only; creates no state.
- **G-5** (`server.py:4164-4165`): lifespan finally runs
  `await shutdown_preserving_epoch()` then `lease.release()` without an inner
  try/finally; an exception in the former skips explicit release (OS releases
  the flock at process exit anyway).

## Deferred (explicitly inventoried, needs a 2nd live account)

- 10 design-§8 live gates, each a named skip stub in
  `tests/test_balanced_deferred_gates.py` {8.1, 8.2, 8.3, 8.4, 8.5, 8.12,
  8.14, 8.15, 8.16, 8.19} with a meta-test pinning the exact set.
- Unified-header ingestion inert until a successful capture: both live probe
  runs hit account-level 429s with NO anthropic-ratelimit-* headers. The
  single account's Fable weekly resets 2026-08-11T21:00Z; 5h window was also
  429ing. Probe: `scripts/capture_unified_headers.py` (default two-sonnet,
  `--include-fable` flag).

## Key references

- State: `.workflow/207f814/state.json` (all 22 tasks done, 16 rounds,
  2 review loops, final verdict recorded)
- Gate verdicts: `.workflow/207f814/reports/pro-*.md` (spec, rounds 1-16,
  final; round 14's two blocking findings + round 16 re-gate)
- Loop-2 evidence: `reports/review-{loop,researcher,validator}-2.json`
- Design: `.docs/research/balanced-mode-design-v2.md` (+ lineage, adjudications)

## Owner rules still in force

Never `git push` without an explicit ask. Never touch the running 8787
daemon/port unless asked. Never print credential values (sha256 fingerprints +
metadata only). `.docs/` and `.workflow/` are git-excluded. zxcv/GHE(사내 tap)
접근 금지 without a new instruction. 시안/화면 전달 시 링크 주소 명시.
