# Dashboard

The gateway serves a runtime dashboard at `GET /` for editing the model map,
toggling Codex Fast mode, checking provider health, and testing model
connections before wiring them.

Opening `http://127.0.0.1:8787/` in a browser serves a dashboard on top of
the same admin API: the Settings tab holds gateway settings behind a
category rail (the General category contains the
[compaction reroute](compaction.md#compaction-reroute) target, Codex Fast mode, and Claude
account routing controls; the Claude accounts category contains
registered-account management), the Status tab shows each
provider's login state and subscription usage — the Kimi and Grok cards (and
their Router add-node buttons) appear only when a local login is detected or
the model map already routes to them; hiding is cosmetic and never affects
routing or `settings.json` — the Log tab tails the gateway's recent
log lines (`GET /admin/logs`) and holds the runtime log-level control
(`PUT /admin/settings/log-level`, applied immediately and persisted), and the Router
tab is a canvas editor — drag a port to wire a model, Apply to `PUT` the
map, and use the connection test box (`POST /admin/test`) to send one
minimal request through the gateway before wiring a new model id. The board
turns view-only when `CLAUDEX_MODEL_MAP` overrides the map and renders mapped
or staged targets as provider-prefixed nodes for the built-in Codex, Kimi, and
Grok providers plus configured custom providers. Add-node suggestions are
backed by live catalogs loaded from `GET /admin/providers/codex/models`,
`GET /admin/providers/kimi/models`, and `GET /admin/providers/grok/models`;
each configured custom provider loads
`GET /admin/providers/custom/{name}/models`. Catalog failures only remove
suggestions, and manually entered model IDs remain valid.

When `CLAUDEX_LOCAL_TOKEN` is set, the dashboard prompts for the token once
per page load and keeps it in memory for the lifetime of that page only — it
is attached as a bearer header to admin requests and never written to the
URL, browser storage, or logs. A wrong token triggers exactly one re-prompt.
