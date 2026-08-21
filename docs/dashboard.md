# Dashboard

The gateway serves a runtime dashboard at `GET /` for editing the model map,
toggling Codex Fast mode, checking provider readiness, and testing model
connections before wiring them.

Opening `http://127.0.0.1:8787/` uses the same guarded admin API as the CLI:

- **Settings** contains the [compaction reroute](compaction.md#compaction-reroute),
  Codex Fast mode, Claude account routing, and registered-account management.
- **Status** shows built-in login and subscription usage plus custom-provider
  readiness. Kimi and Grok cards remain cosmetically hidden until a login or
  mapped route makes them relevant; this never affects routing or settings.
  Custom providers do not receive usage or billing cards.
- **Log** reads `GET /admin/logs` and changes the persisted runtime log level
  through `PUT /admin/settings/log-level`.
- **Router** edits the provider-prefixed model map on a canvas and includes the
  existing manual model input and `POST /admin/test` connection test.

The Router becomes view-only when `CLAUDEX_MODEL_MAP` overrides the persisted
map. Otherwise, Apply sends the complete draft to
`PUT /admin/settings/mapping`.

## Custom-provider metadata and status

`GET /admin/settings/mapping` supplies custom-provider metadata without API
keys. The dashboard uses metadata rather than provider names to choose behavior:

| `wire_kind` | Dashboard label |
| --- | --- |
| `responses` | Responses API |
| `anthropic_messages` | Anthropic Messages |

`catalog_available` controls catalog loading. When it is `true`, the dashboard
may call `GET /admin/providers/custom/{name}/models` for autocomplete. When it
is `false`, the dashboard never calls that endpoint. A missing catalog is a
capability boundary, not a connection failure.

The custom-provider status terms are intentionally different:

- **Connected** means the Responses-family remote catalog verification used by
  health succeeded.
- **Configured** means the provider definition and route binding are ready, but
  no catalog or remote connection was inferred. Static Anthropic-compatible
  providers use this state because their health check performs no remote I/O.
- **Error** means the applicable local binding or remote catalog check failed.
  An unused Responses provider retains the existing neutral state when its
  optional catalog check fails.

Neither configured nor general gateway health proves remote model entitlement
or full Messages compatibility. Enter a complete `provider:model` target in the
connection test, or call `POST /admin/test`, to issue one minimal remote request.
For an Anthropic-compatible provider this is the explicit remote Messages
verification path.

## Model catalogs and manual entry

Built-in add-node suggestions use:

- `GET /admin/providers/codex/models`
- `GET /admin/providers/kimi/models`
- `GET /admin/providers/grok/models`

Catalog-capable custom providers use
`GET /admin/providers/custom/{name}/models`. Catalog failures only remove
autocomplete suggestions. The existing manual model field remains authoritative:
a typed model ID can be staged and mapped even when it did not come from a
catalog. Catalog-less Anthropic-compatible providers therefore require manual
model IDs; the dashboard does not add a separate provider-specific model UI.

Custom provider names can be arbitrary valid configured names, including names
that sound like another API family. Names never determine wire labels, catalog
behavior, or status semantics.

## Local dashboard authentication

When `CLAUDEX_LOCAL_TOKEN` is set, the dashboard prompts for it once per page
load and retains it in an in-memory closure for that page only. It is attached
as a bearer header to admin requests and is never written to the DOM, a URL,
`localStorage`, `sessionStorage`, console output, or gateway logs. A wrong token
triggers exactly one re-prompt.
