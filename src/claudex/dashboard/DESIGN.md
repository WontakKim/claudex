# Ask concurrency row

## Accepted specification

The MCP tab keeps gateway connection setup above the GPT Pro backend card.
Within that card, Ask concurrency stays between session controls and Diagnostics.
Use the shipped Operational TUI colors, typography, borders, and native controls.

- Left: uppercase **Ask concurrency** heading with the description
  **“Limits how many GPT Pro ask tabs can run at once.”** directly below it.
- Right: a 64px native dropdown with integers 1–10. No Apply button, visible
  unit, or permanent helper line. The default is 2, communicated through the
  dropdown title and accessible guidance, not assumed before loading settings.
- Associate the heading and description with the dropdown. Retain accessible
  range/default/rate-limit guidance and a polite status region for feedback.
- Keep copy and dropdown side by side at desktop and phone widths; allow the
  description to wrap on narrow screens rather than clipping it.
- Preserve the environment-lock band. Disabled controls must not imply an
  environment lock when the settings request merely failed.

Selecting a different value immediately saves and applies it through the existing
admin API. Disable editing until settings load, while saving, or when locked.
Adopt only valid server responses. Failed saves restore the last confirmed value;
conflicts lock immediately and require a valid settings refresh to recover.
Late responses from earlier reads must not overwrite a save or its lock state.

Local **FINAL** reference: `_archive/ask-row-oct09/_ask-row-oct09-final.html`.
The HTML probes are not committed. The local reference uses simulated data and
does not perform real saves; the specification above is the repository reference.
Production behavior is documented in [Dashboard](../../../docs/dashboard.md#gpt-pro-mcp).

## Decision Log

1. **2026-10-09 — Adopt the restrained A row.** The owner selected A over the
   stepper (B), exposed numeric choices (C), and edit-on-demand challenger (D),
   then requested a simpler presentation. The final composition retains the
   explanation beneath the heading on the left and places the dropdown on the
   right, following Diagnostics’ heading/description hierarchy. A temporary
   removal of the description and alternate one-/two-line arrangements were
   rejected; explanatory copy is essential, not decorative. “Parallel ask
   tabs.” was replaced because it did not explain the setting’s effect.
   The owner removed Apply, overriding the original explicit-apply constraint;
   the approved implementation therefore saves on selection, with pending
   disablement, response validation, rollback, and conflict recovery. “Modal”
   was clarified to mean the native dropdown, not a separate dialog. The range,
   default, environment precedence, and backend API remain unchanged. Open
   design decisions: none. No further Artifact uploads or updates are allowed.
