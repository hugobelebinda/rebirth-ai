# Phase 18 — MLOps 4: monitoring + data flywheel

> Status: **PLANNED**. Runs once the tool migrates in-language and cross-language (Phases 16–17).
> Previous: [PHASE_17_CROSS_LANGUAGE.md](PHASE_17_CROSS_LANGUAGE.md) · Next: [PHASE_19_LAUNCH.md](PHASE_19_LAUNCH.md)
> Estimated effort: ~2–3 days.

## Goal

See how the tool behaves on real use, catch regressions and drift early, and turn failures into
better training data — the tool improves from its own use. Everything stays local (no telemetry
leaves the machine).

## Scope

| Piece | What |
|-------|------|
| Run telemetry | Every `rebirth migrate` already writes `data/runs/<id>/summary.json` (redacted). Add: model tag served, language/direction, iterations, error classes per iteration, wall time, tokens, sandbox result |
| Dashboard | `rebirth dashboard` (static HTML or Streamlit, local): success rate, mean iterations and top error classes per language and per direction; trends per model version |
| Regression alerts | When a new champion ships (Phase 16), compare live success rate vs the previous version over the next N runs; warn on a drop |
| Drift detection | Compare live inputs vs the training corpus (language/framework version mix, file size, token length, unseen imports); flag when real code drifts away from what the adapters saw |
| Data flywheel | Failed or many-iteration migrations → review queue (`rebirth review`): the user accepts/fixes the final diff → appended to a curated `data/datasets/flywheel/<lang>.jsonl` (license-clean, own code only) → included in the next training round via the Phase 14 pipeline |
| Eval refresh | Frozen eval sets stay frozen; a second, rolling "recent real-world" eval set is built from reviewed flywheel rows |

## Exit criteria

- [ ] Dashboard shows per-language/direction health from real runs
- [ ] Drift + regression warnings implemented and tested on synthetic data
- [ ] Flywheel: review → curated rows → included in a retrain via `dvc repro`
- [ ] Privacy: nothing leaves the machine; redaction verified
