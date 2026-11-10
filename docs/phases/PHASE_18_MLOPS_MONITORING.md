# Phase 18 — MLOps 4: monitoring + data flywheel

> Status: **IN PROGRESS — tooling done 2026-11-10** (telemetry, dashboard, regression + drift checks, review queue → flywheel rows → DVC mix stage; 18 tests on synthetic data). Open, **deferred to after the v1.0 launch** (2026-11-10): the first retrain on flywheel rows; it needs real use (15 dev runs, 0 reviewed rows so far).
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

## Built (2026-11-10)

Code: `src/rebirth_ai/monitor/` (+ `monitor_cli.py`), tests `tests/test_monitor.py` (18, synthetic data;
every test runs with sockets blocked — any network attempt fails it).

| Piece | Command / file | What it does |
|-------|----------------|--------------|
| Run telemetry | `telemetry` block in every `migrate` / `migrate --to` / `migrate-repo` run log | v1 record: task (`modernize` / `translate`), language / direction, backend, **model served**, status, iterations, generations, **error class per attempt**, test verdicts, verify mode, **wall time**, input profile (chars, lines, ~tokens, imports), output size. Old logs without it are derived from their payload + trace (no wall time / input profile) |
| Run artifacts | `data/runs/<id>/artifacts/source.* / draft.*` | Redacted copies of the input file and final draft (`redact_text`, same patterns as the logs) — what review needs. `data/runs/` stays git-ignored |
| Dashboard | `rebirth dashboard [--out data/reports/dashboard.html]` | Terminal table + one self-contained HTML file (no external assets or scripts; light / dark): success per task, per language / direction and per model version, 90 % Wilson CI, mean iterations, median time, top error classes, current warnings |
| Regression alerts | `rebirth monitor check` (exit 1 on a warning), also shown in the dashboard | Per scope (task overall + each language / direction): newest model vs the previous one, once both have `--min-runs` (10). **warn** = success down ≥ `--margin` (10) points with non-overlapping 90 % CIs, or ≥ 2 × margin; **info** = down but within noise, too few runs yet, or mean iterations +1 |
| Drift | `rebirth monitor reference` → `data/monitor/reference/<lang>.json`; checked by `monitor check` | Reference = training inputs (corpus `legacy` sides + stage-3 pool sources): length quantiles / bucket shares + import frequencies. Live (last 50 runs per language, ≥ 20 needed): length **PSI** ≥ 0.25 warn (0.10 info), ≥ 10 % of inputs longer than the training p99 → warn, ≥ 30 % of import mentions unseen in training → warn (top unseen listed) |
| Review queue | `rebirth review list / show / accept --own [--fixed FILE] / reject` | Queue = runs that failed or took ≥ 3 attempts, not yet decided. Accept requires confirming the code is the user's own; row → `data/datasets/flywheel/<lang>.jsonl` (corpus format) or `xlang.jsonl`; split by input hash (~20 % eval). Decision stored in `data/runs/<id>/review.json` |
| Flywheel → training | `training/build_flywheel_mix.py`, DVC stage `flywheel_mix@<lang>` | Base corpus (`data/datasets/p15/<lang>_migration.jsonl`) + train-split flywheel rows (× `flywheel.repeat` = 2), minus rows whose input is in a frozen eval set, minus duplicates → `data/datasets/flywheel_mix/<lang>_migration.jsonl`. `--lang xlang` builds shared-adapter prompt / completion rows. Ran `dvc repro flywheel_mix@swift` (3,620 base rows, 0 flywheel rows yet) |
| Eval refresh | `rebirth review export-eval` → `data/datasets/flywheel/rolling_eval/<lang>.jsonl` | Newest 100 eval-split rows per language, same format as the frozen sets (`eval_behavioral.py --eval-file`); frozen sets untouched |

Privacy: flywheel rows are the user's own code — `data/datasets/flywheel/*` and the dashboard HTML
are git-ignored (this repo is public); nothing in `monitor/` imports a network client.

Reference profiles (2026-11-10): JS/TS 9,105 inputs (median 289 chars), Python 7,034 (333), Go 1,528
(387), Java 4,960 (645), C# 1,562 (461), Rust 1,505 (604), C++ 5,283 (820), Swift 1,535 (495).

## Exit criteria

- [x] Dashboard shows per-language/direction health from real runs (`rebirth dashboard` on `data/runs/`, 2026-11-10; only 15 dev runs so far)
- [x] Drift + regression warnings implemented and tested on synthetic data
- [ ] Flywheel: review → curated rows → included in a retrain via `dvc repro` — review → rows → `flywheel_mix` stage done and tested; **deferred to after launch** (2026-11-10): a training stage that reads `flywheel_mix` needs real reviewed rows first
- [x] Privacy: nothing leaves the machine (tests run with sockets blocked); redaction of stored artifacts verified
