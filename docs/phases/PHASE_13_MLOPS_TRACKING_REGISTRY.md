# Phase 13 — MLOps 1: experiment tracking + model registry

> Status: **COMPLETE (2026-11-05)**. Built on CPU while the GPU ran Phase 12 (no interference: post-hoc design, isolated UI venv, client installed only after the GPU job stopped).
> Previous: [PHASE_12_LOCAL_TUNING_V3.md](PHASE_12_LOCAL_TUNING_V3.md) · Next: [PHASE_14_MLOPS_REPRO_INFRA.md](PHASE_14_MLOPS_REPRO_INFRA.md)
> Estimated effort: ~1–2 days.

## Why now (before the cloud)

Experiments are currently compared by reading `status.jsonl`, `rebirth_meta.json` and report JSON
files by hand. Before paying for cloud GPU hours (Phase 15), every run must land in one place,
comparable, with its exact inputs — otherwise paid runs are hard to compare and easy to lose.

## Scope

| Piece | What | Detail |
|-------|------|--------|
| Tracking server | **MLflow**, local, file/SQLite backend under `mlruns/` (gitignored) | `rebirth mlflow-ui` → http://127.0.0.1:5000 |
| Training runs | `train_lora.py` logs params (recipe, data sha256, split, base model, quant, LoRA targets), per-step loss/lr/VRAM (from `StepLog`), GPU telemetry summary, artifacts (`rebirth_meta.json`, `train_meta.json`) | Opt-in flag `--track`, default on once stable; never blocks training if MLflow is down |
| Eval runs | `eval_behavioral.py` logs soft-exact, parses, symbol Jaccard, token overlap, per-row table as an artifact; linked to the model version it scored | Nested under the training run when scoring a fresh adapter |
| Model registry | One registered model per language (`rebirth-<lang>`), versions = adapters; aliases **champion** / **challenger**; tags: base, recipe, eval-set sha | Replaces manual "promote v3 if it beats v2" |
| Gate as code | `training/lora_gate.py` reads the registry, compares challenger vs champion on the same frozen eval set, sets the alias | Same rule as today: soft-exact ≥, parses ≥ −1 |
| Recipes as config | `configs/recipes/*.yaml` (e.g. `v2.yaml`, `v3_expanded.yaml`, `14b_cloud.yaml`); `train_lora.py --recipe` | CLI flags still override; the resolved config is logged |
| Backfill | Import today's results (v2 ×8, Go A/C/D/D1, Ollama/base references) | So history is in the UI from day one |

## Exit criteria

- [x] Every new train/eval run appears in MLflow with params, metrics and artifacts (hooks in `train_logged.sh` + `eval_behavioral.py`; backfill for history)
- [x] `rebirth-<lang>` registered models with a **champion** alias for all 8 languages (v2 adapters)
- [x] Gate promotes/rejects via the registry; decision logged as version tags
- [x] Recipes in YAML; `wave_v3.sh` uses `--recipe`
- [x] Tests for config loading + gate logic on a temp SQLite store (`tests/test_mlops.py`, 14 tests; full suite 118 passing)
- [x] Docs: how to open the UI and compare runs (below)

## What shipped (2026-11-05)

### Design: post-hoc tracking

Training and eval already write everything to disk (`logs/train/<run>/{run.json,steps.jsonl,telemetry.jsonl,kernel.log,exit.json}`,
`rebirth_meta.json`, eval report JSON). `src/rebirth_ai/tracking.py` **reads those files after a run** and records
them in MLflow — tracking never runs inside the training loop, so it cannot slow down or crash a GPU job, and
the same code backfills history. Every entry point is idempotent (re-tracking the same adapter / report is a no-op).

| File | Role |
|------|------|
| `src/rebirth_ai/tracking.py` | Store config (SQLite `mlruns/mlflow.db`, artifacts `mlruns/artifacts/`), `log_training`, `log_eval`, run-dir discovery (follows `--resume` chains), eval-set ids |
| `src/rebirth_ai/registry.py` | `rebirth-<lang>` registered models; versions = adapters; aliases `champion` / `previous` / `challenger`; `gate`, `promote`, `rollback`, `champion_adapter` |
| `src/rebirth_ai/recipes.py` + `configs/recipes/*.yaml` | YAML recipes → `train_lora.py --recipe <name>` (CLI flags override; unknown keys are errors) |
| `src/rebirth_ai/backfill.py` | Imports existing adapters + reports, sets first champions, dry-run-gates experiments |
| `src/rebirth_ai/mlops_cli.py` | `rebirth track …`, `rebirth registry …`, `rebirth mlflow-ui` |
| `scripts/train_logged.sh` | After a successful run: `rebirth track train <run dir>` (best-effort) |
| `training/eval_behavioral.py` | After writing a report: tracked automatically (`--no-track` to skip, `--model-label` to name it) |
| `tests/test_mlops.py` | Recipes, helpers, resume-chain discovery, full register → eval → gate (incl. tie) → promote → rollback on a temp SQLite store |

### Eval sets are first-class

Scores are only comparable on the **same rows**. Every eval run carries an eval-set id:
`evalset:<lang>:<sha10 of the frozen file>:n<rows>` for the frozen 100-row sets, `heldout_split:<lang>:n24`
for the old 24-shortest-rows suite. The gate refuses to compare across ids (`needs_eval`).

### Gate (now in the registry, with a strict-gain rule)

Challenger replaces champion iff, on the same eval set: soft-exact ≥ champion **+ 1** and parses ≥ champion − 1.
The Phase 11 rule (adapter vs *base model*, soft-exact ≥) let ties through; for challenger vs *champion* a tie
keeps the incumbent — Go D1 tied v2 (14 vs 14) with a 155 MB adapter (vs 20 MB) and 23 % slower steps.
`--min-gain 0` restores tie-promotion.
Decisions are written as version tags (`gate.<set>` = pass/fail, `promoted_reason`, `retired_by`).
The registry champion is the *recorded* decision; serving still reads `data/models/lora_<lang>_7b`
until Phase 16 routes from the registry.

### How to use

```bash
pip install -e ".[mlops]"                 # mlflow-skinny + SQLAlchemy + alembic (client)
rebirth track backfill                    # import everything already on disk (idempotent)
rebirth mlflow-ui                         # http://127.0.0.1:5000 (full MLflow in .venv-mlflow)
rebirth registry list                     # versions, aliases, scores per language
rebirth registry gate --lang go --version 5 --eval-set evalset:go:<sha>:n100 [--dry-run]
rebirth registry rollback --lang go
python training/train_lora.py --lang go --recipe v3_expanded --data data/datasets/expanded/go_migration.jsonl
```

The UI runs from a separate venv (`.venv-mlflow`, full MLflow) because full MLflow would downgrade
`python-dateutil` in the training venv; the training venv only gets the light client.

### Backfill result (2026-11-05)

| What | Count |
|------|------:|
| Training runs (with loss curves, sec/step, VRAM + junction-temp peaks, linked `logs/train` dirs) | 12 (8 × v2, Python v1, Go C / D / D1) |
| Eval runs (24-row held-out, cap-64 bug runs, frozen 100-row sets, base 7B / Ollama 14B references) | 42 |
| Registered models / champions | 8 / 8 (`rebirth-<lang>` → v2 adapter) |
| Store size | ~5 MB (adapter config + weights sha256; weights stay in `data/models`) |

Go challengers vs champion v1 on `evalset:go:a4cc0f3b4c:n100` (all **kept champion**):

| Version | Adapter | Soft-exact / parses | Failed check |
|---------|---------|---------------------|--------------|
| v1 (champion) | `lora_go_7b` (v2 recipe) | 14 / 99 | — |
| v2 | `exp/go_C_attn_r8_exp_e1` | 15 / 97 | parses ≥ champion − 1 |
| v3 | `exp/go_D1_all_r16_exp_e1` | 14 / 98 | soft-exact ≥ champion + 1 |
| v4 | `exp/go_D_all_r16_exp_e2` | 12 / 95 | both |

Install note: full `mlflow` would downgrade `python-dateutil` 3.0.0 → 2.9.0 in the training venv, so the
training venv gets only `mlflow-skinny` + SQLAlchemy + alembic (27 new packages, 0 changed — verified with
`pip freeze` before/after) and the UI lives in `.venv-mlflow`. MLflow 3.18 rejects the plain-folder store,
hence SQLite.
