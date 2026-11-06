# Phase 8 — Adapter Factory + Python Vertical

> Status: **COMPLETE** (2026-11-03)  
> Depends on: Phase 7 + PR #2 merged to `main` (`5f1540e`)  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 8  
> Playbook: [ADDING_A_LANGUAGE.md](../adapters/ADDING_A_LANGUAGE.md)  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md) — Phases 0–10 done; next LoRA Phase 11

## Objectives

- Lock the **repeatable language playbook** (`ADDING_A_LANGUAGE.md` + adapter registry).  
- Ship `PythonAdapter` + pytest Docker sandbox + fixture.  
- Auto-detect project language; closed-loop migrate on Python.  
- Mine first non-JS pairs toward the **15K+** multi-lang corpus.  
- Prove multi-language without rewriting the LangGraph core.

## Exit criteria

- [x] Adapter registry + `rebirth adapters` + `docs/adapters/ADDING_A_LANGUAGE.md`  
- [x] Python fixture migrate ≤3 iterations on Ollama (**1 iter**)  
- [x] `migrate-repo` works on `fixtures/python_sample`  
- [x] Doctor checks Python sandbox image  
- [x] Python migration pairs mined; count recorded in `DATASET.md` (**1503**)  
- [ ] Optional: tag `v0.3.0-js-python-preview` (deferred until commit)

## What shipped

| Path | Role |
|------|------|
| `src/rebirth_ai/adapters/registry.py` | `get_adapter` / `get_adapter_for_project` / detection |
| `src/rebirth_ai/adapters/python.py` | `PythonAdapter` + pytest commands |
| `docker/python_sandbox/Dockerfile` | `rebirth-ai/python-sandbox:latest` |
| `fixtures/python_sample/` | Good source + broken seed + pytest |
| `docs/adapters/ADDING_A_LANGUAGE.md` | Phase 9+ checklist |
| `training/mine_migration_pairs.py --langs python` | CommitPackFT Python mine |
| `data/datasets/python_migration.jsonl` | **1503** pairs (3 synthetic + 1500 mined) |

Graph/CLI: `--lang`, auto-detect, `skipped_wrong_language`, SandboxSpec.`user` / `dockerfile_dir`, Tree-sitter Python chunking.

## How to run

```bash
docker build -t rebirth-ai/python-sandbox:latest docker/python_sandbox
rebirth adapters
rebirth doctor
rebirth sandbox fixtures/python_sample --test tests/test_math_ops.py --lang python

rebirth migrate fixtures/python_sample \
  --file src/math_ops.py --test tests/test_math_ops.py \
  --seed-draft fixtures/python_sample/seeds/math_ops.broken.py \
  --lang python --max-iterations 3

rebirth migrate-repo fixtures/python_sample --lang python \
  --seed 'src/math_ops.py=fixtures/python_sample/seeds/math_ops.broken.py' \
  --report data/reports/migrate_repo_python.json

python training/mine_migration_pairs.py --langs python --max-pairs 1500 --require-parse
```

## Sample outputs

- Sandbox: `PASS … 4 passed`  
- Migrate: `status=success iterations=1` model=`qwen2.5-coder:14b-instruct-q6_K`  
- Mine: scanned 56025 → kept 2459 → capped **1500** + 3 synthetic = **1503**  
- Doctor: `Sandbox (python): ok (rebirth-ai/python-sandbox:latest)`

## Known gaps

- No Python PEFT LoRA yet (Ollama default is enough; training optional).  
- Optional preview tag not cut.  
- Hub clone failures (e.g. is-plain-obj) remain Phase 7 known external gaps — unrelated.

## Decision log

| Date | Decision |
|------|----------|
| 2026-11-03 | PR #2 merged; Phase 8 started on `main` |
| 2026-11-03 | Registry + detection matrix; Python vertical without per-lang LoRA |
| 2026-11-03 | Mine 1500 CommitPackFT Python pairs toward 15K corpus |
