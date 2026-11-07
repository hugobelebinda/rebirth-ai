# Phase 5 — Repository Scale & GitHub Automation

> Status: **COMPLETE**  
> Depends on: Phase 4 single-file loop — **stable** (`rebirth migrate` E2E green on Ollama 14B)  
> Completed: 2026-11-01 · Docs refresh: 2026-11-04  
> Project now: Phases **0–10 complete** on `main`; next = Phase **11** per-lang LoRA — [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

---

## 1. Objectives

- Walk directories via language adapter; migrate file-by-file with shared RAG index.  
- GitHub integration: branch, commit successes, open PR with summary.  
- Failed files listed in PR body without aborting the whole run.  

---

## 2. What shipped

| Path | Responsibility |
|------|----------------|
| `src/rebirth_ai/graph/batch.py` | Discover sources, infer tests, batch LangGraph, JSON report, PR body |
| `src/rebirth_ai/github/client.py` | `gh` auth + git branch/stage/commit/push |
| `src/rebirth_ai/github/pr.py` | `open_migration_pr()` |
| `src/rebirth_ai/cli.py` → `rebirth migrate-repo` | Multi-file CLI + `--open-pr` |
| `fixtures/js_ts_sample/test/format.test.js` | Second migratable target |
| `fixtures/migration_seeds/format.broken.js` | Multi-file seed |
| `tests/test_batch_migrate.py` | Discovery + PR body unit tests |
| `data/reports/phase5_*` | E2E report + migrated snapshots |

### Optional (deferred)

- Remaining v1.0 language adapters (Python → Go/Java/C#/Rust → C++/Swift) + ≥15K corpus  
 
- Supabase / vLLM / AWS sandboxes  

---

## 3. How to run

```bash
source .venv/bin/activate

# Multi-file migrate (continue on failures). Example with two broken seeds:
rebirth migrate-repo fixtures/js_ts_sample \
  --file src/math.js --file src/format.js \
  --seed src/math.js=fixtures/migration_seeds/math.broken.js \
  --seed src/format.js=fixtures/migration_seeds/format.broken.js \
  --write \
  --report data/reports/migrate_repo.json

# Open a PR from successes + report (requires: gh auth login)
rebirth migrate-repo fixtures/js_ts_sample \
  --file src/math.js \
  --seed src/math.js=fixtures/migration_seeds/math.broken.js \
  --write --open-pr --pr-branch rebirth/migrate --pr-base main

pytest tests/test_batch_migrate.py -q
```

### Discovery rules

- Default walk: `src/**` with JS/TS suffixes  
- Test inference: `src/foo.js` → `test/foo.test.js` (also `tests/`, `__tests__/`)  
- Files without a matching test → `skipped_no_test` (does not fail the batch)  

---

## 4. Sample E2E (2026-11-01)

Temp fixture copy + Ollama backend, two seeded files:

```
status summary: Success 2 / Failed 0 / Skipped 0
- src/math.js (iters=1, wrote)
- src/format.js (iters=1, wrote)
```

Demo PR: https://github.com/hugobelebinda/rebirth-ai/pull/1  
Artifacts: `data/reports/phase5_e2e.json`, `data/reports/phase5_migrated/`.

---

## 5. Exit criteria

- [x] Agent opens a real PR with ≥1 migrated file (PR #1 + migrated snapshots)  
- [x] Failed files listed in PR body / report without aborting whole run  
- [x] Docs updated; `GLOBAL.md` marks Phase 5 complete  

### Handoff → production path

Phases **6–7 are complete** on branch/PR #2 — see [SESSION_HANDOFF.md](../SESSION_HANDOFF.md).  
Next after merge: **[Phase 8 — Python](PHASE_8_PYTHON.md)** · [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md).

---

## 6. Decision log

| Date | Decision |
|------|----------|
| 2026-11-01 | Ship Phase 5 on Ollama 14B; PEFT optional / parallel |
| 2026-11-01 | Skip sources with no inferred test rather than guessing `npm test` |
| 2026-11-01 | PR body always lists failures; successes alone are enough to open a PR |
| 2026-11-01 | Use `gh` CLI (not PyGithub) for auth/PR to match existing `gh` setup |
