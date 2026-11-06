# Phase 7 — Real-World JS/TS Validation

> Status: **COMPLETE (100%)** — on `main` via PR #2 (`5f1540e`)  
> Depends on: Phase 6 complete  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 7  
> Last docs refresh: 2026-11-04  
> Project now: Phases **0–10 complete** on `main`; next = Phase **11** per-lang LoRA — [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

## Objectives

- Golden-repo matrix: util, Express+DependencyMissing, mixed JS/TS, monorepo `--under`.  
- External-style full migrate + real Hub clone discovery.  
- Held-out 10+ suite + frozen baselines.  
- Classifier expansion + Unknown/low-confidence telemetry.  
- Safety: secrets / max size / license.

## What shipped

| Repo class | Fixture / clone | Evidence |
|------------|-----------------|----------|
| Tiny util | `fixtures/bench_util` | success report |
| App + deps | `fixtures/bench_express` | **DependencyMissing → install `ms` → PASS** |
| Mixed JS/TS | `fixtures/bench_mixed` | success |
| Monorepo lite | `fixtures/bench_monorepo` | `--under packages/a,packages/b` dry-run + migrate |
| External-style | `fixtures/bench_external_plain` | full migrate success |
| Real clone | `data/bench/is-plain-obj` | dry-run discovery (`--clone-external`) |

Also: `fixtures/eval_suite` (12 cases), `rebirth eval`, `data/reports/baselines/ollama_js_ts.json`, `src/rebirth_ai/safety.py`, classifier **105** logs / acc ≥0.90, heuristic DependencyMissing override + low-confidence telemetry in `classify_error`.

## How to run

```bash
source .venv/bin/activate
export REBIRTH_EMBEDDER=hash   # avoid ST vs DB dim clash during benches

python training/bench_js_repos.py --backend ollama
python training/bench_js_repos.py --dry-run --clone-external --only is-plain-obj
rebirth migrate-repo fixtures/bench_monorepo --under packages/a,packages/b --dry-run
rebirth eval --suite heldout --backend ollama \
  --freeze-baseline data/reports/baselines/ollama_js_ts.json
```

## Verified results

| Gate | Result |
|------|--------|
| Held-out | **12/12** PASS |
| All in-repo benches | **success** (express install_path=True) |
| Monorepo dry-run | only `packages/*/src` sources planned |
| Baseline | frozen under `data/reports/baselines/` |
| Classifier | ≥0.90 held-out |

## Exit criteria

- [x] Tiny util / Express+install / mixed JS/TS / monorepo scoped run  
- [x] ≥2 truthful migrate reports (+ external-style full migrate)  
- [x] Held-out green on Ollama + baseline freeze  
- [x] Safety filters  
- [x] Unknown/low-confidence classify telemetry  
- [x] Real CI-flavored log expansion  

## Next

1. **Merge PR #2** into `main` ([SESSION_HANDOFF.md](../SESSION_HANDOFF.md)).  
2. **[Phase 8 — Adapter factory + Python](PHASE_8_PYTHON.md)**.
