# Bench: monorepo lite (Phase 7)

> Phase 7 **COMPLETE**. Next: merge PR #2 then Phase 8 — [SESSION_HANDOFF.md](../../docs/SESSION_HANDOFF.md).

Scoped discovery:

```bash
rebirth migrate-repo fixtures/bench_monorepo --under packages/a,packages/b --dry-run
python training/bench_js_repos.py --backend ollama --only bench_monorepo
```
