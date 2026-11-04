# Bench: monorepo lite (Phase 7)

Scoped discovery:

```bash
rebirth migrate-repo fixtures/bench_monorepo --under packages/a,packages/b --dry-run
python training/bench_js_repos.py --backend ollama --only bench_monorepo
```
