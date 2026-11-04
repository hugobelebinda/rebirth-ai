# Bench: Express-style + DependencyMissing (Phase 7)

> Phase 7 **COMPLETE** (install-`ms` path proven). Next: merge PR #2 then Phase 8 — [SESSION_HANDOFF.md](../../docs/SESSION_HANDOFF.md).

- `express` is declared in `package.json`.
- Seed `seeds/duration.missing_dep.js` imports **`ms`** (undeclared) so the agent must hit `DependencyMissing` → `install_dependency`.

```bash
python training/bench_js_repos.py --backend ollama --only bench_express
```
