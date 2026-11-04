# Bench: Express-style + DependencyMissing (Phase 7)

- `express` is declared in `package.json`.
- Seed `seeds/duration.missing_dep.js` imports **`ms`** (undeclared) so the agent must hit `DependencyMissing` → `install_dependency`.

```bash
python training/bench_js_repos.py --backend ollama --only bench_express
```
