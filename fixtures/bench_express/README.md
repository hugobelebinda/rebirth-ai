# Bench: Express-style + DependencyMissing (Phase 7, package policy Phase 19)

- `express` is declared in `package.json`; the sandbox's setup step installs it (with network,
  before the draft is written in), so its own dependencies such as `ms` are present too.
- Seed `seeds/duration.missing_dep.js` imports **`pretty-ms`**, which only the draft uses, so the
  tests fail with `DependencyMissing`:
  - `bench_express`: the package policy refuses it (only the draft imports it) and the model has
    to write a draft without it.
  - `bench_express_optin`: `REBIRTH_ALLOW_UNDECLARED_PACKAGES=1` → `install_dependency` queues
    `pretty-ms` for the next run's setup step → tests pass offline.

```bash
python training/bench_js_repos.py --backend ollama --only bench_express --only bench_express_optin
```
