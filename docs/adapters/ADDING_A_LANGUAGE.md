# Adding a language adapter (playbook)

> Status: **live** — all 8 v1.0 languages shipped (Phases 8–9); Phase 10 quality/corpus done (2026-11-04).  
> Next specialization: per-lang LoRA — [PHASE_12_LORA_PER_LANGUAGE.md](../phases/PHASE_12_LORA_PER_LANGUAGE.md).

Every language vertical is **instantiation**, not a rewrite of LangGraph.

## Checklist

1. **Adapter** — `src/rebirth_ai/adapters/<id>.py` implementing `LanguageAdapter`
   - `name`, `SOURCE_SUFFIXES`, `sandbox_spec()`, `is_source_file()`, `parse()`, `test_command_for()`
   - Optional: `install_command(packages)` for DependencyMissing
2. **Registry** — register factory in `src/rebirth_ai/adapters/registry.py` (`_REGISTRY` + detect markers)
3. **Docker sandbox** — `docker/<id>_sandbox/Dockerfile`
   - Image name: `rebirth-ai/<id>-sandbox:latest` (or env override)
   - Non-root user matching `SandboxSpec.user`
   - Doctor must list the image (`rebirth doctor`)
4. **Tree-sitter** — wire grammar in `ast/chunker.py` `_load_language` + classify/name helpers; add dep to `pyproject.toml`
5. **Fixture** — `fixtures/<id>_sample/` with good source + broken seed + tests
6. **Test mapping** — extend `infer_test_path` candidates (document in `TEST_MAPPING.md`)
7. **CLI** — `--lang <id>` works; auto-detect via root markers; `rebirth adapters` lists the row
8. **Mine** — `training/mine_migration_pairs.py --langs …` → `data/datasets/<lang>_migration.jsonl` + stats; update `docs/DATASET.md`
9. **Eval** — at least one behavioral / sandbox case; baseline row when freezing quality
10. **Docs** — phase file + GLOBAL status board + this checklist evidence

## Detection matrix

| Adapter id | Root markers | Extensions |
|------------|--------------|------------|
| `js_ts` | `package.json` | `.js` `.jsx` `.mjs` `.cjs` `.ts` `.tsx` |
| `python` | `pyproject.toml` `requirements.txt` `setup.py` `Pipfile` | `.py` |
| `go` | `go.mod` | `.go` |
| `java` | `pom.xml` `build.gradle*` | `.java` |
| `csharp` | `*.csproj` `*.sln` | `.cs` |
| `rust` | `Cargo.toml` | `.rs` |
| `cpp` | `CMakeLists.txt` | `.cpp` `.cc` `.cxx` `.h` `.hpp` |
| `swift` | `Package.swift` | `.swift` |

Override anytime: `rebirth migrate … --lang python`.

## Smoke after adding a language

```bash
pip install -e ".[dev]"
docker build -t rebirth-ai/<id>-sandbox:latest docker/<id>_sandbox
rebirth adapters
rebirth doctor
rebirth sandbox fixtures/<id>_sample
rebirth migrate fixtures/<id>_sample --file … --test … --seed-draft … --lang <id>
```

Exit when: fixture migrate ≤3 iterations, `migrate-repo` report exists, doctor sees the image, mined pair count recorded.
