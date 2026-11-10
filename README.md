# Rebirth AI

Autonomous, closed-loop agent for migrating legacy codebases without brittle regex scripts or zero-shot-only prompts.

It runs a **Plan → Execute → Verify** loop as a LangGraph Markov Decision Process:

1. **RAG** — Tree-sitter AST + code embeddings retrieve only the interfaces the current file needs  
2. **LLM** — local coder via Ollama (14B default) or optional PEFT LoRA (7B preferred locally)  
3. **Docker sandbox** — ephemeral containers run the real test suite  
4. **Error classifier** — Scikit-Learn routes failures (`DependencyMissing`, `SyntaxError`, …) before hitting the LLM again  

## Status

| Phase | Name | Status |
|-------|------|--------|
| 0–5 | MVP (setup → LangGraph → multi-file PRs) | **Complete** |
| Data | Multi-lang corpus **24 640** (≥15K) + JS/TS PEFT 7B proof | **Complete** (corpus); LoRA waves next |
| 6 | Repo hardening (JS/TS production path) | **Complete (100%)** |
| 7 | Real-world JS/TS validation | **Complete (100%)** |
| 8 | Adapter factory + Python | **Complete** |
| 9 | Full language wave — Go, Java, C#, Rust, C++, Swift | **Complete** |
| 10 | Quality gates + ≥15K corpus | **Complete** (fixtures **8/8**) |
| 11 | Per-language LoRA — harness + v2 7B adapters, all 8 languages | **Complete** (2026-11-05; all 8 pass their gate) |
| **12** | **Local tuning v3 + eval hardening** (expanded corpus, frozen eval sets, recipe experiments) | ✅ Complete (2026-11-06) |
| 13 | MLOps 1 — experiment tracking + model registry (MLflow) + YAML recipes | **Complete** (2026-11-05) |
| 14 | MLOps 2 — reproducible pipelines + infra (DVC, pinned CUDA env, Terraform, cloud scripts, vLLM eval) | **Complete** (2026-11-05) |
| 15 | Cloud — 14B per-language adapters + DPO (AWS g6e.xlarge) | **Complete** (2026-11-08) — stopped at the S2 gate: 14B adapters tie the 7B ones, so the 7B adapters stay |
| 16 | MLOps 3 — model CI/CD + packaging (gates in CI, Ollama adapters, rollback) | Offline part complete (2026-11-06) |
| 17 | Cross-language migration (any → any of 8) + unified adapter | Stages 1–4 done 2026-11-09 — `rebirth migrate --to` with the shared cross-language adapter: whole-program translations verified **208 → 265 / 420** vs the plain 14B |
| 18 | MLOps 4 — monitoring + data flywheel | Planned |
| 19 | Packaging, UX & official **v1.0** launch | Planned (last) |

Phase numbers follow execution order (renumbered 2026-11-05 — map in [`docs/PRODUCTION_ROADMAP.md`](docs/PRODUCTION_ROADMAP.md#phase-renumbering-2026-11-05)).

**Handoff:** [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) — Phases **0–15 complete** (8 langs + **24640** pairs + fixtures 8/8 + v2 LoRA per language); Phase 17 stages 1–4 done.  
**Now:** Phase 17 — `rebirth migrate --to <lang>` translates between any of the 8 languages with a shared cross-language adapter trained on AWS (2026-11-09) on 30K test-verified translation pairs. On 420 whole-program translations (Codeforces problems, judged by their tests) it lifts the local q6_K 14B from 208 to 265 verified ([`PHASE_17_CROSS_LANGUAGE.md`](docs/phases/PHASE_17_CROSS_LANGUAGE.md)). Phase 15 (14B same-language adapters) ended at its gate: they tie the 7B adapters, which stay ([`PHASE_15_CLOUD_14B.md`](docs/phases/PHASE_15_CLOUD_14B.md)). Cloud plan + spend: [`docs/CLOUD_GPU_PLAN.md`](docs/CLOUD_GPU_PLAN.md).  
**Production roadmap:** [`docs/PRODUCTION_ROADMAP.md`](docs/PRODUCTION_ROADMAP.md) · [`docs/GLOBAL.md`](docs/GLOBAL.md) · [`docs/DATASET.md`](docs/DATASET.md)

Live adapters: **JS/TS, Python, Go, Java, C#, Rust, C++, Swift**. Same-language: v2 7B adapter per language, served from Ollama as `rebirth-<lang>:7b` (`--backend auto`, the default; `--backend peft` runs them in-process). Cross-language (`--to`): `rebirth-xlang:14b-v1` (q6_K 14B + shared adapter; build it with `bash scripts/ollama_xlang.sh`), plain Ollama 14B when it is not installed.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Verify tooling (Docker daemon + Ollama must be running)
docker version
ollama list   # expect qwen2.5-coder:14b-instruct-q6_K
docker build -t rebirth-ai/js-ts-sandbox:latest docker/js_ts_sandbox
docker build -t rebirth-ai/python-sandbox:latest docker/python_sandbox
rebirth doctor
rebirth adapters
rebirth parse fixtures/js_ts_sample/src/math.js
rebirth sandbox fixtures/js_ts_sample -t test/math.test.js
rebirth sandbox fixtures/python_sample -t tests/test_math_ops.py --lang python

# Closed-loop single-file migrate (default Ollama 14B)
rebirth migrate fixtures/js_ts_sample \
  --seed-draft fixtures/migration_seeds/math.broken.js \
  --out /tmp/rebirth_math_fixed.js --no-write

rebirth migrate fixtures/python_sample \
  --file src/math_ops.py --test tests/test_math_ops.py \
  --seed-draft fixtures/python_sample/seeds/math_ops.broken.py \
  --lang python --no-write

# Translate to another language (Phase 17): new file under <project>/rebirth_out/<lang>/,
# checked against a target-language project's tests (else a compile check), retried on errors.
# Served by rebirth-xlang:14b-v1 when installed (bash scripts/ollama_xlang.sh), else the plain 14B
rebirth migrate fixtures/python_sample --file src/math_ops.py --to go \
  --target-project fixtures/go_sample --target-file src/mathops.go --target-test src/mathops_test.go

# Multi-file
rebirth migrate-repo fixtures/js_ts_sample \
  --file src/math.js --file src/format.js \
  --seed src/math.js=fixtures/migration_seeds/math.broken.js \
  --seed src/format.js=fixtures/migration_seeds/format.broken.js \
  --write --report data/reports/migrate_repo.json
```

Optional PEFT (defaults to 7B adapter when present):

```bash
pip install -e ".[ml]"
# Unload Ollama first if VRAM is tight: ollama stop qwen2.5-coder:14b-instruct-q6_K
rebirth migrate fixtures/js_ts_sample \
  --seed-draft fixtures/migration_seeds/math.broken.js --backend peft --no-write
```

Behavioral eval (sandbox-pass / AST — preferred over exact string):

```bash
python training/eval_behavioral.py --backend ollama --suite sandbox
```

## Design defaults

- **Languages (v1.0):** all 8 via `LanguageAdapter` (live)  
- **Corpus (v1.0):** ≥15K mined pairs — **24 640** met  
- **Inference:** Ollama + AMD ROCm (`qwen2.5-coder:14b-instruct-q6_K`)  
- **Local PEFT:** per-lang 7B LoRAs (Phase 11); never stack with Ollama+ST on one 16GB card  
- **Vectors:** SQLite first  
- **Cloud GPU:** Optional 14B QLoRA — billing in HF browser settings, not the CLI  
- **No AWS required** for MVP / Phases 6–14  

## License

MIT
