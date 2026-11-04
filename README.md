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
| Data | Dataset + PEFT 7B + behavioral eval (→ **15K+** multi-lang) | **Active** |
| 6 | Repo hardening (JS/TS production path) | **Complete** |
| 7 | Real-world JS/TS validation | **Complete** |
| **8** | Adapter factory + Python | **Next** |
| 9 | Language wave A — Go, Java, C#, Rust | Planned |
| 10 | Wave B — C++, Swift + quality + ≥15K corpus | Planned |
| 11 | Official **v1.0** launch | Planned |
| Optional | Cloud 14B QLoRA | Optional (not required) |

**Production roadmap (detailed):** [`docs/PRODUCTION_ROADMAP.md`](docs/PRODUCTION_ROADMAP.md)  
Living docs: [`docs/GLOBAL.md`](docs/GLOBAL.md) · [`docs/DATASET.md`](docs/DATASET.md) · [`docs/phases/`](docs/phases/)

Today the **live vertical is JavaScript / TypeScript**. **v1.0** targets adapters for **JS/TS, Python, Go, Java, C++, C#, Rust, and Swift** with **≥15K** mined migration pairs. Cloud GPU is optional.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Verify tooling (Docker daemon + Ollama must be running)
docker version
ollama list   # expect qwen2.5-coder:14b-instruct-q6_K
docker build -t rebirth-ai/js-ts-sandbox:latest docker/js_ts_sandbox
rebirth doctor
rebirth parse fixtures/js_ts_sample/src/math.js
rebirth sandbox fixtures/js_ts_sample -t test/math.test.js

# Closed-loop single-file migrate (Phase 4 — default Ollama 14B)
rebirth migrate fixtures/js_ts_sample \
  --seed-draft fixtures/migration_seeds/math.broken.js \
  --out /tmp/rebirth_math_fixed.js --no-write

# Multi-file (Phase 5)
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

- **Languages (v1.0):** JS/TS → Python → Go / Java / C# / Rust → C++ / Swift via `LanguageAdapter`  
- **Corpus (v1.0):** ≥15K mined multi-language migration pairs  
- **Inference:** Ollama + AMD ROCm (`qwen2.5-coder:14b-instruct-q6_K`)  
- **Local PEFT:** 7B LoRA optional; never stack with Ollama+ST on one 16GB card  
- **Vectors:** SQLite first  
- **Cloud GPU:** Optional LoRA upgrade only — billing in HF browser settings, not the CLI  
- **No AWS required** for MVP / Phases 6–11  

## License

MIT
