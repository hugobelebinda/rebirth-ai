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
| 0 | Environment, repo, docs baseline | **Complete** |
| 1 | Docker sandbox + AST | **Complete** |
| 2 | RAG + error classifier | **Complete** |
| 3 | LoRA training + Ollama inference | **Complete** |
| 4 | LangGraph closed loop | **Complete** |
| — | Dataset + PEFT (7B ROCm 1-epoch) | **Active** — see [`docs/DATASET.md`](docs/DATASET.md) |
| 5 | Repo-scale + GitHub PRs | **Complete** — `rebirth migrate-repo` / [PR #1](https://github.com/hugobelebinda/rebirth-ai/pull/1) |

Living docs: [`docs/GLOBAL.md`](docs/GLOBAL.md) · [`docs/DATASET.md`](docs/DATASET.md) · phase notes under [`docs/phases/`](docs/phases/).

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
```

Optional PEFT (7B adapter after local ROCm train):

```bash
pip install -e ".[ml]"
REBIRTH_PEFT_BASE=Qwen/Qwen2.5-Coder-7B-Instruct \
REBIRTH_PEFT_ADAPTER=data/models/lora_js_ts_7b/adapter \
rebirth generate fixtures/js_ts_sample/src/math.js --backend peft --no-rag
```

Docs refresh: when chat context is ~90% full, agents re-read/update all project `.md` files (see `.cursor/rules/docs-context-refresh.mdc`).

## Design defaults

- **First adapter:** JavaScript / TypeScript (pluggable; Python and others later)  
- **Inference:** Ollama + AMD ROCm (`qwen2.5-coder:14b-instruct-q6_K` on RX 7800 XT)  
- **Local PEFT:** `Qwen2.5-Coder-7B-Instruct` LoRA at seq 128 (1-epoch proven); avoid multi-epoch GPU trains while display is on the dGPU  
- **Vectors:** SQLite first (Supabase/pgvector optional later)  
- **No AWS required** for Phases 1–4  

## License

MIT
