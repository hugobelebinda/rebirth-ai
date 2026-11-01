# Rebirth AI

Autonomous, closed-loop agent for migrating legacy codebases without brittle regex scripts or zero-shot-only prompts.

It runs a **Plan → Execute → Verify** loop as a LangGraph Markov Decision Process:

1. **RAG** — Tree-sitter AST + code embeddings retrieve only the interfaces the current file needs  
2. **LoRA LLM** — local coder model (Ollama) with a migration-specialized adapter  
3. **Docker sandbox** — ephemeral containers run the real test suite  
4. **Error classifier** — Scikit-Learn routes failures (`DependencyMissing`, `SyntaxError`, …) before hitting the LLM again  

## Status

| Phase | Name | Status |
|-------|------|--------|
| 0 | Environment, repo, docs baseline | **Complete** |
| 1 | Docker sandbox + AST | Planned |
| 2 | RAG + error classifier | Planned |
| 3 | LoRA training + Ollama inference | Planned |
| 4 | LangGraph closed loop | Planned |
| 5 | Repo-scale + GitHub PRs | Planned |

Living docs: [`docs/GLOBAL.md`](docs/GLOBAL.md) · phase notes under [`docs/phases/`](docs/phases/).

## Quick start (after Phase 0)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Verify tooling (Docker daemon + Ollama must be running)
docker version
ollama list   # expect qwen2.5-coder and/or deepseek-coder
```

## Design defaults

- **First adapter:** JavaScript / TypeScript (pluggable; Python and others later)  
- **Inference:** Ollama local (`qwen2.5-coder:14b` available on this machine)  
- **Vectors:** SQLite first (Supabase/pgvector optional later)  
- **No AWS required** for Phases 1–4  

## License

MIT
