# Phase 0 — Environment, Repo & Documentation Baseline

> Ultra-detailed phase log. Status: **COMPLETE**  
> Date: 2026-10-31 · Docs refresh: 2026-11-04  
> Repo: https://github.com/hugobelebinda/rebirth-ai  
> Project now: Phases **0–10 complete** on `main`; next = Phases **12–14** per-lang LoRA — [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

---

## 1. Objectives

1. Create an independent git repository inside `Rebirth AI/` (not owned by parent multi-project `.git`).  
2. Publish a **public** GitHub repository `hugobelebinda/rebirth-ai`.  
3. Scaffold Python package layout, CLI stub, ignore rules, env example.  
4. Create living docs: `docs/GLOBAL.md` + all phase markdown stubs.  
5. Record machine prerequisites and what is / is not required before Phase 1.

---

## 2. What shipped

### 2.1 Repository scaffold

| Path | Purpose |
|------|---------|
| `README.md` | Project overview + status table |
| `.gitignore` | Python, Node, ML artifacts, secrets |
| `.env.example` | Ollama, sandbox, RAG, agent knobs |
| `pyproject.toml` | Package metadata + deps + optional extras |
| `LICENSE` | MIT |
| `src/rebirth_ai/` | Package root + module packages |
| `src/rebirth_ai/cli.py` | `rebirth version` / `rebirth doctor` |
| `docs/GLOBAL.md` | System bible |
| `docs/phases/PHASE_*.md` | Per-phase ultra-detailed logs |
| `data/**/.gitkeep` | Placeholder dirs for datasets/models |
| `fixtures/js_ts_sample/` | Now a real Node fixture (Phase 1) |
| `docker/js_ts_sandbox/` | Sandbox Dockerfile (Phase 1) |
| `training/.gitkeep` | Reserved for LoRA scripts |
| `tests/test_smoke.py` | Import / version smoke test |

### 2.2 Accounts & tooling (required vs deferred)

#### Must be ready before Phase 1 implementation work

| Item | Why | Phase 0 finding |
|------|-----|-----------------|
| GitHub account + working `gh` | Repo + later PRs | `hugobelebinda` authenticated via keyring |
| Docker Engine running | Sandbox | Daemon OK when host perms available; user in `docker` group |
| Node.js 20+ / npm | JS/TS fixtures & tests | Node `v22.22.2` |
| Python 3.11+ + venv | Agent runtime | Python `3.12.3` |
| Disk for images/models | Sandbox + LLMs | Confirm free space before large pulls |

#### Required by Phase 3 (can wait)

| Item | Why | Phase 0 finding |
|------|-----|-----------------|
| Ollama server + coder model | Local generation | Server up; `qwen2.5-coder:14b`, `deepseek-coder:6.7b` already pulled |
| Hugging Face account (optional) | Download / publish models & datasets | Not blocking |

#### Explicitly NOT required for Phases 1–4

| Item | When it becomes relevant |
|------|--------------------------|
| AWS account | Managed sandboxes / cloud GPUs (optional Phase 5+) |
| Supabase / pgvector | Replace SQLite at scale (optional) |
| vLLM | GPU-serving alternative to Ollama |
| NVIDIA CUDA toolchain | Faster LoRA train / larger models |
| LangSmith | Optional tracing |

### 2.3 Pre-flight verification (copy/paste)

```bash
# Identity
gh auth status
gh api user --jq .login

# Containers
docker info | head -20
docker run --rm hello-world

# LLM
curl -s http://127.0.0.1:11434/api/tags | head
ollama list

# Language toolchains
node -v && npm -v
python3 --version

# Project
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
rebirth doctor
pytest -q
```

---

## 3. Contracts / APIs introduced

### CLI

```text
rebirth version   → prints package version
rebirth doctor    → checks docker binary, docker daemon, ollama, node, python
```

### Env knobs (see `.env.example`)

- `OLLAMA_HOST`, `OLLAMA_MODEL`
- `REBIRTH_SANDBOX_IMAGE`, `REBIRTH_SANDBOX_TIMEOUT_SEC`
- `REBIRTH_EMBEDDING_MODEL`, `REBIRTH_VECTOR_DB_PATH`
- `REBIRTH_MAX_ITERATIONS`

No runtime modules for sandbox/RAG/graph yet — packages are empty placeholders.

---

## 4. How to recreate Phase 0 from scratch

```bash
mkdir -p "/home/hugobele/Documents/Personal Project/Rebirth AI"
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
# (restore files from this commit)
git init -b main
git add -A
git commit -m "chore: phase 0 scaffold for rebirth-ai"
gh repo create rebirth-ai --public --source=. --remote=origin --push
```

**Important:** Parent directory `/home/hugobele/Documents/Personal Project` already has a `.git`. This project uses its **own** `.git` inside `Rebirth AI/` so it is not committed as part of the parent umbrella repo.

---

## 5. Sample outputs (captured 2026-10-31)

```text
$ rebirth version
rebirth-ai 0.1.0

$ rebirth doctor
Python: 3.12.3
docker: ok
ollama: ok
node: ok
Docker daemon: ok
Ollama: ok (6 models)
  - qwen2.5-coder:14b
  - deepseek-coder:6.7b
  - llama3:latest
  - qwen-scripture:latest
  - deepseek-r1:32b
  - gemma4:latest

$ pytest -q
1 passed

$ docker run --rm hello-world
# Hello from Docker! (success)

$ ollama list
qwen2.5-coder:14b   ...
deepseek-coder:6.7b ...
```

---

## 6. Known gaps / follow-ups

- [ ] `fixtures/js_ts_sample` empty until Phase 1  
- [ ] Dockerfiles not yet written  
- [ ] No LoRA / RAG / LangGraph code yet (by design)  
- [ ] Optional: `pip install -e ".[ml,graph,github]"` only when those phases start  
- [ ] If `gh auth status` fails inside a sandbox but works with host permissions, always use host keyring for git/gh operations  

---

## 7. Decision log (Phase 0)

| Decision | Detail |
|----------|--------|
| Repo name | `rebirth-ai` |
| Visibility | Public |
| Package import path | `rebirth_ai` under `src/` |
| Optional dependency groups | `graph`, `ml`, `github`, `dev` |
| Docs protocol | Global + one ultra-detailed file per phase |

---

## 8. Exit criteria checklist

- [x] Independent `git init` in `Rebirth AI/`  
- [x] Public GitHub repo created and initial push completed (`https://github.com/hugobelebinda/rebirth-ai`)  
- [x] Python package scaffold with CLI stub  
- [x] `.gitignore` excludes secrets, venvs, heavy ML artifacts  
- [x] `docs/GLOBAL.md` written  
- [x] All phase markdown files created (0–5)  
- [x] Prerequisites table documents AWS/Ollama/Supabase/etc.  
- [x] Smoke test passes (`pytest`)  
- [x] `rebirth doctor` runnable after `pip install -e ".[dev]"`  

---

## 9. Handoff to Phase 1

Phase 1 should:

1. Add `docker/js_ts_sandbox/Dockerfile` (Node + npm test tooling).  
2. Implement `rebirth_ai.sandbox` using Docker SDK (create → mount → test → logs → destroy).  
3. Implement Tree-sitter JS/TS chunking in `rebirth_ai.ast`.  
4. Define `LanguageAdapter` protocol and `adapters/js_ts`.  
5. Populate `fixtures/js_ts_sample` with a tiny failing/passing migration playground.  
6. Update this file’s sibling `PHASE_1_SANDBOX_AST.md` and sync `GLOBAL.md`.
