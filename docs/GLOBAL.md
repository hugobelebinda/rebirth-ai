# Rebirth AI — Global System Bible

> Living document. **Update after every phase.** Last updated: Phase 0 complete (2026-10-31).  
> Repo: https://github.com/hugobelebinda/rebirth-ai

---

## 1. Mission

Build an autonomous, closed-loop AI agent that migrates legacy codebases (framework transitions, major-version upgrades, architectural pattern enforcement) **without human intervention in the happy path**, using:

- LangGraph MDP orchestration (Plan → Execute → Verify)
- RAG over AST-derived code chunks
- LoRA-adapted local coder LLMs
- Deterministic Scikit-Learn error routing
- Docker sandboxes for verifiable test execution

---

## 2. Locked decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo host | Public GitHub `hugobelebinda/rebirth-ai` | User choice (1A) |
| First language adapter | JavaScript / TypeScript | Concrete E2E path; core stays pluggable |
| Multi-language | Yes — adapters, not a rewrite | User does **not** want “only one forever” |
| Inference (MVP) | Ollama + **AMD ROCm** on RX 7800 XT | `qwen2.5-coder:14b-instruct-q6_K` (~12GB) fully on 16GB VRAM; see `docs/OLLAMA_AMD_GPU.md` |
| Vector store (MVP) | SQLite (local) | No AWS/Supabase required for Phases 1–4 |
| Cloud GPU / AWS | Deferred | 30GB RAM machine; train small LoRA or use HF Jobs later |
| Max fix iterations | 5 then human review | Avoid infinite LLM loops |

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                      │
│  retrieve → generate → execute_tests → classify → route / halt   │
└─────────┬───────────┬────────────┬──────────────┬───────────────┘
          │           │            │              │
          ▼           ▼            ▼              ▼
     RAG (AST +   Ollama +     Docker        Sklearn
     embeddings)  LoRA         sandbox       error classes
          │           │            │              │
          └───────────┴────────────┴──────────────┘
                        Language Adapter
              (JS/TS first → Python later → …)
```

### State object (target — Phase 4)

```python
class MigrationState(TypedDict):
    file_path: str
    language: str
    original_code: str
    current_draft: str
    retrieved_context: list[str]
    test_logs: str
    error_class: str | None
    iteration_count: int
    status: str  # pending | running | success | human_review | failed
```

### Error classes (target — Phase 2)

- `SyntaxError`
- `DependencyMissing`
- `TypeMismatch`
- `TestLogicFailure`

### Routing rules (target — Phase 4)

| Condition | Next node |
|-----------|-----------|
| Tests pass | `success` |
| `DependencyMissing` | `install_dependency` (no LLM) |
| Other error classes & `iteration_count ≤ 5` | `generate_code` with error context |
| `iteration_count > 5` | `halt_human_review` |

---

## 4. Repository layout

```
rebirth-ai/
  docs/
    GLOBAL.md                 ← this file
    phases/PHASE_*.md         ← ultra-detailed per-phase logs
  src/rebirth_ai/
    adapters/                 # language plugins
    ast/                      # Tree-sitter
    rag/                      # embeddings + retrieve
    classify/                 # TF-IDF + RandomForest
    generate/                 # Ollama / LoRA
    sandbox/                  # Docker SDK
    graph/                    # LangGraph
    github/                   # PR automation
    cli.py
  fixtures/js_ts_sample/      # tiny migration playground (Phase 1)
  data/                       # datasets, embeddings DB, model artifacts
  training/                   # LoRA scripts (Phase 3)
  docker/                     # sandbox Dockerfiles (Phase 1)
  tests/
```

---

## 5. Technical stack

| Layer | Technology |
|-------|------------|
| Agent logic | Python 3.11+, Typer CLI |
| Orchestration | LangGraph / LangChain (Phase 4) |
| Parsing | Tree-sitter JS/TS |
| Embeddings | SentenceTransformers (`microsoft/unixcoder-base` candidate) |
| Vectors | SQLite (+ sqlite-vec or numpy cosine MVP) |
| Classifier | `TfidfVectorizer` + `RandomForestClassifier` |
| LLM | Ollama (`qwen2.5-coder:14b-instruct-q6_K`) + PEFT LoRA |
| Sandbox | Docker Python SDK |
| Hosting / PRs | GitHub API (`gh` / PyGithub) |

---

## 6. Prerequisites checklist (machine state)

| Item | Required MVP? | Observed on this machine (Phase 0) |
|------|---------------|-------------------------------------|
| GitHub + `gh` auth | Yes | Logged in as `hugobelebinda` (keyring); scopes gist, read:org, repo, workflow |
| Docker Engine | Yes | Daemon reachable with host permissions; user in `docker` group |
| Ollama + coder model | Phase 3+ | **ROCm enabled** on RX 7800 XT; default `qwen2.5-coder:14b-instruct-q6_K` (~12GB, 100% GPU) |
| Node.js 20+ | Yes (JS/TS) | Node `v22.22.2` via nvm |
| Python 3.11+ | Yes | Python `3.12.3` |
| Hugging Face account | Recommended | Optional until LoRA train / Hub push |
| NVIDIA GPU | No | Not required for MVP |
| AWS | No | Deferred |
| Supabase | No | Deferred |

### Pre-flight commands

```bash
gh auth status
docker info
ollama list
node -v && npm -v
python3 --version
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
rebirth doctor
```

---

## 7. Phase status board

| Phase | Doc | Status | Exit criteria (summary) |
|-------|-----|--------|-------------------------|
| 0 Setup | [PHASE_0_SETUP.md](phases/PHASE_0_SETUP.md) | **Complete** | Public repo live; scaffold + docs; doctor CLI; prereqs documented |
| 1 Sandbox + AST | [PHASE_1_SANDBOX_AST.md](phases/PHASE_1_SANDBOX_AST.md) | Planned | Sandbox round-trip + AST chunks for ≥3 fixtures |
| 2 RAG + Classifier | [PHASE_2_RAG_CLASSIFIER.md](phases/PHASE_2_RAG_CLASSIFIER.md) | Planned | Retrieval demo + classifier on synthetic logs |
| 3 LoRA + Inference | [PHASE_3_LORA_INFERENCE.md](phases/PHASE_3_LORA_INFERENCE.md) | Planned | Train path + Ollama draft generation |
| 4 LangGraph | [PHASE_4_LANGGRAPH.md](phases/PHASE_4_LANGGRAPH.md) | Planned | Single-file fail→classify→rewrite→pass |
| 5 Repo scale | [PHASE_5_REPO_SCALE.md](phases/PHASE_5_REPO_SCALE.md) | Planned | Multi-file + open PR |

---

## 8. Decision log

| Date | Decision | Notes |
|------|----------|-------|
| 2026-10-31 | Public GitHub repo | User: option A |
| 2026-10-31 | Pluggable languages; JS/TS first vertical | User: not locked to one language forever |
| 2026-10-31 | Local-first (Ollama + Docker + SQLite) | No AWS for Phases 1–4 |
| 2026-10-31 | Nested git root under `Rebirth AI/` | Parent `Personal Project` has multi-project `.git`; this repo is independent |
| 2026-10-31 | Phase 0 complete | Public repo pushed; `rebirth doctor` + smoke test green; Ollama/Docker/Node verified |
| 2026-10-31 | AMD ROCm for Ollama | Service drop-in prefers `rocm_v7_2`; discrete RX 7800 XT (`gfx1101`) selected; iGPU dropped |
| 2026-10-31 | Default LLM | `qwen2.5-coder:14b` Q4 — initial fit for 16GB VRAM |
| 2026-10-31 | Upgrade LLM | `qwen2.5-coder:14b-instruct-q6_K` — higher fidelity, still fully on VRAM |

---

## 9. Documentation update protocol

After **each** phase completes:

1. Fill that phase’s `docs/phases/PHASE_N_*.md` with: What shipped, contracts/APIs, how to run, sample outputs, known gaps, decision log, exit-criteria checklist.  
2. Sync this `GLOBAL.md`: status board, decision log, architecture deltas, observed machine state.  
3. Bump README status table.  
4. Prefer evidence (commands + paths) over summaries.

---

## 10. Open risks / watch list

- Docker socket may be invisible inside restricted sandboxes — run Docker-dependent work with full host permissions.  
- LoRA training on CPU will be slow — keep dataset small for MVP; consider HF Jobs if needed.  
- `unixcoder-base` may be heavy on CPU — benchmark; fall back to lighter code embedding if needed.  
- Nested repo under parent `Personal Project` git tree: always operate git commands from `Rebirth AI/` with its own `.git`.
