# Rebirth AI — Global System Bible

> Living document. **Update after every phase** and whenever chat context hits ~90% (full `.md` refresh).  
> Last updated: 2026-11-01 (Phase 5 complete — migrate-repo + PR #1).  
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
| Cloud GPU / AWS | Deferred for agent; optional for heavier LoRA | Local 7B bf16 LoRA works; multi-epoch hung dGPU — HF Jobs for 14B QLoRA |
| Max fix iterations | 5 then human review | Avoid infinite LLM loops |
| Local PEFT default quality | Prefer **7B** adapter over 0.5B | 5/8 exact / ~0.96 overlap; exact understates quality |

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                      │
│  retrieve → generate → execute_tests → classify → route / halt   │
└─────────┬───────────┬────────────┬──────────────┬───────────────┘
          │           │            │              │
          ▼           ▼            ▼              ▼
     RAG (AST +   Ollama 14B   Docker        Sklearn
     embeddings)  or PEFT 7B   sandbox       error classes
          │           │            │              │
          └───────────┴────────────┴──────────────┘
                        Language Adapter
              (JS/TS first → Python later → …)
```

### State object (Phase 4 — live)

```python
class MigrationState(TypedDict, total=False):
    file_path: str
    project_dir: str
    relative_file: str
    relative_test: str
    language: str
    objective: str
    original_code: str
    current_draft: str
    retrieved_context: list[str]
    test_logs: str
    error_class: str | None
    iteration_count: int
    max_iterations: int
    status: str  # pending | running | success | human_review | failed
    trace: list[str]
    extra_packages: list[str]
    test_first: bool
    db_path: str
```

### Error classes (Phase 2 — live)

- `SyntaxError`
- `DependencyMissing`
- `TypeMismatch`
- `TestLogicFailure`

### Routing rules (Phase 4 — live)

| Condition | Next node |
|-----------|-----------|
| Tests pass | `success` |
| `DependencyMissing` | `install_dependency` (queue pkg; re-test with network) |
| Other error classes & `iteration_count < max` | `generate_code` with error context |
| `iteration_count >= max` (default 5) | `halt_human_review` |

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
| Orchestration | LangGraph / LangChain (`rebirth migrate`) |
| Parsing | Tree-sitter JS/TS |
| Embeddings | SentenceTransformers (`st-codesearch-distilroberta-base`; hash embedder for tests) |
| Vectors | SQLite BLOBs + numpy cosine |
| Classifier | `TfidfVectorizer` + `LogisticRegression` (held-out acc ≈ 0.95) |
| LLM | Ollama (`qwen2.5-coder:14b-instruct-q6_K`) default; PEFT LoRA (`--backend peft`, 7B preferred) |
| Train GPU | AMD ROCm 6.4.4 + torch `2.9.1+rocm6.4` on RX 7800 XT |
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
| Hugging Face account | Recommended | Used for CommitPackFT mine + optional Jobs |
| System ROCm + PyTorch HIP | For local 7B LoRA | Installed/verified 2026-11-01 (`scripts/verify_rocm_torch.sh`) |
| NVIDIA GPU | No for MVP | Needed only for local QLoRA / bitsandbytes |
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
| 1 Sandbox + AST | [PHASE_1_SANDBOX_AST.md](phases/PHASE_1_SANDBOX_AST.md) | **Complete** | Sandbox pass/fail + AST chunks verified |
| 2 RAG + Classifier | [PHASE_2_RAG_CLASSIFIER.md](phases/PHASE_2_RAG_CLASSIFIER.md) | **Complete** | SQLite RAG + classifier acc 0.947; CLI index/query/classify |
| 3 LoRA + Inference | [PHASE_3_LORA_INFERENCE.md](phases/PHASE_3_LORA_INFERENCE.md) | **Complete** | Ollama + PEFT path; dataset mined; 0.5B + **7B** adapters |
| 4 LangGraph | [PHASE_4_LANGGRAPH.md](phases/PHASE_4_LANGGRAPH.md) | **Complete** | Single-file fail→classify→rewrite→pass (1 iter E2E) |
| Data (interlude) | [DATASET.md](DATASET.md) | **Active** | 1339 pairs; 7B 1-epoch PASS; stop/decode; no multi-epoch retry |
| 5 Repo scale | [PHASE_5_REPO_SCALE.md](phases/PHASE_5_REPO_SCALE.md) | **Complete** | `migrate-repo` + GitHub PR #1 (multi-file E2E) |

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
| 2026-10-31 | Docs @ 90% context | Always re-read/update every project `.md` (`.cursor/rules/docs-context-refresh.mdc`) |
| 2026-10-31 | Sandbox overlay safety | Never `put_archive` onto bind-mounted project; copytree to tempdir first |
| 2026-10-31 | Phase 2 complete | Hash/ST RAG in SQLite; LogisticRegression error classifier (~95% held-out) |
| 2026-10-31 | Phase 3 complete | Ollama migrate client on Q6_K; LoRA smoke on 0.5B; 59 migration pairs |
| 2026-10-31 | Phase 4 complete | LangGraph closed loop; `rebirth migrate` fixed broken multiply in 1 iteration |
| 2026-10-31 | Web dataset mining | CommitPackFT JS/TS → 1280 real pairs + 59 synthetic = 1339; LoRA 0.5B on 128 ex; classifier acc ≈0.957 |
| 2026-11-01 | PEFT runtime path | `REBIRTH_GENERATOR=peft` / `--backend peft`; full-dataset 0.5B LoRA train kicked off |
| 2026-11-01 | ROCm 7B LoRA smoke | `Qwen2.5-Coder-7B-Instruct` bf16, 5 steps, seq 128 on RX 7800 XT — **PASS** (seq 512 OOM); adapter `data/models/lora_js_ts_7b_smoke` |
| 2026-11-01 | ROCm 7B LoRA 1-epoch | Full 1339 pairs, seq 128, 168 steps, mean loss 1.278 — **PASS**; adapter `data/models/lora_js_ts_7b` |
| 2026-11-01 | ROCm 7B 2-epoch hang | r=16 / 2 epochs hung GPU at step 0 (`GPU Hang`); aborted. Keep 1-epoch adapter; no immediate local retry |
| 2026-11-01 | PEFT stop/decode | `stop_strings` + SFT token cap + leak strip in `peft_client.py`; unit tests pass; 7B eval still 5/8 (semantic misses) |
| 2026-11-01 | Phase 5 complete | `rebirth migrate-repo` + `github/` PR helpers; fixture format tests; E2E 2/2; PR https://github.com/hugobelebinda/rebirth-ai/pull/1 |

---

## 9. Documentation update protocol

After **each** phase completes:

1. Fill that phase’s `docs/phases/PHASE_N_*.md` with: What shipped, contracts/APIs, how to run, sample outputs, known gaps, decision log, exit-criteria checklist.  
2. Sync this `GLOBAL.md`: status board, decision log, architecture deltas, observed machine state.  
3. Bump README status table.  
4. Prefer evidence (commands + paths) over summaries.

**Also** whenever chat context is ≈90% full (or about to compact): re-read **all** project `.md` files and update them to match live reality — rule file: `.cursor/rules/docs-context-refresh.mdc`.

---

## 10. Open risks / watch list

- Docker socket may be invisible inside restricted sandboxes — run Docker-dependent work with full host permissions.  
- **Display on RX 7800 XT:** long/heavier PyTorch LoRA can **GPU Hang** the session (seen on 2-epoch r=16). Prefer short 1-epoch seq≤128 or cloud CUDA.  
- Seq **512** bf16 7B **OOMs** on 16GB with compositor; seq **128** is the proven budget.  
- Nested repo under parent `Personal Project` git tree: always operate git from `Rebirth AI/` with its own `.git`.  
- Bind-mount + container writes can change host ownership — sandbox uses temp copies (Phase 1).  
- Agent default is still **stock Ollama 14B** — 7B LoRA is PEFT-in-process only (not merged into Ollama).  
- Exact-match PEFT smoke **understates** 7B quality (whitespace / rename / `!!` vs `Boolean`).

## 11. Recommended next steps (post Phase 5)

1. **Wire PEFT defaults to 7B** + closed-loop `--backend peft` smoke.  
2. **Better eval** — sandbox-pass / AST metric (not exact string).  
3. **Cloud 14B QLoRA** (~$1–3) matching Ollama runtime, then merge/export.  
4. Optional: Python language adapter; harden `migrate-repo` dependency ordering.  
5. Avoid local multi-epoch 7B until display moves off dGPU or stack is more stable.
