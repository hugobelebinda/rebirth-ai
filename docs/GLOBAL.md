# Rebirth AI — Global System Bible

> Living document. **Update after every phase** and whenever chat context hits ~90% (full `.md` refresh).  
> Last updated: 2026-11-05 (Phases **0–11 complete**; **now = Phase 12** local tuning v3; phases renumbered in execution order — [map](PRODUCTION_ROADMAP.md#phase-renumbering-2026-11-05)).  
> Repo: https://github.com/hugobelebinda/rebirth-ai  
> Roadmap: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md)  
> **Handoff for new agents:** [SESSION_HANDOFF.md](SESSION_HANDOFF.md)

---

## 0. Agent restart — current step

**Phases 0–10 are complete on `main`.** All eight adapters live; corpus **24 640** (≥15K); fixture eval **8/8**; quality CLI (`report`, `eval --suite fixtures`, `data/runs/`) shipped.

**Phase 11 complete (2026-11-05):** recipe-v2 4-bit QLoRA @512 adapters for all 8 languages; all pass their gate vs base 7B and beat/tie Ollama 14B on held-out soft-exact — [PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md).  
**In progress — Phase 12:** [PHASE_12_LOCAL_TUNING_V3.md](phases/PHASE_12_LOCAL_TUNING_V3.md) (expanded corpus, frozen 100-row eval sets, recipe experiments, v3 wave). GPU queue only runs when the user says go.  
**Phase 13 complete (2026-11-05):** MLflow tracking + registry + YAML recipes ([PHASE_13](phases/PHASE_13_MLOPS_TRACKING_REGISTRY.md)).  
**Then:** Phase 14 (MLOps repro + infra) → 15 (cloud 14B, waits on AWS quota) → 16 (model CI/CD) → 17 (cross-language) → 18 (monitoring) → 19 (v1.0 launch).

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
| **v1.0 language set** | JS/TS, Python, Go, Java, C++, C#, Rust, Swift | Locked 2026-11-02; see roadmap language board |
| **v1.0 corpus target** | ≥15 000 mined migration pairs across those languages | **Met: 24 640** (`corpus_manifest.json`) |
| Fixture quality bar | ≥90% sandbox-verified success per language (held-out fixtures) | Launch gate in Phases 10–11 |
| Inference (MVP) | Ollama + **AMD ROCm** on RX 7800 XT | `qwen2.5-coder:14b-instruct-q6_K` (~12GB) fully on 16GB VRAM; see `docs/OLLAMA_AMD_GPU.md` |
| Vector store (MVP) | SQLite (local) | No AWS/Supabase required for Phases 1–4 |
| Cloud GPU / AWS | **Optional** (Track C) | Not required for Phases 6–11; HF Jobs for optional 14B QLoRA only |
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
                        Language Adapter registry
         JS/TS + Python + Go + Java + C# + Rust + C++ + Swift (all live)
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
| Parsing | Tree-sitter for all 8 v1.0 languages |
| Embeddings | SentenceTransformers (`st-codesearch-distilroberta-base`; hash embedder for tests) |
| Vectors | SQLite BLOBs + numpy cosine |
| Classifier | `TfidfVectorizer` + `LogisticRegression` (held-out acc **0.96**, 100 logs) |
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
| Data (interlude) | [DATASET.md](DATASET.md) | **Complete (corpus)** | **24640** multi-lang; JS/TS 7B LoRA proof; v2 LoRA per language in Phase 11 |
| 5 Repo scale | [PHASE_5_REPO_SCALE.md](phases/PHASE_5_REPO_SCALE.md) | **Complete** | `migrate-repo` + GitHub PR #1 (multi-file E2E) |
| 6 Repo hardening | [PHASE_6_REPO_HARDENING.md](phases/PHASE_6_REPO_HARDENING.md) | **Complete (100%)** | Topo, dry-run, VRAM/PEFT doctor, test matrix, gitignore `!` |
| 7 Real-world JS | [PHASE_7_REAL_WORLD_JS.md](phases/PHASE_7_REAL_WORLD_JS.md) | **Complete (100%)** | Full golden matrix + Express install path + held-out 12/12 |
| 8 Adapter factory + Python | [PHASE_8_PYTHON.md](phases/PHASE_8_PYTHON.md) | **Complete** | Registry + pytest sandbox; Python vertical + mine |
| 9 Full language wave | [PHASE_9_MULTI_LANG_QUALITY.md](phases/PHASE_9_MULTI_LANG_QUALITY.md) | **Complete** | Go/Java/C#/Rust/**C++/Swift** adapters + fixtures + mines |
| 10 Quality + 15K | [PHASE_10_RELEASE.md](phases/PHASE_10_RELEASE.md) | **Complete** | Fixtures **8/8**; report+runs; corpus **24640** |
| 11 Per-lang LoRA | [PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md) | **Complete** (2026-11-05) | Harness + v2 7B adapters for all 8 languages, gated vs base 7B |
| **12 Local tuning v3** | [PHASE_12_LOCAL_TUNING_V3.md](phases/PHASE_12_LOCAL_TUNING_V3.md) | **In progress** | Frozen eval sets; recipe decided; v3 wave; GGUF→Ollama adapter test |
| 13 MLOps 1 — tracking + registry | [PHASE_13_MLOPS_TRACKING_REGISTRY.md](phases/PHASE_13_MLOPS_TRACKING_REGISTRY.md) | **Complete** (2026-11-05) | MLflow runs + registry (8 champions) + YAML recipes; `rebirth track/registry/mlflow-ui` |
| 14 MLOps 2 — repro + infra | [PHASE_14_MLOPS_REPRO_INFRA.md](phases/PHASE_14_MLOPS_REPRO_INFRA.md) | Planned | DVC + S3; Docker CUDA image; Terraform; cloud scripts; vLLM eval |
| 15 Cloud 14B adapters | [PHASE_15_CLOUD_14B.md](phases/PHASE_15_CLOUD_14B.md) | Planned (AWS quota) | 14B per-lang adapters + DPO on g6e.xlarge — [CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md) |
| 16 MLOps 3 — CI/CD + packaging | [PHASE_16_MLOPS_CICD_PACKAGING.md](phases/PHASE_16_MLOPS_CICD_PACKAGING.md) | Planned | CI gates, champion/challenger, model cards, Ollama adapters, rollback |
| 17 Cross-language | [PHASE_17_CROSS_LANGUAGE.md](phases/PHASE_17_CROSS_LANGUAGE.md) | Planned | `--to <lang>`, equivalence harness, unified adapter |
| 18 MLOps 4 — monitoring | [PHASE_18_MLOPS_MONITORING.md](phases/PHASE_18_MLOPS_MONITORING.md) | Planned | Run telemetry dashboard, drift, data flywheel |
| 19 Official launch | [PHASE_19_LAUNCH.md](phases/PHASE_19_LAUNCH.md) | Planned (last) | Bootstrap + docs + **v1.0.0** |

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
| 2026-11-01 | PEFT defaults → 7B | Auto-resolve 7B adapter; `--backend peft` closed-loop math **PASS** (1 iter). Avoid ST+7B VRAM stack (killed Cursor once). |
| 2026-11-01 | Behavioral eval | `eval_behavioral.py`: sandbox-pass + AST/soft-exact. PEFT 7B sandbox **2/2**; synth soft-exact **6/8**. |
| 2026-11-01 | Production roadmap | Phases 6–10 + optional cloud LoRA; multi-language via adapters; cloud GPU **not** required |
| 2026-11-02 | v1.0 language + corpus lock | Target adapters: JS/TS, Python, Go, Java, C++, C#, Rust, Swift; corpus **≥15K** pairs; Phases **6–11**; optional `v0.3` after Python only |
| 2026-11-03 | Phase 6 complete | Import topo-order; `--under` multi-root; gitignore/deny; `--dry-run`; embed dim guard; PEFT chat+hash for agent; doctor VRAM warn; PR `--draft` + auto base; Ollama sandbox **2/2** |
| 2026-11-03 | Phase 7 complete | Bench util/mixed + external clone; held-out **12/12**; baselines frozen; safety module; classifier **0.96** / 100 logs |
| 2026-11-03 | Phases 6–7 100% closeout | Express DependencyMissing→install `ms`; monorepo under; external-style migrate; classify override+telemetry; test matrix; commit `ca1c34c` + PR #2 + tag `v0.2.0-mvp` |
| 2026-11-03 | Session handoff | Docs refresh; **next action = merge PR #2**, then Phase 8 |
| 2026-11-03 | PR #2 merged | Phases 6–7 on `main` (`5f1540e`) |
| 2026-11-03 | Phase 8 complete | Registry + PythonAdapter + sandbox; migrate 1 iter; **1503** Python pairs |
| 2026-11-03 | Phase 9 complete | Folded C++/Swift into wave; **8** adapters live; corpus ≈**3244** |
| 2026-11-04 | Phase 10 complete | `eval --suite fixtures` 8/8; `report` + `data/runs/`; corpus **24640** |
| 2026-11-04 | LoRA before launch | Insert Phase **11** (per-lang LoRA); **defer Phase 19** until done |
| 2026-11-04 | Phase 11.1 harness | `train_lora.py --lang`; hash splits (5%, seed 1337) in `data/datasets/splits/`; `rebirth_meta.json`; PEFT routes by `adapter_id`, Ollama fallback when a lang has no adapter |
| 2026-11-04 | 4-bit QLoRA validated | bitsandbytes ROCm: NF4 train OK @512 (8.75 GB); NF4 cached generate broken → train 4-bit, infer bf16; recipe v2 |
| 2026-11-04 | Go seq-256 test | GPU reset at step 1 (memory pressure, not thermal); base-7B ablation shows seq-128 LoRA adds nothing → fix recipe / free VRAM before more trains |
| 2026-11-04 | Python 7B LoRA | 624 steps, loss 1.121; fixture PEFT PASS 1 iter; held-out soft-exact 4/24 vs Ollama 8/24 → experimental; decode leak fix (fence + `<|…|>` tokens) |

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
- Seq **512** bf16 7B **OOMs** on 16GB with compositor; seq **256 → MES timeout + GPU MODE1 reset at step 1** (2026-11-04, `logs/train/go_seq256_*`); seq **128** is the only proven budget — and it truncates the target on most rows (PHASE_11).  
- bf16 7B is capped at 128; use `--quant 4bit` for 512 (inference stays bf16 — ROCm NF4 cached decode is broken).  
- Train only via `scripts/train_logged.sh` (fsync'd step + GPU telemetry logs); after a hang run `scripts/gpu_hang_postmortem.sh`.  
- Nested repo under parent `Personal Project` git tree: always operate git from `Rebirth AI/` with its own `.git`.  
- Bind-mount + container writes can change host ownership — sandbox uses temp copies (Phase 1).  
- Agent default is still **stock Ollama 14B** — 7B LoRA is PEFT-in-process only (not merged into Ollama).  
- Exact-match PEFT smoke **understates** 7B quality (whitespace / rename / `!!` vs `Boolean`).

## 11. Next steps (production path — local-first)

Full detail: **[PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md)** · LoRA plan: **[PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md)**.

| Now | Do this |
|-----|---------|
| **Now** | **Phase 12** — local tuning v3 (v3 wave when the user says go; GGUF→Ollama adapter test) |
| ✅ | **Phase 13** — MLOps 1: MLflow tracking + model registry + YAML recipes (done 2026-11-05) |
| **3** | **Phase 14** — MLOps 2: DVC + S3, Docker CUDA image, Terraform, cloud scripts, vLLM eval |
| **4** | **Phase 15** — cloud 14B per-language adapters + DPO (when the AWS quota is approved) — [CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md) |
| **5** | **Phase 16** — MLOps 3: model CI/CD + Ollama packaging + rollback |
| **6** | **Phase 17** — cross-language migration (any → any of 8) + unified adapter |
| **7** | **Phase 18** — MLOps 4: monitoring + data flywheel |
| **8** | **Phase 19** — bootstrap + docs + **v1.0.0** |

End product: one offline tool — modernize in-language **and** migrate between any of the 8 languages ([CLOUD_GPU_PLAN.md §7](CLOUD_GPU_PLAN.md#7-the-final-product)).
