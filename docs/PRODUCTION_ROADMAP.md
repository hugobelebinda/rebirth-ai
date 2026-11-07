# Rebirth AI — Production-Grade Roadmap

> Last updated: 2026-11-07 (Phases **0–14 complete**, 16 offline done; Phase 15 local prep done, cloud part blocked on AWS quota; phases renumbered in execution order — see Phase map)  
> Scope: take the **local-first JS/TS MVP** (Phases 0–5 + data interlude) to a **production-grade, multi-language** migration product.  
> Constraint: **cloud GPU is optional** — required only for optional 14B LoRA specialization. Everything else runs on this machine (Ollama 14B + Docker + local PEFT 7B).  
> **New agent:** start at [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

---

## 0. Where we are (baseline)

| Capability | Status | Evidence |
|------------|--------|----------|
| Single-file closed loop | Done | `rebirth migrate` + LangGraph |
| Multi-file + GitHub PR | Done | `rebirth migrate-repo`, [PR #1](https://github.com/hugobelebinda/rebirth-ai/pull/1) |
| Repo hardening | **Done (Phase 6)** | Topo-order, dry-run, VRAM doctor, draft PR |
| Real-world JS/TS | **Done (Phase 7)** | Held-out 12/12; full golden matrix; baseline freeze |
| Landing on `main` | **Done** | [PR #2](https://github.com/hugobelebinda/rebirth-ai/pull/2) **merged** |
| Tag | `v0.2.0-mvp` on origin | JS/TS MVP; optional `v0.3.0-js-python-preview` after Phase 8 commit |
| Language verticals | **All 8 v1.0 langs** | Registry + sandboxes + fixtures |
| Agent brain | Ollama `qwen2.5-coder:14b-instruct-q6_K` | Default generator |
| Local specialization | PEFT 7B LoRA (1-epoch, **JS/TS only** so far) | `data/models/lora_js_ts_7b` — Phase **11** extend per lang |
| Eval | Sandbox-pass + AST/soft-exact + fixtures 8/8 | `training/eval_behavioral.py`, `eval_fixture_sandboxes.py` |
| Dataset | **24640** multi-lang (**≥15K met**) | `data/datasets/corpus_manifest.json` |

**Target language set (locked for v1.0):**

| # | Language | Adapter id | Sandbox sketch | Phase |
|---|----------|------------|----------------|-------|
| 1 | JavaScript / TypeScript | `js_ts` | Node / npm test | 0–7 (live) |
| 2 | Python | `python` | pytest | **8 (live)** |
| 3 | Go | `go` | `go test` | **9 (live)** |
| 4 | Java | `java` | Maven/Gradle + JUnit | **9 (live)** |
| 5 | C# | `csharp` | `dotnet test` | **9 (live)** |
| 6 | Rust | `rust` | `cargo test` | **9 (live)** |
| 7 | C++ | `cpp` | CMake + ctest / gtest | **9 (live)** |
| 8 | Swift | `swift` | `swift test` | **9 (live)** |

**Production definition (this roadmap):**

1. Reliably migrates **real** repositories (not only fixtures).  
2. Supports **all 8 target languages** via `LanguageAdapter` (not a rewrite per language).  
3. Training / RAG corpus reaches **≥15K** mined migration pairs across those languages.  
4. Deterministic verify loop (sandbox tests) with clear failure taxonomy.  
5. Safe GitHub automation (branch / PR / partial success).  
6. Measurable quality bars (behavioral eval + held-out suites; target **≥90%** sandbox-verified success on held-out fixtures per language).  
7. Documented ops (VRAM rules, doctor checks, release checklist).  
8. Optional: specialized LoRA merged or PEFT-wired without breaking defaults.

**Intermediate milestone:** after Phase 8, a **JS/TS + Python production-preview** tag is allowed. **Official v1.0 launch** requires Phases 9–11 (full language set + 15K+ pairs).

---

## Guiding principles

1. **Local-first** — ship features without Hugging Face Jobs / paid GPU.  
2. **Adapter-shaped** — every language = `LanguageAdapter` + Docker image + fixture + behavioral case + mined pairs + docs.  
3. **Verify > exact string** — sandbox-pass and AST/API metrics beat token equality.  
4. **Fail soft** — multi-file runs continue; PR body lists failures.  
5. **One GPU owner** — never stack Ollama 14B + PEFT 7B + SentenceTransformers on the 7800 XT.  
6. **Cloud GPU = optional brain upgrade** — card in **browser** billing settings, not the terminal.  
7. **Repeatable vertical** — Phase 8 locks the adapter playbook; Phases 9–10 only instantiate it.

---

## Phase map (post-MVP)

```
Phase 6    Repo hardening (JS/TS production path)     ✅
Phase 7    Real-world JS/TS validation + held-out     ✅
Phase 8    Adapter factory + Python vertical          ✅
Phase 9    Full language wave (Go…Swift)              ✅
Phase 10   Quality gates + ≥15K corpus                ✅
Phase 11   Per-language LoRA — harness + v2 7B adapters (all 8)          ✅ 2026-11-05
Phase 12   Local tuning v3 + eval hardening                               ✅ 2026-11-06
Phase 13   MLOps 1 — experiment tracking + model registry + YAML recipes   ✅ 2026-11-05
Phase 14   MLOps 2 — reproducible pipelines + infra (DVC/S3, Terraform, vLLM)   ✅ 2026-11-05
Phase 15   Cloud — 14B per-language adapters + DPO (AWS g6e.xlarge)    local prep done; waits on AWS quota
Phase 16   MLOps 3 — model CI/CD + packaging (Ollama adapters, rollback)
Phase 17   Cross-language migration (any → any of 8) + unified adapter
Phase 18   MLOps 4 — monitoring + data flywheel
Phase 19   Packaging, UX, official v1.0 launch
```

Phases 0–11 shipped. Numbers now follow **execution order**. Phases 13–14 come before the cloud on
purpose: every paid GPU hour then produces a tracked, reproducible run. If the AWS quota is approved
early, Phase 15 may start once Phase 14's cloud scripts exist.

### Phase renumbering (2026-11-05)

| Old | New | Phase |
|-----|-----|-------|
| 12, 13, 14 | **11** | Per-language LoRA (harness, wave A, wave B) — complete |
| — | **12** | Local tuning v3 + eval hardening (new) |
| — | **13** | MLOps 1 — tracking + registry (new) |
| — | **14** | MLOps 2 — repro + infra (new) |
| "Cloud GPU" | **15** | Cloud 14B adapters ([CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md) S0–S4) |
| — | **16** | MLOps 3 — CI/CD + packaging (new) |
| 15 | **17** | Cross-language migration |
| — | **18** | MLOps 4 — monitoring (new) |
| 11 | **19** | Packaging, UX, v1.0 launch |

Code comments, docs and fixture READMEs use the new numbers. Older commit messages and reports may still use the old ones.

---

## Phase 6 — Repo hardening (JS/TS production path)

**Goal:** Make `migrate-repo` trustworthy on non-toy trees without adding languages yet.

### 6.1 Git / product hygiene

| Task | Detail | Exit |
|------|--------|------|
| Merge PR #1 | Land Phase 5 on `main` | PR merged |
| Sync local WIP | PEFT defaults, behavioral eval, docs — commit or PR | `main` matches reality |
| Release tag | Optional `v0.2.0-mvp` after merge | Tag exists |

### 6.2 Discovery & dependency order

| Task | Detail | Exit |
|------|--------|------|
| Import graph | Parse imports (Tree-sitter) → topological order for migrate | Leaves before dependents |
| Configurable roots | `--under src,lib,app` multi-root | Documented CLI |
| Ignore rules | Honor `.gitignore` + Rebirth deny list (`node_modules`, `dist`, …) | Unit tests |
| Test mapping | Expand heuristics: `*.spec.js`, co-located `__tests__`, `package.json` `test` script fallback with warning | Documented matrix |

### 6.3 RAG under PEFT / VRAM policy

| Task | Detail | Exit |
|------|--------|------|
| Hash vs ST policy | PEFT → hash (or CPU ST); Ollama → ST OK if VRAM free | Encoded in factory/CLI |
| Doctor check | Warn if Ollama model resident + PEFT requested | `rebirth doctor` line |
| Embed DB versioning | Record embedder dim in DB meta; refuse mismatched query | Clear error, not silent skip |

### 6.4 Generation robustness

| Task | Detail | Exit |
|------|--------|------|
| Whole-file = chat | Agent/migrate always chat for multi-hundred-line files | No SFT truncation on sandbox cases |
| Stop / decode | Keep leak strip; add max tokens by file size | Unit tests |
| Idempotent writes | Diff preview; `--dry-run` shows patch without write | CLI flag |

### 6.5 PR UX

| Task | Detail | Exit |
|------|--------|------|
| PR template | Summary, per-file status, sandbox logs excerpt | `open_migration_pr` |
| Draft PRs | `--draft` flag | `gh pr create --draft` |
| Base detection | Auto-detect default branch (`main`/`master`) | Works on both |

**Phase 6 exit checklist**

- [x] PR #1 merged (2026-11-02)  
- [x] `migrate-repo` topo-orders by imports on fixture (`format` → `math`)  
- [x] `--dry-run` works  
- [x] Doctor warns on VRAM stacking / PEFT+Ollama  
- [x] Behavioral sandbox still 2/2 on Ollama  
- [x] Test mapping matrix + gitignore `!` negation  
- [x] Closeout commit + PR #2 opened + tag `v0.2.0-mvp` pushed  
- [x] PR #2 merged into `main` (`5f1540e`, 2026-11-03)  

**Docs:** [phases/PHASE_6_REPO_HARDENING.md](phases/PHASE_6_REPO_HARDENING.md) — **COMPLETE (100%)**

---

## Phase 7 — Real-world JS/TS validation

**Goal:** Prove the product on repositories that look like customer code; freeze the JS/TS quality bar other languages must match.

### 7.1 Golden repos (local clones)

| Repo class | Example targets | Success metric |
|------------|-----------------|----------------|
| Tiny util lib | 5–20 files, Node test | ≥80% files sandbox-green or clean human_review |
| App with deps | Express/Fastify sample | DependencyMissing path installs ≥1 pkg |
| Mixed JS/TS | `.ts` + `.js` | Both suffixes migrate |
| Monorepo lite | `packages/a` + `packages/b` | `--under` scoped run |

Store clones under `data/bench/` (gitignored) or document external paths.

### 7.2 Held-out eval suite

| Task | Detail | Exit |
|------|--------|------|
| Fixture pack | 10+ broken seeds with known tests | `fixtures/eval_suite/` |
| CI job (local script) | `python training/eval_behavioral.py` + suite JSON | Exit non-zero on regression |
| Baseline freeze | Record Ollama 14B sandbox rates | `data/reports/baselines/` |

### 7.3 Classifier & logs

| Task | Detail | Exit |
|------|--------|------|
| Real CI logs | Expand `error_logs.jsonl` from open-source failures | Acc ≥0.90 held-out |
| Unknown class | Route unknown → generate with raw logs (already) + telemetry | Trace field |

### 7.4 Safety / product rules

| Task | Detail | Exit |
|------|--------|------|
| Secret scan | Refuse to commit `.env`, keys in PR paths | Pre-commit / PR helper |
| Max file size | Skip / chunk files > N KB with message | Configurable |
| License note | Skip files with restrictive headers if flagged | Optional filter |

**Phase 7 exit checklist**

- [x] Tiny util + Express **DependencyMissing install** + mixed JS/TS + monorepo `--under`  
- [x] External-style full migrate (`bench_external_plain`) + real clone dry-run (`is-plain-obj`)  
- [x] Held-out suite green on Ollama (**12/12**) + baseline freeze  
- [x] Safety filters + classifier ≥0.90 + Unknown/low-confidence telemetry  

**Docs:** [phases/PHASE_7_REAL_WORLD_JS.md](phases/PHASE_7_REAL_WORLD_JS.md) — **COMPLETE (100%)**

---

## Phase 8 — Adapter factory + Python vertical

**Goal:** Lock the **repeatable language playbook** and ship the second vertical (Python). After this phase, adding Go/Java/… is instantiation, not invention.

### 8.1 Adapter registry (factory)

| Task | Detail | Exit |
|------|--------|------|
| Registry | `get_adapter(lang)` / `get_adapter_for_project(path)` | Unit tests |
| Detection | `package.json` → js_ts; `pyproject.toml` / `requirements.txt` → python | Documented matrix |
| CLI | `rebirth adapters` lists id, extensions, image | Command works |
| How-to doc | `docs/adapters/ADDING_A_LANGUAGE.md` checklist | Used by Phase 9+ |

**Playbook every language must ship:**

1. `src/rebirth_ai/adapters/<id>.py` implementing `LanguageAdapter`  
2. `docker/<id>_sandbox/Dockerfile` + doctor image check  
3. `fixtures/<id>_sample/` + broken seed → tests pass in ≤3 iterations  
4. CommitPackFT (or equiv.) mine filter → pairs under `data/datasets/`  
5. Behavioral eval case + baseline row  
6. Tree-sitter grammar wired in AST/chunker  

### 8.2 Python adapter

| Method | Python behavior |
|--------|-----------------|
| `is_source_file` | `.py` (skip `venv`, `.tox`) |
| `parse` | Tree-sitter Python chunks |
| `sandbox_spec` | Image `rebirth-ai/python-sandbox:latest` |
| `test_command_for` | `pytest {file}` or `pytest` |

| Path | Responsibility |
|------|----------------|
| `src/rebirth_ai/adapters/python.py` | Adapter |
| `docker/python_sandbox/Dockerfile` | pytest + common wheels |
| `fixtures/python_sample/` | Tiny package + failing seed |
| `training/mine_migration_pairs.py` | `--lang python` (or dedicated filter) |
| CLI | Auto-detect + `--lang python` |

### 8.3 Graph / batch

| Task | Detail | Exit |
|------|--------|------|
| Per-file language | `MigrationState.language` drives prompts + sandbox | E2E |
| Mixed repos | Skip non-owned files; report `skipped_wrong_language` | Batch report field |

### 8.4 Eval + data

| Task | Detail | Exit |
|------|--------|------|
| Behavioral | Python broken seed → pytest pass | 1/1 sandbox |
| Mine | ≥1K Python pairs toward 15K corpus (stretch: more) | `data/datasets/` + stats |
| Soft metrics | AST helpers with `language=python` | Tests |

**Phase 8 exit checklist**

- [x] Adapter registry + `rebirth adapters` + ADDING_A_LANGUAGE.md  
- [x] `rebirth migrate` fixes Python fixture in ≤3 iterations (Ollama) — **1 iter**  
- [x] `migrate-repo` on `fixtures/python_sample` produces report  
- [x] Doctor builds/checks Python sandbox image  
- [x] Python pairs mined (count recorded in DATASET.md) — **1503**  
- [ ] Optional tag: `v0.3.0-js-python-preview` (after Phase 8 commit)

**Docs:** [phases/PHASE_8_PYTHON.md](phases/PHASE_8_PYTHON.md)

---

## Phase 9 — Language wave A (Go, Java, C#, Rust)

**Goal:** Ship four high-demand verticals using the Phase 8 playbook.

| Language | Adapter id | Image | Test runner | Extensions / detect |
|----------|------------|-------|-------------|---------------------|
| Go | `go` | `rebirth-ai/go-sandbox` | `go test ./…` | `.go`, `go.mod` |
| Java | `java` | `rebirth-ai/java-sandbox` | Maven/Gradle + JUnit | `.java`, `pom.xml` / `build.gradle*` |
| C# | `csharp` | `rebirth-ai/csharp-sandbox` | `dotnet test` | `.cs`, `*.csproj` / `*.sln` |
| Rust | `rust` | `rebirth-ai/rust-sandbox` | `cargo test` | `.rs`, `Cargo.toml` |

### 9.1 Per-language deliverables (×4)

For **each** of Go, Java, C#, Rust:

- [x] Adapter + registry entry  
- [x] Docker sandbox image + `rebirth doctor` check  
- [x] `fixtures/<id>_sample/` closed-loop migrate ≤3 iterations  
- [x] Mine ≥1–2K filtered pairs (CommitPackFT / language corpus)  
- [x] Behavioral sandbox case in eval suite  
- [x] Tree-sitter parse path green  

### 9.2 Shared work

| Task | Detail | Exit |
|------|--------|------|
| CLI lang flag | `--lang auto\|js_ts\|python\|go\|java\|csharp\|rust\|…` | Documented |
| Prompt templates | Language-aware generate/repair prompts | Unit smoke |
| Dataset layout | `data/datasets/<lang>_migration.jsonl` + combined index | DATASET.md updated |
| Corpus progress | Running total toward **15K+** | Stats JSON |

**Phase 9 exit checklist**

- [x] All four wave-A languages pass fixture E2E on Ollama  
- [x] `rebirth adapters` lists js_ts, python, go, java, csharp, rust  
- [x] Per-language pair counts recorded; corpus trending to 15K  
- [x] Doctor can build/pull all wave-A sandbox images  

**Docs:** [phases/PHASE_9_MULTI_LANG_QUALITY.md](phases/PHASE_9_MULTI_LANG_QUALITY.md) (wave A + orchestration start)

---

## Phase 10 — Language wave B (C++, Swift) + quality + 15K corpus

**Goal:** Finish the locked language set; unify quality gates; hit **≥15K** mined pairs.

### 10.1 Wave B languages

| Language | Adapter id | Image | Test runner | Extensions / detect |
|----------|------------|-------|-------------|---------------------|
| C++ | `cpp` | `rebirth-ai/cpp-sandbox` | CMake + ctest / gtest | `.cpp`/`.cc`/`.hpp`, `CMakeLists.txt` |
| Swift | `swift` | `rebirth-ai/swift-sandbox` | `swift test` | `.swift`, `Package.swift` |

Same playbook as Phase 8/9 (adapter, image, fixture, mine, eval, Tree-sitter).

**Note:** Swift sandbox may require Linux Swift toolchain image or documented macOS host path; record the chosen approach in the phase doc.

### 10.2 Multi-language quality system

| Gate | Rule |
|------|------|
| Pre-merge | Unit tests + fixture sandbox for **every registered** adapter |
| Pre-release | Per-language held-out suite ≥ baseline − 5%; target **≥90%** sandbox-pass on fixtures |
| Per PR from agent | List failed files; never claim 100% if `human_review` > 0 |

| Command | Behavior |
|---------|----------|
| `rebirth migrate-repo --lang auto\|…` | Force or detect (all 8 ids) |
| `rebirth adapters` | List all registered adapters + images |
| `rebirth eval --lang <id>\|all` | Behavioral + held-out suites |
| `rebirth report path.json` | Pretty-print batch / eval JSON |

### 10.3 Observability

| Task | Detail |
|------|--------|
| Structured traces | JSONL run log per migrate (`data/runs/`) |
| Metrics | iterations, error_class histogram, sandbox duration, language |
| Redaction | Strip tokens/env from logs |

### 10.4 Data — hit 15K+

| Task | Detail | Exit |
|------|--------|------|
| Mine wave B | C++ + Swift pairs | Counts in DATASET.md |
| Top up short langs | Re-mine / raise `--max-pairs` where under ~1.5K | Balanced corpus |
| Combined corpus | Concat or manifest of all `*_migration.jsonl` | **≥15 000** pairs total |
| Tags + held-out | `migration_kind`, train/eval split per language | For future LoRA |
| Optional LoRAs | Moved to **Phase 11** (required path before v1.0 per 2026-11-04 decision) | See PHASE_11 |

**Phase 10 exit checklist**

- [x] C++ + Swift fixture E2E pass (Phase 9 + fixtures suite)  
- [x] All **8** adapters registered and doctor-checked  
- [x] `rebirth eval --suite fixtures --lang all` → 8/8 (100%)  
- [x] Combined mined corpus **≥15K** pairs (**24640**); DATASET.md + `corpus_manifest.json`  
- [x] `data/runs/` with redaction + `rebirth report`  

**Docs:** [phases/PHASE_10_RELEASE.md](phases/PHASE_10_RELEASE.md)

---

## Phase 11 — Per-language LoRA ✅ (v2 adapters, all 8 languages, 2026-11-05)

**Full plan:** [phases/PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md)

| Phase | Scope | Exit (summary) |
|-------|--------|----------------|
| Part (old number) | Scope | Result |
|-------------------|--------|--------|
| Harness + Python (12) | `train_lora.py --lang`, splits, meta, routing; Python LoRA | ✅ |
| Wave A (13) | Go, Java, C#, Rust LoRAs | ✅ all pass gate (2026-11-05) |
| Wave B (14) | C++, Swift + PEFT lang→adapter routing + eval | ✅ all pass gate; C++ ties base |

Recipe v2: Qwen2.5-Coder-**7B**-Instruct, **4-bit QLoRA**, seq **512**, EOS + completion-only loss, 1 epoch, unload Ollama first.

## Phases 12–19 — see phase docs

[12](phases/PHASE_12_LOCAL_TUNING_V3.md) · [13](phases/PHASE_13_MLOPS_TRACKING_REGISTRY.md) · [14](phases/PHASE_14_MLOPS_REPRO_INFRA.md) · [15](phases/PHASE_15_CLOUD_14B.md) · [16](phases/PHASE_16_MLOPS_CICD_PACKAGING.md) · [17](phases/PHASE_17_CROSS_LANGUAGE.md) · [18](phases/PHASE_18_MLOPS_MONITORING.md) · [19](phases/PHASE_19_LAUNCH.md)

---

## Phase 19 — Packaging, UX, official v1.0 launch

> **Status: DEFERRED** — last phase (after Phases 12–18).

**Goal:** Someone else can install Rebirth and migrate a repo in any of the **8** target languages (Ollama and/or per-lang PEFT).

### 11.1 Install story

| Task | Detail |
|------|--------|
| One-shot bootstrap | `scripts/bootstrap.sh` — venv, pip, **all** sandbox images, doctor |
| Versioned images | Tag sandbox images with app version |
| `.env.example` | Complete; no secrets |

### 11.2 UX / docs

| Task | Detail |
|------|--------|
| README quickstart | One path per language (or link matrix) |
| Troubleshooting | VRAM stacking, GPU hang, gh auth, Docker group, Swift host notes |
| Architecture diagram | GLOBAL §3 shows full adapter fan-out |

### 11.3 Release checklist

- [ ] `rebirth doctor` green on clean machine notes  
- [ ] Fixture E2E for all 8 languages  
- [ ] ≥1 real-repo migrate report for JS/TS and ≥1 non-JS language  
- [ ] `migrate-repo --open-pr` on throwaway branch  
- [ ] SECURITY: no tokens in reports; sandbox network default `none`  
- [ ] Corpus ≥15K documented  
- [ ] Tag **`v1.0.0`** (official launch; name locked here)

**Docs:** [phases/PHASE_19_LAUNCH.md](phases/PHASE_19_LAUNCH.md)

---

## Optional Track C — Cloud GPU LoRA (non-blocking)

**When:** After Phase 6 (stable product path), if you want a LoRA that matches the **14B** Ollama brain — especially valuable once the multi-lang corpus hits 15K+.

### Billing (no card in the terminal)

1. Browser: [HF billing payment](https://huggingface.co/settings/billing/payment) — add card via Stripe; keep **positive credits** (PRO credits count).  
2. Terminal: `hf auth login` with a **write** token.  
3. Submit Job (`hf jobs …` / TRL); billed **per minute** while Starting/Running.  
4. Push adapter to Hub before job ends (ephemeral disk).  

| Hardware | ~$/hour | Typical short SFT |
|----------|---------|-------------------|
| T4 16GB | $0.40 | ~$1–3 for ~1.3k pairs; scale up for 15K |
| L4 24GB | $0.80 | More headroom |
| A10G | $1.00–1.50 | Comfortable 14B QLoRA |

### Deliverables if pursued

- 14B QLoRA adapter on Hub (multi-lang or per-lang)  
- Merge/export notes → Ollama Modelfile **or** PEFT path with 14B base  
- Eval: behavioral suite ≥ Ollama baseline  

**Not required for Phases 6–11.** Local Ollama 14B + optional local 7B PEFT remains the default stack.

---

## Language status board

| Language | Adapter / sandbox | Corpus pairs | LoRA (Phase 11) |
|----------|-------------------|-------------:|---------------------|
| JS / TS | **Live** (0–7) | 9103 | Refresh in **12** (`lora_js_ts_7b` exists on 1339) |
| Python | **Live** (8) | 5246 | Train in **12** |
| Go | **Live** (9) | 1528 | Train in **13** |
| Java | **Live** (9) | 2649 | Train in **13** |
| C# | **Live** (9) | 1562 | Train in **13** |
| Rust | **Live** (9) | 1505 | Train in **13** |
| C++ | **Live** (9) | 1512 | Train in **14** |
| Swift | **Live** (9) | 1535 | Train in **14** |

Out of scope for v1.0 (unless re-prioritized): Ruby, Kotlin-as-primary (Java adapter may later accept `.kt`), PHP, etc.

Adapter vertical playbook (done): [adapters/ADDING_A_LANGUAGE.md](adapters/ADDING_A_LANGUAGE.md).  
LoRA playbook (next): [phases/PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md).

---

## Execution order

```
Done         Phases 0–11 (8 langs; corpus 24640; fixtures 8/8; v2 LoRA per language)
Now          Phase 15 blocked on AWS quota (local prep done 2026-11-07); Phase 17 stages 1–2 done (zero-shot 89.3 %) → stage 3 decisions
Then         Phase 13 — MLOps 1: MLflow tracking + registry + YAML recipes   (no GPU)
Then         Phase 14 — MLOps 2: DVC/S3, Docker CUDA image, Terraform, cloud scripts, vLLM eval
Then         Phase 15 — cloud 14B adapters + DPO (AWS g6e.xlarge; needs quota)
Then         Phase 16 — MLOps 3: CI gates, champion/challenger, Ollama packaging, rollback
Then         Phase 17 — cross-language migration + unified adapter (AWS for data gen + training)
Then         Phase 18 — MLOps 4: monitoring + data flywheel
Then         Phase 19 — bootstrap + docs + v1.0.0 tag
```

---

## Risk register (production)

| Risk | Mitigation |
|------|------------|
| Display-GPU hang on PEFT/train | No multi-epoch local; `REBIRTH_PEFT_DEVICE=cpu`; stop Ollama before PEFT |
| VRAM stack Ollama+PEFT+ST | Doctor warning; hash RAG with PEFT; unload Ollama for PEFT eval |
| False confidence from exact-match | Behavioral eval only for gates |
| Partial migrations ship as success | PR body must list failures; exit code ≠ 0 if any failed |
| Adapter sprawl | Strict `LanguageAdapter` protocol + registry tests + ADDING_A_LANGUAGE.md |
| Heavy sandboxes (Java/C++/Swift) | Slim images; cache layers; doctor timeout guidance |
| Uneven CommitPackFT coverage | Per-lang mine targets; top-up in Phase 10; document shortfalls |
| Secrets in PRs | Path denylist + scan |

---

## Definition of Done

### A. `v0.3` production-preview (optional, after Phase 8)

1. JS/TS **and** Python fixtures pass closed-loop migrate on Ollama.  
2. Adapter registry + how-to doc exist.  
3. Doctor + bootstrap path for those two sandboxes.

### B. `v1.0` official launch (Phases 6–19)

1. All **8** language fixtures pass closed-loop migrate on Ollama.  
2. Per-language PEFT adapters trained (Phases 11/15) **or** explicit skip per language.  
3. `migrate-repo` on ≥1 real JS/TS repo and ≥1 real non-JS repo produces a truthful report (optional PR).  
4. Behavioral + held-out suites are the release gate; fixture sandbox target **≥90%** per language.  
5. Combined mined corpus **≥15 000** pairs across the 8 languages (**24640** met).  
6. Doctor + bootstrap documented for a second machine (all sandbox images).  
7. VRAM/ops rules documented and enforced in CLI warnings.  
8. Cloud LoRA **not** required to claim v1.0 if local 7B adapters (or Ollama-only skip) are documented.

---

## Doc index

| Doc | Role |
|-----|------|
| [GLOBAL.md](GLOBAL.md) | System bible + status board |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | Restart here |
| [DATASET.md](DATASET.md) | Data + PEFT + eval numbers + 15K target |
| [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) | **This file** |
| [OLLAMA_AMD_GPU.md](OLLAMA_AMD_GPU.md) | Inference / train GPU ops |
| [phases/PHASE_10_RELEASE.md](phases/PHASE_10_RELEASE.md) | Last completed product phase (quality + 15K) |
| [phases/PHASE_11_LORA_PER_LANGUAGE.md](phases/PHASE_11_LORA_PER_LANGUAGE.md) | Per-lang LoRA (complete) |
| [phases/PHASE_12_LOCAL_TUNING_V3.md](phases/PHASE_12_LOCAL_TUNING_V3.md) | ✅ Complete — v3 results, references, Ollama test |
| [phases/PHASE_13_MLOPS_TRACKING_REGISTRY.md](phases/PHASE_13_MLOPS_TRACKING_REGISTRY.md) · [14](phases/PHASE_14_MLOPS_REPRO_INFRA.md) · [16](phases/PHASE_16_MLOPS_CICD_PACKAGING.md) · [18](phases/PHASE_18_MLOPS_MONITORING.md) | MLOps phases |
| [phases/PHASE_15_CLOUD_14B.md](phases/PHASE_15_CLOUD_14B.md) · [CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md) | Cloud 14B adapters |
| [phases/PHASE_17_CROSS_LANGUAGE.md](phases/PHASE_17_CROSS_LANGUAGE.md) | Cross-language migration |
| [phases/PHASE_19_LAUNCH.md](phases/PHASE_19_LAUNCH.md) | Official v1.0 packaging (last) |
| [phases/PHASE_8_PYTHON.md](phases/PHASE_8_PYTHON.md) | Adapter factory + Python |
| [phases/PHASE_9_MULTI_LANG_QUALITY.md](phases/PHASE_9_MULTI_LANG_QUALITY.md) | Remaining language wave |
| [training/export_for_ollama.md](../training/export_for_ollama.md) | Optional LoRA → Ollama |
