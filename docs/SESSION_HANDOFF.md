# Session handoff — 2026-11-10 (restart here)

> **Read this first** in a new agent window. Then sync [GLOBAL.md](GLOBAL.md) + [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md).
> Phases were **renumbered in execution order** on 2026-11-05 (map: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md#phase-renumbering-2026-11-05)).

## Where we are — Phases 0–15 ✅, 16 (offline) ✅, 17 ✅, 18 tooling ✅ (2026-11-10); next: Phase 19; no cloud spend planned

| Item | State |
|------|--------|
| Phases **0–15, 16 (offline)** | **Complete.** Phase 15 ended at its S2 gate (2026-11-08): 14B Go / Python adapters 15 / 19 vs v2 7B 14 / 20 soft-exact per 100 → S3/S4 skipped ([PHASE_15 § Results](phases/PHASE_15_CLOUD_14B.md#results-s1s2-2026-11-08)) |
| Phase 17 | **Complete** (stage 6 not run). Shared cross-language adapter `lora_xlang_14b` (14B, trained on AWS on 14K rows of test-verified translations). Program suite, local q6_K Ollama: plain 14B **208 / 420** → adapter **265 / 420** (17 / 21 directions better, 1 worse by one problem). Cloud bf16: 215 → 274; Qwen3-Coder-30B zero-shot 282 (too big for 16 GB) ([PHASE_17 § Stage 4](phases/PHASE_17_CROSS_LANGUAGE.md#stage-4--shared-adapter--eval-done-2026-11-09-aws--local-check)). Stage 5 (2026-11-10): other 35 directions **343 → 407 / 511**, HumanEval 56 directions **91.1 → 92.6 %** (no regression); one defect: locally the adapter often mixes `bufio.Scanner` into `fmt.Fscan` in Go (17 / 115 drafts, rare in cloud bf16) → re-check after merging the LoRA ([PHASE_17 § Stage 5](phases/PHASE_17_CROSS_LANGUAGE.md#stage-5--the-other-35-directions--humaneval-regression-check-done-2026-11-10-local)) |
| Serving | Same-language: Ollama `rebirth-<lang>:7b` (v2 7B adapters, `configs/routing.yaml`). Cross-language (`migrate --to`, `--backend auto`): **`rebirth-xlang:14b-v1`** (q6_K 14B + f16 GGUF LoRA, built by `scripts/ollama_xlang.sh`), plain 14B if missing; `--backend ollama` = plain 14B (eval baseline); `REBIRTH_XLANG_MODEL` overrides |
| Adapters | `data/models/lora_<lang>_7b` (8, champions) · `lora_{go,python}_14b` (Phase 15, not promoted) · `lora_xlang_14b` (Phase 17) — all DVC-tracked; the three 14B ones are **not pushed to S3 yet** (`scripts/dvc.sh push` needs the user's go) |
| Frozen eval sets | `data/datasets/evalsets/<lang>.jsonl` (+ `rust_v2` / `cpp_v2`); translation: `data/datasets/xlang/humaneval.jsonl` (56 directions) and `programs.jsonl` (163 CodeContests programs) |
| Stage-3 data | `data/datasets/xlang/stage3/` (30,211 pairs, git-ignored, copy in S3) |
| GPU queue | systemd unit `rebirth-gpu-job` (user stops it by hand); idle since 2026-11-10 12:01 (stage 5 done). **Runs only when the user says go in chat.** |
| Cloud | g6e.xlarge in us-east-1 (4-vCPU quota). Spend ≈ **$43 of $120** (S0 + Phase 15 S1–S2 + Phase 17 stages 3–4); instance terminated 2026-11-09 16:32 EST; bucket + IAM remain. Any launch needs the user's go |
| End product | One offline tool: modernize in-language **and** migrate between any of the 8 languages ([CLOUD_GPU_PLAN.md §7](CLOUD_GPU_PLAN.md#7-the-final-product)); **decided 2026-11-10: two models** (7B same-language + 14B cross-language), no unified adapter ([ADR-001](research/ADR-001_UNIFIED_VS_SPECIALISTS.md)) |

## Git / GitHub

| Item | Value |
|------|--------|
| Branch | `main` |
| Last merge | PR #5 (Phase 17 complete + Phase 18 monitoring + ADR-001, 2026-11-10) |
| Branch protection | `main`: CI checks `lint` / `test` / `pipeline` required, PR required (0 approvals), applies to admins — everything lands through a PR |
| Tag | `v0.2.0-mvp` on origin (JS/TS MVP) |

## Exact next step (plan the day with the user)

1. ✅ **Phase 17 stage 5 done 2026-11-10** (local, 5.6 h): 35 bootstrapped directions 343 → 407 / 511, HumanEval 91.1 → 92.6 %. Follow-up: the Go `bufio.Scanner` + `fmt.Fscan` defect — re-count after the LoRA merge (Phase 19), else a Go target hint or stage-6 data.
2. ✅ **Decided 2026-11-10: two models** — 7B adapters (modernize) + 14B cross-language adapter (`--to`), no unified adapter ([ADR-001](research/ADR-001_UNIFIED_VS_SPECIALISTS.md)). Follow-ups: ✅ Ollama swap time measured (7B↔14B 11–17 s warm vs 10 s between two 7B languages; ADR item 2); Phase 19 installer offers both packs; merge the xlang LoRA into the 14B before quantizing.
3. ⏭ **Cloud round declined 2026-11-10:** no more cloud spend unless a jump is guaranteed (the 30B teacher is only on par: 282 vs 274 / 420). Phase 17 is complete with the v1 adapter; ~$77 credit unspent.
4. **Phase 18** — tooling done 2026-11-10 (`rebirth dashboard`, `rebirth monitor check|reference`, `rebirth review …`, DVC stage `flywheel_mix`; [PHASE_18](phases/PHASE_18_MLOPS_MONITORING.md#built-2026-11-10)); first flywheel retrain deferred to after launch (needs real use) → **Phase 19** (installer, docs, v1.0.0) is next.

Never start the GPU queue, launch cloud instances, or commit/push without the user's explicit go-ahead.

## Phase 10 evidence (locked)

- `rebirth eval --suite fixtures --lang all` → **8/8** (`data/reports/eval_fixture_sandboxes.json`)  
- `rebirth report` works on batch + eval JSON  
- Migrate writes `data/runs/<id>/summary.json` (redacted; `data/runs/` gitignored)  
- Corpus: **24640** total, `met_target=true`  

| Lang | Pairs |
|------|------:|
| js_ts | 9103 |
| python | 5246 |
| java | 2649 |
| csharp | 1562 |
| go | 1528 |
| swift | 1535 |
| cpp | 1512 |
| rust | 1505 |

## Do not redo

Phases 0–10 adapters / golden JS benches / 15K mine / Phase 11.1 harness.

## Tests

Full suite green (**212 passed**, 2026-11-10; `ruff check .` clean).

 Prefer `REBIRTH_EMBEDDER=hash` with Ollama 14B resident; unload Ollama before PEFT train/migrate.
