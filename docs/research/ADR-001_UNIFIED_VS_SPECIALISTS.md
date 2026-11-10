# ADR-001: One unified 14B adapter, or specialist models (7B same-language + 14B cross-language)?

**Status:** Accepted (2026-11-10) — owner chose option B: keep the two models (7B + 14B)
**Date:** 2026-11-10
**Deciders:** project owner
**Context docs:** [CLOUD_GPU_PLAN.md §7](../CLOUD_GPU_PLAN.md#7-the-final-product) (option A = unified was the plan's target),
[PHASE_15](../phases/PHASE_15_CLOUD_14B.md#results-s1s2-2026-11-08), [PHASE_17](../phases/PHASE_17_CROSS_LANGUAGE.md#stage-4--shared-adapter--eval-done-2026-11-09-aws--local-check)

## Context

The product is one offline tool, running on a 16 GB GPU (the target is the owner's 7800 XT). It does two jobs: **modernize** code within one of 8 languages, and **translate** code between any two of them (56 directions). The original plan (CLOUD_GPU_PLAN §7) aimed for **option A**: one Qwen2.5-Coder-14B with a single unified LoRA trained on both jobs. **Option B** (specialists) was the fallback "if the unified adapter loses ≥ 3 soft-exact per 100 rows to a specialist".

Measured since then (all on frozen eval sets):

| Evidence | Result |
|----------|--------|
| Same-language, base models (Phase 15 S1, 949 rows over 10 sets) | base 7B **56**, base 14B **56**: the bigger base adds nothing here |
| Same-language, adapters | v2 7B adapters **141 / 949**; 14B adapters (S2 pilot) Go **15** vs 7B 14, Python **19** vs 7B 20 per 100 → the 14B only ties |
| Translation, base models (program suite, 420 translations, retries) | base 7B **122**, base 14B **208**: the 14B is far ahead |
| Translation, adapter | 14B + shared adapter **265 / 420** (+57 over base 14B), on the shipped q6_K runtime |
| Translation, all 56 directions (stage 5, 2026-11-10) | programs 551 → **672 / 931**; HumanEval 91.1 → **92.6 %** (no regression) |
| Translation, a bigger model for reference | Qwen3-Coder-30B zero-shot 282 (does not fit 16 GB at 4-bit) |
| Speed | base 7B ~1.9× faster than base 14B per translation (HumanEval: 1.6 s vs 3.0 s); the unmerged LoRA in Ollama costs the 14B another ~35 % (18.3 vs 13.5 s per program) |
| Size | 7B Q4_K_M 4.7 GB + 8 adapters × 10 MB; 14B q6_K 12 GB + cross-language adapter 132 MB |

**What ships today is B:** `rebirth-<lang>:7b` for modernize, `rebirth-xlang:14b-v1` for `--to`. Cloud credit left: **~$77**.

## Decision (accepted 2026-11-10)

**Keep the specialists (B) — the 7B same-language adapters and the 14B cross-language adapter — and do not train a unified adapter.** Revisit only through a small, gated pilot (see Action items) if install size or model swapping turns out to be a real problem for users.

## Options considered

### Option A: unified 14B adapter (one model for both jobs)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium: one training mix (8 same-language corpora + translation pairs), one Ollama model, simpler routing |
| Cost | **~$38–40 of the $77**: ~13.8 h for the 8 same-language corpora at 1024 tokens (Phase 15 estimate) + ~4.6 h translation rows + ~2 h evals, on a g6e.xlarge. Also needs g6e capacity, which was scarce |
| Expected quality | Translation: probably ≈ the current adapter (265), if the translation rows keep their share. Same-language: **at best ≈ the 7B adapters**, because the per-language 14B adapters, with all their capacity on one language, only tied them; one shared adapter for 9 tasks is more likely to land slightly below |
| Speed | Same-language becomes ~2× slower than the 7B (plus the LoRA overhead) |
| Install / VRAM | One 12 GB base + ~0.3 GB adapter; no model swap between jobs |

**Pros:** single download (~12.3 GB); no swap when a session mixes modernize and translate; one model to version, evaluate and roll back; matches the original product sketch.
**Cons:** spends half the remaining credit on a result the evidence says will at best tie on same-language; same-language gets slower; a regression in one job means retraining both; a further cloud session (capacity risk).

### Option B: specialists (shipping now)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low: already built and packaged (Phase 16 routing + `rebirth-xlang:14b-v1`); two base models to keep compatible |
| Cost | $0 more |
| Expected quality | Best measured on both jobs: 141 / 949 same-language (7B adapters), 265 / 420 translation (14B adapter) |
| Speed | Same-language on the fast 7B; translation on the 14B, where its quality is needed |
| Install / VRAM | Both bases: 4.7 + 12 GB ≈ **17 GB on disk**. They don't fit in 16 GB VRAM together, so Ollama unloads one to load the other when the job changes: **~11–17 s per switch** (measured, warm cache), vs ~10 s for a language change between two 7B models |

**Pros:** no spend; best measured quality on each job; each job can be improved independently (flywheel rows, a v2 translation round) without risking the other; users who only want one job can install only that model.
**Cons:** ~4.7 GB more download than A; a model swap when switching jobs; two release lines in the registry.

### Option C: everything on the 7B (one small base)

Not pursued. The base 7B translates far worse (122 vs 208 / 420), and translation rows run up to ~2K tokens (median ~700, a quarter above 1,024), while the 7800 XT is only proven to train at 512 (4-bit; 1,024 never probed). It would need its own cloud run and would most likely stay well below 265.

## Trade-off analysis

The deciding fact is that **the 14B's advantage is task-specific**. It dominates translation (+86 base, +57 more with the adapter) and adds nothing to same-language modernization (56 = 56 base; 14B adapter ≈ 7B adapter). A unified 14B adapter pays the 14B's price (2× slower, 12 GB) on the job where it buys nothing, and costs ~$40 to find out whether it even ties.

B's real costs are install size (+4.7 GB) and the swap when a user switches jobs. The first is a one-time download and can be split into two optional packs. The second only hits sessions that alternate jobs, and can be measured cheaply.

What would flip the decision:
1. a new base model in the 14B class that beats the 7B adapters on same-language **zero-shot**, which would make one base the better choice for both jobs;
2. users strongly preferring one download, or the model swap proving slow in practice;
3. a pilot (below) showing a unified adapter matches both specialists.

## Consequences

- **Easier:** shipping v1.0 now, with no further spend; improving translation on its own (stage 6 / a teacher round, flywheel `xlang` rows); same-language stays fast.
- **Harder:** the install story (Phase 19) must handle two base models (offer "modernize", "migrate" or both); the registry and dashboard track two model families (already supported: Phase 18 groups by model).
- **Revisit:** when a new base model generation appears, when flywheel data has grown, or before the v2.0 model refresh.

## Action items

1. [x] Owner decided 2026-11-10: B (two models). The unified pilot (item 5) is not run.
2. [x] Measured 2026-11-10 (7800 XT, warm page cache, `OLLAMA_MAX_LOADED_MODELS=1`, num_ctx 4096; first request after the switch, 1 token): **7B → 14B 11.4–17.1 s**, **14B → 7B 10.8–12.1 s**, and **7B → 7B of another language 9.9 s** (each `rebirth-<lang>:7b` is its own Ollama model, so a language change already reloads); same model again 0.2 s. So the second base adds only ~1–7 s per switch over what changing language already costs. Cold cache (first use after boot, read from disk) not measured: it needs root to drop the page cache.
3. [ ] Phase 19: installer offers the two packs separately (modernize 4.7 GB, migrate 12 GB) and both by default.
4. [ ] Phase 19: merge the cross-language LoRA into the 14B weights before quantizing (removes the ~35 % LoRA overhead; one 12 GB file instead of base + adapter). Needs a re-check on the program suite, including the Go defect stage 5 found (`bufio.Scanner` passed to `fmt.Fscan`: 17 / 115 local first-try Go drafts with the unmerged adapter, 3 / 123 in cloud bf16).
5. [ ] ~~*(Only if A is wanted)*~~ not pursued (decision B) — kept for reference: **gated pilot, ~$12–15:** unified 14B adapter on 2 languages' same-language rows (Go + Python) + all translation rows, ~6 h on g6e. Continue to the full unified run only if Go / Python ≥ the 7B adapters (14 / 20 per 100) **and** the program suite ≥ 260 / 420.
