# Phase 19 — Packaging, UX & Official v1.0 Launch

> Status: **IN PROGRESS (started 2026-11-10)** — last phase (was "Phase 11" before the 2026-11-05 renumbering). Release (step 6: Hugging Face upload, `v1.0.0` tag) waits for the owner's review.  
> Previous: [PHASE_18_MLOPS_MONITORING.md](PHASE_18_MLOPS_MONITORING.md)  
> Depends on: Phase 10 complete **and** LoRA specialization waves ([PHASE_11_LORA_PER_LANGUAGE.md](PHASE_11_LORA_PER_LANGUAGE.md)) — or explicit user skip of LoRA  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 19  
> Last docs refresh: 2026-11-11  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

## Why deferred

Phases 0–10 delivered **8 language verticals + ≥15K corpus + quality CLI**.  
Next priority is **train a LoRA adapter per language** (Phase 11, done 2026-11-05; 14B upgrade in Phase 15). Official **`v1.0.0`** packaging waits until those adapters exist (or you choose an Ollama-only launch).

## Objectives (when resumed)

- Bootstrap script installs tooling and **all** sandbox images.  
- README / docs cover every target language (+ PEFT adapter map if shipped).  
- Tag **`v1.0.0`** official launch (not preview).

## Locked launch surface

Languages: **JS/TS, Python, Go, Java, C++, C#, Rust, Swift**  
Corpus: **≥15K** mined multi-language migration pairs (**24 640** already)  
Quality: fixture sandbox target **≥90%** per language; truthful multi-file PR reports  
Preferred brain: Ollama 14B default + **per-lang PEFT** when adapters exist

## Exit criteria

- [ ] Phases 11–18 complete **or** written skip decision  
- [ ] `scripts/bootstrap.sh` + doctor green path documented for second machine  
- [ ] Fixture E2E for all 8 languages  
- [ ] ≥1 real JS/TS repo + ≥1 real non-JS repo migrate report  
- [ ] `migrate-repo --open-pr` on throwaway branch  
- [ ] Security checklist (secrets, sandbox network default `none`)  
- [ ] Corpus ≥15K documented in `DATASET.md`  
- [ ] Tag `v1.0.0`  

## Plan (agreed 2026-11-10)

Two model packs (ADR-001), models hosted on **Hugging Face Hub** (owner's choice 2026-11-10). Steps 1–5 run
now; step 6 (publish + tag) only after a review with the owner.

| # | Step | State |
|---|---|---|
| 1 | Merge PR #5 (Phase 17 + 18) | done (owner merged it 2026-11-10) |
| 2 | Merge the cross-language LoRA into the 14B before quantizing → `rebirth-xlang:14b-v2`, gate vs v1 | done 2026-11-11: gate passed, v2 served, rollback demo done |
| 3 | `scripts/bootstrap.sh` + `rebirth install` + `rebirth doctor --strict` | done, clean-copy test passed |
| 4 | Release checks: fixtures 8/8, real-repo reports, `--open-pr`, security checklist | in progress |
| 5 | Docs: INSTALL, README, SECURITY, Hugging Face model cards | drafts done (both cards) |
| 6 | Hugging Face upload, `configs/models.json` `hf_namespace`, version 1.0.0, tag | **after review** |

## Built (2026-11-10)

### Step 2 — merged cross-language model `rebirth-xlang:14b-v2`

v1 is the q6_K 14B with the f16 LoRA applied at run time (`ADAPTER`). v2 merges the LoRA into the **bf16**
weights first, then quantizes once:

1. `training/merge_lora.py`: merges shard by shard (`W += (α / r) · B·A` in fp32, back to bf16); RAM peak ≈ one
   shard (~5 GB), so it runs on the 30 GB machine where `merge_and_unload()` (whole model in RAM) cannot.
   `tests/test_merge_lora.py` checks it against PEFT's own `merge_and_unload()` on a tiny Qwen2 (weights and
   logits). On the 14B: 336 / 336 LoRA weights merged, spot checks within bf16 rounding (≤ 0.0016).
2. llama.cpp (pinned `6c01da6`): `convert_hf_to_gguf.py --outtype bf16` → `llama-quantize Q6_K` (12.12 GB,
   within 160 bytes of Ollama's official q6_K).
3. Ollama model with the base's chat template, system prompt and license verbatim
   (`configs/ollama/xlang-14b.Modelfile`). Whole pipeline: `scripts/xlang_v2_merge.sh` (~60 GB disk at peak).

Speed on the 7800 XT (same prompt, greedy): **40.5 vs 32.7 tokens/s** generating (+24 %), **466 vs 321
tokens/s** reading the prompt (+45 %).

Gate (`scripts/xlang_v2_eval.sh`, then `rebirth serve publish-xlang`): the same 931 program translations and
2,229 HumanEval translations as v1; v2 must not drop by more than 2 % (programs) / 1 % (HumanEval) of n.
Also watched: the Go `bufio.Scanner` + `fmt.Fscan` defect (v1: 17 / 115 first-try Go drafts).

**Result (2026-11-10 → 11, ~13 h on the 7800 XT, paused overnight after A1 and resumed):**

| Suite | Translations | v1 (adapter at run time) | **v2 (merged)** | Allowed drop |
|---|---:|---:|---:|---:|
| Programs, 21 directions with human sources (A1) | 420 | 265 | **265** | |
| Programs, 35 bootstrapped directions (A2) | 511 | 407 | **402** | |
| **Programs, verified after ≤ 2 retries** | **931** | **672** | **667** | 18 |
| HumanEval, 35 directions (B1) | 1,396 | 1,296 | **1,295** | |
| HumanEval, 21 bootstrapped directions (B2) | 833 | 768 | **770** | |
| **HumanEval, first try** | **2,229** | **2,064** | **2,065** | 22 |

Gate passed: v2 is v1 within noise (per target language, programs move by at most 4 translations either way), and it is
faster. Go defect: **13 / 115** first-try Go drafts (v1 17), 22 / 206 over all rounds (v1 30 / 211).
`xlang_v2_eval.sh` now skips a stage whose report exists, so a stopped run restarts where it left off.

Served (2026-11-11): `rebirth serve publish-xlang` put v1 on the translation route (against the plain 14B:
672 vs 551, 2,064 vs 2,030), then v2 (against v1). Rollback demo: `rebirth serve rollback --lang xlang`
→ `rebirth-xlang:14b` = v1 (Ollama digest `c5f020f8d613`), again → v2 (`9611346cead7`), history keeps v1.
`migrate --to` (`--backend auto`) now follows the route's alias `rebirth-xlang:14b` instead of a fixed
`rebirth-xlang:14b-v1` tag (`REBIRTH_XLANG_MODEL` still overrides).

### Step 3 — install

- `scripts/bootstrap.sh [--pack both|modernize|migrate] [--from DIR | --from-local]`: checks tools (no sudo),
  `.venv` with CPU torch, embedding model cache, the 8 sandbox images (also tagged with the app version),
  `rebirth install`, `rebirth doctor --strict`. Re-runnable.
- `rebirth install` (`src/rebirth_ai/install.py`, `configs/models.json`): per pack, downloads from Hugging Face
  (resumable), checks size + sha256, writes the Modelfile, `ollama create` + alias; skips what is installed.
  `--from DIR` installs from a folder (mirror / USB). `tests/test_install.py` (6).
- `rebirth doctor --strict`: adds GPU + VRAM (nvidia-smi, rocm-smi or the amdgpu driver), disk, per-pack
  model counts, and exits 1 on a missing requirement.
- Translation route: `translation:` in `configs/routing.yaml`, `rebirth serve publish-xlang` (gate on eval
  reports) and `rebirth serve rollback --lang xlang` (swap with the previous version; again = roll forward).
  `tests/test_xlang_serving.py` (4). Gate for v1 vs the plain 14B: programs 672 vs 551, HumanEval 2,064 vs 2,030.
- Fixed: PyYAML is now a core dependency (`rebirth migrate` reads `configs/routing.yaml`; it only arrived
  through `transformers` before).
- Clean-copy test (2026-11-10): fresh copy of the repo without `.venv`, `bootstrap.sh --from <release
  folder>` → 3 min 56 s, `doctor --strict`: all checks passed.

### Step 5 — docs

`docs/INSTALL.md` (requirements, packs, offline install, troubleshooting, uninstall), README "Install",
Hugging Face model card drafts in `docs/release/`.

## What shipped / how to run / samples

_Filled at the release (step 6)._
