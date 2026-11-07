# Phase 12 — Local tuning v3 + eval hardening

> Status: **COMPLETE (2026-11-06)** (started 2026-11-05). Local 7800 XT only. GPU queue runs **only when the
> user says go in chat**; the user starts/stops it by hand (~06:00–23:00, Fri/Sat until 01:00).
> Previous: [PHASE_11_LORA_PER_LANGUAGE.md](PHASE_11_LORA_PER_LANGUAGE.md) · Next: [PHASE_13_MLOPS_TRACKING_REGISTRY.md](PHASE_13_MLOPS_TRACKING_REGISTRY.md)

## Goal

Get the most out of the 7B adapters on local hardware, and make the evaluation trustworthy
enough to make decisions with (the 24-shortest-rows suite is too small and too easy).

## Done so far (2026-11-05)

| Item | Result |
|------|--------|
| Eval token cap | `_sft_max_new_tokens` floor 64 → **256** (`SFT_MIN_NEW_TOKENS`). The 64 floor cut additive rewrites mid-statement and caused **every** v2 parse failure; after the fix all 8 v2 adapters pass their gate |
| Expanded corpus | `data/datasets/expanded/` — every CommitPackFT row passing license/quality/parse filters (originals untouched; superset; no held-out leakage). Miner fix: JSONL split on `\n` only (U+2028 inside strings broke JS/TS rows) |
| Frozen eval sets | `training/make_eval_sets.py` → `data/datasets/evalsets/<lang>.jsonl`: 100 seeded-random held-out rows that fit 512 tokens (Rust 56, C++ 93). `eval_behavioral.py --eval-file` |
| Trainer features | `--lora-targets attn|all`, `--resume` (newest complete checkpoint), alternate `--data` no longer overwrites `splits/<lang>.split.json` |
| Restartable queue | `scripts/gpu_job.sh` → `scripts/wave_v3.sh` (skips finished stages; trainings resume; uses `--recipe v3_expanded` since Phase 13) |

## Go recipe experiment (frozen 100 rows)

| Model | Soft-exact | Parses | Sym | Tok | Train loss |
|-------|-----------:|-------:|----:|----:|-----------:|
| Base 7B (chat) | 5 | 100 | .849 | .783 | — |
| Ollama 14B | 6 | 100 | .869 | .766 | — |
| **A** v2 (attn r8, 1.4K rows, 1 ep) | 14 | 99 | .939 | .854 | .224 |
| **C** attn r8, expanded 2.95K rows, 1 ep | **15** | 97 | .919 | .845 | .182 |
| **D** all-linear r16, expanded, 2 ep | 12 | 95 | .916 | .837 | .152 (overfit) |
| **D1** all-linear r16, expanded, 1 ep | 14 | 98 | .934 | .847 | .180 |

Error buckets (v2): 14 exact, 26 near (token overlap ≥ .95), 24 close, 25 mid, 11 far; **8 no-op
outputs** (legacy returned unchanged) — the target for DPO in Phase 15.

## Remaining

- [x] D1 result → v3 recipe = **C (`configs/recipes/v3_expanded.yaml`)**: D1 ties v2 with a 7.5× larger adapter and 23 % slower steps; bigger LoRA buys nothing at 7B. More data is the only lever with a (small) positive signal
- [x] v3 wave limited to **Python + C#** (2026-11-06). Python v3 (12K rows): soft 19 / parses 99 vs v2 20 / 99 → gate `kept_champion`. C# v3 (7.4K rows): soft 10 / parses 99 vs v2 13 / 100 → `kept_champion` (worse). **Decision (user, 2026-11-06): no v3 for Swift, Java, JS/TS** — Go and Python both show more same-kind data no longer helps at 7B; the GPU time goes to the GGUF→Ollama test and base-7B references instead
- [ ] Promote v3 only where the registry gate passes (`rebirth registry gate`: soft-exact ≥ v2 + 1, parses ≥ v2 − 1); then copy into `data/models/lora_<lang>_7b` (serving path until Phase 16)
- [x] Frozen-set eval of base 7B + v2 for every language (2026-11-06, `scripts/refs_p12.sh`) — table below
- [x] **GGUF → Ollama adapter test (2026-11-06): works — 13/100 on Q8_0 and Q4_K_M vs 14/99 PEFT, ~5 min per 100 rows instead of ~27** (`scripts/ollama_adapter.sh`; was: convert the Go 7B adapter with `llama.cpp/convert_lora_to_gguf.py`,
      Modelfile `FROM qwen2.5-coder:7b-instruct` + `ADAPTER`, re-score on the frozen Go set
      (de-risks Phase 16 packaging; would also make local eval much faster))
- [x] Rust/C++ data: CommitPackFT is exhausted → candidate sources in [../research/RUST_CPP_DATA_SOURCES.md](../research/RUST_CPP_DATA_SOURCES.md) (tool-generated clang-tidy / cargo-fix pairs; full CommitPack ≈100× more commits)

## Frozen-set references, all languages (2026-11-06)

Soft-exact / parses on the frozen sets (100 rows; Rust 56, C++ 93). Base = Qwen2.5-Coder-7B-Instruct, chat, no adapter.

| Lang | Base 7B | v2 adapter (champion) | v3 (expanded) | Gate v2 vs base |
|------|--------:|----------------------:|--------------:|-----------------|
| Go | 5 / 100 | **14 / 99** | 15 / 97 (C) | pass |
| Python | 6 / 99 | **20 / 99** | 19 / 99 | pass |
| C# | 2 / 100 | **13 / 100** | 10 / 99 | pass |
| Swift | 6 / 100 | **11 / 100** | — | pass |
| Java | 10 / 100 | **22 / 100** | — | pass |
| JS/TS | 0 / 99 | **9 / 100** | — | pass |
| Rust | 5 / 56 | **9 / 56** | — | pass |
| C++ | 8 / 93 | **14 / 91** | — | **parses fail** (91 < 93 − 1); symbol Jaccard .906 vs base .929 |

The adapter beats base soft-exact everywhere (≈2× on average). C++ is the weak spot: 2 extra
non-parsing outputs and lower symbol overlap → first target for the Rust/C++ data work.

### GGUF → Ollama (Go v2 adapter, `llama.cpp` 6c01da6 `convert_lora_to_gguf.py`, f16 LoRA 10 MB)

| Engine | Base weights | Soft | Parses | Sym | Tok | Time / 100 rows |
|--------|-------------|-----:|-------:|----:|----:|----------------:|
| PEFT (HF, bf16) | bf16 | 14 | 99 | .939 | .854 | ~27 min |
| Ollama | Q8_0 (8.1 GB) | 13 | 100 | .940 | .865 | ~5 min |
| Ollama | Q4_K_M (4.7 GB) | 13 | 99 | .935 | .863 | ~5 min |

Equivalent within noise, so the offline tool can ship **Q4_K_M base + per-language 10 MB GGUF
adapters** in Ollama. Raw template `{{ .Prompt }}` + the SFT stop strings are required (the
adapters never saw a chat template).

## Exit criteria

- Recipe decision recorded with evidence (this doc).
- Every language has a frozen-set row for v2 (and v3 where trained); promotions done.
- GGUF→Ollama adapter path verified or a documented fallback.
