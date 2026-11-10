# Phase 16 — MLOps 3: model CI/CD + packaging

> Status: **IN PROGRESS** — offline part done 2026-11-06 on the 7B champions. Phase 15 produced no 14B same-language champion (S2 gate, 2026-11-08), so the 7B adapters stay packaged. 2026-11-10: the Phase 17 cross-language adapter is packaged too — `scripts/ollama_xlang.sh` → `rebirth-xlang:14b-v1` (q6_K 14B + f16 GGUF LoRA, base chat template kept), served by `rebirth migrate --to` (`--backend auto`); its serving check is the program-suite run in [PHASE_17](PHASE_17_CROSS_LANGUAGE.md#stage-4--shared-adapter--eval-done-2026-11-09-aws--local-check) (265 / 420 vs 208 for the plain 14B).
> Previous: [PHASE_15_CLOUD_14B.md](PHASE_15_CLOUD_14B.md) · Next: [PHASE_17_CROSS_LANGUAGE.md](PHASE_17_CROSS_LANGUAGE.md)
> Estimated effort: ~2–3 days.

## Goal

Changes to code or models are tested automatically, regressions cannot ship, and models are
served locally through Ollama as versioned, roll-back-able artifacts.

## Scope

| Piece | What |
|-------|------|
| CI (GitHub Actions) | On every PR: lint + full pytest; CPU smoke eval (tiny model, 5 rows) that checks report schema + metrics code; `dvc status` check that data/pipeline are in sync |
| Model CI | When a challenger is registered: gate vs champion on the frozen eval set (soft-exact ≥, parses ≥ −1, fixture sandbox pass); result posted to the PR / registry; no alias change without a pass |
| Model cards | Auto-generated `data/models/lora_<lang>_14b/MODEL_CARD.md` from `rebirth_meta.json` + MLflow (data, recipe, eval table, limitations, license of training data) |
| Ollama packaging | `rebirth package --lang go` → converts the adapter to GGUF (`convert_lora_to_gguf.py`), writes a Modelfile (`FROM qwen2.5-coder:14b-instruct-q6_K` + `ADAPTER`), `ollama create rebirth-go:14b-v4` |
| Versioning + rollback | Ollama tags per version (`rebirth-<lang>:<base>-v<N>`), `:latest` → champion; `rebirth rollback --lang go` re-points to the previous champion |
| Router config | `configs/routing.yaml` (generated from the registry): per language → Ollama model tag or fallback (7B PEFT / plain 14B); `rebirth migrate` reads it |
| Serving gate | Re-score the **served** Ollama model (q6_K base + GGUF adapter) on the frozen set — the shipped artifact, not the training checkpoint, must pass |

## What shipped (2026-11-06, 7B champions)

| Piece | Files | Notes |
|-------|-------|-------|
| Served-model client | `generate/ollama_adapter_client.py` | Same plan / prompts / budgets / cleanup as `PeftClient`; `raw: true`, ChatML written client-side for chat-style plans (matches `apply_chat_template` exactly); greedy, seed 0 |
| Packaging | `packaging.py`, `rebirth package --lang <lang\|all> [--version] [--quant q4\|q8\|q6]` | Registry champion → `convert_lora_to_gguf.py` (llama.cpp pinned in `.tools/llama.cpp`, commit recorded) → Modelfile (`FROM` quantized base, `ADAPTER`, raw template, SFT stops) → `rebirth-<lang>:<size>-v<N>` → serving gate → alias `rebirth-<lang>:<size>` + route + model card |
| Serving gate | `packaging.serve_gate` | Served model scored on the frozen set **through the production client**. Pass = fidelity (soft-exact and parses ≥ PEFT − 2) and soft-exact ≥ base 7B; parse regressions vs base are warnings |
| Served scores | `tracking.registry_set_key` | Ollama scores go to the registry as `eval.served:<label>:<set>.*` — they can never overwrite the PEFT score the promotion gate reads (that happened once on 2026-11-06 via `eval_vllm --adapter`; fixed, Go v1 restored to 14/99) |
| Routing | `routing.py`, `configs/routing.yaml` (generated) | `rebirth migrate` / `migrate-repo` / `generate` default to `--backend auto`: routed language → its packaged adapter; anything else → plain Ollama 14B (unchanged behaviour when nothing is routed) |
| Rollback | `rebirth serve rollback --lang <lang>` | Registry champion ↔ previous **and** the Ollama alias + route together; refuses (and leaves the registry untouched) if the previous version has no passing packaged build. Each route keeps a `history` of previous packaged versions |
| Status | `rebirth serve status` | Route, served vs PEFT vs base, gate, rollback targets |
| Model cards | `docs/models/rebirth-<lang>-<size>.md` | Generated: artifact (tag, base, adapter sha, converter commit), training data + recipe, eval table, gate, limitations |
| Cleaner fix | `prompts.strip_prose_preamble` | Chat replies with ≤ 3 lines of prose before a tagged fence keep their code (Phase 14 known issue). Re-check over all 2,298 stored predictions: 0 change → no re-score needed |
| CI | `.github/workflows/ci.yml`, `[tool.ruff.lint]` in `pyproject.toml` | lint (ruff 0.17.3, pinned rule set; 49 existing findings fixed) + full pytest on CPU torch + `dvc dag`. Suite passes in a clean copy of the repo (152 passed, 3 skipped — local tools) |
| Tests | `tests/test_phase16.py` (15), `tests/test_peft_decode.py` (+3) | Client payloads + fake Ollama server, gate rules, routing fallback, publish history, rollback refusal |

### Served vs validated (Q4_K_M base + f16 GGUF LoRA, frozen sets)

| Lang | Base 7B | PEFT bf16 (champion) | **Served** `rebirth-<lang>:7b` | Gate |
|------|--------:|---------------------:|-------------------------------:|------|
| Java | 10 | 22 / 100 | 20 / 100 | pass |
| Python | 6 | 20 / 99 | 18 / 100 | pass |
| C++ (93) | 8 | 14 / 91 | 16 / 92 | pass |
| Go | 5 | 14 / 99 | 13 / 99 | pass |
| Swift | 6 | 11 / 100 | 13 / 99 | pass |
| C# | 2 | 13 / 100 | 11 / 100 | pass |
| JS/TS | 0 | 9 / 100 | 9 / 100 | pass |
| Rust (56) | 5 | 9 / 56 | 7 / 55 | pass |
| **Total** | 42 | **112** | **107** | 8/8 |

Served ≈ PEFT − 0.6 soft-exact per language (4 languages at −2, two at +2): quantization noise
at n = 100. ~5 min per language instead of ~27. If the −2 cases matter, `--quant q8` (8.1 GB
base) is the knob; Go Q8 scored 13 as well, so Q4 stays the default (4.7 GB).
End-to-end check: `rebirth migrate fixtures/python_sample … --lang python` → routed to
`rebirth-python:7b`, `status=success iterations=1`.

## Exit criteria

- [x] CI green on main; PRs blocked on failing tests or model gate — CI green on main and on PRs #3–#5 (`ci.yml`); **branch protection on `main` since 2026-11-10**: required checks `lint`, `test`, `pipeline` (branch up to date), PR required (0 approvals, solo repo), applies to admins, no force-push / deletion. Model quality stays gated locally (`rebirth registry gate`, serving gate in `rebirth package`): CI has no GPU
- [x] All 8 champions packaged as Ollama models; served-model scores recorded (7B, 2026-11-06; repeat for 14B)
- [ ] Rollback demonstrated on one language — implemented + unit-tested; live demo at the first real promotion (no language has a second packaged champion yet)
- [x] `rebirth migrate` routes through `configs/routing.yaml` (default `--backend auto`)
