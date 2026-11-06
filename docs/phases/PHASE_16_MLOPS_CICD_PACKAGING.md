# Phase 16 — MLOps 3: model CI/CD + packaging

> Status: **PLANNED**. Runs once 14B champions exist (Phase 15).
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

## Exit criteria

- [ ] CI green on main; PRs blocked on failing tests or model gate
- [ ] All 8 champions packaged as Ollama models; served-model scores recorded
- [ ] Rollback demonstrated on one language
- [ ] `rebirth migrate` routes through `configs/routing.yaml`
