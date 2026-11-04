# Held-out eval suite (Phase 7)

> Phase 7 **COMPLETE** — baseline **12/12** frozen. Next: merge PR #2 then Phase 8 — [SESSION_HANDOFF.md](../../docs/SESSION_HANDOFF.md).

12 intentionally broken JS modules + gold sources + `node:test` cases.

- Manifest: [`manifest.json`](manifest.json)
- Run: `rebirth eval --suite heldout --backend ollama --freeze-baseline data/reports/baselines/ollama_js_ts.json`
- Or: `python training/eval_behavioral.py --suite heldout --backend ollama`

Gate: sandbox-pass rate must stay within frozen baseline − 5%.
