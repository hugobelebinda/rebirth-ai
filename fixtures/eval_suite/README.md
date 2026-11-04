# Held-out eval suite (Phase 7)

12 intentionally broken JS modules + gold sources + `node:test` cases.

- Manifest: [`manifest.json`](manifest.json)
- Run: `rebirth eval --suite heldout --backend ollama --freeze-baseline data/reports/baselines/ollama_js_ts.json`
- Or: `python training/eval_behavioral.py --suite heldout --backend ollama`

Gate: sandbox-pass rate must stay within frozen baseline − 5%.
