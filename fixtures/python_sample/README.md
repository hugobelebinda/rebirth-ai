# Python sample fixture (Phase 8)

> Live adapter vertical. Project: Phases 0–10 done; LoRA train in Phase 12 — [SESSION_HANDOFF.md](../../docs/SESSION_HANDOFF.md).

## Layout

- `src/math_ops.py` — correct modern helpers
- `tests/test_math_ops.py` — pytest suite
- `seeds/math_ops.broken.py` — multiply returns sum (closed-loop seed)

## Commands

```bash
# Build sandbox once
docker build -t rebirth-ai/python-sandbox:latest docker/python_sandbox

# Sandbox pass on good sources
rebirth sandbox fixtures/python_sample --test tests/test_math_ops.py

# Closed-loop fix (≤3 iterations expected)
rebirth migrate fixtures/python_sample \
  --file src/math_ops.py \
  --test tests/test_math_ops.py \
  --seed-draft fixtures/python_sample/seeds/math_ops.broken.py \
  --lang python

# Repo walk
rebirth migrate-repo fixtures/python_sample --lang python --dry-run
```

Last updated: 2026-11-03
