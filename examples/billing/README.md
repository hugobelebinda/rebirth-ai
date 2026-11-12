# Example: billing

A small Python project written in an old style, with its tests, used by the README and the demo:

```bash
rebirth migrate examples/billing --file src/invoice.py            # modernize, show the diff
rebirth migrate examples/billing --file src/invoice.py --to go    # translate to Go (written to rebirth_out/go/)
```

Nothing in the project changes unless you add `--write`.
