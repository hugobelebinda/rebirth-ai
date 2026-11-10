"""Cross-language migration (Phase 17): translate a file from one supported language to another.

- :mod:`languages` — language ids, file naming, compile-check / run commands per target sandbox
- :mod:`prompts`   — translation prompts (chat for zero-shot models, SFT format for a future adapter)
- :mod:`verify`    — run a translation in the *target* sandbox (compile check, target tests, programs)
- :mod:`graph`     — closed loop: translate → verify → classify → retry (LangGraph)
- :mod:`multipl_e` — benchmark problems (MultiPL-E tests in every language) for the eval harness
"""
