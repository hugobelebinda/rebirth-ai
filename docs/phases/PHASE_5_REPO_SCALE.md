# Phase 5 — Repository Scale & GitHub Automation

> Status: **PLANNED**  
> Depends on: Phase 4 single-file loop stable  
> Last docs refresh: 2026-10-31

---

## 1. Objectives

- Walk directories via language adapter; migrate file-by-file with shared RAG index.  
- GitHub integration: branch, commit successes, open PR with summary.  
- Optionally add a second language adapter (Python) to prove pluggability.  
- Optionally evaluate Supabase pgvector / GPU vLLM / AWS sandboxes — only if needed.

---

## 2. Planned deliverables

| Path | Responsibility |
|------|----------------|
| `src/rebirth_ai/github/client.py` | Auth via `gh` / token |
| `src/rebirth_ai/github/pr.py` | Branch + commit + PR body |
| `src/rebirth_ai/graph/batch.py` | Multi-file orchestration |
| `adapters/python/` (optional) | Second adapter |

---

## 3. Exit criteria

- [ ] Agent opens a real PR on a sample repo with ≥1 migrated file  
- [ ] Failed files listed in PR body / report without aborting whole run  
- [ ] Docs updated; `GLOBAL.md` marks Phase 5 complete  

---

## 4. What shipped / How to run / Sample outputs / Decision log

_Filled when Phase 5 is implemented._
