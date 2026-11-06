# Phase 2 — RAG & Classification (Data Layers)

> Status: **COMPLETE**  
> Depends on: Phase 1 (AST chunks feed the indexer)  
> Completed: 2026-10-31 · Docs refresh: 2026-11-04 (classifier ~105 logs / ≥0.90 acc)  
> Project now: Phases **0–10 complete** on `main`; next = Phase **11** per-lang LoRA — [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

---

## 1. Objectives

- Embed code chunks (SentenceTransformers) and store in SQLite.  
- Retrieve top-k relevant implementations for a target file’s imports/symbols.  
- Train TF-IDF + classifier on synthetic CI failure logs → discrete error classes.

---

## 2. Deliverables

| Path | Responsibility | Status |
|------|----------------|--------|
| `src/rebirth_ai/rag/embedder.py` | ST embedder + `HashEmbedder` test double | Shipped |
| `src/rebirth_ai/rag/store.py` | SQLite + numpy cosine search | Shipped |
| `src/rebirth_ai/rag/retrieve.py` | Index / query / query_for_file | Shipped |
| `src/rebirth_ai/classify/features.py` | Labels + normalize | Shipped |
| `src/rebirth_ai/classify/model.py` | TF-IDF + LogisticRegression pipeline | Shipped |
| `data/datasets/error_logs.jsonl` | ~74 labeled logs | Shipped |
| `data/models/error_classifier/` | Persisted joblib pipeline | Shipped (local artifact) |
| `data/embeddings.db` | Local vector DB | gitignored; created by `rebirth index` |
| CLI | `index`, `query`, `train-classifier`, `classify` | Shipped |

### Error classes

`SyntaxError` · `DependencyMissing` · `TypeMismatch` · `TestLogicFailure`

---

## 3. Exit criteria

- [x] Index fixture repo; query returns format utilities for math-related queries  
- [x] Classifier ≥ useful accuracy on held-out logs (**0.947** with LogisticRegression)  
- [x] CLI demos for `index`, `query`, `classify`  
- [x] Docs updated (`PHASE_2` + `GLOBAL`)

---

## 4. How to run

```bash
cd "/home/hugobele/Documents/Personal Project/Rebirth AI"
source .venv/bin/activate

# Fast / offline indexing for demos & CI
REBIRTH_EMBEDDER=hash rebirth index fixtures/js_ts_sample
REBIRTH_EMBEDDER=hash rebirth query "formatResult format label"

# Production embedder (downloads model on first use)
# REBIRTH_EMBEDDING_MODEL=flax-sentence-embeddings/st-codesearch-distilroberta-base
rebirth index fixtures/js_ts_sample

rebirth train-classifier
rebirth classify "Error: Cannot find module 'lodash' CODE MODULE_NOT_FOUND"
pytest -q
```

### Sample outputs

```text
$ REBIRTH_EMBEDDER=hash rebirth index fixtures/js_ts_sample
indexed files=3 chunks=16 db=data/embeddings.db

$ rebirth train-classifier
trained accuracy=0.947 → data/models/error_classifier

$ rebirth classify "Error: Cannot find module 'lodash' ..."
DependencyMissing
  DependencyMissing: 0.726
  ...
```

### Decision log

| Decision | Detail |
|----------|--------|
| Vector backend | SQLite BLOBs + numpy cosine (no sqlite-vss yet) |
| Default ST model | `flax-sentence-embeddings/st-codesearch-distilroberta-base` (code-oriented) |
| Test embedder | `HashEmbedder` via `REBIRTH_EMBEDDER=hash` |
| Classifier | LogisticRegression (better than RF on small TF-IDF set) |
| Skip test/ in index | Retrieval corpus focuses on `src/` utilities |

---

## 5. Known gaps

- Production ST model first download can be slow; CI uses hash embedder.  
- Cosine scan is O(n) in Python — fine for fixtures, revisit for large repos (Phase 5).  
- `query_for_file` exclude logic is basename-based (good enough for MVP).  

---

## 6. Handoff to Phase 3

Wire Ollama (`qwen2.5-coder:14b-instruct-q6_K`) with RAG context in prompts; prepare LoRA dataset/train path.
