# Phase 2 — RAG & Classification (Data Layers)

> Status: **PLANNED**  
> Depends on: Phase 1 (AST chunks feed the indexer)

---

## 1. Objectives

- Embed code chunks (SentenceTransformers) and store in SQLite.  
- Retrieve top-k relevant implementations for a target file’s imports/symbols.  
- Train TF-IDF + RandomForest on synthetic CI failure logs → discrete error classes.

---

## 2. Planned deliverables

| Path | Responsibility |
|------|----------------|
| `src/rebirth_ai/rag/embedder.py` | Embedding model wrapper |
| `src/rebirth_ai/rag/store.py` | SQLite vector persistence + cosine search |
| `src/rebirth_ai/rag/retrieve.py` | Query API used by the agent |
| `src/rebirth_ai/classify/features.py` | Log → TF-IDF |
| `src/rebirth_ai/classify/model.py` | Train / load / predict |
| `data/datasets/error_logs.jsonl` | Labeled stack traces |
| `data/models/error_classifier/` | Persisted sklearn pipeline |
| `data/embeddings.db` | Local vector DB (gitignored) |

### Target error classes

`SyntaxError` · `DependencyMissing` · `TypeMismatch` · `TestLogicFailure`

---

## 3. Exit criteria

- [ ] Index fixture repo; query returns expected utility/hook modules  
- [ ] Classifier predicts held-out synthetic logs with useful accuracy  
- [ ] CLI or script demos for `index`, `query`, `classify`  
- [ ] Docs updated (`PHASE_2` + `GLOBAL`)

---

## 4. What shipped / How to run / Sample outputs / Decision log

_Filled when Phase 2 is implemented._
