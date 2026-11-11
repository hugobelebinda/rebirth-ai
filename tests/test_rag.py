"""RAG index/query tests using the deterministic HashEmbedder."""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.rag.embedder import HashEmbedder
from rebirth_ai.rag.retrieve import Retriever

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "js_ts_sample"


def test_index_and_query_finds_format_util(tmp_path: Path) -> None:
    db = tmp_path / "emb.db"
    retriever = Retriever(db_path=db, embedder=HashEmbedder(dim=128))
    stats = retriever.index_project(FIXTURE)
    assert stats.files >= 3
    assert stats.chunks >= 5

    hits = retriever.query("formatResult format label value", top_k=5)
    assert hits
    names = {h.name for h in hits}
    paths = {h.file_path for h in hits}
    assert "formatResult" in names or any("format" in p for p in paths)


def test_query_for_math_prefers_format_module(tmp_path: Path) -> None:
    db = tmp_path / "emb.db"
    retriever = Retriever(db_path=db, embedder=HashEmbedder(dim=128))
    retriever.index_project(FIXTURE)
    hits = retriever.query_for_file(FIXTURE / "src" / "math.js", top_k=5)
    assert hits
    # Should surface format.js utilities rather than only math itself
    assert any("format" in h.file_path or h.name.startswith("format") for h in hits)


def test_query_refuses_embedder_dim_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "emb.db"
    Retriever(db_path=db, embedder=HashEmbedder(dim=64)).index_project(FIXTURE)
    mismatched = Retriever(db_path=db, embedder=HashEmbedder(dim=128))
    try:
        mismatched.query("formatResult")
        raise AssertionError("expected ValueError for dim mismatch")
    except ValueError as exc:
        assert "dim mismatch" in str(exc).lower()


def test_retrieve_context_only_uses_this_projects_index(tmp_path, monkeypatch) -> None:
    """Phase 19 bug hunt: data/embeddings.db built for one project fed its chunks to every other one."""
    from rebirth_ai.graph import nodes
    from rebirth_ai.rag.retrieve import Retriever

    monkeypatch.setenv("REBIRTH_EMBEDDER", "hash")
    a, b = tmp_path / "a", tmp_path / "b"
    for proj in (a, b):
        (proj / "src").mkdir(parents=True)
    (a / "src" / "util.js").write_text("export function helper(x) { return x * 2; }\n")
    (a / "src" / "main.js").write_text("import { helper } from './util.js';\nexport const y = helper(2);\n")
    (b / "src" / "main.js").write_text("import { helper } from './util.js';\nexport const y = helper(2);\n")
    db = tmp_path / "idx.db"
    Retriever(db_path=db).index_project(a)

    def run(proj):
        state = {"file_path": str(proj / "src" / "main.js"), "project_dir": str(proj), "db_path": str(db),
                 "adapter_id": "js_ts", "trace": []}
        return nodes.retrieve_context(state)

    assert run(a)["retrieved_context"]  # its own index: context
    out = run(b)
    assert out["retrieved_context"] == [] and "is for" in out["trace"][-1]  # another project's: skipped
