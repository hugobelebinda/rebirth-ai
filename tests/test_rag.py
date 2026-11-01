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
