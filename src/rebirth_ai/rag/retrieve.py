"""Index a project and retrieve relevant code chunks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rebirth_ai.adapters.base import LanguageAdapter
from rebirth_ai.adapters.js_ts import JsTsAdapter
from rebirth_ai.ast.chunker import CodeChunk
from rebirth_ai.rag.embedder import Embedder, get_embedder
from rebirth_ai.rag.store import StoredChunk, VectorStore

DEFAULT_DB = Path(
    os.environ.get(
        "REBIRTH_VECTOR_DB_PATH",
        "data/embeddings.db",
    )
)


@dataclass
class IndexStats:
    files: int
    chunks: int
    db_path: str


class Retriever:
    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        embedder: Embedder | None = None,
        adapter: LanguageAdapter | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.embedder = embedder or get_embedder()
        self.adapter = adapter or JsTsAdapter()
        self.store = VectorStore(self.db_path)

    def index_project(self, project_dir: Path, *, clear: bool = True) -> IndexStats:
        project_dir = project_dir.resolve()
        chunks: list[CodeChunk] = []
        files = 0
        for path in sorted(project_dir.rglob("*")):
            if not path.is_file():
                continue
            if not self.adapter.is_source_file(path):
                continue
            # Skip tests for retrieval corpus by default (keep src utilities)
            rel = path.relative_to(project_dir).as_posix()
            if "/test/" in f"/{rel}" or rel.startswith("test/"):
                continue
            files += 1
            parsed = self.adapter.parse(path)
            # Prefer relative paths in the index
            for c in parsed:
                chunks.append(
                    CodeChunk(
                        kind=c.kind,
                        name=c.name,
                        start_line=c.start_line,
                        end_line=c.end_line,
                        text=c.text,
                        file_path=rel,
                    )
                )

        if clear:
            self.store.clear()
        self.store.set_meta("embedder_dim", str(self.embedder.dim))
        self.store.set_meta("project_dir", str(project_dir))

        if not chunks:
            return IndexStats(files=files, chunks=0, db_path=str(self.db_path))

        texts = [f"{c.kind} {c.name}\n{c.text}" for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.store.add_chunks(
            file_paths=[c.file_path for c in chunks],
            kinds=[c.kind for c in chunks],
            names=[c.name for c in chunks],
            start_lines=[c.start_line for c in chunks],
            end_lines=[c.end_line for c in chunks],
            texts=[c.text for c in chunks],
            embeddings=embeddings,
        )
        return IndexStats(files=files, chunks=len(chunks), db_path=str(self.db_path))

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        exclude_file: str | None = None,
    ) -> list[StoredChunk]:
        vec = self.embedder.embed([text])[0]
        return self.store.search(vec, top_k=top_k, exclude_file=exclude_file)

    def query_for_file(self, path: Path, *, top_k: int = 5) -> list[StoredChunk]:
        """Retrieve context for migrating `path` using its imports/symbols."""
        chunks = self.adapter.parse(path)
        import_names = [c.name for c in chunks if c.kind == "import"]
        symbol_names = [c.name for c in chunks if c.kind in {"function", "class", "method"}]
        query_bits = import_names + symbol_names
        if not query_bits:
            query_bits = [path.read_text(encoding="utf-8")[:1000]]
        query = "\n".join(query_bits)
        # Exclude the file itself when paths are relative — caller may pass absolute
        exclude = None
        try:
            # Best-effort: if indexed with relative paths, strip common prefixes later
            exclude = path.name
        except Exception:  # noqa: BLE001
            exclude = None
        hits = self.query(query, top_k=top_k * 3, exclude_file=None)
        # Filter out chunks from the same basename
        filtered = [h for h in hits if Path(h.file_path).name != path.name]
        return filtered[:top_k]
