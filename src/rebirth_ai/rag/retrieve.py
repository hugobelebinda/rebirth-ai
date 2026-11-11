"""Index a project and retrieve relevant code chunks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rebirth_ai import paths
from rebirth_ai.adapters.base import LanguageAdapter
from rebirth_ai.adapters.registry import get_adapter_for_project
from rebirth_ai.ast.chunker import CodeChunk
from rebirth_ai.rag.embedder import Embedder, get_embedder
from rebirth_ai.rag.store import StoredChunk, VectorStore

DEFAULT_DB = Path(os.environ.get("REBIRTH_VECTOR_DB_PATH") or paths.DEFAULT_DB)


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
        self.adapter = adapter
        self.store = VectorStore(self.db_path)

    def index_project(self, project_dir: Path, *, clear: bool = True) -> IndexStats:
        project_dir = project_dir.resolve()
        adapter = self.adapter or get_adapter_for_project(project_dir)
        chunks: list[CodeChunk] = []
        files = 0
        for path in sorted(project_dir.rglob("*")):
            if not path.is_file():
                continue
            if not adapter.is_source_file(path):
                continue
            # Skip tests for retrieval corpus by default (keep src utilities)
            rel = path.relative_to(project_dir).as_posix()
            if (
                "/test/" in f"/{rel}"
                or "/tests/" in f"/{rel}"
                or rel.startswith("test/")
                or rel.startswith("tests/")
            ):
                continue
            files += 1
            parsed = adapter.parse(path)
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
        embedder_name = getattr(self.embedder, "model_name", None) or type(self.embedder).__name__
        self.store.set_meta("embedder_dim", str(self.embedder.dim))
        self.store.set_meta("embedder_name", str(embedder_name))
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

    def _assert_embedder_compatible(self) -> None:
        """Refuse mismatched embedder dims (Phase 6) — clear error, not silent skip."""
        stored = self.store.get_meta("embedder_dim")
        if stored is None:
            return
        try:
            stored_dim = int(stored)
        except ValueError as exc:
            raise ValueError(
                f"Corrupt embeddings DB meta embedder_dim={stored!r} at {self.db_path}"
            ) from exc
        if stored_dim != int(self.embedder.dim):
            raise ValueError(
                f"Embedder dim mismatch for {self.db_path}: "
                f"db={stored_dim} current={self.embedder.dim}. "
                "Re-index with `rebirth index` using the same REBIRTH_EMBEDDER."
            )
        stored_name = self.store.get_meta("embedder_name")
        current_name = getattr(self.embedder, "model_name", None) or type(self.embedder).__name__
        if stored_name and stored_name != current_name and stored_name not in {
            "HashEmbedder",
            type(self.embedder).__name__,
        }:
            # Soft signal via meta only when dims already match; still refuse hash↔ST flips
            if ("Hash" in stored_name) != ("Hash" in current_name):
                raise ValueError(
                    f"Embedder family mismatch for {self.db_path}: "
                    f"db={stored_name} current={current_name}. Re-index required."
                )

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        exclude_file: str | None = None,
    ) -> list[StoredChunk]:
        self._assert_embedder_compatible()
        vec = self.embedder.embed([text])[0]
        return self.store.search(vec, top_k=top_k, exclude_file=exclude_file)

    def query_for_file(self, path: Path, *, top_k: int = 5) -> list[StoredChunk]:
        """Retrieve context for migrating `path` using its imports/symbols."""
        from rebirth_ai.adapters.registry import get_adapter

        adapter = self.adapter or get_adapter(
            "python" if path.suffix.lower() == ".py" else "js_ts"
        )
        chunks = adapter.parse(path)
        import_names = [c.name for c in chunks if c.kind == "import"]
        symbol_names = [c.name for c in chunks if c.kind in {"function", "class", "method"}]
        query_bits = import_names + symbol_names
        if not query_bits:
            query_bits = [path.read_text(encoding="utf-8")[:1000]]
        query = "\n".join(query_bits)
        hits = self.query(query, top_k=top_k * 3, exclude_file=None)
        # Filter out chunks from the same basename
        filtered = [h for h in hits if Path(h.file_path).name != path.name]
        return filtered[:top_k]
