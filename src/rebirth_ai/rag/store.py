"""SQLite-backed vector store with numpy cosine search."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StoredChunk:
    id: int
    file_path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "kind": self.kind,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "score": self.score,
        }


class VectorStore:
    """Persist code chunks + embeddings in SQLite; search with cosine similarity."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              file_path TEXT NOT NULL,
              kind TEXT NOT NULL,
              name TEXT NOT NULL,
              start_line INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);
            """
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def add_chunks(
        self,
        *,
        file_paths: list[str],
        kinds: list[str],
        names: list[str],
        start_lines: list[int],
        end_lines: list[int],
        texts: list[str],
        embeddings: np.ndarray,
    ) -> int:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        n = len(texts)
        assert embeddings.shape[0] == n
        rows = [
            (
                file_paths[i],
                kinds[i],
                names[i],
                int(start_lines[i]),
                int(end_lines[i]),
                texts[i],
                embeddings[i].tobytes(),
            )
            for i in range(n)
        ]
        self._conn.executemany(
            "INSERT INTO chunks(file_path, kind, name, start_line, end_line, text, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return n

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
        return int(row["c"])

    def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 5,
        exclude_file: str | None = None,
    ) -> list[StoredChunk]:
        if query_vec.ndim != 1:
            query_vec = query_vec.reshape(-1)
        query_vec = query_vec.astype(np.float32)
        qnorm = np.linalg.norm(query_vec)
        if qnorm > 0:
            query_vec = query_vec / qnorm

        sql = "SELECT id, file_path, kind, name, start_line, end_line, text, embedding FROM chunks"
        params: list[object] = []
        if exclude_file:
            sql += " WHERE file_path != ?"
            params.append(exclude_file)

        scored: list[StoredChunk] = []
        for row in self._conn.execute(sql, params):
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            score = float(np.dot(query_vec, emb))
            scored.append(
                StoredChunk(
                    id=int(row["id"]),
                    file_path=str(row["file_path"]),
                    kind=str(row["kind"]),
                    name=str(row["name"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    text=str(row["text"]),
                    score=score,
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
