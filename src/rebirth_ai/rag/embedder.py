"""Code embedding wrappers (SentenceTransformers + test double)."""

from __future__ import annotations

import hashlib
import os
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape (n, dim) float32 L2-normalized vectors."""
        ...


class HashEmbedder:
    """Deterministic bag-of-hashes embedder for unit tests (no model download)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().replace("/", " ").replace(".", " ").split()
            if not tokens:
                tokens = ["empty"]
            for tok in tokens:
                h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    """Production embedder via sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.environ.get(
            "REBIRTH_EMBEDDING_MODEL",
            "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        )
        self._model = SentenceTransformer(self.model_name)
        # Probe dim
        probe = self._model.encode(["dim"], normalize_embeddings=True)
        self.dim = int(probe.shape[-1])

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(*, for_tests: bool = False) -> Embedder:
    if for_tests or os.environ.get("REBIRTH_EMBEDDER") == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder()
