"""Maintain an in-memory FAISS index and map vector IDs back to source chunks."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from app.chunker import CodeChunk


class FAISSStore:
    """Store and retrieve code chunks using a FAISS index."""

    def __init__(self, dimension: int):
        # Inner product on normalized vectors is equivalent to cosine similarity.
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: dict[int, CodeChunk] = {}
        self.next_vector_id = 0

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        if not chunks:
            return

        # Validate vector shape before passing embeddings into FAISS.
        vectors = np.asarray([np.asarray(chunk.embedding, dtype=np.float32) for chunk in chunks], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of shape (n, {self.dimension}), got {vectors.shape}.")

        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        # FAISS stores vectors only, so keep a parallel ID-to-chunk mapping for citations.
        for chunk in chunks:
            self.metadata[self.next_vector_id] = chunk
            self.next_vector_id += 1

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        # Normalize the question vector using the same cosine-similarity convention.
        query_vector = np.asarray([query_embedding], dtype=np.float32)
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Expected query vector length {self.dimension}, got {query_vector.shape[1]}.")

        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results: list[dict[str, Any]] = []
        for score, vector_id in zip(scores[0], indices[0]):
            if vector_id < 0:
                continue
            # Resolve FAISS IDs back to chunks so answers can cite file paths and lines.
            chunk = self.metadata.get(int(vector_id))
            if chunk is None:
                continue
            results.append({"chunk": chunk, "score": float(score)})
        return results
