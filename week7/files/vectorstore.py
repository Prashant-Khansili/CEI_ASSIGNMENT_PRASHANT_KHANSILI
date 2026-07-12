"""
vectorstore.py
--------------
Embeds text chunks and stores/searches them using FAISS (local vector
similarity search - no cloud account needed).

Uses embedder.get_embedder() so it works with either the production
sentence-transformers model or the offline TF-IDF fallback, transparently.
"""

from typing import List, Tuple
import numpy as np
import faiss

from embedder import get_embedder


class VectorStore:
    def __init__(self, embedder=None, embedder_preference: str = "auto"):
        """
        Args:
            embedder: an already-constructed embedder (optional).
            embedder_preference: "auto" | "sentence-transformers" | "tfidf",
                used only if `embedder` is not provided.
        """
        self.embedder = embedder or get_embedder(embedder_preference)
        self.index = None
        self.chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        if not chunks:
            raise ValueError("No text chunks provided. Is the document empty or unreadable?")

        self.chunks = chunks
        embeddings = self.embedder.encode(chunks)

        faiss.normalize_L2(embeddings)  # cosine similarity via inner product
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Vector store is empty. Call build() first.")

        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
