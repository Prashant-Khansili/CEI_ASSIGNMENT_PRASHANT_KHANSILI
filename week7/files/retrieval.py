"""
retrieval.py
------------
Retrieval module (pipeline steps 5-6 and the optimization experiments in
step 8): converts a query into a vector, retrieves candidates, and offers
two optional upgrades over plain vector search:

- Hybrid search: blends vector similarity with keyword overlap (BM25-style
  scoring), which helps when a query uses exact terms/names/numbers that
  embeddings alone can under-weight.
- Re-ranking: a lightweight second pass that re-scores the top candidates
  using lexical overlap with the query, promoting chunks that are more
  precisely on-topic even if their raw embedding score was a bit lower.
"""

from typing import List, Tuple
import re
from collections import Counter

from vectorstore import VectorStore


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_score(query: str, chunk: str) -> float:
    """Simple normalized keyword-overlap score between 0 and 1."""
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0
    c_counts = Counter(_tokenize(chunk))
    overlap = sum(1 for t in q_tokens if c_counts[t] > 0)
    return overlap / len(q_tokens)


def retrieve(
    store: VectorStore,
    query: str,
    top_k: int = 4,
    fetch_k: int = 10,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    vector_weight: float = 0.7,
) -> List[Tuple[str, float]]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        store: a built VectorStore.
        query: the user's question.
        top_k: number of chunks to return.
        fetch_k: number of candidates to pull from the vector index before
            hybrid scoring / re-ranking (should be >= top_k).
        use_hybrid: blend vector similarity with keyword overlap.
        use_rerank: apply a lexical-overlap re-ranking pass on the blended
            candidates (a cheap stand-in for a cross-encoder re-ranker).
        vector_weight: weight given to vector similarity vs keyword score
            when use_hybrid=True (0-1).

    Returns:
        List of (chunk_text, final_score) tuples, best first.
    """
    candidates = store.search(query, top_k=max(fetch_k, top_k))

    if use_hybrid:
        scored = []
        for chunk, vec_score in candidates:
            kw_score = keyword_score(query, chunk)
            blended = vector_weight * vec_score + (1 - vector_weight) * kw_score
            scored.append((chunk, blended))
    else:
        scored = candidates

    if use_rerank:
        # Cheap re-ranking pass: nudge scores using keyword overlap again,
        # simulating what a cross-encoder re-ranker would do (re-scoring
        # the shortlist using a signal that's more query-aware than raw
        # cosine similarity).
        reranked = []
        for chunk, score in scored:
            rerank_bonus = 0.1 * keyword_score(query, chunk)
            reranked.append((chunk, score + rerank_bonus))
        scored = reranked

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
