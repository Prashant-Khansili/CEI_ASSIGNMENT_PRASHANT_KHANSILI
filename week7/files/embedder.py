"""
embedder.py
-----------
Pluggable embedding backends behind one common interface:
    encode(list_of_texts) -> np.ndarray [n_texts, dim]

Two implementations:
- SentenceTransformerEmbedder: production embedder (all-MiniLM-L6-v2, 384-dim).
  Requires downloading the model from huggingface.co the first time.
- TfidfEmbedder: pure local fallback (scikit-learn), no downloads needed.
  Used automatically if the sentence-transformers model can't be reached,
  e.g. in a network-restricted sandbox.

This lets the rest of the pipeline (vectorstore, retrieval, chatbot) stay
identical regardless of which embedder is active.
"""

from typing import List
import numpy as np


class SentenceTransformerEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")


class TfidfEmbedder:
    """
    Local, dependency-light fallback embedder using TF-IDF + SVD to produce
    fixed-size dense vectors. Not as semantically rich as a transformer
    embedding model, but requires no network access, which makes it useful
    for offline testing/validation and CI environments.
    """
    name = "tfidf-svd-local-fallback"

    def __init__(self, dimension: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.dimension = dimension
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._svd = TruncatedSVD(n_components=dimension, random_state=42)
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        tfidf = self._vectorizer.fit_transform(texts)
        n_components = min(self.dimension, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_components = max(n_components, 1)
        if n_components != self.dimension:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.dimension = n_components
        self._svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        tfidf = self._vectorizer.transform(texts)
        vectors = self._svd.transform(tfidf)
        return vectors.astype("float32")


def get_embedder(prefer: str = "auto"):
    """
    Return a ready embedder, preferring sentence-transformers and falling
    back to the local TF-IDF embedder if the model can't be downloaded
    (e.g. no network access to huggingface.co).

    Args:
        prefer: "auto" (default), "sentence-transformers", or "tfidf".
    """
    if prefer == "tfidf":
        return TfidfEmbedder()

    if prefer in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as e:
            if prefer == "sentence-transformers":
                raise
            print(f"[embedder] Falling back to local TF-IDF embedder "
                  f"(sentence-transformers unavailable: {e.__class__.__name__}).")
            return TfidfEmbedder()

    raise ValueError(f"Unknown embedder preference: {prefer}")
