# System Metrics Report

*Generated from an actual pipeline + validation run on `week7_project.txt`
(the text extracted from the assignment brief), executed in this
environment on 2026-07-12.*

## 1. Chunking Profile

| Parameter | Value | Rationale |
|---|---|---|
| Method | Character-based sliding window with overlap | Simple, deterministic, works identically for PDF and .txt input |
| Chunk size | 800 characters (default, configurable) | Small enough to keep retrieval precise, large enough to preserve sentence-level context |
| Chunk overlap | 150 characters (default, configurable) | Prevents relevant content from being split across a chunk boundary |
| Result on test document | 3,355 characters → 6 chunks | Matches expected count: (3355 / (800-150)) ≈ 5-6 chunks |
| Configurable via | `document_loader.load_and_chunk(chunk_size=, chunk_overlap=)` and the Streamlit sidebar slider | Lets you tune the size/precision tradeoff per document type without touching code |

## 2. Embedding Configuration

Two interchangeable backends behind one interface (`embedder.py`):

| Backend | Dimension | Model | When used |
|---|---|---|---|
| **Production** | 384 | `sentence-transformers/all-MiniLM-L6-v2` | Default when internet access to huggingface.co is available. Real semantic embeddings. |
| **Offline fallback** | Up to 128 (capped by corpus size via TruncatedSVD — resolved to **5** on this small 6-chunk test document) | TF-IDF + Truncated SVD (scikit-learn, local) | Automatic fallback with no network access; used for this validation run since the sandbox this report was generated in cannot reach huggingface.co |

**Note on dimension:** the offline embedder's dimensionality is capped by `min(target_dim, vocab_size-1, num_chunks-1)`, which is why a target of 128 collapsed to 5 on this small test document. On a realistically sized document (dozens+ chunks, larger vocabulary) it will use closer to the full 128 dimensions. The production embedder's 384 dimensions are fixed regardless of corpus size.

## 3. Vector Store

| Property | Value |
|---|---|
| Library | FAISS (`faiss-cpu`) |
| Index type | `IndexFlatIP` (exact inner-product search) |
| Similarity metric | Cosine similarity (achieved by L2-normalizing vectors before inner product) |
| Storage | In-memory, rebuilt per run (no persistence layer yet — listed as a future extension) |
| Vectors indexed (test run) | 6 |

## 4. Retrieval Configuration

| Property | Value |
|---|---|
| Base retrieval | Vector similarity search, `fetch_k=10` candidates pulled before scoring |
| Hybrid search | Enabled by default — blends vector similarity (weight 0.7) with keyword overlap (weight 0.3) |
| Re-ranking | Enabled by default — lightweight lexical-overlap re-ranking pass on the shortlist (stand-in for a cross-encoder re-ranker) |
| Chunks returned to LLM (`top_k`) | 4 (configurable via Streamlit slider, 2-8) |

## 5. Language Model / Generation Setup

| Property | Value |
|---|---|

| Production generator | Ollama, configured via `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, temperature 0.2 |
| API handling | No hosted API key required; all generation stays local through Ollama |
| System prompt constraint | Answers restricted to retrieved context; instructed to say "not enough information" rather than hallucinate |
| Offline fallback generator | Extractive sentence selection from the top retrieved chunk, ranked by keyword overlap with the question (`generate_answer_offline`) — kept only as a non-LLM validation helper |

## 6. Validation Results Summary

(Full detail in `validation_log.md`)

| Metric | Value |
|---|---|
| Test cases run | 5 dynamic sample questions |
| Passed (≥50% expected keywords retrieved) | 5 / 5 |
| Average top retrieval score | 0.914 |
| Score range | 0.719 - 0.969 |

## 7. Known Limitations of This Run

- Validation numbers above reflect the **offline TF-IDF embedder**, since this environment cannot reach huggingface.co. Retrieval scores and answer fluency will differ (generally improve) with the production sentence-transformers + Ollama stack — re-run `eval.py` and `pipeline.py` on your own machine (with `.env` set up) to get production metrics.
- The test document (6 chunks) is small; retrieval scores at this scale don't stress-test hybrid search or re-ranking the way a larger, noisier document would.
- No persistence layer — the index is rebuilt on every run. Fine for a course project; a production system would persist the FAISS index to disk or use Pinecone/Chroma/Weaviate.
