"""
eval.py
-------
Validation harness for retrieval quality. Runs a set of dynamic sample
questions against the indexed document, checks whether the retrieved
context actually contains the expected answer keywords, and writes a
timestamped log to validation_log.md.

This is a lightweight substitute for a full labeled QA eval set: each
question is paired with keywords that MUST appear somewhere in the
retrieved chunks if retrieval worked correctly. It measures retrieval
accuracy, not generation quality (generation is a separate, LLM-dependent
concern handled in chatbot.py).

Usage:
    python eval.py <path_to_pdf_or_txt>
"""

import argparse
import sys
from datetime import datetime, timezone

from document_loader import load_and_chunk
from vectorstore import VectorStore
from retrieval import retrieve, keyword_score

# Dynamic sample questions with the keywords a correct retrieval should surface.
# Edit/extend this list to validate against your own document's content.
TEST_CASES = [
    {
        "question": "What does the retrieval step do?",
        "expected_keywords": ["retrieval", "relevant", "chunks"],
    },
    {
        "question": "What is a vector database used for?",
        "expected_keywords": ["vector", "embeddings", "database"],
    },
    {
        "question": "What are the stages of the system architecture?",
        "expected_keywords": ["ingestion", "chunking", "embedding"],
    },
    {
        "question": "How is text split before embedding?",
        "expected_keywords": ["chunk", "text", "split"],
    },
    {
        "question": "What kind of data sources does the system accept?",
        "expected_keywords": ["pdf", "text", "documents"],
    },
]


def evaluate(file_path: str, top_k: int = 4):
    chunks = load_and_chunk(file_path)
    store = VectorStore(embedder_preference="tfidf")  # offline/deterministic for reproducible validation
    store.build(chunks)

    rows = []
    for case in TEST_CASES:
        question = case["question"]
        expected = [k.lower() for k in case["expected_keywords"]]

        retrieved = retrieve(store, question, top_k=top_k)
        combined_text = " ".join(chunk for chunk, _score in retrieved).lower()

        hits = [k for k in expected if k in combined_text]
        hit_rate = len(hits) / len(expected) if expected else 0.0
        top_score = retrieved[0][1] if retrieved else 0.0
        passed = hit_rate >= 0.5  # at least half the expected keywords surfaced

        rows.append({
            "question": question,
            "expected_keywords": expected,
            "keywords_found": hits,
            "hit_rate": hit_rate,
            "top_retrieval_score": top_score,
            "chunks_retrieved": len(retrieved),
            "passed": passed,
        })

    return rows, len(chunks), store.embedder.dimension


def write_log(rows, num_chunks, embed_dim, file_path, log_path="validation_log.md"):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)
    avg_score = sum(r["top_retrieval_score"] for r in rows) / total if total else 0.0

    lines = []
    lines.append("# Retrieval Validation Log")
    lines.append("")
    lines.append(f"- **Run timestamp:** {timestamp}")
    lines.append(f"- **Source document:** `{file_path}`")
    lines.append(f"- **Chunks indexed:** {num_chunks}")
    lines.append(f"- **Embedding dimension:** {embed_dim}")
    lines.append(f"- **Test cases:** {total}")
    lines.append(f"- **Passed (>=50% expected keywords retrieved):** {passed}/{total}")
    lines.append(f"- **Average top retrieval score:** {avg_score:.3f}")
    lines.append("")
    lines.append("| # | Question | Hit rate | Top score | Chunks found | Result |")
    lines.append("|---|----------|----------|-----------|--------------|--------|")

    for i, r in enumerate(rows, start=1):
        result = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"| {i} | {r['question']} | {r['hit_rate']:.2f} "
            f"({', '.join(r['keywords_found']) or 'none'}) | "
            f"{r['top_retrieval_score']:.3f} | {r['chunks_retrieved']} | {result} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Hit rate measures whether the retrieved chunks contained the expected "
        "keywords for each question - a proxy for retrieval accuracy, independent "
        "of the language model's final phrasing."
    )
    lines.append(
        "- This run used the offline TF-IDF embedder for reproducibility. Re-run "
        "with the production sentence-transformers embedder for a semantic-similarity "
        "validation pass."
    )

    with open(log_path, "w") as f:
        f.write("\n".join(lines))

    return log_path


def main():
    parser = argparse.ArgumentParser(description="Validate retrieval accuracy against sample questions.")
    parser.add_argument("file_path", help="Path to a .pdf or .txt document")
    parser.add_argument("--log-path", default="validation_log.md")
    args = parser.parse_args()

    rows, num_chunks, embed_dim = evaluate(args.file_path)
    log_path = write_log(rows, num_chunks, embed_dim, args.file_path, args.log_path)

    passed = sum(1 for r in rows if r["passed"])
    print(f"Validation complete: {passed}/{len(rows)} test cases passed.")
    print(f"Log written to: {log_path}")


if __name__ == "__main__":
    sys.exit(main())
