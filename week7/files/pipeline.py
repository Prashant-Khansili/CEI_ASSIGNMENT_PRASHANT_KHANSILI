"""
pipeline.py
-----------
Operational end-to-end RAG pipeline. Wires together every stage:

  ingest -> chunk -> embed -> index -> query -> retrieve -> generate

and prints grounded, context-aware answers to the console.

Usage:
    python pipeline.py <path_to_pdf_or_txt>

This pipeline uses local embeddings and Ollama for answer generation.
"""

import argparse
import sys

from dotenv import load_dotenv

from document_loader import load_and_chunk
from vectorstore import VectorStore
from retrieval import retrieve
from chatbot import generate_answer

load_dotenv()

DEFAULT_SAMPLE_QUESTIONS = [
    "What is the main goal of this system?",
    "What technologies are used to store and search embeddings?",
    "How does the system generate its final answer?",
]


def run_pipeline(file_path: str, questions, top_k: int = 4):
    print(f"[1/5] Ingesting document: {file_path}")
    chunks = load_and_chunk(file_path)
    print(f"       -> extracted {len(chunks)} chunks")

    print("[2/5] Embedding chunks (sentence-transformers)")
    store = VectorStore(embedder_preference="sentence-transformers")
    store.build(chunks)
    print(f"       -> embedding dimension: {store.embedder.dimension}")

    print(f"[3/5] Vector index built (FAISS IndexFlatIP, {len(chunks)} vectors)")

    print("[4/5] Running queries through retrieval + generation:\n")
    results = []

    for question in questions:
        retrieved = retrieve(store, question, top_k=top_k)
        top_score = retrieved[0][1] if retrieved else 0.0

        answer = generate_answer(question, retrieved)

        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"   (top retrieval score: {top_score:.3f}, chunks used: {len(retrieved)})\n")

        results.append({
            "question": question,
            "answer": answer,
            "top_score": top_score,
            "num_chunks_retrieved": len(retrieved),
        })

    print("[5/5] Pipeline run complete.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end RAG pipeline.")
    parser.add_argument("file_path", help="Path to a .pdf or .txt document")
    parser.add_argument("--questions", nargs="*", default=None,
                         help="Custom questions to ask (defaults to a built-in sample set)")
    args = parser.parse_args()

    questions = args.questions or DEFAULT_SAMPLE_QUESTIONS
    run_pipeline(args.file_path, questions)


if __name__ == "__main__":
    sys.exit(main())
