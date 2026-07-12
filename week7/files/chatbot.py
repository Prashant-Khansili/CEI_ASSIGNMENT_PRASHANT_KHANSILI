"""
chatbot.py
----------
Takes retrieved document chunks + the user's question and generates a
final grounded answer using Ollama.

This project is local-LLM only: no hosted API backends are supported here.
"""

import os
from typing import List, Tuple

import requests
from retrieval import keyword_score

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided document context. If the answer isn't in the context, say "
    "you don't have enough information in the document to answer - do not "
    "make things up."
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")  # default to Qwen-3.4B if unset


def build_prompt(question: str, retrieved_chunks):
    context_text = "\n\n".join(chunk for chunk, _ in retrieved_chunks)

    return f"""
Document Context:

{context_text}

Question:
{question}
"""


def _generate_with_ollama(prompt: str, model: str = None, base_url: str = None) -> str:
    """
    Generate an answer using Ollama's /api/generate endpoint.
    """

    url = f"{(base_url or OLLAMA_BASE_URL).rstrip('/')}/api/generate"

    final_prompt = f"""
{SYSTEM_PROMPT}

Context:
{prompt}

Instructions:
- Answer ONLY using the provided context.
- If the answer is not present, reply:
  "I don't have enough information in the document."
- Keep the answer concise.
- If the user asks for important points, summarize them in 4-6 bullet points.
"""

    response = requests.post(
        url,
        json={
            "model": model or OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "").strip()




def generate_answer(
    question: str,
    retrieved_chunks: List[Tuple[str, float]],
    model: str = None,
) -> str:
    prompt = build_prompt(question, retrieved_chunks)

    return _generate_with_ollama(prompt, model=model)


def generate_answer_offline(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """
    Extractive fallback answerer that requires no LLM, API key, or network
    access at all. Used for pipeline validation/testing (e.g. in sandboxed
    or offline environments). It returns the most query-relevant sentence
    from the top retrieved chunk, so the answer stays strictly grounded in
    the retrieved context. Use generate_answer() for real natural-language
    generation once Ollama is set up.
    """
    if not retrieved_chunks:
        return "I don't have enough information in the document to answer that."

    top_chunk, _score = retrieved_chunks[0]
    sentences = [s.strip() for s in top_chunk.replace("\n", " ").split(".") if s.strip()]

    if not sentences:
        return "I don't have enough information in the document to answer that."

    ranked = sorted(sentences, key=lambda s: keyword_score(question, s), reverse=True)
    best = ranked[0]
    return best if best.endswith(".") else best + "."
