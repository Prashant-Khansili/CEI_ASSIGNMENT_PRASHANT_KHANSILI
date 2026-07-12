"""
document_loader.py
-------------------
Handles loading documents (PDF or plain text) and splitting them into
smaller overlapping chunks that are easier to embed and retrieve accurately.
"""

from pathlib import Path
from typing import List
from pypdf import PdfReader


def load_document_text(file_path: str) -> str:
    """
    Load raw text from a PDF or .txt file.

    Args:
        file_path: Path to a .pdf or .txt file.

    Returns:
        The full extracted text as a single string.
    """
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
        return "\n".join(pages_text)

    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .pdf or .txt")


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks (by characters).

    Overlap helps preserve context that might otherwise be cut in half
    at a chunk boundary, which improves retrieval quality.

    Args:
        text: The full document text.
        chunk_size: Max number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap  # move forward, keeping overlap

    return chunks


def load_and_chunk(file_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Convenience function: load a document and return its text chunks."""
    text = load_document_text(file_path)
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_huggingface_dataset(
    dataset_name: str,
    text_column: str,
    split: str = "train",
    max_records: int = 200,
) -> str:
    """
    Load a domain-specific Hugging Face dataset and concatenate its text
    column into a single document, e.g. for the vectara/open_ragbench
    dataset referenced in the project brief.

    Requires internet access to huggingface.co and the `datasets` package
    (`pip install datasets`).

    Args:
        dataset_name: e.g. "vectara/open_ragbench".
        text_column: name of the column containing the text to index.
        split: dataset split to load (default "train").
        max_records: cap on number of records to concatenate (datasets can
            be large; this keeps indexing time reasonable for a demo).

    Returns:
        A single string containing the concatenated text records.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    n = min(max_records, len(ds))
    records = ds.select(range(n))

    if text_column not in records.column_names:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {records.column_names}"
        )

    texts = [str(r[text_column]) for r in records if r[text_column]]
    return "\n\n".join(texts)


def load_and_chunk_from_huggingface(
    dataset_name: str,
    text_column: str,
    split: str = "train",
    max_records: int = 200,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """Convenience function: load a HF dataset and return its text chunks."""
    text = load_huggingface_dataset(dataset_name, text_column, split, max_records)
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
