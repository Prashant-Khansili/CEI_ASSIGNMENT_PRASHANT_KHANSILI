"""
app.py
------
Streamlit interface for the RAG Document Question Answering system.

Works with a local LLM via Ollama.


Run with:
    cd src
    streamlit run app.py
"""

import os
import tempfile

import requests
import streamlit as st
from dotenv import load_dotenv

from document_loader import load_and_chunk
from vectorstore import VectorStore
from retrieval import retrieve
from chatbot import generate_answer

load_dotenv()  # reads .env in the project root (or current dir) into os.environ

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Question Answering")
st.caption("Upload a document, then ask questions about it. Answers are grounded "
           "in the document's content using retrieval-augmented generation.")

# --- Sidebar: settings + provider status ---
with st.sidebar:
    st.header("Settings")

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
    st.info(f"Using local LLM via Ollama\n\nModel: `{model}`\nServer: `{base_url}`")
    st.caption("Make sure `ollama serve` is running and the model is pulled.")

    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk size (characters)", min_value=300, max_value=1500, value=800, step=100)

# --- Session state setup ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Step 1: Upload + index document ---
uploaded_file = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Reading and indexing document..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_chunk(tmp_path, chunk_size=chunk_size)
            os.unlink(tmp_path)

            store = VectorStore(embedder_preference="sentence-transformers")
            store.build(chunks)
            st.session_state.vector_store = store
            st.session_state.chat_history = []

        st.success(f"Document indexed into {len(chunks)} chunks. Ask a question below!")

# --- Step 2: Ask questions ---
if st.session_state.vector_store is not None:
    question = st.text_input("Ask a question about the document")

    if st.button("Get Answer") and question:
        with st.spinner("Retrieving relevant context and generating answer..."):
            retrieved = retrieve(st.session_state.vector_store, question, top_k=top_k)
            try:
                answer = generate_answer(question, retrieved)
            except (ConnectionError, requests.RequestException) as e:
                answer = f"⚠️ Ollama request failed: {e}"

        st.session_state.chat_history.append((question, answer, retrieved))

    # --- Display chat history (most recent first) ---
    for q, a, retrieved in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.divider()
else:
    st.info("Upload a document and click 'Process Document' to get started.")
