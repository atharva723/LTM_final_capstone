# Ollama wrapper
# ─────────────────────────────────────────────────────
# backend/rag/embedder.py
#
# PURPOSE:
#   Provides the embedding function used to convert
#   text chunks into vectors for FAISS storage.
#
#   Primary:  OllamaEmbeddings (nomic-embed-text)
#             — runs locally, no API key needed
#   Fallback: HuggingFaceEmbeddings
#             — used if Ollama is not running
#
# WHY nomic-embed-text?
#   - 768-dim dense embeddings, strong for technical text
#   - Runs fully offline via Ollama
#   - Fast enough for batch indexing of our ~50 chunks
# ─────────────────────────────────────────────────────

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embeddings():
    """
    Returns an embeddings object compatible with LangChain FAISS.

    Tries Ollama first (preferred for this project).
    Falls back to HuggingFace sentence-transformers if Ollama is offline.
    """
    # ── Try Ollama embeddings first ───────────────────
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            from langchain_ollama import OllamaEmbeddings
            print(f"  [Embedder] Using Ollama → {EMBEDDING_MODEL}")
            return OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
    except Exception:
        pass

    # ── Fallback: HuggingFace (offline) ──────────────
    print("  [Embedder] Ollama not reachable — falling back to HuggingFace")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",   # small, fast, good quality
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    embeddings = get_embeddings()
    test_texts = [
        "What PPE is required for CNC machine operation?",
        "Error E01 spindle overload troubleshooting steps",
    ]
    vecs = embeddings.embed_documents(test_texts)
    print(f"  Embedding dim: {len(vecs[0])}")
    print(f"  First 5 values: {vecs[0][:5]}")