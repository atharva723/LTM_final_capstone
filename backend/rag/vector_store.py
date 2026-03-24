# FAISS store - build and search
# ─────────────────────────────────────────────────────
# backend/rag/vector_store.py
#
# PURPOSE:
#   Builds and persists the FAISS vector store from
#   embedded document chunks. Also loads it back for
#   querying at runtime.
#
# WHY FAISS?
#   ✔ No server needed — pure local file-based index
#   ✔ Very fast similarity search (< 10ms for our corpus)
#   ✔ Easy to persist and reload with LangChain
#   ✔ Ideal for factory floor (no internet dependency)
#
#   ChromaDB alternative would be easier to inspect
#   but requires a running server. FAISS wins for our
#   offline industrial deployment scenario.
# ─────────────────────────────────────────────────────

import sys
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import VECTOR_STORE_DIR, VECTOR_STORE_NAME, RETRIEVER_K
from backend.rag.embedder import get_embeddings


FAISS_INDEX_PATH = Path(VECTOR_STORE_DIR) / VECTOR_STORE_NAME


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed all chunks and build the FAISS index.
    Saves the index to disk for reuse.

    Args:
        chunks: List of Document objects from document_loader.py

    Returns:
        FAISS vector store object
    """
    print(f"\n── Building FAISS Vector Store ────────────────")
    print(f"  Chunks to embed: {len(chunks)}")
    print(f"  Save path: {FAISS_INDEX_PATH}")

    embeddings = get_embeddings()

    t0 = time.time()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    elapsed = round(time.time() - t0, 2)
    print(f"  Embedding complete in {elapsed}s")

    # Persist to disk
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"  ✓ Vector store saved to: {FAISS_INDEX_PATH}")

    return vectorstore


def load_vector_store() -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.
    Returns None if index doesn't exist yet.
    """
    index_file = FAISS_INDEX_PATH / "index.faiss"

    if not index_file.exists():
        print(f"  [VectorStore] No index found at {FAISS_INDEX_PATH}")
        print(f"  [VectorStore] Run: python build_rag.py to create it first")
        return None

    print(f"  [VectorStore] Loading index from {FAISS_INDEX_PATH}")
    embeddings  = get_embeddings()
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,   # safe — our own files
    )
    print(f"  [VectorStore] ✓ Index loaded successfully")
    return vectorstore


def get_retriever(vectorstore: Optional[FAISS] = None, k: int = RETRIEVER_K):
    """
    Returns a LangChain retriever from the vector store.
    Loads from disk if vectorstore not passed in.

    Args:
        vectorstore: Optional pre-loaded FAISS object
        k: Number of documents to retrieve per query

    Returns:
        LangChain VectorStoreRetriever
    """
    if vectorstore is None:
        vectorstore = load_vector_store()
        if vectorstore is None:
            raise RuntimeError(
                "Vector store not found. Run build_rag.py first."
            )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


def similarity_search(query: str, k: int = RETRIEVER_K, machine_id: str = None) -> List[Document]:
    """
    Direct similarity search — returns top-k docs for a query.
    Optionally filters by machine_id metadata.

    Args:
        query:      Natural language query
        k:          Number of results
        machine_id: Optional filter (e.g. "CNC-M01")

    Returns:
        List of relevant Document chunks with metadata
    """
    vectorstore = load_vector_store()
    if vectorstore is None:
        return []

    # Fetch more docs if filtering, since filter reduces results
    fetch_k = k * 3 if machine_id else k

    results = vectorstore.similarity_search(query, k=fetch_k)

    # Post-filter by machine_id if specified
    if machine_id:
        results = [
            doc for doc in results
            if doc.metadata.get("machine_id") in (machine_id, "ALL")
        ][:k]

    return results


def format_retrieved_context(docs: List[Document]) -> str:
    """
    Format retrieved docs into a clean string for LLM context injection.
    Includes source metadata so the LLM can cite sources.
    """
    if not docs:
        return "No relevant documents found in knowledge base."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = (
            f"[Source {i}] {meta.get('doc_type', 'Document')} | "
            f"File: {meta.get('filename', 'unknown')} | "
            f"Machine: {meta.get('machine_id', 'ALL')}"
        )
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n" + "\n\n---\n\n".join(context_parts)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    results = similarity_search(
        "What are the steps when error E01 spindle overload occurs?",
        k=3
    )
    print(f"\nTop {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.metadata}")
        print(doc.page_content[:200])