# ─────────────────────────────────────────────────────
# build_rag.py  (run this ONCE to build the vector store)
#
# USAGE:
#   python build_rag.py
#
# WHAT IT DOES:
#   1. Loads all .txt documents from data/ directories
#   2. Splits them into chunks
#   3. Embeds each chunk using Ollama / HuggingFace
#   4. Saves FAISS index to vector_store/
#
# WHEN TO RE-RUN:
#   - After adding new SOP or document files to data/
#   - After changing CHUNK_SIZE in settings.py
# ─────────────────────────────────────────────────────

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from backend.rag.document_loader import load_and_split
from backend.rag.vector_store import build_vector_store


def main():
    print("=" * 55)
    print("  Manufacturing Assistant — RAG Index Builder")
    print("=" * 55)

    t_start = time.time()

    # Step 1: Load and split all documents
    chunks = load_and_split()

    if not chunks:
        print("\n[ERROR] No documents found. Check data/ directories.")
        sys.exit(1)

    # Step 2: Embed and build FAISS index
    vectorstore = build_vector_store(chunks)

    # Step 3: Quick smoke test
    print("\n── Smoke Test ─────────────────────────────────")
    test_queries = [
        "What PPE is required for CNC machine maintenance?",
        "Troubleshooting steps for conveyor belt misalignment E23",
        "Hydraulic pump overpressure error E11 steps",
        "Daily maintenance checklist for boiler",
        "LOTO lockout tagout procedure",
    ]

    for q in test_queries:
        results = vectorstore.similarity_search(q, k=2)
        top_source = results[0].metadata.get("filename", "?") if results else "NO RESULT"
        print(f"  Q: {q[:55]:<55} → {top_source}")

    elapsed = round(time.time() - t_start, 2)
    print(f"\n{'='*55}")
    print(f"  ✓ RAG index built in {elapsed}s")
    print(f"  ✓ Chunks indexed: {len(chunks)}")
    print(f"  ✓ Index saved to: vector_store/manufacturing_faiss/")
    print(f"  Next step: Run FastAPI → python backend/main.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()