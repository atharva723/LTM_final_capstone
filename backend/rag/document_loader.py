# ─────────────────────────────────────────────────────
# backend/rag/document_loader.py
#
# PURPOSE:
#   Loads all .txt files from RAG document directories,
#   attaches metadata (source, machine_id, doc_type),
#   and splits them into chunks for embedding.
#
# USED BY: embedder.py → vector_store.py
# ─────────────────────────────────────────────────────

import os
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAG_DOCUMENT_DIRS, CHUNK_SIZE, CHUNK_OVERLAP

# ── Map folder names to human-readable doc types ─────
FOLDER_TO_DOCTYPE = {
    "sops":            "Standard Operating Procedure",
    "troubleshooting": "Troubleshooting Handbook",
    "maintenance":     "Preventive Maintenance Schedule",
    "safety":          "Safety Manual",
    "fmea":            "Failure Mode & Effects Analysis",
    "manuals":         "Machine Manual",
}

# ── Map file name keywords to machine IDs ─────────────
FILENAME_TO_MACHINE = {
    "cnc":       "CNC-M01",
    "hydraulic": "HYD-P02",
    "conveyor":  "CVB-003",
    "boiler":    "BLR-004",
    "robotic":   "ROB-005",
    "safety":    "ALL",
    "pm_schedule": "ALL",
    "troubleshooting": "ALL",
}


def infer_machine_id(filename: str) -> str:
    """Infer machine ID from filename keywords."""
    fname_lower = filename.lower()
    for keyword, machine_id in FILENAME_TO_MACHINE.items():
        if keyword in fname_lower:
            return machine_id
    return "UNKNOWN"


def load_documents() -> List[Document]:
    """
    Walk all RAG document directories, load .txt files,
    and return a list of LangChain Document objects with metadata.
    """
    raw_docs: List[Document] = []

    for doc_dir in RAG_DOCUMENT_DIRS:
        doc_dir = Path(doc_dir)
        if not doc_dir.exists():
            print(f"  [WARN] Directory not found: {doc_dir}")
            continue

        folder_name = doc_dir.name
        doc_type = FOLDER_TO_DOCTYPE.get(folder_name, folder_name.title())

        for file_path in sorted(doc_dir.glob("*.txt")):
            try:
                content = file_path.read_text(encoding="utf-8")
                machine_id = infer_machine_id(file_path.stem)

                doc = Document(
                    page_content=content,
                    metadata={
                        "source":      str(file_path),
                        "filename":    file_path.name,
                        "doc_type":    doc_type,
                        "machine_id":  machine_id,
                        "folder":      folder_name,
                    }
                )
                raw_docs.append(doc)
                print(f"  [LOADED] {file_path.name}  ({len(content)} chars)  machine={machine_id}")

            except Exception as e:
                print(f"  [ERROR] Could not load {file_path.name}: {e}")

    print(f"\n  Total documents loaded: {len(raw_docs)}")
    return raw_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Improved section-aware chunking.

    Root causes fixed vs naive 600-char splitting:
    1. chunk_size=1000  — keeps section header + steps in ONE chunk
                          (was 600 — header and steps split into separate chunks)
    2. chunk_overlap=200 — prevents step lists from being orphaned at boundaries
                          (was 100)
    3. SOP separator priority — splits on '=========' lines first, so each
       numbered section (STARTUP PROCEDURE, SHUTDOWN, etc.) stays whole
    4. Header injection — prepends nearest section title to chunks that
       contain step lists but no title, so embedding carries the topic
    """
    import re

    # Priority separators: SOP section dividers first, then paragraphs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 1000,
        chunk_overlap = 200,
        separators    = [
            "\n=========",   # SOP section dividers — split here first
            "\n─────────",   # secondary dividers
            "\n\n",          # paragraph breaks
            "\n",
            ". ", " ", "",
        ],
        length_function = len,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        # Remove pure separator lines (=====) that waste embedding space
        cleaned = re.sub(r'\n[=─]{8,}\n', '\n\n', doc.page_content)
        cleaned_doc = Document(page_content=cleaned, metadata=doc.metadata.copy())

        sub_chunks = splitter.split_documents([cleaned_doc])

        # Header injection pass
        last_section_title = ""
        for chunk in sub_chunks:
            content = chunk.page_content.strip()

            # Detect if this chunk opens with a section number — record it
            title_match = re.match(r'^(\d+\.\s+[A-Z][A-Z\s/&\-]+)', content)
            if title_match:
                last_section_title = title_match.group(1).strip()[:80]

            # Detect level-A/B/C/D maintenance blocks
            level_match = re.match(r'^(LEVEL [A-D] —[^\n]+)', content)
            if level_match:
                last_section_title = level_match.group(1).strip()[:80]

            # Inject section title into step-containing chunks that lack one
            has_steps = bool(re.search(
                r'Step \d+:|^\[\s*\]|\b(must|should|check|verify|inspect|replace)\b',
                content, re.IGNORECASE | re.MULTILINE
            ))
            has_own_title = bool(re.match(r'^\d+\.', content))

            if has_steps and not has_own_title and last_section_title:
                chunk.page_content = f"[Section: {last_section_title}]\n{chunk.page_content}"

            all_chunks.append(chunk)

    # Assign global chunk IDs
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"]     = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    print(f"  Total chunks after splitting: {len(all_chunks)}")
    return all_chunks


def load_and_split() -> List[Document]:
    """Convenience function — load + split in one call."""
    print("\n── Loading RAG Documents ──────────────────────")
    docs   = load_documents()
    print("\n── Splitting into Chunks ──────────────────────")
    chunks = split_documents(docs)
    return chunks


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    chunks = load_and_split()
    print(f"\nSample chunk:\n{'-'*50}")
    print(chunks[0].page_content[:300])
    print(f"\nMetadata: {chunks[0].metadata}")