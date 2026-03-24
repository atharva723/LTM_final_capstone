# RAG chain logic
# ─────────────────────────────────────────────────────
# backend/rag/retriever.py
#
# PURPOSE:
#   Builds the full RAG chain:
#     User Query → Retrieve Docs → Inject Context → LLM → Answer
#
#   Also provides a standalone retrieval function used
#   by individual tools (e.g. safety checker, fault tool).
# ─────────────────────────────────────────────────────

import sys
from pathlib import Path
from typing import Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LLM_MODEL, OLLAMA_BASE_URL, RETRIEVER_K
from backend.rag.vector_store import (
    load_vector_store,
    get_retriever,
    similarity_search,
    format_retrieved_context,
)

# ─────────────────────────────────────────────────────
# SYSTEM PROMPT TEMPLATE
# Injected into every RAG query so the LLM knows its role
# ─────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are an expert Manufacturing Plant AI Assistant.
You help plant operators, maintenance engineers, and shift supervisors
with machine operation, troubleshooting, safety procedures, and maintenance.

Use the retrieved context below to answer the question accurately.
If the context does not contain the answer, say:
"I don't have specific information on this in the knowledge base.
Please refer to the machine manual or contact your maintenance engineer."

Always:
- Give step-by-step answers when troubleshooting
- Mention PPE requirements when discussing maintenance tasks
- Cite which document or section your answer comes from
- Highlight any CRITICAL or SAFETY warnings prominently
- Be concise and practical — operators need fast answers

─────────────────────────────────────────────────────
RETRIEVED CONTEXT:
{context}
─────────────────────────────────────────────────────

OPERATOR QUESTION: {question}

ANSWER:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


def get_llm(temperature: float = 0.1) -> ChatOllama:
    """Return configured Ollama LLM."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,   # low temp = factual, consistent answers
    )


def build_rag_chain(vectorstore=None) -> RetrievalQA:
    """
    Build the full RetrievalQA chain.

    Flow:
      query → retriever (FAISS) → top-k docs →
      prompt template → Mistral LLM → answer

    Args:
        vectorstore: Optional pre-loaded FAISS store

    Returns:
        Configured RetrievalQA chain
    """
    if vectorstore is None:
        vectorstore = load_vector_store()
        if vectorstore is None:
            raise RuntimeError("Vector store not found. Run build_rag.py first.")

    retriever = get_retriever(vectorstore, k=RETRIEVER_K)
    llm       = get_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # stuff = inject all retrieved docs into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    print("  [RAG] Chain built successfully")
    return chain


def rag_query(
    query: str,
    chain: Optional[RetrievalQA] = None,
    machine_id: Optional[str] = None,
) -> dict:
    """
    Run a RAG query and return answer + sources.

    Args:
        query:      Operator's natural language question
        chain:      Optional pre-built chain (avoids rebuilding each call)
        machine_id: Optional machine filter for retrieval

    Returns:
        {
          "answer":   str,
          "sources":  list of source metadata dicts,
          "context":  formatted context string
        }
    """
    # If a machine_id filter is needed, use direct similarity search
    # instead of the chain retriever (chain retriever can't filter metadata)
    if machine_id:
        docs    = similarity_search(query, k=RETRIEVER_K, machine_id=machine_id)
        context = format_retrieved_context(docs)
        llm     = get_llm()

        filled_prompt = RAG_PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", query)
        response = llm.invoke(filled_prompt)
        answer   = response.content if hasattr(response, "content") else str(response)

        sources = [doc.metadata for doc in docs]
        return {"answer": answer, "sources": sources, "context": context}

    # Standard chain query (no machine filter)
    if chain is None:
        chain = build_rag_chain()

    result = chain.invoke({"query": query})

    sources = [
        doc.metadata for doc in result.get("source_documents", [])
    ]

    return {
        "answer":  result.get("result", ""),
        "sources": sources,
        "context": format_retrieved_context(result.get("source_documents", [])),
    }


# ── Convenience wrappers for specific use cases ──────

def get_safety_steps(machine_id: str, task: str) -> dict:
    """Retrieve safety/PPE steps for a specific task on a machine."""
    query = f"What are the safety steps and PPE required for {task} on {machine_id}?"
    return rag_query(query, machine_id=machine_id)


def get_troubleshooting_steps(machine_id: str, error_code: str) -> dict:
    """Retrieve troubleshooting steps for a given error code."""
    query = f"What are the troubleshooting steps for error code {error_code} on machine {machine_id}?"
    return rag_query(query, machine_id=machine_id)


def get_maintenance_schedule(machine_id: str) -> dict:
    """Retrieve PM schedule for a machine."""
    query = f"What is the preventive maintenance schedule for machine {machine_id}? List all levels."
    return rag_query(query, machine_id=machine_id)


def get_startup_procedure(machine_id: str) -> dict:
    """Retrieve startup procedure from SOP."""
    query = f"What is the startup procedure for machine {machine_id}? Give step by step."
    return rag_query(query, machine_id=machine_id)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    print("\n── Testing RAG Retriever ──────────────────────")
    result = get_troubleshooting_steps("CNC-M01", "E01")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {[s['filename'] for s in result['sources']]}")