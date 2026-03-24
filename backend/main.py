# ─────────────────────────────────────────────────────
# backend/main.py
# FastAPI application entry point
#
# START:
#   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
#
# DOCS:
#   http://localhost:8000/docs   (Swagger UI)
#   http://localhost:8000/redoc  (ReDoc)
# ─────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



from config.settings import API_HOST, API_PORT, API_TITLE
from backend.api.routes import router


# ─────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before accepting requests."""
    print("\n" + "=" * 55)
    print(f"  {API_TITLE}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Pre-warm: try loading vector store at startup
    try:
        from backend.rag.vector_store import load_vector_store
        vs = load_vector_store()
        if vs:
            print("  ✓ RAG vector store loaded")
        else:
            print("  ⚠ RAG vector store not found — run build_rag.py first")
    except Exception as e:
        print(f"  ⚠ RAG load failed: {e}")

    print("  ✓ API ready — visit http://localhost:8000/docs\n")
    yield

    print("\n  API shutting down...")


# ─────────────────────────────────────────────────────
# APP CREATION
# ─────────────────────────────────────────────────────
app = FastAPI(
    title       = API_TITLE,
    description = (
        "AI-powered manufacturing plant assistant. "
        "Provides real-time sensor monitoring, fault diagnosis, "
        "maintenance scheduling, safety guidance, and production KPIs "
        "via llama3.2 LLM + LangChain + FAISS RAG."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow Streamlit frontend to call API ───────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to frontend URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Register all routes ───────────────────────────────
app.include_router(router, prefix="/api/v1")


# ── Root endpoint ──────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "service":  API_TITLE,
        "version":  "1.0.0",
        "status":   "running",
        "docs":     "/docs",
        "endpoints": [
            "POST /api/v1/chat",
            "GET  /api/v1/sensors",
            "GET  /api/v1/sensors/{machine_id}",
            "POST /api/v1/fault",
            "GET  /api/v1/parts",
            "POST /api/v1/maintenance",
            "POST /api/v1/rag",
            "POST /api/v1/analyze_log",
            "POST /api/v1/safety",
            "GET  /api/v1/metrics",
            "GET  /api/v1/alerts",
            "GET  /api/v1/health",
        ],
    }


# ── Run directly ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = True,
        workers = 1,
    )
