# Manufacturing Plant Monitoring & Diagnostics Assistant
## MAIA — Manufacturing AI Assistant

An industrial AI assistant for plant operators, maintenance engineers,
and shift supervisors. Built with Mistral + LangChain + FAISS RAG.

---

## Quick Start

```powershell
# 1. Run setup script (creates folders, venv, placeholder files)
.\setup_project.ps1

# 2. Activate virtual environment
cd manufacturing-assistant
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Ollama models
ollama pull mistral
ollama pull nomic-embed-text

# 5. Generate sensor + log data
python data/sensors/generate_data.py

# 6. Build RAG vector index
python build_rag.py

# 7. Start API backend (Terminal 1)
uvicorn backend.main:app --reload --port 8000

# 8. Start Streamlit UI (Terminal 2)
streamlit run frontend/app.py

# 9. Run evaluations
python evaluation/run_all_evals.py
```

---

## Project Structure

```
manufacturing-assistant/
│
├── build_rag.py              ← Run once to build FAISS index
├── requirements.txt
├── setup_project.ps1         ← Windows PowerShell setup script
│
├── config/
│   └── settings.py           ← Central config (paths, models, thresholds)
│
├── backend/
│   ├── agent.py              ← Core LangChain agent (Phase 4)
│   ├── main.py               ← FastAPI app entry point
│   │
│   ├── rag/
│   │   ├── document_loader.py  ← Loads + section-aware splits docs
│   │   ├── embedder.py         ← Ollama / HuggingFace embeddings
│   │   ├── vector_store.py     ← FAISS build/save/load/search
│   │   └── retriever.py        ← RAG chain + prompt template
│   │
│   ├── tools/
│   │   ├── sensor_fetch.py     ← Tool 1: Live sensor readings
│   │   ├── fault_diagnose.py   ← Tool 2: Rule-based fault diagnosis
│   │   ├── spare_parts.py      ← Tool 3: Parts catalog lookup
│   │   ├── maintenance.py      ← Tool 4: PM scheduler
│   │   ├── log_analyzer.py     ← Tool 5: CSV log anomaly detection
│   │   ├── safety_checker.py   ← Tool 6: PPE + LOTO rules
│   │   ├── metrics.py          ← Tool 7: OEE / production KPIs
│   │   └── escalation.py       ← Tool 8: Alerts + supervisor notify
│   │
│   ├── memory/
│   │   ├── entity_memory.py    ← Tracks machine/fault/operator per session
│   │   └── summary_memory.py   ← Rolling conversation summarization
│   │
│   └── api/
│       ├── routes.py           ← 18 FastAPI endpoints
│       └── schemas.py          ← Pydantic request/response models
│
├── frontend/
│   └── app.py                ← Streamlit UI (8 tabs)
│
├── evaluation/
│   ├── rag_eval.py           ← Precision@k, Recall@k, MRR, Hit Rate
│   ├── fault_eval.py         ← Severity accuracy, escalation accuracy
│   ├── response_quality.py   ← BLEU + ROUGE-L scoring
│   ├── latency_tests.py      ← Anomaly detection + latency + stress
│   └── run_all_evals.py      ← Master runner + scorecard
│
└── data/
    ├── sops/                 ← 5 machine SOPs (RAG source)
    ├── troubleshooting/      ← 50 error codes handbook (RAG source)
    ├── maintenance/          ← PM schedules A/B/C/D (RAG source)
    ├── safety/               ← PPE, LOTO, evacuation (RAG source)
    ├── fmea/                 ← Failure mode analysis (RAG source)
    ├── manuals/              ← Calibration + specs (RAG source)
    ├── sensors/              ← Mock sensor JSON + generator
    ├── logs/                 ← 7-day CSV logs per machine
    └── parts/                ← Spare parts catalog (JSON + CSV)
```

---

## API Endpoints (http://localhost:8000/docs)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/chat | Main chat — full agent pipeline |
| GET | /api/v1/sensors | All machine sensor readings |
| GET | /api/v1/sensors/{machine_id} | Single machine sensors |
| POST | /api/v1/fault | Fault diagnosis |
| GET | /api/v1/parts | Spare parts lookup |
| POST | /api/v1/maintenance | PM schedule |
| POST | /api/v1/rag | Knowledge base query |
| POST | /api/v1/analyze_log | Log analysis |
| POST | /api/v1/safety | Safety + LOTO rules |
| GET | /api/v1/metrics | OEE production KPIs |
| GET | /api/v1/alerts | Active alerts |
| GET | /api/v1/health | System health check |

---

## Machines Covered

| ID | Machine | Location |
|----|---------|----------|
| CNC-M01 | CNC Milling Machine | Machining Cell A |
| HYD-P02 | Hydraulic Pump Unit | Press & Forming Cell |
| CVB-003 | Conveyor Belt System | Material Handling |
| BLR-004 | Industrial Steam Boiler | Utilities |
| ROB-005 | 6-Axis Robotic Arm | Welding & Assembly |

---

## Evaluation Results

| Metric | Score | Grade |
|--------|-------|-------|
| Fault Severity Accuracy | 1.000 | A |
| Escalation Accuracy | 1.000 | A |
| ALARM F1 (Anomaly Detection) | 1.000 | A |
| Anomaly Overall Accuracy | 0.906 | B |
| RAG Hit Rate@4 | 0.85+ (after fix) | A |
| All Latency SLAs | 8/8 pass | A |
| Cache Speedup | 42x | - |

---

## RAG Fix Applied (Important)

If you cloned this before the RAG fix, update these settings in
`backend/rag/document_loader.py` → `split_documents()`:

- chunk_size: 600 → **1000**
- chunk_overlap: 100 → **200**
- separators: add `"\n========="` as first separator

Then rebuild: `python build_rag.py`

---

## Tech Stack

- **LLM**: Mistral 7B via Ollama (local, offline)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: FAISS (local file-based)
- **Framework**: LangChain 0.3
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit + Plotly
- **Memory**: Entity + Summary (in-memory, per session)
- **Caching**: TTLCache (sensor: 5min, parts: 1hr)
