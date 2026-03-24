# ─────────────────────────────────────────────────────
# backend/api/routes.py
# All FastAPI route handlers
# ─────────────────────────────────────────────────────

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MACHINES, LLM_MODEL
from backend.api.schemas import (
    ChatRequest, ChatResponse,
    FaultRequest, FaultResponse,
    MaintenanceRequest, MaintenanceResponse,
    PartsRequest, SafetyRequest,
    RAGRequest, LogAnalysisRequest,
    SensorResponse, MetricsResponse,
    AlertSummary, HealthResponse,
)
from backend.agent import get_agent, clear_session

router = APIRouter()


# ─────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API, LLM, and RAG are operational."""
    from backend.rag.vector_store import load_vector_store
    rag_ready = load_vector_store() is not None

    return HealthResponse(
        status    = "ok",
        llm_model = LLM_MODEL,
        rag_ready = rag_ready,
        machines  = list(MACHINES.keys()),
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ─────────────────────────────────────────────────────
# CHAT — Main conversational endpoint
# ─────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts operator messages and returns
    AI responses with tool data + RAG context integrated.
    """
    try:
        agent  = get_agent(request.session_id)
        result = agent.chat(
            user_message  = request.message,
            operator_name = request.operator_name,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/{session_id}", tags=["Chat"])
def clear_chat_session(session_id: str):
    """Clear conversation memory for a session."""
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.get("/chat/{session_id}/state", tags=["Chat"])
def get_session_state(session_id: str):
    """Get current session state (entity memory, turn count, etc.)"""
    agent = get_agent(session_id)
    return agent.get_session_state()


# ─────────────────────────────────────────────────────
# SENSORS
# ─────────────────────────────────────────────────────
@router.get("/sensors", tags=["Sensors"])
def get_all_sensors():
    """Get sensor readings for all machines."""
    from backend.tools.sensor_fetch import get_all_sensors
    return get_all_sensors()


@router.get("/sensors/{machine_id}", tags=["Sensors"])
def get_machine_sensors(machine_id: str, use_cache: bool = Query(default=True)):
    """Get current sensor readings for a specific machine."""
    from backend.tools.sensor_fetch import get_sensor_data
    machine_id = machine_id.upper()
    if machine_id not in MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    return get_sensor_data(machine_id, use_cache=use_cache)


@router.get("/sensors/summary/fleet", tags=["Sensors"])
def get_fleet_summary():
    """Get a quick status summary for all machines — for dashboard header."""
    from backend.tools.sensor_fetch import get_sensor_summary
    return get_sensor_summary()


# ─────────────────────────────────────────────────────
# FAULT DIAGNOSIS
# ─────────────────────────────────────────────────────
@router.post("/fault", tags=["Fault"])
def diagnose(request: FaultRequest):
    """
    Diagnose a fault from error code and/or sensor readings.
    Returns severity, failure mode, troubleshooting steps, parts.
    """
    from backend.tools.fault_diagnose import diagnose_fault
    machine_id = request.machine_id.upper()
    if machine_id not in MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    return diagnose_fault(machine_id, request.error_code, request.sensor_readings)


@router.get("/fault/{machine_id}/{error_code}", tags=["Fault"])
def get_fault_info(machine_id: str, error_code: str):
    """Quick GET endpoint to look up a specific error code."""
    from backend.tools.fault_diagnose import diagnose_fault
    return diagnose_fault(machine_id.upper(), error_code.upper())


# ─────────────────────────────────────────────────────
# SPARE PARTS
# ─────────────────────────────────────────────────────
@router.get("/parts", tags=["Parts"])
def get_parts(
    machine_id: Optional[str] = Query(default=None),
    query:      Optional[str] = Query(default=None),
):
    """Search spare parts by machine or keyword."""
    from backend.tools.spare_parts import lookup_parts_by_machine, search_parts
    if machine_id and not query:
        return lookup_parts_by_machine(machine_id.upper())
    elif query:
        return search_parts(query, machine_id.upper() if machine_id else None)
    else:
        # Return full catalog
        from backend.tools.spare_parts import _load_catalog
        return _load_catalog()


@router.get("/parts/low-stock", tags=["Parts"])
def get_low_stock(machine_id: Optional[str] = Query(default=None)):
    """Return all parts at or below reorder level."""
    from backend.tools.spare_parts import get_low_stock_parts
    return get_low_stock_parts(machine_id.upper() if machine_id else None)


# ─────────────────────────────────────────────────────
# MAINTENANCE
# ─────────────────────────────────────────────────────
@router.post("/maintenance", tags=["Maintenance"])
def get_maintenance_schedule(request: MaintenanceRequest):
    """
    Calculate what PM is due for a machine.
    Returns checklists for each overdue/due level.
    """
    from backend.tools.maintenance import calculate_pm_due
    machine_id = request.machine_id.upper()
    if machine_id not in MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    return calculate_pm_due(
        machine_id    = machine_id,
        current_hours = request.current_hours,
        last_pm_hours = request.last_pm_hours,
        last_pm_date  = request.last_pm_date,
    )


@router.get("/maintenance/{machine_id}", tags=["Maintenance"])
def get_quick_pm_status(machine_id: str, hours: int = Query(default=5000)):
    """Quick PM status check with estimated hours."""
    from backend.tools.maintenance import calculate_pm_due
    from backend.tools.sensor_fetch import get_sensor_data
    sensor = get_sensor_data(machine_id.upper())
    curr_hours = sensor.get("operating_hours", hours)
    return calculate_pm_due(machine_id.upper(), curr_hours)


# ─────────────────────────────────────────────────────
# RAG / KNOWLEDGE BASE
# ─────────────────────────────────────────────────────
@router.post("/rag", tags=["RAG"])
def rag_query(request: RAGRequest):
    """
    Direct RAG query against the knowledge base.
    Returns top-k relevant document chunks + LLM answer.
    """
    try:
        from backend.rag.retriever import rag_query as do_rag_query
        result = do_rag_query(request.query, machine_id=request.machine_id)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/rag/search", tags=["RAG"])
def rag_search(
    q:          str           = Query(..., description="Search query"),
    machine_id: Optional[str] = Query(default=None),
    k:          int           = Query(default=4),
):
    """Search knowledge base and return raw document chunks."""
    from backend.rag.vector_store import similarity_search
    docs = similarity_search(q, k=k, machine_id=machine_id)
    return [{"content": d.page_content[:500], "metadata": d.metadata} for d in docs]


# ─────────────────────────────────────────────────────
# LOG ANALYSIS
# ─────────────────────────────────────────────────────
@router.post("/analyze_log", tags=["Logs"])
def analyze_log_endpoint(request: LogAnalysisRequest):
    """
    Analyze machine log data.
    Provide file_content (CSV string) or leave empty to use stored 7-day log.
    """
    from backend.tools.log_analyzer import analyze_log
    machine_id = request.machine_id.upper()
    if machine_id not in MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    return analyze_log(machine_id, request.file_content)


@router.post("/analyze_log/upload/{machine_id}", tags=["Logs"])
async def upload_and_analyze(machine_id: str, file: UploadFile = File(...)):
    """Upload a CSV log file and analyze it."""
    from backend.tools.log_analyzer import analyze_log
    content = await file.read()
    return analyze_log(machine_id.upper(), content)


# ─────────────────────────────────────────────────────
# SAFETY
# ─────────────────────────────────────────────────────
@router.post("/safety", tags=["Safety"])
def get_safety_rules(request: SafetyRequest):
    """Get PPE requirements, LOTO steps, and hazard info for a task."""
    from backend.tools.safety_checker import check_safety
    machine_id = request.machine_id.upper()
    if machine_id not in MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    return check_safety(machine_id, request.task)


@router.get("/safety/{machine_id}", tags=["Safety"])
def get_machine_safety_profile(machine_id: str):
    """Get full safety profile for a machine."""
    from backend.tools.safety_checker import check_safety
    return check_safety(machine_id.upper(), "general maintenance")


# ─────────────────────────────────────────────────────
# METRICS / OEE
# ─────────────────────────────────────────────────────
@router.get("/metrics", tags=["Metrics"])
def get_fleet_metrics():
    """Get OEE and KPIs for all machines."""
    from backend.tools.metrics import get_fleet_metrics
    return get_fleet_metrics()


@router.get("/metrics/{machine_id}", tags=["Metrics"])
def get_machine_metrics(machine_id: str):
    """Get detailed OEE breakdown for a specific machine."""
    from backend.tools.metrics import compute_oee
    machine_id = machine_id.upper()
    if machine_id not in MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    return compute_oee(machine_id)


# ─────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────
@router.get("/alerts", tags=["Alerts"])
def get_alerts(
    machine_id:          Optional[str] = Query(default=None),
    unacknowledged_only: bool          = Query(default=False),
):
    """Return active alert log."""
    from backend.tools.escalation import get_active_alerts
    return get_active_alerts(machine_id, unacknowledged_only)


@router.get("/alerts/summary", tags=["Alerts"])
def get_alert_summary():
    """Return alert count summary for dashboard header."""
    from backend.tools.escalation import get_alert_summary
    return get_alert_summary()


@router.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
def acknowledge_alert(alert_id: str):
    """Mark an alert as acknowledged."""
    from backend.tools.escalation import acknowledge_alert
    success = acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"message": f"Alert {alert_id} acknowledged"}
