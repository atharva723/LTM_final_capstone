# ─────────────────────────────────────────────────────
# backend/api/schemas.py
# Pydantic models for all API request/response bodies
# ─────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ─────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:       str            = Field(..., description="Operator's question or message")
    session_id:    str            = Field(default="default", description="Session identifier")
    operator_name: Optional[str] = Field(default=None, description="Name of operator sending message")
    machine_id:    Optional[str] = Field(default=None, description="Override machine context")

    class Config:
        json_schema_extra = {
            "example": {
                "message":       "Error E01 on CNC machine — what do I do?",
                "session_id":    "shift_morning_001",
                "operator_name": "Rajesh Kumar",
                "machine_id":    "CNC-M01",
            }
        }


class FaultRequest(BaseModel):
    machine_id:      str                       = Field(..., description="Machine ID e.g. CNC-M01")
    error_code:      Optional[str]             = Field(default=None, description="Error code e.g. E01")
    sensor_readings: Optional[Dict[str, float]] = Field(default=None, description="Current sensor values")

    class Config:
        json_schema_extra = {
            "example": {
                "machine_id":   "CNC-M01",
                "error_code":   "E01",
                "sensor_readings": {"temperature_c": 78.5, "vibration_mm_s": 4.2},
            }
        }


class MaintenanceRequest(BaseModel):
    machine_id:     str           = Field(..., description="Machine ID")
    current_hours:  int           = Field(..., description="Total operating hours on machine")
    last_pm_hours:  Optional[int] = Field(default=None, description="Hours at last PM (optional)")
    last_pm_date:   Optional[str] = Field(default=None, description="Date of last PM YYYY-MM-DD")


class PartsRequest(BaseModel):
    machine_id:  Optional[str] = Field(default=None, description="Filter by machine")
    query:       Optional[str] = Field(default=None, description="Search keyword")
    part_ids:    Optional[List[str]] = Field(default=None, description="Specific part IDs to look up")


class SafetyRequest(BaseModel):
    machine_id:  str = Field(..., description="Machine ID")
    task:        str = Field(..., description="Task description e.g. 'tool change', 'hose replacement'")


class RAGRequest(BaseModel):
    query:      str           = Field(..., description="Natural language question")
    machine_id: Optional[str] = Field(default=None, description="Optional machine filter")
    k:          int           = Field(default=4, description="Number of documents to retrieve")


class LogAnalysisRequest(BaseModel):
    machine_id: str           = Field(..., description="Machine ID")
    file_content: Optional[str] = Field(default=None, description="CSV content as string (optional — uses stored log if not provided)")


# ─────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    response:   str
    intents:    List[str]
    tools_used: List[str]
    sources:    List[str]
    machine_id: Optional[str]
    alerts:     List[Dict[str, Any]]
    metadata:   Dict[str, Any]


class SensorResponse(BaseModel):
    machine_id:      str
    machine_name:    str
    overall_status:  str
    timestamp:       str
    sensor_readings: Dict[str, Any]
    active_error_codes: List[str]
    alerts:          List[Dict[str, Any]]


class FaultResponse(BaseModel):
    machine_id:          str
    status:              str
    severity:            str
    severity_guidance:   Optional[str]
    diagnoses:           List[Dict[str, Any]]
    recommended_parts:   List[str]
    escalate:            bool
    summary:             str
    timestamp:           str


class MaintenanceResponse(BaseModel):
    machine_id:     str
    machine_name:   str
    urgency:        str
    levels_due:     List[Dict[str, Any]]
    schedule:       List[Dict[str, Any]]
    due_count:      int


class MetricsResponse(BaseModel):
    machine_id:   str
    machine_name: str
    oee:          Dict[str, Any]
    availability: Dict[str, Any]
    performance:  Dict[str, Any]
    quality:      Dict[str, Any]


class AlertSummary(BaseModel):
    total:     int
    unacked:   int
    critical:  int
    high:      int
    medium:    int
    last_alert: Optional[str]


class HealthResponse(BaseModel):
    status:    str
    llm_model: str
    rag_ready: bool
    machines:  List[str]
    timestamp: str
