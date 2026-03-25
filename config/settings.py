# ─────────────────────────────────────────────────────
# config/settings.py
# Central configuration for the Manufacturing Assistant
# ─────────────────────────────────────────────────────

import os
from pathlib import Path

# ── Base Paths ──────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
CACHE_DIR       = BASE_DIR / "cache"

# ── RAG Document Sources ─────────────────────────────
RAG_DOCUMENT_DIRS = [
    DATA_DIR / "sops",
    DATA_DIR / "troubleshooting",
    DATA_DIR / "maintenance",
    DATA_DIR / "safety",
    DATA_DIR / "fmea",
    DATA_DIR / "manuals",
]

# ── Ollama / LLM Settings ────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL        = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ── RAG Settings ─────────────────────────────────────
CHUNK_SIZE       = 600       # characters per chunk
CHUNK_OVERLAP    = 100       # overlap between chunks
RETRIEVER_K      = 4         # number of docs to retrieve per query
VECTOR_STORE_NAME = "manufacturing_faiss"

# ── Sensor Data ──────────────────────────────────────
SENSOR_DATA_PATH = DATA_DIR / "sensors" / "mock_sensor_data.json"
PARTS_JSON_PATH  = DATA_DIR / "parts" / "spare_parts_catalog.json"
PARTS_CSV_PATH   = DATA_DIR / "parts" / "spare_parts_catalog.csv"
LOGS_DIR         = DATA_DIR / "logs"

# ── Supported Machines ───────────────────────────────
MACHINES = {
    "CNC-M01":  "CNC Milling Machine",
    "HYD-P02":  "Hydraulic Pump Unit",
    "CVB-003":  "Conveyor Belt System",
    "BLR-004":  "Industrial Steam Boiler",
    "ROB-005":  "6-Axis Robotic Arm",
}

# ── Sensor Thresholds (for anomaly detection) ────────
SENSOR_THRESHOLDS = {
    "CNC-M01": {
        "temperature_c":      {"warning": 65,   "alarm": 80},
        "vibration_mm_s":     {"warning": 3.5,  "alarm": 5.0},
        "spindle_rpm":        {"warning": 8200, "alarm": 8500},
        "coolant_flow_l_min": {"warning": 6,    "alarm": 3},
        "power_kw":           {"warning": 18,   "alarm": 22},
    },
    "HYD-P02": {
        "pressure_bar":       {"warning": 260,  "alarm": 280},
        "temperature_c":      {"warning": 65,   "alarm": 75},
        "flow_l_min":         {"warning": 15,   "alarm": 8},
        "motor_current_a":    {"warning": 45,   "alarm": 55},
        "vibration_mm_s":     {"warning": 4.0,  "alarm": 6.0},
    },
    "CVB-003": {
        "belt_speed_m_s":     {"warning": 1.6,  "alarm": 1.8},
        "motor_temp_c":       {"warning": 75,   "alarm": 85},
        "motor_current_a":    {"warning": 28,   "alarm": 35},
        "belt_tension_n":     {"warning": 600,  "alarm": 400},
        "vibration_mm_s":     {"warning": 3.0,  "alarm": 5.0},
    },
    "BLR-004": {
        "steam_pressure_bar": {"warning": 10.5, "alarm": 11.0},
        "water_level_pct":    {"warning": 30,   "alarm": 15},
        "flue_gas_temp_c":    {"warning": 260,  "alarm": 300},
        "fuel_pressure_mbar": {"warning": 80,   "alarm": 60},
        "feedwater_temp_c":   {"warning": 70,   "alarm": 60},
    },
    "ROB-005": {
        "joint_temp_c":       {"warning": 65,   "alarm": 75},
        "motor_current_a":    {"warning": 20,   "alarm": 25},
        "tcp_speed_mm_s":     {"warning": 1900, "alarm": 2100},
        "air_pressure_bar":   {"warning": 4.5,  "alarm": 4.0},
        "vibration_mm_s":     {"warning": 2.0,  "alarm": 3.5},
    },
}

# ── Memory Settings ──────────────────────────────────
MEMORY_MAX_TOKENS        = 2000
SUMMARY_TRIGGER_MESSAGES = 10    # summarize after N messages

# ── Cache Settings ───────────────────────────────────
CACHE_TTL_SECONDS        = 300   # 5 minutes for sensor cache
CACHE_MAX_SIZE           = 128

# ── FastAPI Settings ─────────────────────────────────
API_HOST  = "0.0.0.0"
API_PORT  = 8000
API_TITLE = "Manufacturing Assistant API"

# ── Escalation Settings ──────────────────────────────
ESCALATION_EMAIL         = "supervisor@plant.local"
ESCALATION_SEVERITY      = "HIGH"   # escalate if severity >= this