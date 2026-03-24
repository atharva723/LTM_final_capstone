# ─────────────────────────────────────────────────────
# backend/tools/sensor_fetch.py
#
# TOOL 1 — Sensor Data Fetch Tool
#
# PURPOSE:
#   Reads mock_sensor_data.json to return current sensor
#   readings for a given machine. Simulates a live IoT
#   sensor API. In production, replace json read with
#   real API call (ThingSpeak, Azure IoT Hub, etc.)
#
# USED BY: agent.py, FastAPI /sensors endpoint
# ─────────────────────────────────────────────────────

import json
import time
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from cachetools import TTLCache

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SENSOR_DATA_PATH, SENSOR_THRESHOLDS, CACHE_TTL_SECONDS, MACHINES

# ── TTL Cache: sensor readings cached for 5 minutes ──
# Simulates real IoT polling — avoids redundant reads
_sensor_cache = TTLCache(maxsize=20, ttl=CACHE_TTL_SECONDS)


def _add_live_noise(value: float, pct: float = 0.02) -> float:
    """Add ±2% noise to simulate real-time sensor fluctuation."""
    return round(value * (1 + random.uniform(-pct, pct)), 2)


def _classify_reading(machine_id: str, sensor: str, value: float) -> dict:
    """
    Compare a reading against thresholds and return status + message.
    Returns: { status, message, action_needed }
    """
    thresholds = SENSOR_THRESHOLDS.get(machine_id, {}).get(sensor, {})
    if not thresholds:
        return {"status": "UNKNOWN", "message": "No threshold defined", "action_needed": False}

    alarm_val   = thresholds.get("alarm")
    warning_val = thresholds.get("warning")

    # Some sensors are dangerous when LOW (e.g. coolant_flow, belt_tension, water_level)
    low_is_bad = sensor in (
        "coolant_flow_l_min", "belt_tension_n", "water_level_pct",
        "flow_l_min", "fuel_pressure_mbar", "feedwater_temp_c", "air_pressure_bar"
    )

    if low_is_bad:
        if value <= alarm_val:
            return {"status": "ALARM",   "message": f"{sensor} critically low: {value}", "action_needed": True}
        elif value <= warning_val:
            return {"status": "WARNING", "message": f"{sensor} low: {value}", "action_needed": True}
    else:
        if value >= alarm_val:
            return {"status": "ALARM",   "message": f"{sensor} critically high: {value}", "action_needed": True}
        elif value >= warning_val:
            return {"status": "WARNING", "message": f"{sensor} elevated: {value}", "action_needed": True}

    return {"status": "NORMAL", "message": f"{sensor} within normal range", "action_needed": False}


def get_sensor_data(machine_id: str, use_cache: bool = True) -> dict:
    """
    Fetch current sensor readings for a machine.

    Args:
        machine_id:  e.g. "CNC-M01", "HYD-P02", "CVB-003"
        use_cache:   Return cached value if within TTL window

    Returns:
        Full sensor data dict with readings, statuses, and alerts
    """
    machine_id = machine_id.upper().strip()

    # Validate machine ID
    if machine_id not in MACHINES:
        return {
            "error": f"Unknown machine ID: {machine_id}",
            "valid_machines": list(MACHINES.keys()),
        }

    # Check cache first
    cache_key = f"sensor_{machine_id}"
    if use_cache and cache_key in _sensor_cache:
        cached = _sensor_cache[cache_key].copy()
        cached["from_cache"] = True
        cached["cache_age_s"] = round(time.time() - cached.get("_fetched_at", time.time()), 1)
        return cached

    # Load from JSON data file
    try:
        with open(SENSOR_DATA_PATH, "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        return {"error": f"Sensor data file not found: {SENSOR_DATA_PATH}"}
    except json.JSONDecodeError as e:
        return {"error": f"Sensor data file corrupted: {e}"}

    if machine_id not in all_data:
        return {"error": f"Machine {machine_id} not found in sensor data"}

    machine_data = all_data[machine_id].copy()

    # Add live noise to sensor readings to simulate real-time fluctuation
    enriched_readings = {}
    sensor_alerts     = []
    overall_status    = "NORMAL"

    for sensor, value in machine_data["sensor_readings"].items():
        live_value = _add_live_noise(value)
        classification = _classify_reading(machine_id, sensor, live_value)

        enriched_readings[sensor] = {
            "value":          live_value,
            "status":         classification["status"],
            "message":        classification["message"],
            "action_needed":  classification["action_needed"],
        }

        if classification["status"] == "ALARM":
            overall_status = "ALARM"
            sensor_alerts.append({
                "severity": "ALARM",
                "sensor":   sensor,
                "value":    live_value,
                "message":  classification["message"],
            })
        elif classification["status"] == "WARNING" and overall_status != "ALARM":
            overall_status = "WARNING"
            sensor_alerts.append({
                "severity": "WARNING",
                "sensor":   sensor,
                "value":    live_value,
                "message":  classification["message"],
            })

    result = {
        "machine_id":          machine_id,
        "machine_name":        machine_data["machine_name"],
        "location":            machine_data["location"],
        "overall_status":      overall_status,
        "status":              machine_data["status"],
        "operator":            machine_data["operator"],
        "shift":               machine_data["shift"],
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operating_hours":     machine_data["operating_hours_total"],
        "hours_since_last_pm": machine_data["hours_since_last_pm"],
        "last_maintenance":    machine_data["last_maintenance_date"],
        "sensor_readings":     enriched_readings,
        "active_error_codes":  machine_data["active_error_codes"],
        "alerts":              sensor_alerts,
        "alert_count":         len(sensor_alerts),
        "from_cache":          False,
        "_fetched_at":         time.time(),
    }

    # Store in cache
    _sensor_cache[cache_key] = result

    return result


def get_all_sensors() -> dict:
    """Fetch sensor readings for ALL machines at once."""
    results = {}
    for machine_id in MACHINES:
        results[machine_id] = get_sensor_data(machine_id)
    return results


def get_sensor_summary() -> list:
    """
    Returns a concise status summary of all machines.
    Good for dashboard overview and shift briefings.
    """
    summary = []
    for machine_id in MACHINES:
        data = get_sensor_data(machine_id)
        if "error" in data:
            summary.append({"machine_id": machine_id, "status": "ERROR", "alerts": 0})
        else:
            summary.append({
                "machine_id":     machine_id,
                "machine_name":   data["machine_name"],
                "overall_status": data["overall_status"],
                "alert_count":    data["alert_count"],
                "operator":       data["operator"],
                "shift":          data["shift"],
                "error_codes":    data["active_error_codes"],
            })
    return summary


def format_sensor_report(machine_id: str) -> str:
    """
    Returns a human-readable sensor report string.
    Used by the LLM agent to inject sensor context into responses.
    """
    data = get_sensor_data(machine_id)
    if "error" in data:
        return f"Error fetching sensor data for {machine_id}: {data['error']}"

    lines = [
        f"=== Sensor Report: {data['machine_name']} ({machine_id}) ===",
        f"Status: {data['overall_status']}  |  Operator: {data['operator']}  |  Shift: {data['shift']}",
        f"Timestamp: {data['timestamp']}  |  Total Hours: {data['operating_hours']}  |  Hours Since PM: {data['hours_since_last_pm']}",
        f"Active Error Codes: {', '.join(data['active_error_codes']) or 'None'}",
        "",
        "Sensor Readings:",
    ]

    for sensor, info in data["sensor_readings"].items():
        status_icon = {"NORMAL": "✓", "WARNING": "⚠", "ALARM": "✗"}.get(info["status"], "?")
        lines.append(f"  {status_icon} {sensor:<25} {info['value']:>10}   [{info['status']}]")

    if data["alerts"]:
        lines.append("")
        lines.append("ACTIVE ALERTS:")
        for alert in data["alerts"]:
            lines.append(f"  [{alert['severity']}] {alert['sensor']}: {alert['message']}")

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    print(format_sensor_report("CNC-M01"))
    print()
    print("=== Fleet Summary ===")
    for item in get_sensor_summary():
        print(f"  {item['machine_id']}  {item['overall_status']:<8}  alerts={item['alert_count']}")
