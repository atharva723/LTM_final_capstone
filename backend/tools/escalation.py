# ─────────────────────────────────────────────────────
# backend/tools/escalation.py
#
# TOOL 8 — Notification / Escalation Tool
#
# PURPOSE:
#   Evaluates fault severity and decides whether to
#   escalate. Sends notifications via:
#     • Console log (always — for demo)
#     • Email simulation (configurable)
#     • In-memory alert log (visible in dashboard)
#
#   Auto-triggers when:
#     • Fault severity == CRITICAL
#     • Same error code repeats > 3 times in 1 hour
#     • Operator has not acknowledged alert in 5 min
#
# USED BY: agent.py, sensor_fetch.py, fault_diagnose.py
# ─────────────────────────────────────────────────────

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import ESCALATION_EMAIL, ESCALATION_SEVERITY, MACHINES

# ── In-memory alert log (persists for session) ────────
# Stores last 100 alerts — visible in Streamlit dashboard
_alert_log: deque = deque(maxlen=100)

# ── Repeat fault tracker ──────────────────────────────
# Tracks error code occurrences to detect repeat faults
_fault_tracker: dict = {}   # {machine_id: {error_code: [timestamps]}}

# ── Severity ranking ──────────────────────────────────
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
ESCALATION_THRESHOLD_RANK = SEVERITY_RANK.get(ESCALATION_SEVERITY, 2)

# ── Contact directory ─────────────────────────────────
CONTACTS = {
    "maintenance_supervisor": {"name": "Maintenance Supervisor", "ext": "201", "email": "maintenance@plant.local"},
    "shift_supervisor":       {"name": "Shift Supervisor",       "ext": "301", "email": "shift@plant.local"},
    "hse_officer":            {"name": "HSE Officer",            "ext": "101", "email": "hse@plant.local"},
    "boiler_inspector":       {"name": "Boiler Inspector",       "ext": "410", "email": "boiler@plant.local"},
}

# Machine → primary contact mapping
MACHINE_CONTACTS = {
    "CNC-M01":  "maintenance_supervisor",
    "HYD-P02":  "maintenance_supervisor",
    "CVB-003":  "maintenance_supervisor",
    "BLR-004":  "boiler_inspector",
    "ROB-005":  "maintenance_supervisor",
}


def _get_contact(machine_id: str) -> dict:
    """Get the primary contact for a machine."""
    contact_key = MACHINE_CONTACTS.get(machine_id, "maintenance_supervisor")
    return CONTACTS.get(contact_key, CONTACTS["maintenance_supervisor"])


def _track_fault(machine_id: str, error_code: str) -> int:
    """
    Track fault occurrences and return count in last 60 minutes.
    Used to detect repeating faults.
    """
    now = datetime.now()
    key = machine_id

    if key not in _fault_tracker:
        _fault_tracker[key] = {}
    if error_code not in _fault_tracker[key]:
        _fault_tracker[key][error_code] = []

    # Append current time
    _fault_tracker[key][error_code].append(now)

    # Prune entries older than 60 minutes
    cutoff = now - timedelta(hours=1)
    _fault_tracker[key][error_code] = [
        t for t in _fault_tracker[key][error_code] if t > cutoff
    ]

    return len(_fault_tracker[key][error_code])


def _log_alert(alert: dict):
    """Add alert to in-memory log and print to console."""
    _alert_log.appendleft(alert)

    # Console output (simulates notification system)
    border = "═" * 60
    print(f"\n{border}")
    print(f"  🚨 PLANT ALERT — {alert['severity']}")
    print(f"  Machine:  {alert['machine_id']} — {alert.get('machine_name', '')}")
    print(f"  Time:     {alert['timestamp']}")
    print(f"  Issue:    {alert['message']}")
    if alert.get("error_code"):
        print(f"  Error:    {alert['error_code']}")
    print(f"  Notify:   {alert.get('contact_name')} (Ext: {alert.get('contact_ext')})")
    if alert.get("action_required"):
        print(f"  Action:   {alert['action_required']}")
    print(f"{border}\n")


def _simulate_email(to: str, subject: str, body: str):
    """Simulate sending an email (console log only — replace with SMTP in production)."""
    print(f"\n[EMAIL SIMULATION]")
    print(f"  TO:      {to}")
    print(f"  SUBJECT: {subject}")
    print(f"  BODY:    {body[:200]}...")


def evaluate_and_escalate(
    machine_id: str,
    severity: str,
    message: str,
    error_code: Optional[str] = None,
    operator_name: Optional[str] = None,
    send_email: bool = False,
) -> dict:
    """
    Core escalation logic. Decides whether to escalate
    based on severity, and records the alert.

    Args:
        machine_id:    e.g. "CNC-M01"
        severity:      "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
        message:       Description of the fault
        error_code:    Optional error code e.g. "E01"
        operator_name: Current operator's name
        send_email:    Whether to simulate email send

    Returns:
        Escalation result dict
    """
    machine_id = machine_id.upper().strip()
    severity   = severity.upper().strip()
    contact    = _get_contact(machine_id)

    # Track repeat faults
    repeat_count = _track_fault(machine_id, error_code or "GENERIC") if error_code else 0
    repeat_fault = repeat_count >= 3

    # Decide if escalation is needed
    should_escalate = (
        SEVERITY_RANK.get(severity, 0) >= ESCALATION_THRESHOLD_RANK
        or repeat_fault
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine action based on severity
    action_map = {
        "CRITICAL": "STOP machine immediately. Evacuate if needed. Call supervisor NOW.",
        "HIGH":     "Stop machine. Do not restart without maintenance clearance.",
        "MEDIUM":   "Monitor closely. Schedule correction within current shift.",
        "LOW":      "Log and monitor. Schedule for next planned maintenance.",
    }

    alert = {
        "alert_id":       f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{machine_id}",
        "machine_id":     machine_id,
        "machine_name":   MACHINES.get(machine_id, machine_id),
        "severity":       severity,
        "message":        message,
        "error_code":     error_code,
        "operator":       operator_name or "Unknown",
        "timestamp":      timestamp,
        "escalated":      should_escalate,
        "repeat_fault":   repeat_fault,
        "repeat_count":   repeat_count,
        "contact_name":   contact["name"],
        "contact_ext":    contact["ext"],
        "contact_email":  contact["email"],
        "action_required": action_map.get(severity, "Review and log"),
        "acknowledged":   False,
    }

    # Log the alert
    _log_alert(alert)

    # Send email simulation for HIGH/CRITICAL
    if send_email and should_escalate:
        subject = f"[{severity}] Plant Alert: {machine_id} — {message[:50]}"
        body = (
            f"Machine: {machine_id} — {MACHINES.get(machine_id, '')}\n"
            f"Time: {timestamp}\n"
            f"Severity: {severity}\n"
            f"Issue: {message}\n"
            f"Error Code: {error_code or 'N/A'}\n"
            f"Operator: {operator_name or 'Unknown'}\n"
            f"Repeat Fault: {'YES — ' + str(repeat_count) + ' times in last hour' if repeat_fault else 'No'}\n"
            f"Action Required: {action_map.get(severity, 'Review')}\n"
        )
        _simulate_email(contact["email"], subject, body)

    return {
        "escalated":      should_escalate,
        "alert_id":       alert["alert_id"],
        "severity":       severity,
        "contact":        contact,
        "repeat_fault":   repeat_fault,
        "repeat_count":   repeat_count,
        "action_required": action_map.get(severity, "Review and log"),
        "message":        f"Alert logged. {'ESCALATED to ' + contact['name'] + ' (Ext: ' + contact['ext'] + ')' if should_escalate else 'No escalation required.'}",
    }


def auto_escalate_from_diagnosis(diagnosis: dict, operator_name: Optional[str] = None) -> dict:
    """
    Convenience: feed a fault_diagnose.py result directly
    and auto-escalate if the diagnosis warrants it.
    """
    if diagnosis.get("status") == "NO_FAULT":
        return {"escalated": False, "message": "No fault detected — no escalation needed"}

    machine_id  = diagnosis.get("machine_id", "UNKNOWN")
    severity    = diagnosis.get("severity", "LOW")
    summary     = diagnosis.get("summary", "Fault detected")
    top_diag    = diagnosis.get("diagnoses", [{}])[0]
    error_code  = top_diag.get("code")

    return evaluate_and_escalate(
        machine_id    = machine_id,
        severity      = severity,
        message       = summary,
        error_code    = error_code,
        operator_name = operator_name,
        send_email    = severity in ("HIGH", "CRITICAL"),
    )


def get_active_alerts(machine_id: Optional[str] = None, unacknowledged_only: bool = False) -> list:
    """Return current alert log, optionally filtered."""
    alerts = list(_alert_log)
    if machine_id:
        alerts = [a for a in alerts if a["machine_id"] == machine_id.upper()]
    if unacknowledged_only:
        alerts = [a for a in alerts if not a["acknowledged"]]
    return alerts


def acknowledge_alert(alert_id: str) -> bool:
    """Mark an alert as acknowledged by operator."""
    for alert in _alert_log:
        if alert["alert_id"] == alert_id:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
    return False


def get_alert_summary() -> dict:
    """Summary of all active alerts — for dashboard header."""
    alerts  = list(_alert_log)
    critical = sum(1 for a in alerts if a["severity"] == "CRITICAL" and not a["acknowledged"])
    high     = sum(1 for a in alerts if a["severity"] == "HIGH"     and not a["acknowledged"])
    medium   = sum(1 for a in alerts if a["severity"] == "MEDIUM"   and not a["acknowledged"])
    return {
        "total":     len(alerts),
        "unacked":   sum(1 for a in alerts if not a["acknowledged"]),
        "critical":  critical,
        "high":      high,
        "medium":    medium,
        "last_alert": alerts[0]["timestamp"] if alerts else None,
    }


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    # Test critical escalation
    result = evaluate_and_escalate(
        machine_id="CNC-M01",
        severity="CRITICAL",
        message="Spindle bearing temperature exceeded 80°C — thermal runaway risk",
        error_code="E08",
        operator_name="Rajesh Kumar",
        send_email=True,
    )
    print(f"\nEscalation result: {json.dumps(result, indent=2)}")

    # Test medium — no escalation
    result2 = evaluate_and_escalate(
        machine_id="CVB-003",
        severity="MEDIUM",
        message="Belt misalignment detected — drifting 25mm to right",
        error_code="E23",
    )
    print(f"\nEscalation result: {json.dumps(result2, indent=2)}")

    print(f"\nAlert Summary: {json.dumps(get_alert_summary(), indent=2)}")
