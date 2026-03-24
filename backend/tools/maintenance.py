# ─────────────────────────────────────────────────────
# backend/tools/maintenance.py
#
# TOOL 4 — Maintenance Scheduler Tool
#
# PURPOSE:
#   Given a machine's operating hours and last PM date,
#   calculates what maintenance is due now, and returns
#   the specific checklist for each level (A/B/C/D).
#
# USED BY: agent.py, FastAPI /maintenance endpoint
# ─────────────────────────────────────────────────────

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MACHINES

# ─────────────────────────────────────────────────────
# PM INTERVALS PER MACHINE (hours)
# ─────────────────────────────────────────────────────
PM_INTERVALS = {
    "CNC-M01": {
        "A_daily":     8,      # every shift
        "A_weekly":    40,
        "B_monthly":   160,
        "C_quarterly": 500,
        "D_annual":    2000,
    },
    "HYD-P02": {
        "A_daily":     8,
        "A_weekly":    40,
        "B_monthly":   160,
        "C_quarterly": 500,
        "D_annual":    2000,
    },
    "CVB-003": {
        "A_daily":     8,
        "A_weekly":    40,
        "B_monthly":   160,
        "C_quarterly": 500,
        "D_annual":    2000,
    },
    "BLR-004": {
        "A_daily":     8,
        "A_weekly":    40,
        "B_monthly":   160,
        "C_quarterly": 500,
        "D_annual":    2000,
    },
    "ROB-005": {
        "A_daily":     8,
        "A_weekly":    40,
        "B_monthly":   160,
        "C_quarterly": 500,
        "D_annual":    2000,
    },
}

# ─────────────────────────────────────────────────────
# PM CHECKLISTS PER MACHINE AND LEVEL
# ─────────────────────────────────────────────────────
PM_CHECKLISTS = {
    "CNC-M01": {
        "A_daily": [
            "Clean machine bed and chip tray",
            "Check coolant level — top up if below MIN",
            "Check spindle oil sight glass level",
            "Wipe guideways — apply light oil film",
            "Inspect tooling for wear or damage",
            "Verify door interlock and E-STOP function",
            "Log machine hours in logbook",
        ],
        "A_weekly": [
            "Check lubrication oil tank level — top up",
            "Clean coolant filter strainer",
            "Inspect axis bellows for damage",
            "Check tool holders for corrosion",
            "Clean control panel — verify all buttons",
        ],
        "B_monthly": [
            "Check axis backlash — measure with dial gauge (max 0.01mm)",
            "Inspect servo motor cables and connectors",
            "Clean electrical cabinet interior",
            "Check coolant concentration — adjust to 8–10%",
            "Lubricate all grease nipples per lubrication chart",
            "Check spindle runout — max 0.005mm",
            "Review and address recurring alarms",
        ],
        "C_quarterly": [
            "Replace coolant filter cartridge",
            "Check and adjust axis ballscrew preload",
            "Spindle warm-up test — run to 5000 RPM, check temp",
            "Replace automatic lubrication oil",
            "Calibrate tool length probe",
            "Thermographic inspection of electrical cabinet",
        ],
        "D_annual": [
            "Full spindle bearing replacement",
            "Replace all axis linear guideway pads",
            "Full coolant system flush and fresh fill",
            "Geometry recalibration — laser alignment",
            "Electrical wiring inspection — megger test",
            "Safety circuit function test",
        ],
    },

    "HYD-P02": {
        "A_daily": [
            "Check hydraulic fluid level — sight glass",
            "Inspect all hoses and fittings for leaks",
            "Check system pressure at operating gauge",
            "Listen for unusual pump noise",
            "Verify filter indicator — replace if RED",
            "Log operating hours",
        ],
        "A_weekly": [
            "Drain filter bowl — remove water accumulation",
            "Check all actuator seals for external leaks",
            "Wipe down pump unit and motor exterior",
        ],
        "B_monthly": [
            "Sample hydraulic fluid — send for lab analysis",
            "Check motor coupling for wear",
            "Inspect suction strainer — clean if blocked",
            "Verify PRV setting with calibrated gauge",
            "Check all solenoid valve function",
            "Inspect cylinder rod surfaces for scoring",
        ],
        "C_quarterly": [
            "Replace hydraulic return filter element",
            "Check pump flow rate vs nameplate",
            "Inspect pump shaft seal",
            "Clean oil cooler externally",
            "Check hose assemblies — replace any > 4 years",
            "Thermographic scan of motor and pump",
        ],
        "D_annual": [
            "Full hydraulic fluid drain and replacement",
            "Pump overhaul if efficiency < 80%",
            "Full seal replacement on all cylinders",
            "Replace all hoses regardless of condition",
            "Clean oil cooler internally",
            "PRV recalibration and certification",
        ],
    },

    "CVB-003": {
        "A_daily": [
            "Walk full conveyor length — check for belt damage",
            "Verify belt tracking — centered ±10mm",
            "Check drive pulley and belt for debris",
            "Test emergency pull cord (no need to trigger)",
            "Check gearbox oil level — dipstick",
            "Log runtime hours",
        ],
        "A_weekly": [
            "Clean conveyor frame and belt underside",
            "Inspect all carry rollers — spin by hand",
            "Check belt tension — visual and deflection test",
            "Lubricate return roller bearings",
        ],
        "B_monthly": [
            "Measure belt tension with tension meter",
            "Inspect drive and tail pulley lagging",
            "Check all roller bearing temperatures",
            "Verify speed sensor reading vs actual",
            "Test all pull cords and E-STOP buttons",
            "Check photocell alignment and cleanliness",
        ],
        "C_quarterly": [
            "Replace any seized or noisy rollers",
            "Regrease all sealed roller bearings",
            "Align conveyor frame squareness",
            "Measure belt thickness — replace if < 70% original",
            "Gearbox oil sample analysis",
        ],
        "D_annual": [
            "Full belt replacement or NDT inspection",
            "Gearbox overhaul or replacement",
            "Replace drive and tail pulley bearings",
            "Renew all roller sets if > 5 years",
            "Electrical inspection of control panel and cables",
        ],
    },
}


def calculate_pm_due(
    machine_id: str,
    current_hours: int,
    last_pm_hours: Optional[int] = None,
    last_pm_date: Optional[str] = None,
) -> dict:
    """
    Calculate which PM levels are currently due.

    Args:
        machine_id:    e.g. "CNC-M01"
        current_hours: Total operating hours on the machine
        last_pm_hours: Hours at which last PM was done (optional)
        last_pm_date:  Date of last PM as "YYYY-MM-DD" (optional)

    Returns:
        PM due report with overdue/due/upcoming levels + checklists
    """
    machine_id = machine_id.upper().strip()

    if machine_id not in PM_INTERVALS:
        return {"error": f"No PM data for machine {machine_id}"}

    intervals  = PM_INTERVALS[machine_id]
    checklists = PM_CHECKLISTS.get(machine_id, {})

    # Default: assume last PM was done 160 hours ago if not specified
    hours_since_pm = current_hours - (last_pm_hours or max(0, current_hours - 160))

    levels_due    = []
    levels_ok     = []
    next_due_info = []

    level_labels = {
        "A_daily":     "Level A — Daily",
        "A_weekly":    "Level A — Weekly",
        "B_monthly":   "Level B — Monthly",
        "C_quarterly": "Level C — Quarterly",
        "D_annual":    "Level D — Annual Overhaul",
    }

    for level_key, interval in intervals.items():
        hours_since = hours_since_pm
        is_due   = hours_since >= interval
        overdue  = hours_since >= interval * 1.2  # 20% overrun
        hrs_left = max(0, interval - hours_since)

        entry = {
            "level":          level_key,
            "label":          level_labels.get(level_key, level_key),
            "interval_hours": interval,
            "hours_since_pm": round(hours_since, 1),
            "hours_until_due": round(hrs_left, 1),
            "is_due":         is_due,
            "is_overdue":     overdue,
            "checklist":      checklists.get(level_key, []),
        }

        if is_due:
            levels_due.append(entry)
        else:
            levels_ok.append(entry)

        # Estimated calendar date
        hours_per_day = 16  # assume 2 shifts
        days_until    = hrs_left / hours_per_day
        due_date      = (datetime.now() + timedelta(days=days_until)).strftime("%Y-%m-%d")
        next_due_info.append({
            "level":    level_labels.get(level_key, level_key),
            "due_date": due_date if not is_due else "NOW",
            "status":   "OVERDUE" if overdue else ("DUE NOW" if is_due else "OK"),
        })

    # Determine overall urgency
    if any(e["is_overdue"] for e in levels_due):
        urgency = "OVERDUE — Schedule maintenance immediately"
    elif levels_due:
        urgency = "MAINTENANCE DUE — Plan within 24 hours"
    else:
        urgency = "UP TO DATE — Next PM coming up"

    return {
        "machine_id":      machine_id,
        "machine_name":    MACHINES.get(machine_id, machine_id),
        "current_hours":   current_hours,
        "hours_since_pm":  round(hours_since_pm, 1),
        "last_pm_date":    last_pm_date or "Unknown",
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "urgency":         urgency,
        "levels_due":      levels_due,
        "levels_ok":       levels_ok,
        "schedule":        next_due_info,
        "due_count":       len(levels_due),
    }


def format_pm_report(machine_id: str, current_hours: int, last_pm_hours: int = None) -> str:
    """Return a human-readable PM report string for the LLM agent."""
    report = calculate_pm_due(machine_id, current_hours, last_pm_hours)
    if "error" in report:
        return report["error"]

    lines = [
        f"=== PM Status: {report['machine_name']} ({machine_id}) ===",
        f"Current Hours: {report['current_hours']}  |  Hours Since PM: {report['hours_since_pm']}",
        f"Overall Status: {report['urgency']}",
        "",
    ]

    if report["levels_due"]:
        lines.append("── MAINTENANCE DUE ──────────────────────────────")
        for lvl in report["levels_due"]:
            tag = "⚠ OVERDUE" if lvl["is_overdue"] else "✓ DUE"
            lines.append(f"  [{tag}] {lvl['label']} (interval: {lvl['interval_hours']}h)")
            lines.append(f"  Checklist:")
            for task in lvl["checklist"][:5]:
                lines.append(f"    □ {task}")
            if len(lvl["checklist"]) > 5:
                lines.append(f"    ... +{len(lvl['checklist'])-5} more tasks")
            lines.append("")

    lines.append("── UPCOMING ─────────────────────────────────────")
    for s in report["schedule"]:
        lines.append(f"  {s['level']:<30} {s['due_date']:<14} [{s['status']}]")

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    print(format_pm_report("CNC-M01", current_hours=5420, last_pm_hours=5260))
    print()
    print(format_pm_report("CVB-003", current_hours=3800, last_pm_hours=3600))
