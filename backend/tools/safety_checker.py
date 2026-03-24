# ─────────────────────────────────────────────────────
# backend/tools/safety_checker.py
#
# TOOL 6 — Safety Rule Checker Tool
#
# PURPOSE:
#   Given a machine and task type, returns:
#     • Required PPE
#     • Lockout/tagout (LOTO) steps
#     • Hazard level
#     • Specific warnings for that machine + task
#
# USED BY: agent.py, FastAPI /safety endpoint,
#          retriever.py (supplements RAG safety answers)
# ─────────────────────────────────────────────────────

import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MACHINES

# ─────────────────────────────────────────────────────
# MACHINE-SPECIFIC SAFETY PROFILES
# ─────────────────────────────────────────────────────
SAFETY_PROFILES = {
    "CNC-M01": {
        "area":             "Machining Cell A",
        "mandatory_ppe":    ["Safety glasses (ANSI Z87.1)", "Steel-toed boots (S3)"],
        "task_ppe": {
            "operation":        ["Safety glasses", "Steel-toed boots"],
            "tool_change":      ["Safety glasses", "Steel-toed boots", "Cut-resistant gloves"],
            "maintenance":      ["Safety glasses", "Steel-toed boots", "Cut-resistant gloves"],
            "electrical":       ["Safety glasses", "Steel-toed boots", "Insulated gloves", "Face shield"],
            "coolant":          ["Safety glasses", "Steel-toed boots", "Nitrile gloves", "Apron"],
            "cleaning":         ["Safety glasses", "Steel-toed boots", "Nitrile gloves"],
        },
        "prohibited_ppe":   ["Loose gloves during spindle rotation", "Open-toe shoes", "Loose clothing"],
        "energy_sources":   ["Electrical 415V", "Pneumatic 6 bar", "Coolant pressure"],
        "loto_required":    True,
        "hazards":          ["Rotating spindle", "Sharp cutting tools", "Coolant splash", "Chip ejection", "Pinch points on axes"],
        "special_warnings": [
            "NEVER put hands inside machine during spindle rotation",
            "Always use chip brush — never bare hands for chip removal",
            "Coolant concentrate is a skin sensitizer — wear gloves",
        ],
    },
    "HYD-P02": {
        "area":             "Press & Forming Cell",
        "mandatory_ppe":    ["Safety glasses", "Steel-toed boots", "Oil-resistant gloves"],
        "task_ppe": {
            "operation":        ["Safety glasses", "Steel-toed boots"],
            "hose_change":      ["Safety glasses", "Steel-toed boots", "Face shield", "FR coveralls"],
            "maintenance":      ["Safety glasses", "Steel-toed boots", "Oil-resistant gloves"],
            "electrical":       ["Safety glasses", "Steel-toed boots", "Insulated gloves"],
            "fluid_change":     ["Safety glasses", "Steel-toed boots", "Oil-resistant gloves", "Apron"],
            "valve_work":       ["Safety glasses", "Steel-toed boots", "Face shield", "FR coveralls"],
        },
        "prohibited_ppe":   ["Open-toe shoes"],
        "energy_sources":   ["Electrical 415V", "Hydraulic pressure up to 280 bar", "Stored energy in accumulators"],
        "loto_required":    True,
        "hazards":          ["High-pressure fluid injection", "Crush hazard from cylinders", "Hot hydraulic fluid", "Slip hazard from oil spills"],
        "special_warnings": [
            "HIGH PRESSURE FLUID INJECTION IS FATAL — never approach a pressurized leak",
            "ALWAYS verify zero pressure on gauge before any hydraulic work",
            "Accumulators store pressure even when pump is off — must be discharged",
            "Hydraulic oil fires — use CO2 or dry powder only",
        ],
    },
    "CVB-003": {
        "area":             "Material Handling & Assembly",
        "mandatory_ppe":    ["Safety glasses", "Steel-toed boots", "Hi-Vis vest"],
        "task_ppe": {
            "operation":        ["Safety glasses", "Steel-toed boots", "Hi-Vis vest"],
            "belt_tracking":    ["Safety glasses", "Steel-toed boots", "Hi-Vis vest"],
            "roller_change":    ["Safety glasses", "Steel-toed boots", "Hi-Vis vest"],
            "maintenance":      ["Safety glasses", "Steel-toed boots", "Hi-Vis vest"],
            "electrical":       ["Safety glasses", "Steel-toed boots", "Insulated gloves"],
            "belt_repair":      ["Safety glasses", "Steel-toed boots", "Hi-Vis vest"],
        },
        "prohibited_ppe":   ["Gloves near moving conveyor belt (entanglement)", "Loose clothing"],
        "energy_sources":   ["Electrical 415V", "Pneumatic 4 bar (if pneumatic tensioner)"],
        "loto_required":    True,
        "hazards":          ["Belt and roller entanglement", "Nip points at pulleys", "Moving products falling from belt", "Slip on oil or product spillage"],
        "special_warnings": [
            "NEVER wear gloves near a moving conveyor — entanglement risk",
            "Sound warning horn before starting conveyor",
            "Pull cord test must be done every shift — mandatory",
            "No entry under raised belt without mechanical support",
        ],
    },
    "BLR-004": {
        "area":             "Utilities & Steam Generation",
        "mandatory_ppe":    ["Safety glasses", "Steel-toed boots", "Heat-resistant gloves", "FR coveralls"],
        "task_ppe": {
            "operation":        ["Safety glasses", "Steel-toed boots", "Heat-resistant gloves", "FR coveralls"],
            "blowdown":         ["Safety glasses", "Steel-toed boots", "Heat-resistant gloves", "Face shield", "FR coveralls"],
            "maintenance":      ["Safety glasses", "Steel-toed boots", "Heat-resistant gloves", "FR coveralls"],
            "tube_inspection":  ["Safety glasses", "Steel-toed boots", "Hard hat", "Hearing protection", "FR coveralls"],
            "burner_work":      ["Safety glasses", "Steel-toed boots", "Heat-resistant gloves", "FR coveralls"],
            "chemical_dosing":  ["Safety glasses", "Steel-toed boots", "Chemical gloves", "Face shield", "Apron"],
        },
        "prohibited_ppe":   ["Synthetic clothing (fire hazard)"],
        "energy_sources":   ["Steam up to 11 bar", "Gas fuel supply", "Electrical 415V", "Hot water / feedwater"],
        "loto_required":    True,
        "hazards":          ["High pressure steam burns", "Explosion risk (overpressure)", "Gas fire", "Chemical burns (boiler chemicals)", "High temperature surfaces"],
        "special_warnings": [
            "ONLY certified boiler operators may operate BLR-004",
            "NEVER block or wire shut the safety valve",
            "Do not enter boiler shell without permit-to-work and gas test",
            "Chemical anti-scale is corrosive — face shield mandatory",
            "Blowdown involves high-velocity hot water — maintain 2m exclusion zone",
        ],
    },
    "ROB-005": {
        "area":             "Welding & Assembly Automation",
        "mandatory_ppe":    ["Safety glasses", "Steel-toed boots"],
        "task_ppe": {
            "auto_mode":        ["Safety glasses", "Steel-toed boots", "Hi-Vis vest (outside cell)"],
            "teach_mode":       ["Safety glasses", "Steel-toed boots"],
            "welding":          ["Safety glasses", "Steel-toed boots", "Auto-darkening welding shield (Shade 10+)", "FR gloves", "FR coveralls"],
            "maintenance":      ["Safety glasses", "Steel-toed boots"],
            "programming":      ["Safety glasses", "Steel-toed boots"],
            "end_effector":     ["Safety glasses", "Steel-toed boots", "Cut-resistant gloves"],
        },
        "prohibited_ppe":   ["No entry to cell during AUTO operation"],
        "energy_sources":   ["Electrical 415V", "Pneumatic 6 bar (gripper)", "Welding power supply"],
        "loto_required":    True,
        "hazards":          ["High-speed robot arm collision", "Welding arc flash", "Welding fume", "Crush/pinch at joints", "UV radiation from welding"],
        "special_warnings": [
            "NEVER enter robot cell during AUTO operation",
            "In TEACH mode: stay aware — robot can move unexpectedly at 250 mm/s",
            "Robot E-STOP buttons are on ALL sides of cell — know your nearest one",
            "Welding fume: ensure extraction system is ON before welding",
            "After collision: DO NOT restart until TCP recalibration is done",
        ],
    },
}

# Task keyword mapping for auto-detection
TASK_KEYWORDS = {
    "tool change":    "tool_change",
    "maintenance":    "maintenance",
    "cleaning":       "cleaning",
    "hose":           "hose_change",
    "electrical":     "electrical",
    "blowdown":       "blowdown",
    "chemical":       "chemical_dosing",
    "teach":          "teach_mode",
    "welding":        "welding",
    "programming":    "programming",
    "belt":           "belt_tracking",
    "roller":         "roller_change",
    "fluid":          "fluid_change",
    "valve":          "valve_work",
    "operation":      "operation",
    "auto":           "auto_mode",
}

STANDARD_LOTO_STEPS = [
    "1. NOTIFY — Inform supervisor and affected operators of planned maintenance",
    "2. IDENTIFY — Identify ALL energy sources (electrical, hydraulic, pneumatic, stored)",
    "3. SHUTDOWN — Use normal stopping procedure to stop machine",
    "4. ISOLATE — Turn isolator to OFF, close hydraulic supply valve, shut air supply",
    "5. LOCK — Apply personal padlock to each energy isolation point",
    "6. TAG — Attach RED 'DANGER — DO NOT OPERATE' tag to each lock",
    "7. VERIFY — Test that machine CANNOT start (check voltage=0, pressure=0)",
    "8. WORK — Proceed with maintenance task",
    "9. RESTORE — Remove tools, confirm all personnel clear",
    "10. REMOVE LOCK — Each person removes their OWN lock only",
    "11. NOTIFY — Inform operators that machine is restored",
]


def _detect_task_type(task_description: str) -> str:
    """Auto-detect task type from a description string."""
    task_lower = task_description.lower()
    for keyword, task_type in TASK_KEYWORDS.items():
        if keyword in task_lower:
            return task_type
    return "maintenance"   # default


def check_safety(machine_id: str, task: str) -> dict:
    """
    Return full safety requirements for a machine + task combination.

    Args:
        machine_id: e.g. "CNC-M01"
        task:       e.g. "tool change", "hydraulic hose replacement", "welding"

    Returns:
        Safety requirements dict with PPE, LOTO, hazards, warnings
    """
    machine_id = machine_id.upper().strip()

    if machine_id not in SAFETY_PROFILES:
        return {"error": f"No safety profile for machine {machine_id}"}

    profile   = SAFETY_PROFILES[machine_id]
    task_type = _detect_task_type(task)
    task_ppe  = profile["task_ppe"].get(task_type, profile["task_ppe"].get("maintenance", []))

    # Combine mandatory + task-specific PPE, deduplicate
    all_ppe = list(dict.fromkeys(profile["mandatory_ppe"] + task_ppe))

    # Determine hazard level based on energy sources and task
    high_risk_tasks = {"hose_change", "valve_work", "blowdown", "tube_inspection", "chemical_dosing", "electrical"}
    hazard_level = "HIGH" if task_type in high_risk_tasks else "MEDIUM"
    if machine_id == "BLR-004":
        hazard_level = "HIGH"  # boiler is always HIGH

    return {
        "machine_id":       machine_id,
        "machine_name":     MACHINES.get(machine_id, machine_id),
        "area":             profile["area"],
        "task":             task,
        "task_type":        task_type,
        "hazard_level":     hazard_level,
        "loto_required":    profile["loto_required"],
        "required_ppe":     all_ppe,
        "prohibited_ppe":   profile["prohibited_ppe"],
        "energy_sources":   profile["energy_sources"],
        "hazards":          profile["hazards"],
        "special_warnings": profile["special_warnings"],
        "loto_steps":       STANDARD_LOTO_STEPS if profile["loto_required"] else [],
        "permit_required":  hazard_level == "HIGH",
    }


def format_safety_report(machine_id: str, task: str) -> str:
    """Return formatted safety briefing string for LLM/operator."""
    safety = check_safety(machine_id, task)
    if "error" in safety:
        return safety["error"]

    lines = [
        f"╔══ SAFETY BRIEFING: {safety['machine_name']} ({machine_id}) ══",
        f"  Task:         {task}",
        f"  Area:         {safety['area']}",
        f"  Hazard Level: ⚠ {safety['hazard_level']}",
        f"  LOTO Required: {'YES — mandatory' if safety['loto_required'] else 'No'}",
        f"  Permit Required: {'YES — get PTW approval' if safety['permit_required'] else 'No'}",
        "",
        "── Required PPE ─────────────────────────────────",
    ]
    for ppe in safety["required_ppe"]:
        lines.append(f"  ✓ {ppe}")

    if safety["prohibited_ppe"]:
        lines.append("\n── Prohibited ───────────────────────────────────")
        for p in safety["prohibited_ppe"]:
            lines.append(f"  ✗ {p}")

    lines.append("\n── Hazards ──────────────────────────────────────")
    for h in safety["hazards"]:
        lines.append(f"  • {h}")

    lines.append("\n── Critical Warnings ────────────────────────────")
    for w in safety["special_warnings"]:
        lines.append(f"  ⚠ {w}")

    if safety["loto_steps"]:
        lines.append("\n── LOTO Steps (mandatory before work) ──────────")
        for step in safety["loto_steps"]:
            lines.append(f"  {step}")

    lines.append("╚" + "═" * 55)
    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    print(format_safety_report("CNC-M01", "tool change"))
    print()
    print(format_safety_report("HYD-P02", "hydraulic hose replacement"))
