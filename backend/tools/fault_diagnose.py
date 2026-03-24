# ─────────────────────────────────────────────────────
# backend/tools/fault_diagnose.py
#
# TOOL 2 — Fault Diagnosis Tool
#
# PURPOSE:
#   Diagnoses faults using:
#     1. Rule-based engine  — fast, deterministic,
#        uses sensor thresholds + error code lookup
#     2. Pattern matching   — cross-sensor correlation
#        (e.g. high temp + high vibration = bearing)
#
#   Returns: failure mode, severity, first-level steps,
#            recommended spare parts, escalation flag
#
# USED BY: agent.py, FastAPI /fault endpoint
# ─────────────────────────────────────────────────────

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SENSOR_THRESHOLDS, MACHINES

# ─────────────────────────────────────────────────────
# ERROR CODE KNOWLEDGE BASE
# Maps error code → diagnosis metadata
# ─────────────────────────────────────────────────────
ERROR_CODE_DB = {
    # CNC-M01
    "E01": {"machine": "CNC-M01", "title": "Spindle Overload",         "severity": "HIGH",     "failure_mode": "Tool wear or incorrect cutting parameters", "escalate": False, "parts": ["SP-CNC-T01", "SP-CNC-B01"], "first_steps": ["Press E-STOP", "Inspect cutting tool for wear/breakage", "Reduce feed rate by 20%", "Check spindle motor current at drive panel"]},
    "E02": {"machine": "CNC-M01", "title": "Axis Drive Fault",         "severity": "HIGH",     "failure_mode": "Servo drive alarm or encoder issue",          "escalate": False, "parts": ["SP-CNC-E01", "SP-CNC-D01"], "first_steps": ["Record which axis faulted", "Check servo drive LED status", "Press FAULT RESET on drive", "Home machine after reset"]},
    "E03": {"machine": "CNC-M01", "title": "Coolant Low",              "severity": "MEDIUM",   "failure_mode": "Coolant leak or high consumption",             "escalate": False, "parts": ["SP-CNC-C01", "SP-CNC-F01"], "first_steps": ["Check reservoir sight glass", "Inspect for leaks at tank and nozzles", "Refill with 10% concentration coolant mix"]},
    "E04": {"machine": "CNC-M01", "title": "Door Interlock Open",      "severity": "LOW",      "failure_mode": "Door not closed or switch fault",              "escalate": False, "parts": ["SP-CNC-I01"],               "first_steps": ["Close machine door firmly", "Inspect interlock switch alignment", "Check wiring connector at switch"]},
    "E05": {"machine": "CNC-M01", "title": "Tool Length Error",        "severity": "MEDIUM",   "failure_mode": "Incorrect tool loaded or offset not updated",  "escalate": False, "parts": ["SP-CNC-H01", "SP-CNC-T01"], "first_steps": ["Stop machine", "Verify tool number matches program", "Re-probe tool length", "Update offset in controller table"]},
    "E06": {"machine": "CNC-M01", "title": "Lubrication Fault",       "severity": "HIGH",     "failure_mode": "Lube pump failure or blocked line",            "escalate": False, "parts": ["SP-CNC-L01", "SP-CNC-P01"], "first_steps": ["Check lube oil tank level", "Press manual lube button", "Do NOT run without lubrication"]},
    "E07": {"machine": "CNC-M01", "title": "Program Syntax Error",    "severity": "LOW",      "failure_mode": "G-code syntax or post-processor error",        "escalate": False, "parts": [],                           "first_steps": ["Note error line number", "Open G-code editor", "Fix syntax and re-simulate"]},
    "E08": {"machine": "CNC-M01", "title": "Spindle Bearing Overheat","severity": "CRITICAL", "failure_mode": "Spindle bearing failure or over-lubrication",  "escalate": True,  "parts": ["SP-CNC-B01", "SP-CNC-G01"], "first_steps": ["STOP machine immediately", "Do not restart until temp < 50°C", "Call maintenance engineer"]},
    "E09": {"machine": "CNC-M01", "title": "Feed Rate Override Low",  "severity": "LOW",      "failure_mode": "Override dial set below 50%",                  "escalate": False, "parts": [],                           "first_steps": ["Check feed rate override dial", "Set to 100% for production", "Log if intentional"]},
    "E10": {"machine": "CNC-M01", "title": "Memory Full",             "severity": "LOW",      "failure_mode": "Controller program memory exhausted",          "escalate": False, "parts": [],                           "first_steps": ["Back up programs to USB", "Delete obsolete programs", "Transfer to DNC server"]},

    # HYD-P02
    "E11": {"machine": "HYD-P02", "title": "Overpressure",            "severity": "CRITICAL", "failure_mode": "PRV stuck or blocked actuator",                "escalate": True,  "parts": ["SP-HYD-R01", "SP-HYD-T01"], "first_steps": ["STOP pump immediately", "Open unloading valve to relieve pressure", "Test PRV by manual lift", "Check all cylinders for mechanical lock"]},
    "E12": {"machine": "HYD-P02", "title": "Low Fluid Level",         "severity": "HIGH",     "failure_mode": "Hydraulic fluid leak",                         "escalate": False, "parts": ["SP-HYD-O01", "SP-HYD-S01"], "first_steps": ["STOP pump — running low cavitates pump", "Inspect all hoses and fittings", "Do not restart until leak is fixed"]},
    "E13": {"machine": "HYD-P02", "title": "High Fluid Temperature",  "severity": "HIGH",     "failure_mode": "Fouled cooler or continuous high load",        "escalate": False, "parts": ["SP-HYD-C01", "SP-HYD-V01"], "first_steps": ["Reduce system load immediately", "Check oil cooler for fouling", "Allow fluid to cool below 55°C"]},
    "E14": {"machine": "HYD-P02", "title": "Filter Blocked",          "severity": "MEDIUM",   "failure_mode": "Filter element saturated",                     "escalate": False, "parts": ["SP-HYD-F01"],               "first_steps": ["Change hydraulic filter element", "Check for metal particles in used filter", "Adjust PM schedule if interval too long"]},
    "E15": {"machine": "HYD-P02", "title": "Motor Overload",          "severity": "HIGH",     "failure_mode": "Overpressure or motor winding degradation",    "escalate": False, "parts": ["SP-HYD-M01"],               "first_steps": ["Allow motor to cool 10 min", "Reset overload relay", "Reduce pressure setpoint by 10%", "Megger test if fault repeats"]},
    "E16": {"machine": "HYD-P02", "title": "Pump Cavitation",         "severity": "HIGH",     "failure_mode": "Suction starvation or blocked inlet",          "escalate": False, "parts": ["SP-HYD-SS01"],              "first_steps": ["Check fluid temperature — cold fluid is viscous", "Clean suction strainer", "Check suction line for kinks"]},
    "E17": {"machine": "HYD-P02", "title": "Solenoid Valve Fault",    "severity": "MEDIUM",   "failure_mode": "Coil failure or spool contamination",          "escalate": False, "parts": ["SP-HYD-SV01"],              "first_steps": ["Check PLC output signal to solenoid", "Verify 24VDC at coil terminals", "Replace coil if open circuit"]},
    "E18": {"machine": "HYD-P02", "title": "Cylinder Leakage",        "severity": "MEDIUM",   "failure_mode": "Piston rod seal wear",                         "escalate": False, "parts": ["SP-HYD-S01"],               "first_steps": ["Observe cylinder drift under load", "Inspect piston rod for scoring", "Replace seal kit at next downtime"]},
    "E19": {"machine": "HYD-P02", "title": "Pressure Transducer Fault","severity": "MEDIUM",  "failure_mode": "Wiring fault or transducer failure",           "escalate": False, "parts": ["SP-HYD-T01"],               "first_steps": ["Compare vs analog gauge reading", "Check wiring connectors", "Replace if deviation > 5%"]},
    "E20": {"machine": "HYD-P02", "title": "Hose Burst Warning",      "severity": "CRITICAL", "failure_mode": "High pressure hose rupture",                   "escalate": True,  "parts": ["SP-HYD-H01"],               "first_steps": ["STOP pump IMMEDIATELY", "EVACUATE area — fluid injection is fatal", "Replace hose with rated assembly only"]},

    # CVB-003
    "E21": {"machine": "CVB-003", "title": "Belt Slip",               "severity": "MEDIUM",   "failure_mode": "Drive pulley lagging worn or belt overloaded", "escalate": False, "parts": ["SP-CVB-L01", "SP-CVB-B01"], "first_steps": ["Reduce conveyor load", "Increase belt tension via tail pulley", "Inspect drive pulley lagging for wear"]},
    "E22": {"machine": "CVB-003", "title": "Motor Overload",          "severity": "HIGH",     "failure_mode": "Mechanical jam or motor winding fault",        "escalate": False, "parts": ["SP-CVB-M01"],               "first_steps": ["Stop conveyor and remove product load", "Cool motor 15 minutes", "Check for roller jam", "Reset overload relay"]},
    "E23": {"machine": "CVB-003", "title": "Belt Misalignment",       "severity": "MEDIUM",   "failure_mode": "Uneven load or tracking adjustment needed",    "escalate": False, "parts": ["SP-CVB-B02"],               "first_steps": ["Observe which side belt drifts to", "Stop conveyor safely", "Adjust tail pulley tracking bolts per SOP Section 6", "Check for off-center product loading"]},
    "E24": {"machine": "CVB-003", "title": "Roller Jam",              "severity": "HIGH",     "failure_mode": "Debris in roller bearing",                     "escalate": False, "parts": ["SP-CVB-R01"],               "first_steps": ["STOP conveyor immediately", "Lock out main panel", "Locate and clear jammed roller", "Replace if bearing is seized"]},
    "E25": {"machine": "CVB-003", "title": "Drive Gearbox Fault",     "severity": "HIGH",     "failure_mode": "Low oil or gearbox bearing wear",              "escalate": False, "parts": ["SP-CVB-G01", "SP-CVB-GBX01"],"first_steps": ["Stop conveyor", "Check gearbox oil via dipstick", "Listen for grinding or whining", "Replace if oil contains metal particles"]},
    "E26": {"machine": "CVB-003", "title": "Speed Sensor Fault",      "severity": "MEDIUM",   "failure_mode": "Sensor loose, coated, or wiring fault",        "escalate": False, "parts": ["SP-CVB-SS01"],              "first_steps": ["Check sensor mounting", "Clean sensor face", "Verify 2-5mm gap to target", "Replace if fault persists"]},
    "E27": {"machine": "CVB-003", "title": "Pull Cord Activated",     "severity": "HIGH",     "failure_mode": "Emergency pull cord triggered",                "escalate": True,  "parts": [],                           "first_steps": ["Identify which section triggered", "Investigate for injury or entrapment", "Mandatory supervisor notification", "Clear cause before restart"]},
    "E28": {"machine": "CVB-003", "title": "Photocell Sensor Fault",  "severity": "LOW",      "failure_mode": "Dirty lens or misalignment",                   "escalate": False, "parts": ["SP-CVB-PC01"],              "first_steps": ["Clean photocell lens", "Check emitter/receiver alignment", "Verify 24VDC power to sensor"]},
    "E29": {"machine": "CVB-003", "title": "Belt Tear Detected",      "severity": "CRITICAL", "failure_mode": "Belt tear or physical damage",                 "escalate": True,  "parts": ["SP-CVB-BLT01", "SP-CVB-VK01"],"first_steps": ["STOP immediately", "Lock out conveyor", "Locate damage — assess repair vs replacement", "Do not run with any open tear"]},
    "E30": {"machine": "CVB-003", "title": "Gearbox Oil Leak",        "severity": "MEDIUM",   "failure_mode": "Shaft seal or gasket failure",                 "escalate": False, "parts": ["SP-CVB-OS01"],              "first_steps": ["Identify leak point", "Clean and apply dye test if needed", "Replace shaft seal or gasket", "Monitor oil level daily until repaired"]},
}

# ─────────────────────────────────────────────────────
# CROSS-SENSOR PATTERN RULES
# Multi-sensor combinations that indicate specific failures
# ─────────────────────────────────────────────────────
PATTERN_RULES = [
    {
        "id": "CNC_BEARING_FAIL",
        "machine": "CNC-M01",
        "conditions": {"temperature_c": ">70", "vibration_mm_s": ">3.5"},
        "diagnosis":  "Probable spindle bearing failure — high temp + vibration combination",
        "severity":   "CRITICAL",
        "escalate":   True,
        "steps":      ["Stop machine immediately", "Do not restart", "Call maintenance engineer for bearing inspection"],
    },
    {
        "id": "CNC_TOOL_WEAR",
        "machine": "CNC-M01",
        "conditions": {"power_kw": ">15", "vibration_mm_s": ">3.0"},
        "diagnosis":  "Likely tool wear or breakage — increased cutting force + vibration",
        "severity":   "HIGH",
        "escalate":   False,
        "steps":      ["Inspect cutting tool", "Replace if worn", "Reduce depth of cut"],
    },
    {
        "id": "HYD_PUMP_CAVITATION",
        "machine": "HYD-P02",
        "conditions": {"pressure_bar": "<80", "vibration_mm_s": ">4.0"},
        "diagnosis":  "Pump cavitation — low pressure + excessive vibration",
        "severity":   "HIGH",
        "escalate":   False,
        "steps":      ["Stop pump", "Check suction strainer", "Verify fluid level and temperature"],
    },
    {
        "id": "HYD_OVERLOAD",
        "machine": "HYD-P02",
        "conditions": {"pressure_bar": ">260", "motor_current_a": ">45"},
        "diagnosis":  "Hydraulic system overload — high pressure + high motor current",
        "severity":   "CRITICAL",
        "escalate":   True,
        "steps":      ["Stop pump", "Open unloading valve", "Check all cylinders for mechanical lock"],
    },
    {
        "id": "CVB_BELT_PROBLEM",
        "machine": "CVB-003",
        "conditions": {"motor_current_a": ">28", "belt_tension_n": "<600"},
        "diagnosis":  "Belt slipping under load — high motor current + low tension",
        "severity":   "HIGH",
        "escalate":   False,
        "steps":      ["Reduce belt load", "Increase belt tension via tail pulley adjustment", "Inspect belt surface for contamination"],
    },
    {
        "id": "CVB_MOTOR_OVERHEAT",
        "machine": "CVB-003",
        "conditions": {"motor_temp_c": ">75", "motor_current_a": ">28"},
        "diagnosis":  "Motor overheating under excessive load",
        "severity":   "HIGH",
        "escalate":   False,
        "steps":      ["Stop conveyor", "Remove product load", "Allow motor to cool 15 minutes"],
    },
]


def _check_pattern_rules(machine_id: str, sensor_readings: dict) -> list:
    """Check sensor readings against cross-sensor pattern rules."""
    triggered = []
    for rule in PATTERN_RULES:
        if rule["machine"] != machine_id:
            continue
        match = True
        for sensor, condition in rule["conditions"].items():
            val = sensor_readings.get(sensor)
            if val is None:
                match = False
                break
            op, threshold = condition[0], float(condition[1:])
            if op == ">" and not (val > threshold):
                match = False; break
            if op == "<" and not (val < threshold):
                match = False; break
        if match:
            triggered.append(rule)
    return triggered


def diagnose_fault(
    machine_id: str,
    error_code: Optional[str] = None,
    sensor_readings: Optional[dict] = None,
) -> dict:
    """
    Diagnose a fault from error code and/or sensor readings.

    Args:
        machine_id:      e.g. "CNC-M01"
        error_code:      e.g. "E01" (optional)
        sensor_readings: dict of sensor name → value (optional)

    Returns:
        Diagnosis report dict with severity, steps, parts, escalate flag
    """
    machine_id = machine_id.upper().strip()
    diagnoses  = []
    parts      = []
    escalate   = False
    severity   = "LOW"
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    # ── 1. Error code lookup ──────────────────────────
    if error_code:
        code = error_code.upper().strip()
        if code in ERROR_CODE_DB:
            entry = ERROR_CODE_DB[code]
            diagnoses.append({
                "source":       "error_code",
                "code":         code,
                "title":        entry["title"],
                "severity":     entry["severity"],
                "failure_mode": entry["failure_mode"],
                "steps":        entry["first_steps"],
            })
            parts    += entry["parts"]
            escalate  = escalate or entry["escalate"]
            if severity_rank[entry["severity"]] > severity_rank[severity]:
                severity = entry["severity"]
        else:
            diagnoses.append({
                "source":   "error_code",
                "code":     code,
                "title":    "Unknown Error Code",
                "severity": "MEDIUM",
                "failure_mode": "Error code not found in knowledge base",
                "steps":    ["Check machine HMI for detailed description", "Refer to machine manual", "Contact maintenance engineer"],
            })

    # ── 2. Sensor pattern analysis ────────────────────
    if sensor_readings:
        patterns = _check_pattern_rules(machine_id, sensor_readings)
        for rule in patterns:
            diagnoses.append({
                "source":       "sensor_pattern",
                "rule_id":      rule["id"],
                "title":        rule["diagnosis"][:60],
                "severity":     rule["severity"],
                "failure_mode": rule["diagnosis"],
                "steps":        rule["steps"],
            })
            escalate = escalate or rule["escalate"]
            if severity_rank[rule["severity"]] > severity_rank[severity]:
                severity = rule["severity"]

    # ── 3. Threshold breach scan ──────────────────────
    if sensor_readings:
        from config.settings import SENSOR_THRESHOLDS
        thresholds = SENSOR_THRESHOLDS.get(machine_id, {})
        low_bad = {"coolant_flow_l_min", "belt_tension_n", "water_level_pct",
                   "flow_l_min", "fuel_pressure_mbar", "feedwater_temp_c", "air_pressure_bar"}

        for sensor, value in sensor_readings.items():
            thresh = thresholds.get(sensor, {})
            if not thresh:
                continue
            alarm_val   = thresh["alarm"]
            warning_val = thresh["warning"]
            is_low_bad  = sensor in low_bad

            breach_sev = None
            if is_low_bad:
                if value <= alarm_val:   breach_sev = "ALARM"
                elif value <= warning_val: breach_sev = "WARNING"
            else:
                if value >= alarm_val:   breach_sev = "ALARM"
                elif value >= warning_val: breach_sev = "WARNING"

            if breach_sev == "ALARM":
                diagnoses.append({
                    "source":       "threshold_breach",
                    "sensor":       sensor,
                    "value":        value,
                    "severity":     "CRITICAL",
                    "failure_mode": f"{sensor} in ALARM zone: {value}",
                    "steps":        [f"Immediately investigate {sensor} reading", "Check sensor calibration", "Inspect related components"],
                })
                if severity_rank["CRITICAL"] > severity_rank[severity]:
                    severity = "CRITICAL"
                escalate = True

    # ── 4. Build final report ─────────────────────────
    if not diagnoses:
        return {
            "machine_id":  machine_id,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status":      "NO_FAULT",
            "severity":    "LOW",
            "diagnoses":   [],
            "recommended_parts": [],
            "escalate":    False,
            "summary":     "No faults detected. All readings and error codes within normal parameters.",
        }

    # Consolidate unique parts
    parts = list(dict.fromkeys(parts))

    # Determine top severity label
    severity_label = {
        "LOW":      "Monitor — no immediate action required",
        "MEDIUM":   "Schedule correction within current shift",
        "HIGH":     "Stop machine — fix before restarting",
        "CRITICAL": "EMERGENCY STOP — evacuate if needed, call supervisor",
    }

    return {
        "machine_id":        machine_id,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status":            "FAULT_DETECTED",
        "severity":          severity,
        "severity_guidance": severity_label[severity],
        "diagnoses":         diagnoses,
        "recommended_parts": parts,
        "escalate":          escalate,
        "diagnosis_count":   len(diagnoses),
        "summary": (
            f"{len(diagnoses)} fault(s) detected on {machine_id}. "
            f"Highest severity: {severity}. "
            f"{'⚠ ESCALATION REQUIRED.' if escalate else 'Escalation not required.'}"
        ),
    }


def get_fault_history_summary(machine_id: str, log_df=None) -> dict:
    """
    Summarize fault frequency from a log dataframe.
    Pass in a pandas DataFrame from log_analyzer or None for mock.
    """
    if log_df is None:
        # Return mock frequency data
        import random
        codes = {
            "CNC-M01": ["E01", "E03", "E02"],
            "HYD-P02": ["E13", "E14", "E15"],
            "CVB-003": ["E23", "E21", "E22"],
        }.get(machine_id, ["E01"])
        return {
            "machine_id":    machine_id,
            "top_faults":    [{"code": c, "count": random.randint(2, 15)} for c in codes],
            "total_faults":  random.randint(10, 50),
            "analysis_days": 7,
        }

    error_col = log_df[log_df["error_code"] != ""]["error_code"]
    counts    = error_col.value_counts().head(5).to_dict()
    return {
        "machine_id":   machine_id,
        "top_faults":   [{"code": k, "count": v} for k, v in counts.items()],
        "total_faults": len(error_col),
        "analysis_days": 7,
    }


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    import json

    # Test 1 — error code lookup
    result = diagnose_fault("CNC-M01", error_code="E01")
    print("=== Error Code Diagnosis ===")
    print(json.dumps(result, indent=2))

    # Test 2 — sensor pattern
    result2 = diagnose_fault(
        "CNC-M01",
        sensor_readings={"temperature_c": 78, "vibration_mm_s": 4.2, "power_kw": 17}
    )
    print("\n=== Sensor Pattern Diagnosis ===")
    print(json.dumps(result2, indent=2))
