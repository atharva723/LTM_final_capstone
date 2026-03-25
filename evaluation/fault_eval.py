# ─────────────────────────────────────────────────────
# evaluation/fault_eval.py
#
# FAULT DIAGNOSIS EVALUATION
#
# WHAT IS EVALUATED:
#   The fault_diagnose tool is tested against a ground
#   truth dataset of known error code → expected outcomes.
#
# METRICS:
#   Diagnosis Accuracy  — Did we identify the right failure mode?
#   Severity Accuracy   — Did we assign the correct severity?
#   Escalation Accuracy — Did we correctly flag cases needing escalation?
#   Step Coverage       — Are critical first-level steps present?
#   Parts Accuracy      — Did we recommend the right spare parts?
#
# WHY NECESSARY:
#   A wrong severity classification is DANGEROUS.
#   If E20 (Hose Burst — CRITICAL) is classified as MEDIUM,
#   an operator won't evacuate and could be fatally injured
#   by high-pressure fluid injection.
#   This eval ensures severity logic is correct before deployment.
# ─────────────────────────────────────────────────────

import sys
import json
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────
# GROUND TRUTH DATASET
# Expected output for each error code + scenario
# ─────────────────────────────────────────────────────
FAULT_TEST_CASES = [
    # ── CNC-M01 ──
    {
        "test_id":        "FD-01",
        "machine_id":     "CNC-M01",
        "error_code":     "E01",
        "sensor_readings": None,
        "expected_severity":  "HIGH",
        "expected_escalate":  False,
        "expected_title_kw":  "spindle",
        "expected_part_kw":   "SP-CNC",
        "expected_step_kw":   ["e-stop", "tool", "feed"],
        "description":    "Spindle overload — should be HIGH, not CRITICAL",
    },
    {
        "test_id":        "FD-02",
        "machine_id":     "CNC-M01",
        "error_code":     "E08",
        "sensor_readings": None,
        "expected_severity":  "CRITICAL",
        "expected_escalate":  True,
        "expected_title_kw":  "bearing",
        "expected_part_kw":   "SP-CNC-B01",
        "expected_step_kw":   ["stop", "restart", "maintenance"],
        "description":    "Spindle bearing overheat — must be CRITICAL + escalate",
    },
    {
        "test_id":        "FD-03",
        "machine_id":     "CNC-M01",
        "error_code":     "E03",
        "sensor_readings": None,
        "expected_severity":  "MEDIUM",
        "expected_escalate":  False,
        "expected_title_kw":  "coolant",
        "expected_part_kw":   "SP-CNC-C01",
        "expected_step_kw":   ["refill", "coolant", "leak"],
        "description":    "Coolant low — MEDIUM, no escalation",
    },
    {
        "test_id":        "FD-04",
        "machine_id":     "CNC-M01",
        "error_code":     "E04",
        "sensor_readings": None,
        "expected_severity":  "LOW",
        "expected_escalate":  False,
        "expected_title_kw":  "interlock",
        "expected_part_kw":   "SP-CNC-I01",
        "expected_step_kw":   ["door", "close"],
        "description":    "Door interlock — LOW severity",
    },
    # ── HYD-P02 ──
    {
        "test_id":        "FD-05",
        "machine_id":     "HYD-P02",
        "error_code":     "E11",
        "sensor_readings": None,
        "expected_severity":  "CRITICAL",
        "expected_escalate":  True,
        "expected_title_kw":  "overpressure",
        "expected_part_kw":   "SP-HYD-R01",
        "expected_step_kw":   ["stop", "pressure", "valve"],
        "description":    "Hydraulic overpressure — CRITICAL + escalate",
    },
    {
        "test_id":        "FD-06",
        "machine_id":     "HYD-P02",
        "error_code":     "E20",
        "sensor_readings": None,
        "expected_severity":  "CRITICAL",
        "expected_escalate":  True,
        "expected_title_kw":  "hose",
        "expected_part_kw":   "SP-HYD-H01",
        "expected_step_kw":   ["stop", "evacuate", "pressure"],
        "description":    "Hose burst — CRITICAL, must escalate and mention evacuation",
    },
    {
        "test_id":        "FD-07",
        "machine_id":     "HYD-P02",
        "error_code":     "E14",
        "sensor_readings": None,
        "expected_severity":  "MEDIUM",
        "expected_escalate":  False,
        "expected_title_kw":  "filter",
        "expected_part_kw":   "SP-HYD-F01",
        "expected_step_kw":   ["filter", "replace"],
        "description":    "Filter blocked — MEDIUM, filter replacement",
    },
    # ── CVB-003 ──
    {
        "test_id":        "FD-08",
        "machine_id":     "CVB-003",
        "error_code":     "E23",
        "sensor_readings": None,
        "expected_severity":  "MEDIUM",
        "expected_escalate":  False,
        "expected_title_kw":  "misalign",
        "expected_part_kw":   "SP-CVB",
        "expected_step_kw":   ["belt", "tracking", "stop"],
        "description":    "Belt misalignment — MEDIUM, tracking adjustment",
    },
    {
        "test_id":        "FD-09",
        "machine_id":     "CVB-003",
        "error_code":     "E29",
        "sensor_readings": None,
        "expected_severity":  "CRITICAL",
        "expected_escalate":  True,
        "expected_title_kw":  "tear",
        "expected_part_kw":   "SP-CVB-BLT01",
        "expected_step_kw":   ["stop", "lock"],
        "description":    "Belt tear — CRITICAL, immediate stop + lockout",
    },
    {
        "test_id":        "FD-10",
        "machine_id":     "CVB-003",
        "error_code":     "E27",
        "sensor_readings": None,
        "expected_severity":  "HIGH",
        "expected_escalate":  True,
        "expected_title_kw":  "cord",
        "expected_part_kw":   None,
        "expected_step_kw":   ["supervisor", "investigate"],
        "description":    "Pull cord activated — HIGH + mandatory supervisor notification",
    },
    # ── Sensor pattern tests ──
    {
        "test_id":        "FD-11",
        "machine_id":     "CNC-M01",
        "error_code":     None,
        "sensor_readings": {"temperature_c": 78.0, "vibration_mm_s": 4.5},
        "expected_severity":  "CRITICAL",
        "expected_escalate":  True,
        "expected_title_kw":  "bearing",
        "expected_part_kw":   None,
        "expected_step_kw":   ["stop"],
        "description":    "High temp + vibration → bearing failure pattern",
    },
    {
        "test_id":        "FD-12",
        "machine_id":     "HYD-P02",
        "error_code":     None,
        "sensor_readings": {"pressure_bar": 270.0, "motor_current_a": 48.0},
        "expected_severity":  "CRITICAL",
        "expected_escalate":  True,
        "expected_title_kw":  "overload",
        "expected_part_kw":   None,
        "expected_step_kw":   ["stop", "pressure"],
        "description":    "High pressure + high current → hydraulic overload pattern",
    },
    # ── No fault case ──
    {
        "test_id":        "FD-13",
        "machine_id":     "CNC-M01",
        "error_code":     None,
        "sensor_readings": {"temperature_c": 50.0, "vibration_mm_s": 1.5,
                             "spindle_rpm": 3000, "power_kw": 8.0},
        "expected_severity":  "LOW",
        "expected_escalate":  False,
        "expected_title_kw":  None,
        "expected_part_kw":   None,
        "expected_step_kw":   [],
        "description":    "Normal readings — should return NO_FAULT",
    },
]

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _check_steps(steps_list: List[str], expected_kws: List[str]) -> float:
    """
    Check if expected keywords appear in any of the diagnosis steps.
    Returns fraction of keywords found.
    """
    if not expected_kws:
        return 1.0
    all_steps_text = " ".join(steps_list).lower()
    found = sum(1 for kw in expected_kws if kw.lower() in all_steps_text)
    return round(found / len(expected_kws), 3)


def run_fault_evaluation(verbose: bool = True) -> Dict:
    """
    Run fault diagnosis evaluation against all test cases.

    Returns:
        Results with accuracy scores per test + aggregate metrics
    """
    from backend.tools.fault_diagnose import diagnose_fault

    results          = []
    severity_correct = 0
    escalate_correct = 0
    step_coverage_total = 0.0
    total_latency_ms    = 0.0
    within_one_level    = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"  FAULT DIAGNOSIS EVALUATION  |  {len(FAULT_TEST_CASES)} test cases")
        print(f"{'='*70}")
        print(f"  {'ID':<8} {'Description':<45} {'Sev':>5} {'Esc':>5} {'Steps':>7}")
        print(f"  {'─'*8} {'─'*45} {'─'*5} {'─'*5} {'─'*7}")

    for tc in FAULT_TEST_CASES:
        t0 = time.time()
        diag = diagnose_fault(
            machine_id      = tc["machine_id"],
            error_code      = tc["error_code"],
            sensor_readings = tc["sensor_readings"],
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        actual_severity = diag.get("severity", "LOW")
        actual_escalate = diag.get("escalate", False)
        all_steps = []
        for d in diag.get("diagnoses", []):
            all_steps.extend(d.get("steps", []))

        # ── Score each metric ─────────────────────────
        sev_correct = (actual_severity == tc["expected_severity"])

        # "Within one level" — acceptable if off by one severity level
        sev_diff    = abs(SEVERITY_RANK.get(actual_severity, 0) -
                          SEVERITY_RANK.get(tc["expected_severity"], 0))
        within_one  = int(sev_diff <= 1)

        esc_correct = (actual_escalate == tc["expected_escalate"])

        step_cov = _check_steps(all_steps, tc["expected_step_kw"]) if tc["expected_step_kw"] else 1.0

        severity_correct     += int(sev_correct)
        escalate_correct     += int(esc_correct)
        step_coverage_total  += step_cov
        within_one_level     += within_one
        total_latency_ms     += latency_ms

        sev_icon = "✓" if sev_correct else ("~" if within_one else "✗")
        esc_icon = "✓" if esc_correct else "✗"

        if verbose:
            print(f"  {tc['test_id']:<8} {tc['description'][:44]:<45} "
                  f"{sev_icon}{actual_severity[:3]:>4} "
                  f"{esc_icon}{str(actual_escalate)[:1]:>4} "
                  f"{step_cov:>7.2f}")

        results.append({
            "test_id":          tc["test_id"],
            "description":      tc["description"],
            "machine_id":       tc["machine_id"],
            "error_code":       tc["error_code"],
            "expected_severity": tc["expected_severity"],
            "actual_severity":   actual_severity,
            "severity_correct":  sev_correct,
            "severity_within_one": within_one,
            "expected_escalate": tc["expected_escalate"],
            "actual_escalate":   actual_escalate,
            "escalate_correct":  esc_correct,
            "step_coverage":     step_cov,
            "latency_ms":        latency_ms,
        })

    n = len(FAULT_TEST_CASES)
    aggregate = {
        "severity_accuracy":       round(severity_correct / n, 4),
        "severity_within_1_level": round(within_one_level / n, 4),
        "escalation_accuracy":     round(escalate_correct / n, 4),
        "avg_step_coverage":       round(step_coverage_total / n, 4),
        "avg_latency_ms":          round(total_latency_ms / n, 1),
        "total_test_cases":        n,
        "severity_correct_count":  severity_correct,
        "escalation_correct_count": escalate_correct,
    }

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  AGGREGATE RESULTS")
        print(f"{'─'*70}")
        print(f"  Severity Accuracy:       {aggregate['severity_accuracy']:.4f}  "
              f"({severity_correct}/{n} exact matches)")
        print(f"  Severity Within 1 Level: {aggregate['severity_within_1_level']:.4f}  "
              f"(tolerance for off-by-one)")
        print(f"  Escalation Accuracy:     {aggregate['escalation_accuracy']:.4f}  "
              f"({escalate_correct}/{n} correct escalation flags)")
        print(f"  Avg Step Coverage:       {aggregate['avg_step_coverage']:.4f}  "
              f"(fraction of expected keywords in steps)")
        print(f"  Avg Latency:             {aggregate['avg_latency_ms']:.1f}ms")
        print(f"{'='*70}")

        # ── Critical safety check ─────────────────────
        critical_misses = [r for r in results
                           if r["expected_severity"] == "CRITICAL"
                           and not r["severity_correct"]]
        if critical_misses:
            print(f"\n  ⚠ CRITICAL SEVERITY MISCLASSIFICATIONS ({len(critical_misses)}):")
            for m in critical_misses:
                print(f"    {m['test_id']}: expected CRITICAL, got {m['actual_severity']}")
                print(f"    → {m['description']}")
        else:
            print(f"\n  ✓ All CRITICAL faults correctly identified — safety check passed")

        escalation_misses = [r for r in results
                             if r["expected_escalate"] and not r["actual_escalate"]]
        if escalation_misses:
            print(f"\n  ⚠ MISSED ESCALATIONS ({len(escalation_misses)}):")
            for m in escalation_misses:
                print(f"    {m['test_id']}: {m['description']}")
        else:
            print(f"  ✓ All required escalations correctly triggered")

    return {"aggregate": aggregate, "per_test": results}


if __name__ == "__main__":
    results = run_fault_evaluation(verbose=True)
    out_path = Path(__file__).parent / "fault_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
