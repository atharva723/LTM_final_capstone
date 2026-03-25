# ─────────────────────────────────────────────────────
# evaluation/latency_tests.py
#
# SENSOR ANOMALY + LATENCY + STRESS EVALUATION
#
# SECTION A — Sensor Anomaly Detection Evaluation
#   Tests whether the threshold logic correctly classifies
#   normal vs warning vs alarm readings.
#   Metrics: Precision, Recall, F1 for each severity class.
#
# SECTION B — Latency Tests
#   Measures response time for each tool and the
#   end-to-end agent pipeline.
#   Target SLAs for a factory floor:
#     Sensor fetch:    < 100ms
#     Fault diagnosis: < 200ms
#     Parts lookup:    < 100ms
#     Maintenance:     < 150ms
#     Log analysis:    < 500ms
#     Agent pipeline:  < 2000ms (with tool calls, no LLM)
#
# SECTION C — Stress Tests
#   Simulates concurrent load:
#     • 5 machines polled every second for 10 seconds
#     • Multiple fault diagnoses in rapid succession
#     • Cache effectiveness measurement
#
# WHY NECESSARY:
#   A factory operator cannot wait 30 seconds for an answer
#   while a machine is alarming. Latency tests ensure the
#   system meets real-time requirements.
# ─────────────────────────────────────────────────────

import sys
import time
import json
import statistics
import threading
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════
# SECTION A — SENSOR ANOMALY DETECTION EVALUATION
# ══════════════════════════════════════════════════════

ANOMALY_TEST_CASES = [
    # (machine_id, sensor, value, expected_status)
    # CNC-M01 — normal readings
    ("CNC-M01", "temperature_c",      50.0,  "NORMAL"),
    ("CNC-M01", "temperature_c",      62.0,  "NORMAL"),
    ("CNC-M01", "vibration_mm_s",      1.5,  "NORMAL"),
    ("CNC-M01", "spindle_rpm",       4000.0,  "NORMAL"),
    ("CNC-M01", "coolant_flow_l_min", 10.0,  "NORMAL"),
    ("CNC-M01", "power_kw",            8.0,  "NORMAL"),
    # CNC-M01 — warning readings
    ("CNC-M01", "temperature_c",      68.0,  "WARNING"),
    ("CNC-M01", "vibration_mm_s",      3.8,  "WARNING"),
    ("CNC-M01", "coolant_flow_l_min",  5.0,  "WARNING"),
    ("CNC-M01", "power_kw",           16.0,  "WARNING"),
    # CNC-M01 — alarm readings
    ("CNC-M01", "temperature_c",      85.0,  "ALARM"),
    ("CNC-M01", "vibration_mm_s",      5.5,  "ALARM"),
    ("CNC-M01", "coolant_flow_l_min",  2.0,  "ALARM"),
    ("CNC-M01", "power_kw",           23.0,  "ALARM"),
    # HYD-P02 — normal
    ("HYD-P02", "pressure_bar",      180.0,  "NORMAL"),
    ("HYD-P02", "temperature_c",      50.0,  "NORMAL"),
    ("HYD-P02", "flow_l_min",         45.0,  "NORMAL"),
    # HYD-P02 — warning
    ("HYD-P02", "pressure_bar",      255.0,  "WARNING"),
    ("HYD-P02", "temperature_c",      67.0,  "WARNING"),
    ("HYD-P02", "flow_l_min",         12.0,  "WARNING"),
    # HYD-P02 — alarm
    ("HYD-P02", "pressure_bar",      285.0,  "ALARM"),
    ("HYD-P02", "temperature_c",      78.0,  "ALARM"),
    ("HYD-P02", "flow_l_min",          5.0,  "ALARM"),
    # CVB-003 — normal
    ("CVB-003", "belt_speed_m_s",      1.0,  "NORMAL"),
    ("CVB-003", "motor_temp_c",        55.0,  "NORMAL"),
    ("CVB-003", "belt_tension_n",     900.0,  "NORMAL"),
    # CVB-003 — warning
    ("CVB-003", "belt_speed_m_s",      1.65,  "WARNING"),
    ("CVB-003", "motor_temp_c",        78.0,  "WARNING"),
    ("CVB-003", "belt_tension_n",     700.0,  "WARNING"),
    # CVB-003 — alarm
    ("CVB-003", "belt_speed_m_s",      1.9,  "ALARM"),
    ("CVB-003", "motor_temp_c",        90.0,  "ALARM"),
    ("CVB-003", "belt_tension_n",     350.0,  "ALARM"),
]


def run_anomaly_evaluation(verbose: bool = True) -> Dict:
    """
    Evaluate sensor anomaly detection accuracy.
    Tests whether _classify_reading() assigns correct status.
    """
    from backend.tools.sensor_fetch import _classify_reading

    tp = {"NORMAL": 0, "WARNING": 0, "ALARM": 0}
    fp = {"NORMAL": 0, "WARNING": 0, "ALARM": 0}
    fn = {"NORMAL": 0, "WARNING": 0, "ALARM": 0}

    correct = 0
    results = []

    if verbose:
        print(f"\n{'='*65}")
        print(f"  SENSOR ANOMALY DETECTION EVALUATION  |  {len(ANOMALY_TEST_CASES)} cases")
        print(f"{'='*65}")

    for machine_id, sensor, value, expected in ANOMALY_TEST_CASES:
        classification = _classify_reading(machine_id, sensor, value)
        actual = classification["status"]

        is_correct = (actual == expected)
        if is_correct:
            correct += 1
            tp[expected] += 1
        else:
            fn[expected] += 1
            fp[actual]   += 1

        results.append({
            "machine_id": machine_id,
            "sensor":     sensor,
            "value":      value,
            "expected":   expected,
            "actual":     actual,
            "correct":    is_correct,
        })

    # Per-class F1
    class_metrics = {}
    for cls in ["NORMAL", "WARNING", "ALARM"]:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        class_metrics[cls] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "support":   tp[cls] + fn[cls],
        }

    n = len(ANOMALY_TEST_CASES)
    accuracy = round(correct / n, 4)

    if verbose:
        print(f"\n  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
        for cls, m in class_metrics.items():
            flag = " ⚠" if cls == "ALARM" and m["f1"] < 0.9 else ""
            print(f"  {cls:<10} {m['precision']:>10.4f} {m['recall']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['support']:>8}{flag}")
        print(f"\n  Overall Accuracy: {accuracy:.4f}  ({correct}/{n})")

        misses = [r for r in results if not r["correct"] and r["expected"] == "ALARM"]
        if misses:
            print(f"\n  ⚠ Missed ALARM detections ({len(misses)}):")
            for m in misses:
                print(f"    {m['machine_id']} {m['sensor']}={m['value']} "
                      f"expected=ALARM got={m['actual']}")
        else:
            print(f"  ✓ All ALARM readings correctly detected — safety check passed")
        print(f"{'='*65}")

    return {
        "accuracy":       accuracy,
        "class_metrics":  class_metrics,
        "correct":        correct,
        "total":          n,
        "per_test":       results,
    }


# ══════════════════════════════════════════════════════
# SECTION B — LATENCY TESTS
# ══════════════════════════════════════════════════════

LATENCY_SLA_MS = {
    "sensor_fetch":    100,
    "fault_diagnose":  200,
    "spare_parts":     100,
    "maintenance":     150,
    "log_analyzer":    500,
    "safety_checker":  100,
    "metrics":         200,
    "agent_pipeline":  2000,
}


def _measure_latency(fn, *args, **kwargs) -> Tuple[any, float]:
    """Run a function and return (result, elapsed_ms)."""
    t0     = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, round(elapsed, 2)


def run_latency_tests(n_runs: int = 5, verbose: bool = True) -> Dict:
    """
    Measure latency for each tool over n_runs iterations.
    Reports mean, median, p95, and SLA pass/fail.
    """
    from backend.tools.sensor_fetch   import get_sensor_data
    from backend.tools.fault_diagnose import diagnose_fault
    from backend.tools.spare_parts    import lookup_parts_by_machine
    from backend.tools.maintenance    import calculate_pm_due
    from backend.tools.log_analyzer   import analyze_log
    from backend.tools.safety_checker import check_safety
    from backend.tools.metrics        import compute_oee
    from backend.agent                import get_agent

    test_fns = {
        "sensor_fetch":   lambda: get_sensor_data("CNC-M01", use_cache=False),
        "fault_diagnose": lambda: diagnose_fault("CNC-M01", "E01"),
        "spare_parts":    lambda: lookup_parts_by_machine("CNC-M01"),
        "maintenance":    lambda: calculate_pm_due("CNC-M01", 5000),
        "log_analyzer":   lambda: analyze_log("CNC-M01"),
        "safety_checker": lambda: check_safety("CNC-M01", "maintenance"),
        "metrics":        lambda: compute_oee("CNC-M01"),
    }

    results = {}

    if verbose:
        print(f"\n{'='*65}")
        print(f"  LATENCY TESTS  |  {n_runs} runs per tool")
        print(f"{'='*65}")
        print(f"  {'Tool':<18} {'Mean':>8} {'Median':>8} {'P95':>8} {'SLA':>8} {'Pass':>6}")
        print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

    for tool_name, fn in test_fns.items():
        latencies = []
        for _ in range(n_runs):
            _, elapsed = _measure_latency(fn)
            latencies.append(elapsed)

        mean_ms   = round(statistics.mean(latencies), 1)
        median_ms = round(statistics.median(latencies), 1)
        p95_ms    = round(sorted(latencies)[int(n_runs * 0.95)], 1)
        sla_ms    = LATENCY_SLA_MS.get(tool_name, 500)
        passed    = mean_ms <= sla_ms

        results[tool_name] = {
            "mean_ms":   mean_ms,
            "median_ms": median_ms,
            "p95_ms":    p95_ms,
            "sla_ms":    sla_ms,
            "passed":    passed,
            "all_runs":  latencies,
        }

        if verbose:
            icon = "✓" if passed else "✗"
            print(f"  {tool_name:<18} {mean_ms:>7.1f}ms {median_ms:>7.1f}ms "
                  f"{p95_ms:>7.1f}ms {sla_ms:>7}ms {icon:>6}")

    # Agent pipeline test (tool calls only, no LLM)
    agent = get_agent("latency_test")
    agent._initialized = True
    agent.llm = None
    agent_latencies = []
    for _ in range(min(3, n_runs)):
        _, elapsed = _measure_latency(agent.chat, "What is the sensor status of CNC-M01?")
        agent_latencies.append(elapsed)

    agent_mean = round(statistics.mean(agent_latencies), 1)
    agent_sla  = LATENCY_SLA_MS["agent_pipeline"]
    results["agent_pipeline"] = {
        "mean_ms":  agent_mean,
        "sla_ms":   agent_sla,
        "passed":   agent_mean <= agent_sla,
        "all_runs": agent_latencies,
    }
    if verbose:
        icon = "✓" if agent_mean <= agent_sla else "✗"
        print(f"  {'agent_pipeline':<18} {agent_mean:>7.1f}ms {'─':>7} "
              f"{'─':>7} {agent_sla:>7}ms {icon:>6}")
        passing = sum(1 for v in results.values() if v["passed"])
        print(f"\n  SLA Pass Rate: {passing}/{len(results)} tools")
        print(f"{'='*65}")

    return results


# ══════════════════════════════════════════════════════
# SECTION C — STRESS TESTS
# ══════════════════════════════════════════════════════

def run_stress_tests(verbose: bool = True) -> Dict:
    """
    Stress test: concurrent machine polling + cache effectiveness.
    """
    from backend.tools.sensor_fetch import get_sensor_data
    from backend.tools.fault_diagnose import diagnose_fault

    machines = ["CNC-M01", "HYD-P02", "CVB-003"]

    if verbose:
        print(f"\n{'='*65}")
        print(f"  STRESS TESTS")
        print(f"{'='*65}")

    # ── Test 1: Concurrent sensor polling ─────────────
    if verbose:
        print(f"\n  [1] Concurrent sensor polling — {len(machines)} machines × 10 rounds")

    results_concurrent = []
    errors = 0

    def poll_machine(mid, results_list):
        try:
            t0 = time.perf_counter()
            get_sensor_data(mid, use_cache=False)
            results_list.append((time.perf_counter() - t0) * 1000)
        except Exception:
            global errors
            errors += 1

    for round_num in range(10):
        threads   = []
        round_ms  = []
        t_round   = time.perf_counter()
        for mid in machines:
            t = threading.Thread(target=poll_machine, args=(mid, round_ms))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        results_concurrent.extend(round_ms)

    all_ms    = results_concurrent
    mean_conc = round(statistics.mean(all_ms), 1) if all_ms else 0
    p95_conc  = round(sorted(all_ms)[int(len(all_ms) * 0.95)], 1) if all_ms else 0

    if verbose:
        print(f"    Total requests: {len(all_ms)}  |  Errors: {errors}")
        print(f"    Mean latency:   {mean_conc}ms  |  P95: {p95_conc}ms")
        print(f"    Result: {'✓ PASS' if mean_conc < 200 else '✗ FAIL'}")

    # ── Test 2: Cache effectiveness ───────────────────
    if verbose:
        print(f"\n  [2] Cache effectiveness test")

    _, t_no_cache = _measure_latency(get_sensor_data, "CNC-M01", use_cache=False)
    _, t_cached   = _measure_latency(get_sensor_data, "CNC-M01", use_cache=True)
    speedup       = round(t_no_cache / t_cached, 1) if t_cached > 0 else 1.0

    if verbose:
        print(f"    Without cache: {t_no_cache:.1f}ms")
        print(f"    With cache:    {t_cached:.1f}ms")
        print(f"    Cache speedup: {speedup}×")

    # ── Test 3: Rapid fault diagnoses ─────────────────
    if verbose:
        print(f"\n  [3] Rapid fault diagnosis — 20 queries in sequence")

    codes = ["E01","E02","E03","E08","E11","E20","E23","E29"] * 3
    fault_times = []
    for code in codes[:20]:
        machine = "CNC-M01" if code.startswith("E0") else ("HYD-P02" if code in ("E11","E20") else "CVB-003")
        _, elapsed = _measure_latency(diagnose_fault, machine, code)
        fault_times.append(elapsed)

    mean_fault = round(statistics.mean(fault_times), 1)
    if verbose:
        print(f"    20 diagnoses completed")
        print(f"    Mean latency: {mean_fault}ms  |  "
              f"Max: {round(max(fault_times),1)}ms")
        print(f"    Result: {'✓ PASS' if mean_fault < 300 else '✗ FAIL'}")
        print(f"\n{'='*65}")

    return {
        "concurrent_polling": {
            "total_requests": len(all_ms),
            "errors":         errors,
            "mean_ms":        mean_conc,
            "p95_ms":         p95_conc,
            "passed":         mean_conc < 200,
        },
        "cache_effectiveness": {
            "no_cache_ms":   t_no_cache,
            "cached_ms":     t_cached,
            "speedup_factor": speedup,
        },
        "rapid_fault_diagnosis": {
            "queries":   len(fault_times),
            "mean_ms":   mean_fault,
            "max_ms":    round(max(fault_times), 1),
            "passed":    mean_fault < 300,
        },
    }


# ── Run all sections ──────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*65)
    print("  MANUFACTURING ASSISTANT — FULL EVALUATION SUITE")
    print("═"*65)

    print("\n--- Section A: Anomaly Detection ---")
    anomaly_results = run_anomaly_evaluation(verbose=True)

    print("\n--- Section B: Latency Tests ---")
    latency_results = run_latency_tests(n_runs=5, verbose=True)

    print("\n--- Section C: Stress Tests ---")
    stress_results = run_stress_tests(verbose=True)

    # Save all results
    all_results = {
        "anomaly_detection": anomaly_results,
        "latency_tests":     latency_results,
        "stress_tests":      stress_results,
    }
    out_path = Path(__file__).parent / "latency_stress_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ All results saved to: {out_path}")
