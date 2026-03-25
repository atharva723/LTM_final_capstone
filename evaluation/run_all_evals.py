# ─────────────────────────────────────────────────────
# evaluation/run_all_evals.py
#
# MASTER EVALUATION RUNNER
# Runs all 4 evaluation modules and prints a final
# consolidated report with pass/fail for each metric.
#
# USAGE:  python evaluation/run_all_evals.py
# ─────────────────────────────────────────────────────

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def print_section(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def grade(score: float, thresholds: dict) -> str:
    """Return letter grade based on score thresholds."""
    if score >= thresholds.get("A", 0.85): return "A ✓"
    if score >= thresholds.get("B", 0.70): return "B ✓"
    if score >= thresholds.get("C", 0.55): return "C ~"
    return "F ✗"


def run_all_evaluations(save_report: bool = True) -> dict:
    t_start = time.time()

    print("\n" + "═"*65)
    print("  MANUFACTURING ASSISTANT — COMPLETE EVALUATION REPORT")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*65)

    all_results = {}

    # ── 1. RAG Evaluation ────────────────────────────
    print_section("1/4  RAG PIPELINE EVALUATION")
    try:
        from evaluation.rag_eval import run_rag_evaluation
        rag = run_rag_evaluation(k=4, verbose=True)
        all_results["rag"] = rag["aggregate"]
    except Exception as e:
        print(f"  RAG eval failed: {e}")
        all_results["rag"] = {"error": str(e)}

    # ── 2. Fault Diagnosis Evaluation ────────────────
    print_section("2/4  FAULT DIAGNOSIS EVALUATION")
    try:
        from evaluation.fault_eval import run_fault_evaluation
        fault = run_fault_evaluation(verbose=True)
        all_results["fault"] = fault["aggregate"]
    except Exception as e:
        print(f"  Fault eval failed: {e}")
        all_results["fault"] = {"error": str(e)}

    # ── 3. Response Quality Evaluation ───────────────
    print_section("3/4  RESPONSE QUALITY (BLEU + ROUGE-L)")
    try:
        from evaluation.response_quality import run_response_quality_evaluation
        rq = run_response_quality_evaluation(verbose=True)
        all_results["response_quality"] = rq["aggregate"]
    except Exception as e:
        print(f"  Response quality eval failed: {e}")
        all_results["response_quality"] = {"error": str(e)}

    # ── 4. Latency + Anomaly + Stress ────────────────
    print_section("4/4  LATENCY + ANOMALY DETECTION + STRESS TESTS")
    try:
        from evaluation.latency_tests import (
            run_anomaly_evaluation, run_latency_tests, run_stress_tests
        )
        anomaly = run_anomaly_evaluation(verbose=True)
        latency = run_latency_tests(n_runs=5, verbose=True)
        stress  = run_stress_tests(verbose=True)
        all_results["anomaly_detection"] = {
            "accuracy": anomaly["accuracy"],
            "alarm_f1": anomaly["class_metrics"]["ALARM"]["f1"],
        }
        all_results["latency"] = {
            k: v.get("passed", False) for k, v in latency.items()
        }
        all_results["stress"] = {
            k: v.get("passed", v.get("speedup_factor", False))
            for k, v in stress.items()
        }
    except Exception as e:
        print(f"  Latency/stress eval failed: {e}")
        all_results["latency"] = {"error": str(e)}

    # ── FINAL SCORECARD ──────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    print_section("FINAL EVALUATION SCORECARD")

    scorecard_rows = []

    # RAG scores
    rag_agg = all_results.get("rag", {})
    if "error" not in rag_agg:
        rows = [
            ("RAG Precision@4",   rag_agg.get("mean_precision_at_k", 0),   {"A":.70,"B":.55,"C":.40}),
            ("RAG Recall@4",      rag_agg.get("mean_recall_at_k", 0),      {"A":.70,"B":.55,"C":.40}),
            ("RAG Hit Rate@4",    rag_agg.get("hit_rate_at_k", 0),         {"A":.85,"B":.70,"C":.55}),
            ("RAG MRR",           rag_agg.get("mean_reciprocal_rank", 0),  {"A":.70,"B":.55,"C":.40}),
        ]
        scorecard_rows.extend(rows)

    # Fault scores
    fault_agg = all_results.get("fault", {})
    if "error" not in fault_agg:
        rows = [
            ("Fault Severity Accuracy",  fault_agg.get("severity_accuracy", 0),       {"A":.90,"B":.80,"C":.70}),
            ("Fault Within 1 Level",     fault_agg.get("severity_within_1_level", 0), {"A":.95,"B":.85,"C":.75}),
            ("Escalation Accuracy",      fault_agg.get("escalation_accuracy", 0),     {"A":.95,"B":.85,"C":.75}),
            ("Step Coverage",            fault_agg.get("avg_step_coverage", 0),       {"A":.80,"B":.65,"C":.50}),
        ]
        scorecard_rows.extend(rows)

    # Response quality
    rq_agg = all_results.get("response_quality", {})
    if "error" not in rq_agg:
        rows = [
            ("ROUGE-L F1",   rq_agg.get("mean_rouge_l_f1", 0), {"A":.50,"B":.35,"C":.20}),
            ("BLEU-1",       rq_agg.get("mean_bleu_1", 0),      {"A":.50,"B":.35,"C":.20}),
        ]
        scorecard_rows.extend(rows)

    # Anomaly detection
    anom_agg = all_results.get("anomaly_detection", {})
    if anom_agg and "error" not in anom_agg:
        rows = [
            ("Anomaly Accuracy",  anom_agg.get("accuracy", 0),  {"A":.95,"B":.85,"C":.75}),
            ("ALARM F1 Score",    anom_agg.get("alarm_f1", 0),  {"A":.95,"B":.85,"C":.75}),
        ]
        scorecard_rows.extend(rows)

    print(f"\n  {'Metric':<30} {'Score':>8} {'Grade':>8}")
    print(f"  {'─'*30} {'─'*8} {'─'*8}")
    grades_list = []
    for metric, score, thresholds in scorecard_rows:
        g = grade(score, thresholds)
        grades_list.append(g)
        print(f"  {metric:<30} {score:>8.4f} {g:>8}")

    # Overall health
    pass_count  = sum(1 for g in grades_list if "✓" in g)
    total_count = len(grades_list)
    overall_pct = round(pass_count / total_count * 100, 1) if total_count > 0 else 0

    print(f"\n  {'─'*50}")
    print(f"  Overall Score: {pass_count}/{total_count} metrics passing  ({overall_pct}%)")
    if overall_pct >= 80:
        print(f"  ✅ System READY for deployment")
    elif overall_pct >= 60:
        print(f"  ⚠️  System needs improvement before deployment")
    else:
        print(f"  ❌ System needs significant work — do not deploy")

    print(f"\n  Total evaluation time: {elapsed}s")
    print(f"{'═'*65}\n")

    if save_report:
        report = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":    elapsed,
            "scorecard":    [{"metric": m, "score": s} for m, s, _ in scorecard_rows],
            "pass_rate":    overall_pct,
            "all_results":  all_results,
        }
        out_path = Path(__file__).parent / "final_eval_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved → {out_path}")

    return all_results


if __name__ == "__main__":
    run_all_evaluations(save_report=True)
