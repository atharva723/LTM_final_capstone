# ─────────────────────────────────────────────────────
# evaluation/response_quality.py
#
# RESPONSE QUALITY EVALUATION  —  BLEU + ROUGE-L
#
# WHAT IS EVALUATED:
#   The quality of text answers produced by tools and the
#   agent compared to reference "ideal" answers.
#
# METRICS:
#
#   BLEU (Bilingual Evaluation Understudy)
#   ─────────────────────────────────────
#   Measures n-gram overlap between generated and reference text.
#   BLEU-1 = unigram overlap  (word level)
#   BLEU-4 = 4-gram overlap   (phrase level)
#   Range: 0–1  (higher = more similar to reference)
#   Limitation: penalizes paraphrasing even if meaning is same.
#
#   ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
#   ─────────────────────────────────────────────────────────────
#   Measures Longest Common Subsequence (LCS) between texts.
#   More flexible than BLEU — doesn't need exact n-gram matches.
#   ROUGE-L F1 = harmonic mean of precision and recall of LCS.
#   Range: 0–1  (higher = better coverage of key phrases)
#
#   INTERPRETATION for this project:
#   > 0.40 ROUGE-L = Good     (enough key content covered)
#   > 0.25 ROUGE-L = Adequate (core message present)
#   < 0.20 ROUGE-L = Poor     (missing important content)
#
# WHY NECESSARY:
#   Without this, we don't know if "reduce feed rate, inspect
#   tool, check spindle" is actually in the response or if the
#   LLM hallucinated "reboot the controller".
# ─────────────────────────────────────────────────────

import sys
import re
import json
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────
# REFERENCE ANSWER DATASET
# These are "ideal" answers written by a domain expert.
# The system's output is compared against these.
# ─────────────────────────────────────────────────────
RESPONSE_TEST_CASES = [
    {
        "test_id":   "RQ-01",
        "query":     "What are the troubleshooting steps for Error E01 spindle overload on CNC-M01?",
        "tool":      "fault_diagnose",
        "tool_args": {"machine_id": "CNC-M01", "error_code": "E01"},
        "reference": (
            "Press E-STOP immediately and wait for spindle to fully stop. "
            "Open machine door and inspect cutting tool for breakage or wear. "
            "If tool is worn replace with new tool and update offset. "
            "Review G-code and verify feed rate and depth of cut match job card. "
            "Reduce feed rate by 20 percent and retry. "
            "Check spindle motor current at drive panel if overload repeats. "
            "Call maintenance to inspect spindle bearings if current exceeds rated value."
        ),
        "category":  "fault_troubleshooting",
    },
    {
        "test_id":   "RQ-02",
        "query":     "What PPE is required for tool change on CNC machine?",
        "tool":      "safety_checker",
        "tool_args": {"machine_id": "CNC-M01", "task": "tool change"},
        "reference": (
            "Safety glasses are mandatory at all times. "
            "Steel-toed safety boots are required. "
            "Cut-resistant gloves must be worn during tool handling. "
            "Loose gloves must not be worn during spindle rotation. "
            "LOTO lockout tagout is required before tool change maintenance."
        ),
        "category":  "safety_ppe",
    },
    {
        "test_id":   "RQ-03",
        "query":     "What is overdue on conveyor belt CVB-003 maintenance?",
        "tool":      "maintenance",
        "tool_args": {"machine_id": "CVB-003", "current_hours": 5500, "last_pm_hours": 5000},
        "reference": (
            "Level B monthly maintenance is overdue at interval of 160 hours. "
            "Tasks include measuring belt tension with tension meter. "
            "Inspect drive and tail pulley lagging condition. "
            "Check all roller bearing temperatures. "
            "Verify speed sensor reading versus actual speed. "
            "Test all safety pull cord and E-STOP buttons. "
            "Level C quarterly maintenance is also due at 500 hour interval."
        ),
        "category":  "maintenance",
    },
    {
        "test_id":   "RQ-04",
        "query":     "What is the OEE score and availability for CNC-M01?",
        "tool":      "metrics",
        "tool_args": {"machine_id": "CNC-M01"},
        "reference": (
            "OEE is calculated as availability times performance times quality. "
            "Availability measures run time divided by planned time. "
            "Performance compares actual cycle time to ideal cycle time. "
            "Quality measures good parts produced versus total parts. "
            "Downtime percentage shows time lost to faults and idle periods."
        ),
        "category":  "metrics",
    },
    {
        "test_id":   "RQ-05",
        "query":     "Error E20 hose burst on hydraulic pump what to do?",
        "tool":      "fault_diagnose",
        "tool_args": {"machine_id": "HYD-P02", "error_code": "E20"},
        "reference": (
            "Stop pump immediately. "
            "Evacuate area because high pressure fluid injection is fatal. "
            "Do not approach the area until system is fully depressurized. "
            "Identify the ruptured hose after depressurization. "
            "Replace with rated hose assembly and do not use temporary repairs. "
            "Report incident to safety officer."
        ),
        "category":  "critical_fault",
    },
    {
        "test_id":   "RQ-06",
        "query":     "What spare parts are needed for conveyor belt E23 misalignment fault?",
        "tool":      "fault_diagnose",
        "tool_args": {"machine_id": "CVB-003", "error_code": "E23"},
        "reference": (
            "Belt tracking guide set part SP-CVB-B02 is recommended for belt misalignment repair. "
            "Check availability and stock quantity before ordering. "
            "Lead time and supplier information should be verified with procurement."
        ),
        "category":  "spare_parts",
    },
    {
        "test_id":   "RQ-07",
        "query":     "Daily maintenance checklist for hydraulic pump",
        "tool":      "maintenance",
        "tool_args": {"machine_id": "HYD-P02", "current_hours": 3200},
        "reference": (
            "Check hydraulic fluid level using sight glass. "
            "Inspect all hoses and fittings for leaks. "
            "Check system pressure at operating gauge. "
            "Listen for unusual pump noise such as knocking or whining. "
            "Verify filter indicator and replace if in red zone. "
            "Log operating hours in maintenance logbook."
        ),
        "category":  "maintenance",
    },
]


# ─────────────────────────────────────────────────────
# METRIC IMPLEMENTATIONS
# ─────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def compute_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    """Count n-grams in a token list."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-max_n scores.
    Uses clipped precision (standard BLEU).
    """
    import math

    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if not hyp_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    scores = {}
    for n in range(1, max_n + 1):
        hyp_ngrams = compute_ngrams(hyp_tokens, n)
        ref_ngrams = compute_ngrams(ref_tokens, n)

        if not hyp_ngrams:
            scores[f"bleu_{n}"] = 0.0
            continue

        # Clipped count
        clipped = sum(
            min(count, ref_ngrams.get(gram, 0))
            for gram, count in hyp_ngrams.items()
        )
        total = sum(hyp_ngrams.values())
        precision = clipped / total if total > 0 else 0

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))
        scores[f"bleu_{n}"] = round(bp * precision, 4)

    return scores


def rouge_l_score(hypothesis: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L F1 using Longest Common Subsequence (LCS).
    More robust than BLEU for paraphrased answers.
    """
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    # LCS via dynamic programming
    m, n   = len(ref_tokens), len(hyp_tokens)
    dp     = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0
    recall    = lcs_len / m if m > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return {
        "rouge_l_precision": round(precision, 4),
        "rouge_l_recall":    round(recall, 4),
        "rouge_l_f1":        round(f1, 4),
    }


def interpret_rouge(score: float) -> str:
    if score >= 0.50: return "Excellent"
    if score >= 0.40: return "Good"
    if score >= 0.25: return "Adequate"
    return "Poor"


# ─────────────────────────────────────────────────────
# TOOL OUTPUT EXTRACTOR
# Gets the text output from each tool for comparison
# ─────────────────────────────────────────────────────

def _get_tool_output(tool: str, tool_args: dict) -> str:
    """Run the specified tool and return its text output."""
    try:
        if tool == "fault_diagnose":
            from backend.tools.fault_diagnose import diagnose_fault
            result = diagnose_fault(**tool_args)
            # Flatten diagnosis to text
            lines = [result.get("summary", "")]
            for d in result.get("diagnoses", []):
                lines.append(d.get("failure_mode", ""))
                lines.extend(d.get("steps", []))
            return " ".join(lines)

        elif tool == "safety_checker":
            from backend.tools.safety_checker import format_safety_report
            return format_safety_report(
                tool_args["machine_id"], tool_args["task"]
            )

        elif tool == "maintenance":
            from backend.tools.maintenance import format_pm_report
            return format_pm_report(
                tool_args["machine_id"],
                tool_args.get("current_hours", 5000),
                tool_args.get("last_pm_hours"),
            )

        elif tool == "metrics":
            from backend.tools.metrics import format_metrics_report
            return format_metrics_report(tool_args["machine_id"])

        else:
            return f"Unknown tool: {tool}"
    except Exception as e:
        return f"Tool error: {e}"


# ─────────────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────────────

def run_response_quality_evaluation(verbose: bool = True) -> Dict:
    """
    Run BLEU + ROUGE-L evaluation on tool outputs vs reference answers.
    """
    results           = []
    total_rouge_l_f1  = 0.0
    total_bleu_1      = 0.0
    total_bleu_4      = 0.0
    total_latency_ms  = 0.0

    category_scores: Dict[str, list] = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"  RESPONSE QUALITY EVALUATION  |  BLEU + ROUGE-L")
        print(f"  {len(RESPONSE_TEST_CASES)} test cases")
        print(f"{'='*70}")
        print(f"  {'ID':<8} {'Category':<22} {'BLEU-1':>7} {'BLEU-4':>7} {'ROUGE-L':>8} {'Quality':<12}")
        print(f"  {'─'*8} {'─'*22} {'─'*7} {'─'*7} {'─'*8} {'─'*12}")

    for tc in RESPONSE_TEST_CASES:
        t0         = time.time()
        hypothesis = _get_tool_output(tc["tool"], tc["tool_args"])
        latency_ms = round((time.time() - t0) * 1000, 1)

        bleu   = bleu_score(hypothesis, tc["reference"])
        rouge  = rouge_l_score(hypothesis, tc["reference"])

        b1  = bleu.get("bleu_1", 0)
        b4  = bleu.get("bleu_4", 0)
        rl  = rouge["rouge_l_f1"]
        qual = interpret_rouge(rl)

        total_rouge_l_f1  += rl
        total_bleu_1      += b1
        total_bleu_4      += b4
        total_latency_ms  += latency_ms

        cat = tc["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(rl)

        if verbose:
            print(f"  {tc['test_id']:<8} {tc['category']:<22} "
                  f"{b1:>7.4f} {b4:>7.4f} {rl:>8.4f} {qual:<12}")

        results.append({
            "test_id":      tc["test_id"],
            "query":        tc["query"][:60],
            "category":     cat,
            "tool":         tc["tool"],
            "bleu_1":       b1,
            "bleu_4":       b4,
            "rouge_l_f1":   rl,
            "rouge_l_p":    rouge["rouge_l_precision"],
            "rouge_l_r":    rouge["rouge_l_recall"],
            "quality":      qual,
            "latency_ms":   latency_ms,
            "hypothesis_len": len(hypothesis.split()),
            "reference_len":  len(tc["reference"].split()),
        })

    n = len(RESPONSE_TEST_CASES)
    aggregate = {
        "mean_bleu_1":     round(total_bleu_1 / n, 4),
        "mean_bleu_4":     round(total_bleu_4 / n, 4),
        "mean_rouge_l_f1": round(total_rouge_l_f1 / n, 4),
        "quality_rating":  interpret_rouge(total_rouge_l_f1 / n),
        "avg_latency_ms":  round(total_latency_ms / n, 1),
        "total_test_cases": n,
    }

    category_summary = {
        cat: {"avg_rouge_l": round(sum(s)/len(s), 3), "count": len(s)}
        for cat, s in category_scores.items()
    }

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  AGGREGATE RESULTS")
        print(f"{'─'*70}")
        print(f"  Mean BLEU-1:       {aggregate['mean_bleu_1']:.4f}")
        print(f"  Mean BLEU-4:       {aggregate['mean_bleu_4']:.4f}")
        print(f"  Mean ROUGE-L F1:   {aggregate['mean_rouge_l_f1']:.4f}  "
              f"→ {aggregate['quality_rating']}")
        print(f"  Avg Tool Latency:  {aggregate['avg_latency_ms']:.1f}ms")
        print(f"\n  CATEGORY BREAKDOWN (ROUGE-L):")
        for cat, cs in sorted(category_summary.items()):
            bar  = "█" * int(cs["avg_rouge_l"] * 20)
            print(f"    {cat:<28} {cs['avg_rouge_l']:.3f}  {bar}")
        print(f"{'='*70}")

    return {
        "aggregate":        aggregate,
        "category_summary": category_summary,
        "per_test":         results,
    }


if __name__ == "__main__":
    results = run_response_quality_evaluation(verbose=True)
    out_path = Path(__file__).parent / "response_quality_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
