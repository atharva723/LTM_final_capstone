# ─────────────────────────────────────────────────────
# backend/tools/metrics.py
#
# TOOL 7 — Production Metrics Tool
#
# PURPOSE:
#   Computes production KPIs from mock production logs:
#     • OEE (Overall Equipment Effectiveness)
#     • Availability, Performance, Quality components
#     • Downtime percentage and breakdown
#     • Throughput rate
#     • Cycle time deviation
#
# OEE Formula:
#   OEE = Availability × Performance × Quality
#   Availability = Run Time / Planned Time
#   Performance  = (Ideal Cycle Time × Parts) / Run Time
#   Quality      = Good Parts / Total Parts
#
# USED BY: agent.py, FastAPI /metrics endpoint,
#          Streamlit KPI dashboard panel
# ─────────────────────────────────────────────────────

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MACHINES, LOGS_DIR

# ── Ideal cycle times per machine (seconds) ──────────
IDEAL_CYCLE_TIMES = {
    "CNC-M01":  28.0,    # 28 sec per part (ideal)
    "HYD-P02":  15.0,    # 15 sec per press cycle
    "CVB-003":  5.0,     # 5 sec per item transported
    "BLR-004":  None,    # continuous process — no cycle time
    "ROB-005":  22.0,    # 22 sec per robot cycle
}

# ── Shift definitions ─────────────────────────────────
SHIFT_HOURS = 8          # hours per shift
PLANNED_HOURS_PER_DAY = 16   # 2 production shifts


def _load_log_df(machine_id: str):
    """Load machine log CSV into a pandas DataFrame."""
    try:
        import pandas as pd
        log_path = Path(LOGS_DIR) / f"{machine_id}_7day_log.csv"
        if not log_path.exists():
            return None
        return pd.read_csv(log_path, parse_dates=["timestamp"])
    except ImportError:
        return None


def compute_oee(
    machine_id: str,
    planned_time_hrs: float = PLANNED_HOURS_PER_DAY * 7,
    df=None,
) -> dict:
    """
    Compute full OEE breakdown for a machine.

    Args:
        machine_id:       e.g. "CNC-M01"
        planned_time_hrs: Total planned production time in hours
        df:               Optional pre-loaded DataFrame

    Returns:
        OEE components dict with scores and KPIs
    """
    machine_id = machine_id.upper().strip()

    if machine_id not in MACHINES:
        return {"error": f"Unknown machine: {machine_id}"}

    # Try loading real log data
    if df is None:
        df = _load_log_df(machine_id)

    if df is not None and not df.empty:
        return _compute_from_log(machine_id, df, planned_time_hrs)
    else:
        return _compute_mock_oee(machine_id, planned_time_hrs)


def _compute_from_log(machine_id: str, df, planned_time_hrs: float) -> dict:
    """Compute OEE from actual log data."""
    import pandas as pd

    total_rows   = len(df)
    running_rows = len(df[df["status"] == "Running"])
    fault_rows   = len(df[df["status"] == "Fault"])
    idle_rows    = len(df[df["status"] == "Idle"])

    availability = round(running_rows / total_rows, 4) if total_rows > 0 else 0

    # Performance: compare actual vs ideal cycle time
    ideal_ct = IDEAL_CYCLE_TIMES.get(machine_id)
    if ideal_ct and "cycle_time_s" in df.columns:
        running_df  = df[df["status"] == "Running"]["cycle_time_s"]
        actual_ct   = running_df.mean() if len(running_df) > 0 else ideal_ct
        performance = round(min(1.0, ideal_ct / actual_ct) if actual_ct > 0 else 0, 4)
    else:
        performance = round(random.uniform(0.82, 0.96), 4)

    # Quality: use log oee_pct as proxy (it includes quality factor)
    avg_log_oee = df["oee_pct"].mean() / 100 if "oee_pct" in df.columns else 0.9
    quality = round(min(1.0, avg_log_oee / (availability * performance + 0.001)), 4)
    quality = min(1.0, max(0.8, quality))   # clamp to realistic range

    oee = round(availability * performance * quality, 4)

    # Downtime breakdown
    interval_mins = 10   # each row = 10 minutes
    total_mins    = total_rows * interval_mins
    downtime_mins = fault_rows * interval_mins
    idle_mins     = idle_rows * interval_mins

    # Throughput
    if ideal_ct and ideal_ct > 0:
        run_time_secs  = running_rows * interval_mins * 60
        actual_parts   = int(run_time_secs / IDEAL_CYCLE_TIMES[machine_id])
        ideal_parts    = int(planned_time_hrs * 3600 / ideal_ct)
    else:
        actual_parts = ideal_parts = 0

    return _build_result(
        machine_id, oee, availability, performance, quality,
        downtime_mins, idle_mins, total_mins,
        actual_parts, ideal_parts, "log_data"
    )


def _compute_mock_oee(machine_id: str, planned_time_hrs: float) -> dict:
    """Generate realistic mock OEE when no log data is available."""
    random.seed(hash(machine_id) % 1000)
    availability = round(random.uniform(0.82, 0.95), 4)
    performance  = round(random.uniform(0.83, 0.97), 4)
    quality      = round(random.uniform(0.94, 0.99), 4)
    oee          = round(availability * performance * quality, 4)

    total_mins    = int(planned_time_hrs * 60)
    downtime_mins = int(total_mins * (1 - availability) * 0.7)
    idle_mins     = int(total_mins * (1 - availability) * 0.3)

    ideal_ct = IDEAL_CYCLE_TIMES.get(machine_id, 30)
    if ideal_ct:
        run_secs     = (total_mins - downtime_mins - idle_mins) * 60
        actual_parts = int(run_secs / ideal_ct)
        ideal_parts  = int(planned_time_hrs * 3600 / ideal_ct)
    else:
        actual_parts = ideal_parts = 0

    return _build_result(
        machine_id, oee, availability, performance, quality,
        downtime_mins, idle_mins, total_mins,
        actual_parts, ideal_parts, "estimated"
    )


def _build_result(
    machine_id, oee, availability, performance, quality,
    downtime_mins, idle_mins, total_mins,
    actual_parts, ideal_parts, data_source
) -> dict:
    """Assemble the final OEE result dict."""

    run_mins     = total_mins - downtime_mins - idle_mins
    throughput   = round(actual_parts / (total_mins / 60), 1) if total_mins > 0 else 0

    oee_rating = (
        "World Class (≥85%)" if oee >= 0.85 else
        "Good (70–84%)"      if oee >= 0.70 else
        "Average (60–69%)"   if oee >= 0.60 else
        "Below Average (<60%) — improvement needed"
    )

    return {
        "machine_id":       machine_id,
        "machine_name":     MACHINES.get(machine_id, machine_id),
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source":      data_source,
        "period_hours":     round(total_mins / 60, 1),
        "oee": {
            "score":        oee,
            "score_pct":    round(oee * 100, 1),
            "rating":       oee_rating,
        },
        "availability": {
            "score":        availability,
            "score_pct":    round(availability * 100, 1),
            "run_hrs":      round(run_mins / 60, 1),
            "downtime_hrs": round(downtime_mins / 60, 1),
            "downtime_pct": round(downtime_mins / total_mins * 100, 1) if total_mins > 0 else 0,
        },
        "performance": {
            "score":        performance,
            "score_pct":    round(performance * 100, 1),
            "ideal_ct_s":   IDEAL_CYCLE_TIMES.get(machine_id),
        },
        "quality": {
            "score":        quality,
            "score_pct":    round(quality * 100, 1),
            "actual_parts": actual_parts,
            "ideal_parts":  ideal_parts,
            "throughput_per_hr": throughput,
        },
        "downtime_breakdown": {
            "fault_mins":       downtime_mins,
            "idle_mins":        idle_mins,
            "fault_pct":        round(downtime_mins / total_mins * 100, 1) if total_mins > 0 else 0,
            "idle_pct":         round(idle_mins / total_mins * 100, 1) if total_mins > 0 else 0,
        },
    }


def get_fleet_metrics() -> list:
    """Compute OEE for all machines and return as a list."""
    results = []
    for machine_id in MACHINES:
        oee = compute_oee(machine_id)
        if "error" not in oee:
            results.append(oee)
    return results


def format_metrics_report(machine_id: str) -> str:
    """Return a human-readable OEE and KPI report."""
    m = compute_oee(machine_id)
    if "error" in m:
        return m["error"]

    lines = [
        f"=== Production Metrics: {m['machine_name']} ({machine_id}) ===",
        f"Period: {m['period_hours']} hours  |  Source: {m['data_source']}",
        "",
        f"  ┌─────────────────────────────────────────────┐",
        f"  │  OEE Score:  {m['oee']['score_pct']:>5.1f}%   {m['oee']['rating']:<28}│",
        f"  └─────────────────────────────────────────────┘",
        "",
        f"  Availability:   {m['availability']['score_pct']:>5.1f}%  (Run: {m['availability']['run_hrs']}h  |  Down: {m['availability']['downtime_hrs']}h)",
        f"  Performance:    {m['performance']['score_pct']:>5.1f}%  (Ideal CT: {m['performance']['ideal_ct_s']}s)",
        f"  Quality:        {m['quality']['score_pct']:>5.1f}%",
        "",
        f"  Throughput:     {m['quality']['throughput_per_hr']} parts/hr",
        f"  Parts Produced: {m['quality']['actual_parts']} / {m['quality']['ideal_parts']} (ideal)",
        f"  Downtime:       {m['availability']['downtime_pct']}%  ({m['downtime_breakdown']['fault_pct']}% fault  +  {m['downtime_breakdown']['idle_pct']}% idle)",
    ]
    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    for mid in ["CNC-M01", "HYD-P02", "CVB-003"]:
        print(format_metrics_report(mid))
        print()
