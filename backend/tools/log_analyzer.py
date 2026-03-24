# ─────────────────────────────────────────────────────
# backend/tools/log_analyzer.py
#
# TOOL 5 — Log File Analyzer Tool
#
# PURPOSE:
#   Reads 7-day machine log CSVs (or uploaded CSVs) and:
#     1. Detects anomalies using threshold-based detection
#     2. Extracts fault patterns and error code frequency
#     3. Computes OEE, downtime %, and cycle time stats
#     4. Summarizes key findings for the operator/LLM
#
# USED BY: agent.py, FastAPI /analyze_log endpoint,
#          Streamlit file upload widget
# ─────────────────────────────────────────────────────

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import io

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LOGS_DIR, SENSOR_THRESHOLDS, MACHINES

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def _load_log(machine_id: str) -> Optional["pd.DataFrame"]:
    """Load the 7-day log CSV for a machine."""
    if not PANDAS_AVAILABLE:
        return None

    log_path = Path(LOGS_DIR) / f"{machine_id}_7day_log.csv"
    if not log_path.exists():
        return None

    df = pd.read_csv(log_path, parse_dates=["timestamp"])
    return df


def load_uploaded_log(file_content: Union[str, bytes]) -> Optional["pd.DataFrame"]:
    """
    Parse an uploaded CSV log from Streamlit file uploader.

    Args:
        file_content: Raw bytes or string content of uploaded CSV

    Returns:
        Parsed DataFrame or None if parsing fails
    """
    if not PANDAS_AVAILABLE:
        return None
    try:
        if isinstance(file_content, bytes):
            file_content = file_content.decode("utf-8")
        df = pd.read_csv(io.StringIO(file_content))
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        print(f"[LogAnalyzer] Failed to parse uploaded file: {e}")
        return None


def detect_anomalies(df: "pd.DataFrame", machine_id: str) -> list:
    """
    Threshold-based anomaly detection.
    Scans each sensor column against alarm thresholds.

    Returns list of anomaly events with timestamps.
    """
    if df is None or df.empty:
        return []

    thresholds = SENSOR_THRESHOLDS.get(machine_id.upper(), {})
    if not thresholds:
        return []

    low_bad = {
        "coolant_flow_l_min", "belt_tension_n", "water_level_pct",
        "flow_l_min", "fuel_pressure_mbar", "feedwater_temp_c", "air_pressure_bar"
    }

    anomalies = []
    for sensor, thresh in thresholds.items():
        if sensor not in df.columns:
            continue

        alarm_val   = thresh["alarm"]
        warning_val = thresh["warning"]
        is_low_bad  = sensor in low_bad

        for _, row in df.iterrows():
            val = row[sensor]
            if pd.isna(val):
                continue

            level = None
            if is_low_bad:
                if val <= alarm_val:    level = "ALARM"
                elif val <= warning_val: level = "WARNING"
            else:
                if val >= alarm_val:    level = "ALARM"
                elif val >= warning_val: level = "WARNING"

            if level:
                anomalies.append({
                    "timestamp": str(row.get("timestamp", "unknown")),
                    "sensor":    sensor,
                    "value":     round(float(val), 2),
                    "level":     level,
                    "threshold": alarm_val if level == "ALARM" else warning_val,
                })

    # Sort by timestamp
    anomalies.sort(key=lambda x: x["timestamp"])
    return anomalies


def compute_oee_stats(df: "pd.DataFrame") -> dict:
    """
    Compute OEE and related production KPIs from log data.
    OEE = Availability × Performance × Quality

    Simplified calculation from log data:
    - Availability = Running time / Total time
    - Performance  = Avg OEE from log
    - Downtime     = Time in Fault or Idle status
    """
    if df is None or df.empty:
        return {}

    total_rows = len(df)
    if total_rows == 0:
        return {}

    # Status distribution
    status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    running       = status_counts.get("Running", 0)
    faulted       = status_counts.get("Fault", 0)
    idle          = status_counts.get("Idle", 0)

    availability  = round(running / total_rows * 100, 1) if total_rows > 0 else 0
    downtime_pct  = round(faulted / total_rows * 100, 1) if total_rows > 0 else 0
    idle_pct      = round(idle / total_rows * 100, 1) if total_rows > 0 else 0

    # OEE from log
    avg_oee = round(df["oee_pct"].mean(), 1) if "oee_pct" in df.columns else 0
    min_oee = round(df["oee_pct"].min(), 1) if "oee_pct" in df.columns else 0

    # Cycle time stats
    ct_col = df["cycle_time_s"] if "cycle_time_s" in df.columns else None
    ct_running = ct_col[df["status"] == "Running"] if ct_col is not None else None

    cycle_time_avg = round(float(ct_running.mean()), 1) if ct_running is not None and len(ct_running) > 0 else 0
    cycle_time_std = round(float(ct_running.std()), 2) if ct_running is not None and len(ct_running) > 0 else 0

    return {
        "total_readings":   total_rows,
        "availability_pct": availability,
        "downtime_pct":     downtime_pct,
        "idle_pct":         idle_pct,
        "avg_oee_pct":      avg_oee,
        "min_oee_pct":      min_oee,
        "cycle_time_avg_s": cycle_time_avg,
        "cycle_time_std_s": cycle_time_std,
        "status_breakdown": status_counts,
    }


def extract_fault_patterns(df: "pd.DataFrame") -> dict:
    """Extract error code frequency and time-clustering of faults."""
    if df is None or df.empty or "error_code" not in df.columns:
        return {}

    error_rows = df[df["error_code"].notna() & (df["error_code"] != "")]
    if error_rows.empty:
        return {"total_faults": 0, "top_errors": []}

    # Frequency analysis
    freq = error_rows["error_code"].value_counts().head(10)
    top_errors = [{"code": k, "count": int(v)} for k, v in freq.items()]

    # Find fault clusters — consecutive faults within 1 hour
    fault_times  = pd.to_datetime(error_rows["timestamp"], errors="coerce").dropna()
    clusters     = 0
    if len(fault_times) > 1:
        time_diffs  = fault_times.diff().dt.total_seconds().fillna(9999)
        clusters    = int((time_diffs < 3600).sum())

    return {
        "total_faults":         len(error_rows),
        "unique_error_codes":   int(error_rows["error_code"].nunique()),
        "top_errors":           top_errors,
        "fault_clusters":       clusters,
        "fault_rate_per_hour":  round(len(error_rows) / max(1, len(df) / 6), 2),
    }


def analyze_log(
    machine_id: str,
    file_content: Optional[Union[str, bytes]] = None,
) -> dict:
    """
    Full log analysis pipeline.

    Args:
        machine_id:   e.g. "CNC-M01"
        file_content: Optional uploaded CSV bytes (if None, uses stored log)

    Returns:
        Complete analysis dict with anomalies, OEE, fault patterns, summary
    """
    machine_id = machine_id.upper().strip()

    if not PANDAS_AVAILABLE:
        return {"error": "pandas not installed. Run: pip install pandas numpy"}

    # Load data
    if file_content:
        df = load_uploaded_log(file_content)
        source = "uploaded_file"
    else:
        df = _load_log(machine_id)
        source = "stored_7day_log"

    if df is None or df.empty:
        return {"error": f"No log data found for {machine_id}"}

    # Run all analyses
    anomalies     = detect_anomalies(df, machine_id)
    oee_stats     = compute_oee_stats(df)
    fault_patterns = extract_fault_patterns(df)

    # Count anomaly levels
    alarm_count   = sum(1 for a in anomalies if a["level"] == "ALARM")
    warning_count = sum(1 for a in anomalies if a["level"] == "WARNING")

    # Most problematic sensor
    if anomalies:
        from collections import Counter
        sensor_freq = Counter(a["sensor"] for a in anomalies if a["level"] == "ALARM")
        top_sensor  = sensor_freq.most_common(1)[0][0] if sensor_freq else None
    else:
        top_sensor = None

    # Date range
    if "timestamp" in df.columns:
        valid_ts   = df["timestamp"].dropna()
        date_start = str(valid_ts.min())[:10] if len(valid_ts) > 0 else "unknown"
        date_end   = str(valid_ts.max())[:10] if len(valid_ts) > 0 else "unknown"
    else:
        date_start = date_end = "unknown"

    return {
        "machine_id":     machine_id,
        "machine_name":   MACHINES.get(machine_id, machine_id),
        "data_source":    source,
        "date_range":     {"start": date_start, "end": date_end},
        "total_rows":     len(df),
        "anomaly_summary": {
            "total":   len(anomalies),
            "alarms":  alarm_count,
            "warnings": warning_count,
            "top_problem_sensor": top_sensor,
            "events":  anomalies[:20],   # cap at 20 for response size
        },
        "oee_stats":      oee_stats,
        "fault_patterns": fault_patterns,
        "health_score":   _compute_health_score(oee_stats, alarm_count, fault_patterns),
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _compute_health_score(oee_stats: dict, alarm_count: int, fault_patterns: dict) -> int:
    """
    Compute a 0–100 machine health score.
    Higher = healthier. Used for dashboard indicator.
    """
    score = 100
    score -= min(30, oee_stats.get("downtime_pct", 0) * 2)
    score -= min(20, alarm_count * 0.5)
    score -= min(20, fault_patterns.get("total_faults", 0) * 0.3)
    if oee_stats.get("avg_oee_pct", 100) < 70:
        score -= 10
    return max(0, int(score))


def format_log_summary(machine_id: str) -> str:
    """Return a concise, human-readable log analysis summary."""
    result = analyze_log(machine_id)
    if "error" in result:
        return f"Log analysis error: {result['error']}"

    oee  = result["oee_stats"]
    flt  = result["fault_patterns"]
    anom = result["anomaly_summary"]

    lines = [
        f"=== 7-Day Log Analysis: {result['machine_name']} ({machine_id}) ===",
        f"Period: {result['date_range']['start']} to {result['date_range']['end']}  |  Rows: {result['total_rows']}",
        f"Health Score: {result['health_score']}/100",
        "",
        "── Production KPIs ──────────────────────────────",
        f"  Availability:  {oee.get('availability_pct', 0):.1f}%",
        f"  Avg OEE:       {oee.get('avg_oee_pct', 0):.1f}%  (min: {oee.get('min_oee_pct', 0):.1f}%)",
        f"  Downtime:      {oee.get('downtime_pct', 0):.1f}%",
        f"  Avg Cycle Time:{oee.get('cycle_time_avg_s', 0):.1f}s  (σ={oee.get('cycle_time_std_s', 0):.2f}s)",
        "",
        "── Anomaly Detection ────────────────────────────",
        f"  Total events:  {anom['total']}  (Alarms: {anom['alarms']}, Warnings: {anom['warnings']})",
        f"  Most affected: {anom.get('top_problem_sensor', 'None')}",
        "",
        "── Fault Patterns ───────────────────────────────",
        f"  Total faults:  {flt.get('total_faults', 0)}",
        f"  Unique codes:  {flt.get('unique_error_codes', 0)}",
        f"  Fault clusters:{flt.get('fault_clusters', 0)}  (faults within 1hr window)",
    ]

    if flt.get("top_errors"):
        lines.append("  Top errors:")
        for e in flt["top_errors"][:3]:
            lines.append(f"    {e['code']}: {e['count']} occurrences")

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    print(format_log_summary("CNC-M01"))
    print()
    print(format_log_summary("CVB-003"))
