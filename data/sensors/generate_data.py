"""
generate_data.py
Generates:
  1. mock_sensor_data.json  — current snapshot of all 5 machines
  2. machine_logs/*.csv     — 7-day time-series logs per machine with anomalies
"""

import json
import random
import csv
import os
from datetime import datetime, timedelta

random.seed(42)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(OUTPUT_DIR, "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. MACHINE SENSOR PROFILES
# ─────────────────────────────────────────────
MACHINES = {
    "CNC-M01": {
        "name": "CNC Milling Machine",
        "location": "Machining Cell A",
        "sensors": {
            "temperature_c":      {"normal": (45, 65),  "warning": (65, 80),  "alarm": (80, 95)},
            "vibration_mm_s":     {"normal": (0.5, 2.5), "warning": (2.5, 4.0), "alarm": (4.0, 6.0)},
            "spindle_rpm":        {"normal": (2000, 6000), "warning": (6000, 8200), "alarm": (8200, 9000)},
            "coolant_flow_l_min": {"normal": (8, 15),   "warning": (4, 8),    "alarm": (0, 4)},
            "power_kw":           {"normal": (5, 15),   "warning": (15, 20),  "alarm": (20, 25)},
        },
        "error_codes": ["E01","E02","E03","E04","E05","E06","E07","E08","E09","E10"]
    },
    "HYD-P02": {
        "name": "Hydraulic Pump Unit",
        "location": "Press & Forming Cell",
        "sensors": {
            "pressure_bar":       {"normal": (100, 250), "warning": (250, 270), "alarm": (270, 300)},
            "temperature_c":      {"normal": (35, 60),  "warning": (60, 75),  "alarm": (75, 90)},
            "flow_l_min":         {"normal": (20, 80),  "warning": (10, 20),  "alarm": (0, 10)},
            "motor_current_a":    {"normal": (10, 40),  "warning": (40, 50),  "alarm": (50, 60)},
            "vibration_mm_s":     {"normal": (0.5, 3.0), "warning": (3.0, 5.0), "alarm": (5.0, 8.0)},
        },
        "error_codes": ["E11","E12","E13","E14","E15","E16","E17","E18","E19","E20"]
    },
    "CVB-003": {
        "name": "Conveyor Belt System",
        "location": "Material Handling & Assembly",
        "sensors": {
            "belt_speed_m_s":     {"normal": (0.5, 1.5), "warning": (1.5, 1.8), "alarm": (1.8, 2.2)},
            "motor_temp_c":       {"normal": (30, 70),  "warning": (70, 85),  "alarm": (85, 100)},
            "motor_current_a":    {"normal": (5, 25),   "warning": (25, 32),  "alarm": (32, 40)},
            "belt_tension_n":     {"normal": (800, 1200), "warning": (500, 800), "alarm": (200, 500)},
            "vibration_mm_s":     {"normal": (0.3, 2.0), "warning": (2.0, 3.5), "alarm": (3.5, 6.0)},
        },
        "error_codes": ["E21","E22","E23","E24","E25","E26","E27","E28","E29","E30"]
    },
    "BLR-004": {
        "name": "Industrial Steam Boiler",
        "location": "Utilities & Steam Generation",
        "sensors": {
            "steam_pressure_bar": {"normal": (7, 10),   "warning": (10, 11),  "alarm": (11, 13)},
            "water_level_pct":    {"normal": (40, 80),  "warning": (20, 40),  "alarm": (0, 20)},
            "flue_gas_temp_c":    {"normal": (180, 240), "warning": (240, 280), "alarm": (280, 320)},
            "fuel_pressure_mbar": {"normal": (100, 200), "warning": (70, 100), "alarm": (40, 70)},
            "feedwater_temp_c":   {"normal": (80, 105), "warning": (65, 80),  "alarm": (50, 65)},
        },
        "error_codes": ["E31","E32","E33","E34","E35","E36","E37","E38","E39","E40"]
    },
    "ROB-005": {
        "name": "6-Axis Robotic Arm",
        "location": "Welding & Assembly Automation",
        "sensors": {
            "joint_temp_c":       {"normal": (25, 55),  "warning": (55, 70),  "alarm": (70, 85)},
            "motor_current_a":    {"normal": (2, 18),   "warning": (18, 23),  "alarm": (23, 28)},
            "tcp_speed_mm_s":     {"normal": (100, 1800), "warning": (1800, 2100), "alarm": (2100, 2500)},
            "air_pressure_bar":   {"normal": (5.0, 6.0), "warning": (4.0, 5.0), "alarm": (3.0, 4.0)},
            "vibration_mm_s":     {"normal": (0.1, 1.5), "warning": (1.5, 2.5), "alarm": (2.5, 4.0)},
        },
        "error_codes": ["E41","E42","E43","E44","E45","E46","E47","E48","E49","E50"]
    }
}

OPERATORS = ["Rajesh Kumar", "Priya Sharma", "Anil Verma", "Sunita Patel", "Mohan Das"]
SHIFTS = ["Morning (6AM-2PM)", "Afternoon (2PM-10PM)", "Night (10PM-6AM)"]
STATUSES = ["Running", "Idle", "Maintenance", "Fault"]

def rand_in_range(r):
    return round(random.uniform(r[0], r[1]), 2)

def pick_zone(sensor_profile, fault_probability=0.1):
    """Pick a sensor value — mostly normal, occasionally warning/alarm."""
    roll = random.random()
    if roll < fault_probability:
        return rand_in_range(sensor_profile["alarm"]), "ALARM"
    elif roll < fault_probability * 3:
        return rand_in_range(sensor_profile["warning"]), "WARNING"
    else:
        return rand_in_range(sensor_profile["normal"]), "NORMAL"

# ─────────────────────────────────────────────
# 2. GENERATE CURRENT SNAPSHOT JSON
# ─────────────────────────────────────────────
snapshot = {}

for machine_id, config in MACHINES.items():
    readings = {}
    statuses_per_sensor = {}
    has_fault = False

    for sensor_name, profile in config["sensors"].items():
        value, status = pick_zone(profile, fault_probability=0.08)
        readings[sensor_name] = value
        statuses_per_sensor[sensor_name] = status
        if status == "ALARM":
            has_fault = True

    active_errors = []
    if has_fault:
        active_errors = random.sample(config["error_codes"], k=random.randint(1, 2))

    machine_status = "Fault" if has_fault else random.choice(["Running", "Running", "Running", "Idle"])

    snapshot[machine_id] = {
        "machine_id": machine_id,
        "machine_name": config["name"],
        "location": config["location"],
        "status": machine_status,
        "operator": random.choice(OPERATORS),
        "shift": random.choice(SHIFTS),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operating_hours_total": random.randint(2000, 15000),
        "hours_since_last_pm": random.randint(10, 720),
        "sensor_readings": readings,
        "sensor_statuses": statuses_per_sensor,
        "active_error_codes": active_errors,
        "last_maintenance_date": (datetime.now() - timedelta(days=random.randint(10, 90))).strftime("%Y-%m-%d"),
    }

snap_path = os.path.join(OUTPUT_DIR, "mock_sensor_data.json")
with open(snap_path, "w") as f:
    json.dump(snapshot, f, indent=2)
print(f"✓ Snapshot saved: {snap_path}")

# ─────────────────────────────────────────────
# 3. GENERATE 7-DAY TIME-SERIES LOG CSVs
# ─────────────────────────────────────────────
START_TIME = datetime.now() - timedelta(days=7)
INTERVAL_MINUTES = 10   # one reading every 10 minutes
TOTAL_POINTS = 7 * 24 * 6   # 1008 rows per machine

for machine_id, config in MACHINES.items():
    sensor_names = list(config["sensors"].keys())
    fieldnames = ["timestamp", "machine_id", "status"] + sensor_names + ["error_code", "cycle_time_s", "oee_pct"]

    log_path = os.path.join(LOGS_DIR, f"{machine_id}_7day_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(TOTAL_POINTS):
            ts = START_TIME + timedelta(minutes=i * INTERVAL_MINUTES)

            # Simulate a fault window around hour 72-80 (day 3)
            in_fault_window = (72 * 6) <= i <= (80 * 6)
            fault_prob = 0.6 if in_fault_window else 0.04

            row = {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "machine_id": machine_id,
            }

            any_alarm = False
            for sensor_name, profile in config["sensors"].items():
                value, status = pick_zone(profile, fault_probability=fault_prob)
                row[sensor_name] = value
                if status == "ALARM":
                    any_alarm = True

            if any_alarm and random.random() < 0.7:
                row["error_code"] = random.choice(config["error_codes"])
                row["status"] = "Fault"
            else:
                row["error_code"] = ""
                row["status"] = "Running" if random.random() > 0.05 else "Idle"

            row["cycle_time_s"] = round(random.uniform(28, 45) if row["status"] == "Running" else 0, 1)
            row["oee_pct"] = round(random.uniform(60, 95) if row["status"] == "Running" else 0, 1)

            writer.writerow(row)

    print(f"✓ Log saved: {log_path}  ({TOTAL_POINTS} rows)")

print("\n✅ Phase 1 data generation complete.")
print(f"   Snapshot:  data/sensors/mock_sensor_data.json")
print(f"   Log CSVs:  data/logs/<machine_id>_7day_log.csv")
