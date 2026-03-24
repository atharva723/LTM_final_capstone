# ─────────────────────────────────────────────────────
# backend/memory/entity_memory.py
#
# PURPOSE:
#   Tracks key entities mentioned during a conversation:
#     • Machine currently being discussed
#     • Fault codes that came up
#     • Sensor readings observed this session
#     • Operator name and shift
#     • Troubleshooting steps already attempted
#
#   This lets the agent say things like:
#   "Based on the E01 fault we discussed earlier on CNC-M01..."
#   instead of treating every message as a fresh start.
# ─────────────────────────────────────────────────────

import re
from datetime import datetime
from typing import Optional


class EntityMemory:
    """
    Lightweight in-memory entity store for one conversation session.
    Not backed by a database — resets on server restart.
    For persistent history, swap this for a Redis or SQLite store.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all entities — call at start of new session."""
        self.machine_id:    Optional[str]  = None
        self.machine_name:  Optional[str]  = None
        self.operator_name: Optional[str]  = None
        self.shift:         Optional[str]  = None
        self.fault_codes:   list           = []     # list of error codes seen
        self.sensor_readings: dict         = {}     # latest sensor snapshot
        self.attempted_steps: list         = []     # troubleshooting steps done
        self.active_issues:   list         = []     # current open faults
        self.session_start:   str          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.message_count:   int          = 0
        self.last_updated:    str          = self.session_start

    # ── Machine tracking ──────────────────────────────
    def set_machine(self, machine_id: str, machine_name: str = ""):
        self.machine_id   = machine_id.upper()
        self.machine_name = machine_name
        self._touch()

    def get_machine(self) -> Optional[str]:
        return self.machine_id

    # ── Fault code tracking ───────────────────────────
    def add_fault_code(self, code: str):
        code = code.upper()
        if code not in self.fault_codes:
            self.fault_codes.append(code)
        self._touch()

    def get_fault_codes(self) -> list:
        return self.fault_codes

    # ── Operator tracking ─────────────────────────────
    def set_operator(self, name: str, shift: str = ""):
        self.operator_name = name
        self.shift         = shift
        self._touch()

    # ── Sensor snapshot ───────────────────────────────
    def update_sensor_readings(self, readings: dict):
        self.sensor_readings.update(readings)
        self._touch()

    # ── Troubleshooting step tracking ─────────────────
    def add_attempted_step(self, step: str):
        if step not in self.attempted_steps:
            self.attempted_steps.append(step)
        self._touch()

    def get_attempted_steps(self) -> list:
        return self.attempted_steps

    # ── Active issue tracking ─────────────────────────
    def add_issue(self, issue: str):
        if issue not in self.active_issues:
            self.active_issues.append(issue)
        self._touch()

    def resolve_issue(self, issue: str):
        self.active_issues = [i for i in self.active_issues if i != issue]
        self._touch()

    # ── Auto-extract entities from user message ────────
    def extract_from_message(self, message: str):
        """
        Parse a user message and auto-update entity store.
        Looks for machine IDs and error codes in text.
        """
        msg_upper = message.upper()

        # Detect machine IDs
        machine_patterns = {
            "CNC-M01": ["CNC", "CNC-M01", "MILLING"],
            "HYD-P02": ["HYDRAULIC", "HYD", "HYD-P02", "PUMP"],
            "CVB-003": ["CONVEYOR", "CVB", "BELT", "CVB-003"],
            "BLR-004": ["BOILER", "BLR", "STEAM", "BLR-004"],
            "ROB-005": ["ROBOT", "ROB", "ARM", "ROB-005"],
        }
        for machine_id, keywords in machine_patterns.items():
            if any(kw in msg_upper for kw in keywords):
                if not self.machine_id:   # only set if not already set
                    self.set_machine(machine_id)
                break

        # Detect error codes (pattern: E followed by 1–2 digits)
        error_codes = re.findall(r'\bE\d{1,2}\b', msg_upper)
        for code in error_codes:
            self.add_fault_code(code)

        self.message_count += 1
        self._touch()

    def _touch(self):
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_context_string(self) -> str:
        """
        Serialize entity state to a compact string injected
        into the LLM system prompt for context continuity.
        """
        parts = []
        if self.machine_id:
            parts.append(f"Current machine: {self.machine_id} ({self.machine_name or ''})")
        if self.operator_name:
            parts.append(f"Operator: {self.operator_name} | Shift: {self.shift or 'unknown'}")
        if self.fault_codes:
            parts.append(f"Fault codes discussed: {', '.join(self.fault_codes)}")
        if self.sensor_readings:
            sensor_str = ", ".join(f"{k}={v}" for k, v in list(self.sensor_readings.items())[:4])
            parts.append(f"Recent sensor readings: {sensor_str}")
        if self.attempted_steps:
            parts.append(f"Troubleshooting already attempted: {'; '.join(self.attempted_steps[-3:])}")
        if self.active_issues:
            parts.append(f"Open issues: {'; '.join(self.active_issues)}")
        if not parts:
            return "No prior context in this session."
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "machine_id":       self.machine_id,
            "machine_name":     self.machine_name,
            "operator_name":    self.operator_name,
            "shift":            self.shift,
            "fault_codes":      self.fault_codes,
            "sensor_readings":  self.sensor_readings,
            "attempted_steps":  self.attempted_steps,
            "active_issues":    self.active_issues,
            "message_count":    self.message_count,
            "session_start":    self.session_start,
            "last_updated":     self.last_updated,
        }
