# ─────────────────────────────────────────────────────
# backend/agent.py
#
# PURPOSE:
#   The brain of the system. Orchestrates:
#     • Mistral LLM (via Ollama)
#     • RAG retrieval (FAISS vector store)
#     • All 8 tools
#     • Entity + Summary memory
#
#   Flow for every user message:
#     1. Extract entities (machine, error code) from message
#     2. Classify intent (sensor / fault / safety / parts /
#        maintenance / metrics / general / rag_only)
#     3. Run relevant tool(s)
#     4. Retrieve relevant RAG context
#     5. Build enriched prompt with tool output + RAG + memory
#     6. Call Mistral LLM → get response
#     7. Update memory with new message + response
#     8. Return response + metadata
# ─────────────────────────────────────────────────────

import sys
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import LLM_MODEL, OLLAMA_BASE_URL, MACHINES
from backend.memory.entity_memory import EntityMemory
from backend.memory.summary_memory import SummaryMemory

# Tools
from backend.tools.sensor_fetch    import get_sensor_data, format_sensor_report
from backend.tools.fault_diagnose  import diagnose_fault
from backend.tools.spare_parts     import get_parts_for_fault, search_parts
from backend.tools.maintenance     import format_pm_report, calculate_pm_due
from backend.tools.log_analyzer    import analyze_log, format_log_summary
from backend.tools.safety_checker  import check_safety, format_safety_report
from backend.tools.metrics         import compute_oee, format_metrics_report
from backend.tools.escalation      import evaluate_and_escalate, auto_escalate_from_diagnosis, get_active_alerts
from guardrails                    import guardrail_middleware

# ─────────────────────────────────────────────────────
# INTENT CLASSIFICATION
# Determines which tools to invoke for a given query
# ─────────────────────────────────────────────────────
INTENT_KEYWORDS = {
    "sensor":      ["sensor", "temperature", "vibration", "rpm", "pressure", "reading",
                    "current reading", "what is the", "status", "live", "now", "monitor"],
    "fault":       ["error", "fault", "alarm", "broken", "failed", "code", "e0", "e1", "e2",
                    "e3", "e4", "e5", "not working", "stopped", "trip", "diagnosis", "diagnose"],
    "safety":      ["safety", "ppe", "gloves", "helmet", "loto", "lockout", "tagout",
                    "hazard", "risk", "protective", "safe to", "precaution", "permit"],
    "parts":       ["spare", "part", "component", "replace", "stock", "availability",
                    "supplier", "order", "buy", "catalog", "price"],
    "maintenance": ["maintenance", "pm", "schedule", "checklist", "overdue", "service",
                    "lubricate", "inspect", "when is", "last pm", "next pm"],
    "metrics":     ["oee", "downtime", "throughput", "production", "efficiency", "kpi",
                    "performance", "availability", "quality rate", "cycle time"],
    "log":         ["log", "csv", "history", "anomaly", "analyze", "trend", "7 day",
                    "past week", "data", "pattern", "report"],
    "escalate":    ["escalate", "supervisor", "notify", "alert", "emergency", "critical",
                    "call", "urgent", "help"],
    "startup":     ["start", "startup", "turn on", "boot", "initialize", "power on"],
    "shutdown":    ["shut", "shutdown", "turn off", "power off", "stop machine"],
    "sop":         ["procedure", "sop", "how to", "steps", "instruction", "manual", "guide"],
}


def classify_intent(message: str) -> list:
    """
    Return list of intents detected in the message.
    A message can have multiple intents (e.g. fault + parts).
    Ordered by priority.
    """
    msg_lower = message.lower()
    detected  = []

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            detected.append(intent)

    # Default to RAG/general if nothing specific detected
    if not detected:
        detected.append("general")

    return detected


# ─────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Manufacturing Plant AI Assistant named MAIA
(Manufacturing AI Assistant) deployed at an industrial plant.

You help plant operators, maintenance engineers, shift supervisors, and quality
control personnel with:
- Machine operation questions (CNC, Hydraulic Pump, Conveyor Belt, Boiler, Robot)
- Real-time sensor monitoring and fault diagnosis
- Step-by-step troubleshooting with safety guidance
- Preventive maintenance schedules and checklists
- Spare parts lookup and procurement advice
- Safety procedures, PPE requirements, LOTO
- Production KPIs (OEE, downtime, throughput)

Response guidelines:
- Be CONCISE and PRACTICAL — operators need fast, actionable answers
- Always mention PPE when discussing maintenance tasks
- Highlight CRITICAL warnings prominently with ⚠
- Give numbered step-by-step instructions for procedures
- Cite which SOP or document your answer is based on
- If a fault is CRITICAL, recommend escalation immediately
- Use simple language — avoid jargon unless technical context is clear"""


# ─────────────────────────────────────────────────────
# MANUFACTURING AGENT
# ─────────────────────────────────────────────────────
class ManufacturingAgent:
    """
    Core agent class. One instance per user session.
    Maintains entity memory and conversation history.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id     = session_id
        self.entity_memory  = EntityMemory()
        self.conversation   = SummaryMemory()
        self.llm            = None
        self.vectorstore    = None
        self._initialized   = False

    def initialize(self):
        """
        Lazy initialization — load LLM and vector store on first use.
        Separated from __init__ so the FastAPI app starts instantly.
        """
        if self._initialized:
            return

        print(f"  [Agent] Initializing session: {self.session_id}")

        # Load LLM
        try:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
            )
            self.conversation.llm = self.llm
            print(f"  [Agent] LLM loaded: {LLM_MODEL}")
        except Exception as e:
            print(f"  [Agent] LLM init failed: {e} — will use tool-only mode")
            self.llm = None

        # Load vector store
        try:
            from backend.rag.vector_store import load_vector_store
            self.vectorstore = load_vector_store()
        except Exception as e:
            print(f"  [Agent] RAG init failed: {e} — will skip RAG")
            self.vectorstore = None

        self._initialized = True

    # ── Main entry point ──────────────────────────────
    def chat(self, user_message: str, operator_name: str = None) -> dict:
        """
        Process a user message and return the agent response.

        Args:
            user_message:  The operator's question or statement
            operator_name: Optional operator name for context

        Returns:
            {
              "response":    str  — final answer to show operator
              "intents":     list — detected intents
              "tools_used":  list — tools that were called
              "sources":     list — RAG source documents
              "machine_id":  str  — machine in context
              "alerts":      list — any new alerts raised
              "metadata":    dict — timing and session info
            }
        """
        self.initialize()

        t_start = datetime.now()

        # ── Step 0: Guardrail check ───────────────────
        # Runs BEFORE any tool call, RAG retrieval, or LLM invocation.
        # Blocks unsafe, abusive, off-topic, or PII-containing messages.
        allowed, block_msg, safe_input = guardrail_middleware(user_message)
        if not allowed:
            return {
                "response":   block_msg,
                "intents":    ["blocked"],
                "tools_used": [],
                "sources":    [],
                "machine_id": None,
                "alerts":     [],
                "metadata": {
                    "session_id":    self.session_id,
                    "turn":          self.entity_memory.message_count,
                    "elapsed_ms":    0,
                    "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "guardrail":     "BLOCKED",
                    "llm_available": self.llm is not None,
                    "rag_available": self.vectorstore is not None,
                },
            }
        # Use sanitized input for all downstream processing
        user_message = safe_input

        # ── Step 1: Extract entities from message ─────
        self.entity_memory.extract_from_message(user_message)
        if operator_name:
            self.entity_memory.set_operator(operator_name)

        machine_id = self.entity_memory.get_machine()
        intents    = classify_intent(user_message)
        tools_used = []
        tool_outputs = {}
        new_alerts   = []

        # ── Step 2: Run relevant tools ────────────────
        try:
            # Sensor data
            if "sensor" in intents and machine_id:
                sensor_data = get_sensor_data(machine_id)
                tool_outputs["sensor"] = format_sensor_report(machine_id)
                tools_used.append("sensor_fetch")
                # Update entity memory with fresh readings
                if "sensor_readings" in sensor_data:
                    readings = {k: v["value"] for k, v in sensor_data["sensor_readings"].items()}
                    self.entity_memory.update_sensor_readings(readings)
                # Auto-escalate if ALARM
                if sensor_data.get("overall_status") == "ALARM":
                    esc = evaluate_and_escalate(
                        machine_id    = machine_id,
                        severity      = "HIGH",
                        message       = f"Sensor alarm detected on {machine_id}",
                        operator_name = operator_name,
                    )
                    new_alerts.append(esc)

            # Fault diagnosis
            if "fault" in intents:
                error_codes = self.entity_memory.get_fault_codes()
                latest_code = error_codes[-1] if error_codes else None
                sensor_vals = self.entity_memory.sensor_readings or None
                diagnosis   = diagnose_fault(machine_id or "CNC-M01", latest_code, sensor_vals)
                tool_outputs["fault"] = self._format_diagnosis(diagnosis)
                tools_used.append("fault_diagnose")
                # Auto-escalate critical faults
                if diagnosis.get("severity") in ("HIGH", "CRITICAL"):
                    esc = auto_escalate_from_diagnosis(diagnosis, operator_name)
                    new_alerts.append(esc)
                    tools_used.append("escalation")
                # Get parts for this fault
                if diagnosis.get("recommended_parts"):
                    tool_outputs["parts"] = get_parts_for_fault(diagnosis["recommended_parts"])
                    tools_used.append("spare_parts")

            # Safety check
            if "safety" in intents and machine_id:
                task_desc = self._extract_task(user_message)
                tool_outputs["safety"] = format_safety_report(machine_id, task_desc)
                tools_used.append("safety_checker")

            # Spare parts (standalone query)
            if "parts" in intents and "fault" not in intents:
                query = self._extract_part_query(user_message)
                parts = search_parts(query, machine_id)
                if parts:
                    from backend.tools.spare_parts import format_parts_report
                    tool_outputs["parts"] = format_parts_report(parts, f"Parts for '{query}'")
                tools_used.append("spare_parts")

            # Maintenance schedule
            if "maintenance" in intents and machine_id:
                sensor_snap = get_sensor_data(machine_id)
                curr_hours  = sensor_snap.get("operating_hours", 5000)
                tool_outputs["maintenance"] = format_pm_report(machine_id, curr_hours)
                tools_used.append("maintenance")

            # Production metrics / OEE
            if "metrics" in intents:
                mid = machine_id or "CNC-M01"
                tool_outputs["metrics"] = format_metrics_report(mid)
                tools_used.append("metrics")

            # Log analysis
            if "log" in intents and machine_id:
                tool_outputs["log"] = format_log_summary(machine_id)
                tools_used.append("log_analyzer")

        except Exception as e:
            tool_outputs["error"] = f"Tool execution error: {str(e)}"

        # ── Step 3: RAG retrieval ─────────────────────
        rag_context = ""
        rag_sources = []
        if self.vectorstore:
            try:
                from backend.rag.vector_store import similarity_search, format_retrieved_context
                docs = similarity_search(user_message, k=3, machine_id=machine_id)
                rag_context = format_retrieved_context(docs)
                rag_sources = [d.metadata.get("filename", "?") for d in docs]
            except Exception as e:
                rag_context = ""

        # ── Step 4: Build enriched prompt ─────────────
        enriched_prompt = self._build_prompt(
            user_message  = user_message,
            tool_outputs  = tool_outputs,
            rag_context   = rag_context,
            entity_context = self.entity_memory.to_context_string(),
        )

        # ── Step 5: Call LLM ──────────────────────────
        response_text = self._call_llm(enriched_prompt)

        # ── Step 6: Update memory ─────────────────────
        self.conversation.add_user_message(user_message)
        self.conversation.add_assistant_message(response_text)

        elapsed_ms = int((datetime.now() - t_start).total_seconds() * 1000)

        return {
            "response":   response_text,
            "intents":    intents,
            "tools_used": list(set(tools_used)),
            "sources":    rag_sources,
            "machine_id": machine_id,
            "alerts":     new_alerts,
            "metadata": {
                "session_id":   self.session_id,
                "turn":         self.entity_memory.message_count,
                "elapsed_ms":   elapsed_ms,
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "llm_available": self.llm is not None,
                "rag_available": self.vectorstore is not None,
            },
        }

    # ── Prompt builder ────────────────────────────────
    def _build_prompt(
        self,
        user_message: str,
        tool_outputs: dict,
        rag_context: str,
        entity_context: str,
    ) -> str:

        sections = [SYSTEM_PROMPT, ""]

        # Session context from entity memory
        if entity_context and entity_context != "No prior context in this session.":
            sections.append(f"── SESSION CONTEXT ──────────────────────────────\n{entity_context}\n")

        # Conversation history
        history = self.conversation.get_context_for_prompt()
        if history:
            sections.append(f"── CONVERSATION HISTORY ─────────────────────────\n{history}\n")

        # Tool outputs
        if tool_outputs:
            sections.append("── LIVE DATA FROM PLANT SYSTEMS ─────────────────")
            for tool_name, output in tool_outputs.items():
                if tool_name != "error":
                    sections.append(f"[{tool_name.upper()}]\n{output}")
            sections.append("")

        # RAG context
        if rag_context:
            sections.append(f"── KNOWLEDGE BASE (SOPs & MANUALS) ──────────────\n{rag_context}\n")

        # Current question
        sections.append(f"── OPERATOR QUESTION ────────────────────────────\n{user_message}")
        sections.append("\n── YOUR RESPONSE (be concise and practical) ─────")

        return "\n".join(sections)

    # ── LLM caller ────────────────────────────────────
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM or return a tool-only fallback."""
        if self.llm is None:
            return self._tool_only_response()

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            # If Ollama is unreachable, return tool data as-is
            return f"[LLM unavailable: {e}]\n\n{self._tool_only_response()}"

    def _tool_only_response(self) -> str:
        """Fallback when LLM is unavailable — return raw tool data."""
        return (
            "Ollama LLM is not reachable. Please ensure Ollama is running:\n"
            "  ollama serve\n"
            "  ollama pull mistral\n\n"
            "Tool data has been collected — start Ollama to get AI-interpreted responses."
        )

    # ── Helpers ───────────────────────────────────────
    def _format_diagnosis(self, diagnosis: dict) -> str:
        if diagnosis.get("status") == "NO_FAULT":
            return "No faults detected — all readings within normal parameters."
        lines = [f"Severity: {diagnosis['severity']} — {diagnosis.get('severity_guidance', '')}"]
        for d in diagnosis.get("diagnoses", []):
            lines.append(f"• {d.get('title', 'Fault')}: {d.get('failure_mode', '')}")
            for step in d.get("steps", [])[:3]:
                lines.append(f"  → {step}")
        if diagnosis.get("escalate"):
            lines.append("⚠ ESCALATION REQUIRED — notify supervisor immediately")
        return "\n".join(lines)

    def _extract_task(self, message: str) -> str:
        """Extract task type from message for safety checker."""
        task_keywords = ["tool change", "maintenance", "cleaning", "hose", "blowdown",
                         "electrical", "welding", "programming", "belt", "roller"]
        msg_lower = message.lower()
        for kw in task_keywords:
            if kw in msg_lower:
                return kw
        return "maintenance"

    def _extract_part_query(self, message: str) -> str:
        """Extract part search term from message."""
        stop_words = {"the", "a", "an", "for", "of", "in", "on", "spare", "part", "need", "want", "get"}
        words = [w for w in message.lower().split() if w not in stop_words and len(w) > 2]
        return " ".join(words[:3]) if words else "filter"

    def reset_session(self):
        """Start a fresh session — clears memory."""
        self.entity_memory.reset()
        self.conversation.clear()

    def get_session_state(self) -> dict:
        """Return current session state for debugging/dashboard."""
        return {
            "session_id": self.session_id,
            "entity_memory": self.entity_memory.to_dict(),
            "turn_count": self.entity_memory.message_count,
            "summary": self.conversation.summary,
            "active_alerts": len(get_active_alerts()),
        }


# ─────────────────────────────────────────────────────
# SESSION REGISTRY
# One agent per session — managed by FastAPI
# ─────────────────────────────────────────────────────
_sessions: dict = {}

def get_agent(session_id: str = "default") -> ManufacturingAgent:
    """Get or create an agent for a session."""
    if session_id not in _sessions:
        _sessions[session_id] = ManufacturingAgent(session_id)
    return _sessions[session_id]

def clear_session(session_id: str):
    """Remove a session."""
    if session_id in _sessions:
        del _sessions[session_id]


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    agent = get_agent("test_session")
    agent.initialize()

    test_questions = [
        ("What is the current sensor status of CNC-M01?", "Rajesh Kumar"),
        ("Error E01 on the CNC machine — what do I do?",  "Rajesh Kumar"),
        ("What PPE do I need for tool change on CNC?",    "Rajesh Kumar"),
        ("Is maintenance overdue on the conveyor belt?",  "Priya Sharma"),
    ]

    for q, operator in test_questions:
        print(f"\n{'='*60}")
        print(f"Q ({operator}): {q}")
        result = agent.chat(q, operator_name=operator)
        print(f"Intents: {result['intents']}")
        print(f"Tools:   {result['tools_used']}")
        print(f"Time:    {result['metadata']['elapsed_ms']}ms")
        print(f"\nA: {result['response'][:400]}...")