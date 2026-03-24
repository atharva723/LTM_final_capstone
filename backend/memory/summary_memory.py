# ─────────────────────────────────────────────────────
# backend/memory/summary_memory.py
#
# PURPOSE:
#   Manages the full conversation history and compresses
#   it into a rolling summary when it gets too long.
#
#   Why needed:
#   Mistral has a context window limit. A long diagnostic
#   session (10+ messages) would overflow it. This memory
#   keeps a recent window + a compressed summary of older
#   messages so the agent always has context.
#
#   Strategy:
#   - Keep last N messages verbatim (RECENT_WINDOW = 6)
#   - Summarize anything older using the LLM itself
#   - Inject summary + recent messages into every prompt
# ─────────────────────────────────────────────────────

from datetime import datetime
from typing import Optional


RECENT_WINDOW = 6       # number of recent messages kept verbatim
SUMMARY_TRIGGER = 10    # summarize when total messages exceed this


class ConversationMessage:
    def __init__(self, role: str, content: str):
        self.role      = role       # "user" | "assistant" | "system"
        self.content   = content
        self.timestamp = datetime.now().strftime("%H:%M:%S")

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    def __repr__(self):
        return f"[{self.timestamp}] {self.role.upper()}: {self.content[:80]}..."


class SummaryMemory:
    """
    Manages conversation history with auto-summarization.

    Maintains:
      self.messages     → recent messages (last RECENT_WINDOW)
      self.summary      → compressed summary of older messages
      self.full_history → complete message log (for evaluation)
    """

    def __init__(self, llm=None):
        self.messages:     list  = []    # recent messages
        self.full_history: list  = []    # all messages ever
        self.summary:      str   = ""    # rolling summary of older context
        self.llm                 = llm   # Ollama LLM for summarization
        self.turn_count:   int   = 0

    def add_message(self, role: str, content: str):
        """Add a message to history and trigger summarization if needed."""
        msg = ConversationMessage(role, content)
        self.messages.append(msg)
        self.full_history.append(msg)
        self.turn_count += 1

        # Trigger summarization when history exceeds threshold
        if len(self.messages) > SUMMARY_TRIGGER and self.llm:
            self._compress()

    def add_user_message(self, content: str):
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        self.add_message("assistant", content)

    def _compress(self):
        """
        Summarize oldest messages and keep only recent window verbatim.
        Uses the LLM to create a concise summary of what was discussed.
        """
        # Split: oldest to summarize | recent to keep
        to_summarize = self.messages[:-RECENT_WINDOW]
        self.messages = self.messages[-RECENT_WINDOW:]

        if not to_summarize:
            return

        # Build conversation text for summarization
        convo_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in to_summarize
        )

        summary_prompt = f"""Summarize this factory floor troubleshooting conversation in 3–5 bullet points.
Focus on: which machine, what faults were found, what steps were taken, what was resolved.
Be concise — this is used as context for the AI assistant.

Previous summary: {self.summary or 'None'}

Conversation to summarize:
{convo_text}

New summary:"""

        try:
            response = self.llm.invoke(summary_prompt)
            new_summary = response.content if hasattr(response, "content") else str(response)
            # Prepend existing summary if there is one
            if self.summary:
                self.summary = f"{self.summary}\n\nAdditional context:\n{new_summary}"
            else:
                self.summary = new_summary
        except Exception as e:
            # Fallback: simple text concatenation if LLM fails
            self.summary = (self.summary + "\n" + convo_text[:500]) if self.summary else convo_text[:500]

    def get_context_for_prompt(self) -> str:
        """
        Returns memory context string to inject into the LLM prompt.
        Includes summary of older messages + recent verbatim messages.
        """
        parts = []

        if self.summary:
            parts.append(f"[CONVERSATION SUMMARY]\n{self.summary}")

        if self.messages:
            recent_lines = []
            for m in self.messages[-RECENT_WINDOW:]:
                if m.role != "system":
                    recent_lines.append(f"{m.role.upper()} [{m.timestamp}]: {m.content}")
            if recent_lines:
                parts.append("[RECENT MESSAGES]\n" + "\n".join(recent_lines))

        return "\n\n".join(parts) if parts else ""

    def get_messages_for_llm(self) -> list:
        """
        Returns messages in the format expected by ChatOllama:
        [{"role": "user"|"assistant", "content": "..."}]
        Includes summary as a system message if present.
        """
        messages = []

        if self.summary:
            messages.append({
                "role":    "system",
                "content": f"Previous conversation summary:\n{self.summary}",
            })

        for m in self.messages[-RECENT_WINDOW:]:
            if m.role in ("user", "assistant"):
                messages.append({"role": m.role, "content": m.content})

        return messages

    def clear(self):
        """Reset memory for a new conversation session."""
        self.messages     = []
        self.full_history = []
        self.summary      = ""
        self.turn_count   = 0

    def get_full_history_text(self) -> str:
        """Return complete conversation log — used for evaluation."""
        return "\n".join(
            f"[{m.timestamp}] {m.role.upper()}: {m.content}"
            for m in self.full_history
        )

    def __len__(self):
        return len(self.full_history)
