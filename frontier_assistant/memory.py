"""
Conversation memory for the frontier assistant (Claude Sonnet).
Supports:
  - Full history up to max_pairs turns
  - Automatic summarization of older turns when limit is reached
  - System prompt injection
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Full-history memory with optional summarization.
    When history exceeds max_pairs, older turns are compressed into a summary
    that's injected back into the system prompt.
    """

    def __init__(
        self,
        max_pairs: int = 50,
        system_prompt: str = "",
        summarize_fn=None,          # callable(older_messages) -> str
    ):
        self.max_pairs = max_pairs
        self.base_system_prompt = system_prompt
        self.summarize_fn = summarize_fn
        self._history: List[Turn] = []
        self._summary: Optional[str] = None   # summary of compressed turns

    def add_user(self, content: str) -> None:
        self._history.append(Turn(role="user", content=content))
        self._maybe_compress()

    def add_assistant(self, content: str) -> None:
        self._history.append(Turn(role="assistant", content=content))

    def _maybe_compress(self) -> None:
        max_entries = self.max_pairs * 2
        if len(self._history) <= max_entries:
            return
        # Keep the most recent half; summarize the rest
        split = len(self._history) // 2
        older  = self._history[:split]
        recent = self._history[split:]
        if self.summarize_fn:
            try:
                self._summary = self.summarize_fn(
                    [{"role": t.role, "content": t.content} for t in older]
                )
            except Exception:
                self._summary = f"[Earlier conversation ({len(older)} messages) was summarized]"
        else:
            self._summary = f"[Earlier conversation ({len(older)} messages) was truncated]"
        self._history = recent

    def to_messages(self) -> List[Dict[str, str]]:
        """Build the messages list for the Anthropic API."""
        system = self.base_system_prompt
        if self._summary:
            system += f"\n\n[Summary of earlier conversation]\n{self._summary}"

        messages = []
        for turn in self._history:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def get_system(self) -> str:
        system = self.base_system_prompt
        if self._summary:
            system += f"\n\n[Summary of earlier conversation]\n{self._summary}"
        return system

    def clear(self) -> None:
        self._history = []
        self._summary = None

    def __len__(self) -> int:
        return len(self._history)
