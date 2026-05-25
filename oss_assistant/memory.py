"""
Short-term conversational memory for the OSS assistant.
Uses a sliding window of the last N turns to stay within the model's context limit.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import time


@dataclass
class Turn:
    role: str       # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Sliding-window memory: keeps the last `max_turns` full turns.
    A "turn" = one user message + one assistant response = 2 entries.
    """

    def __init__(self, max_turns: int = 10, system_prompt: str = ""):
        self.max_turns = max_turns          # max user+assistant pairs
        self.system_prompt = system_prompt
        self._history: List[Turn] = []

    def add_user(self, content: str) -> None:
        self._history.append(Turn(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._history.append(Turn(role="assistant", content=content))

    def _trim(self) -> None:
        """Keep only the last max_turns * 2 entries (pairs)."""
        max_entries = self.max_turns * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

    def to_messages(self) -> List[Dict[str, str]]:
        """
        Return the message list in the format expected by the chat template.
        Includes a system message if one is set.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for turn in self._history:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def clear(self) -> None:
        self._history = []

    def __len__(self) -> int:
        return len(self._history)

    def summary(self) -> str:
        return f"Memory: {len(self._history)} messages ({len(self._history)//2} turns)"
