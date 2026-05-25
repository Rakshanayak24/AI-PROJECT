"""
Lightweight guardrails for the OSS assistant.
Since small OSS models lack built-in Constitutional AI, we add:
  1. Input filter  — block clearly harmful prompts before inference
  2. Output filter — scan responses for unsafe content
"""

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Blocked input patterns (case-insensitive)
# ---------------------------------------------------------------------------
BLOCKED_INPUT_PATTERNS = [
    # Weapons of mass destruction
    r"\b(synthesize|make|build|create|manufacture)\b.{0,40}\b(bomb|explosive|bioweapon|nerve.?agent|sarin|vx\b|ricin|anthrax)\b",
    # CSAM
    r"\b(child|minor|underage|kid).{0,20}(naked|nude|sex|porn|explicit)\b",
    r"\b(loli|shota)\b",
    # Malware / hacking harmful
    r"\b(write|create|build).{0,30}\b(ransomware|keylogger|rootkit|botnet|worm|virus)\b",
    # Self-harm facilitation
    r"\b(how to|steps to|guide.{0,10})(kill yourself|commit suicide|end your life|self.harm)\b",
    # PII exfiltration
    r"\b(dump|leak|steal).{0,30}(database|passwords|credit.?card|ssn|social.security)\b",
]

# ---------------------------------------------------------------------------
# Output warning patterns (flag but still return, with a caveat)
# ---------------------------------------------------------------------------
OUTPUT_WARNING_PATTERNS = [
    r"\b(kill|murder|harm|hurt).{0,20}(them|him|her|you|people)\b",
    r"\b(instructions|steps|recipe).{0,30}(poison|drug|weapon)\b",
    r"\b(i hate|all \w+ are|those people deserve)\b",
]

_compiled_blocked  = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in BLOCKED_INPUT_PATTERNS]
_compiled_warnings = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in OUTPUT_WARNING_PATTERNS]

REFUSAL_MESSAGE = (
    "I'm sorry, I can't help with that request. "
    "If you have another question, I'm happy to assist!"
)

OUTPUT_CAVEAT = (
    "\n\n⚠️ *Note: This response has been flagged for review. "
    "Please verify any sensitive information independently.*"
)


def check_input(text: str) -> Tuple[bool, str]:
    """
    Returns (is_safe, message).
    If is_safe is False, message contains the refusal text.
    """
    for pattern in _compiled_blocked:
        if pattern.search(text):
            return False, REFUSAL_MESSAGE
    return True, ""


def check_output(text: str) -> str:
    """
    Returns the (possibly annotated) output text.
    Appends a caveat if a warning pattern fires.
    """
    for pattern in _compiled_warnings:
        if pattern.search(text):
            return text + OUTPUT_CAVEAT
    return text


def sanitize_pair(user_input: str, assistant_output: str) -> Tuple[bool, str, str]:
    """
    Convenience wrapper: returns (input_ok, safe_input, safe_output).
    If input_ok is False, safe_output is the refusal message.
    """
    ok, refusal = check_input(user_input)
    if not ok:
        return False, user_input, refusal
    safe_output = check_output(assistant_output)
    return True, user_input, safe_output
