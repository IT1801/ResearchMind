from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")

from nemoguardrails import LLMRails, RailsConfig

BLOCKED_MESSAGE = (
    "I can't help with that request because it contains sensitive personal "
    "information or unsafe content. Please remove private details and ask a "
    "safe research question."
)

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE)),
    ("phone number", re.compile(r"(?<!\d)(?:\+?\d[\d(). -]{7,}\d)(?!\d)")),
    ("US social security number", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("credit card number", re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")),
    ("secret or API key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,}|AIza[A-Za-z0-9_-]{20,})\b")),
    ("prompt injection", re.compile(r"\b(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b|\b(?:reveal|show|print)\s+(?:the\s+)?system\s+prompt\b|\bdeveloper\s+message\b", re.IGNORECASE)),
    ("malware or credential theft", re.compile(r"\b(?:steal|exfiltrate|dump)\s+(?:passwords?|tokens?|credentials?)\b|\b(?:create|write|deploy)\s+(?:a\s+)?(?:keylogger|ransomware|credential\s+stealer)\b", re.IGNORECASE)),
    ("violent wrongdoing", re.compile(r"\b(?:build|make|create)\s+(?:a\s+)?(?:bomb|explosive|weapon)\b|\bhow\s+to\s+(?:kill|assassinate|poison|hurt)\b", re.IGNORECASE)),
    ("self-harm instructions", re.compile(r"\bhow\s+to\s+(?:commit\s+suicide|kill\s+myself|self[- ]harm)\b", re.IGNORECASE)),
    ("sexual exploitation", re.compile(r"\b(?:sexual|nude)\s+(?:content|images?)\s+(?:of|involving)\s+(?:a\s+)?minor\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    category: str | None = None
    message: str | None = None


class ResearchMindGuardrails:
    """NeMo-configured input/output guardrails with local sensitive-data checks."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path or Path(__file__).parent)
        self.config = RailsConfig.from_path(str(path))
        self.rails = LLMRails(self.config)
        self.rails.register_action(self.check_input, name="check_input")
        self.rails.register_action(self.check_output, name="check_output")

    @staticmethod
    def _check(text: str) -> GuardrailDecision:
        for category, pattern in _RULES:
            if pattern.search(text):
                return GuardrailDecision(False, category, BLOCKED_MESSAGE)
        return GuardrailDecision(True)

    async def check_input(self, text: str) -> GuardrailDecision:
        return self._check(text)

    async def check_output(self, text: str) -> GuardrailDecision:
        return self._check(text)

    def validate_input(self, text: str) -> GuardrailDecision:
        return self._check(text)

    def validate_output(self, text: str) -> GuardrailDecision:
        return self._check(text)


__all__ = ["BLOCKED_MESSAGE", "GuardrailDecision", "ResearchMindGuardrails"]
