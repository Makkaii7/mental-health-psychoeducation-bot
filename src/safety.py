"""
Four-tier safety routing + sticky crisis session state.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml

Tier = Literal[1, 2, 3, 4]

CRISIS_KEYWORDS: list[str] = [
    "want to die",
    "kill myself",
    "suicide",
    "hurt myself",
    "end my life",
    "no point in living",
    "better off dead",
    "harm myself",
    "suicidal",
]

UAE_CRISIS_RESOURCES: dict[str, str] = {
    "Estijaba": "800 1717",
    "HOPE_line": "800 4673",
    "Emergency": "999",
}

_OUT_OF_SCOPE_PATTERNS = [
    r"\bdiagnos(e|is)\s+me\b",
    r"\bdo i have\b.*\b(adhd|bipolar|ptsd|ocd|depression|anxiety disorder)\b",
    r"\bwhat medication\b",
    r"\bshould i take\b",
    r"\b(prescribe|prescription)\b",
    r"\btherapy session\b",
    r"\bbe my therapist\b",
    r"\btell me exactly what to do\b",
    r"\bgive me a treatment plan\b",
]
_OUT_OF_SCOPE_RE = re.compile("|".join(f"(?:{p})" for p in _OUT_OF_SCOPE_PATTERNS), re.IGNORECASE)

_WITH_CARE_PATTERNS = [
    r"\b(can't cope|cant cope)\b",
    r"\boverwhelmed\b",
    r"\bpanic attack\b",
    r"\bself[- ]harm\b",  # ideation phrasing; explicit methods still hit crisis keywords first
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\blow mood\b",
    r"\b(anxious|anxiety)\b",
    r"\blonely\b",
]
_WITH_CARE_RE = re.compile("|".join(f"(?:{p})" for p in _WITH_CARE_PATTERNS), re.IGNORECASE)


def _load_keywords_from_config() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "config.yaml"
    if not cfg_path.exists():
        return list(CRISIS_KEYWORDS)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    rel = ((cfg.get("safety") or {}).get("crisis_keywords_file")) or "config/crisis_keywords.txt"
    path = root / rel
    if not path.is_file():
        return list(CRISIS_KEYWORDS)
    extra = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    merged = list(dict.fromkeys([*CRISIS_KEYWORDS, *extra]))
    return merged


def classify_tier(message: str, crisis_keywords: list[str] | None = None) -> Tier:
    """
    Return tier 1–4:
    4 = crisis, 3 = out of scope, 2 = in scope with care, 1 = in scope.
    Order: crisis lexical → out-of-scope regex → with-care heuristics → default in-scope.
    """
    text = message.lower()
    kws = crisis_keywords or _load_keywords_from_config()
    if any(k in text for k in kws):
        return 4
    if _OUT_OF_SCOPE_RE.search(message):
        return 3
    if _WITH_CARE_RE.search(message):
        return 2
    return 1


def get_crisis_response() -> str:
    lines = [
        "I'm really concerned about what you've shared. This chatbot is not able to assess risk or keep you safe.",
        "Please contact a trained responder or emergency service right now:",
        f"- Estijaba (mental health support): {UAE_CRISIS_RESOURCES['Estijaba']}",
        f"- HOPE Line: {UAE_CRISIS_RESOURCES['HOPE_line']}",
        f"- Emergency (immediate danger): {UAE_CRISIS_RESOURCES['Emergency']}",
        "If you can, stay with someone you trust while you reach out.",
    ]
    return "\n".join(lines)


def is_out_of_scope(message: str) -> bool:
    return classify_tier(message) == 3


def get_redirect_response() -> str:
    return (
        "I can't provide therapy, diagnoses, or medical advice, and I won't tell you what you "
        "must do. I can still help with psychoeducation-style reflection: exploring how stress "
        "shows up in the body and mind, values, coping ideas, and questions that help you think "
        "clearly. What part of your situation feels safest to explore in that way?"
    )


class StickySession:
    """
    Tracks whether crisis mode was triggered; once active, normal chat stays blocked
    for the remainder of the session (configurable).
    """

    def __init__(self, enable_sticky: bool = True) -> None:
        self.enable_sticky = enable_sticky
        self.crisis_triggered = False

    def note_tier(self, tier: Tier) -> None:
        if tier == 4:
            self.crisis_triggered = True

    def blocked_from_normal_chat(self) -> bool:
        return self.enable_sticky and self.crisis_triggered

    def reset(self) -> None:
        self.crisis_triggered = False
