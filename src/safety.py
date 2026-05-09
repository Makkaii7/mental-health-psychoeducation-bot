"""
Four-tier safety routing + sticky crisis session state.
Templates load from ``prompts/safety_prompts.txt`` (YAML).
Crisis phrases use word-boundary regex after Unicode/whitespace normalization.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

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

# Common false positives for phrase-based crisis matching
_CRISIS_FALSE_POSITIVE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwant\s+to\s+die\s+my\s+hair\b", re.IGNORECASE),
    re.compile(r"\bwant\s+to\s+dye\b", re.IGNORECASE),
)

_JAILBREAK_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(?:previous|prior|all)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:previous|prior|all)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(?:to\s+be|you\s+are)\b", re.IGNORECASE),
    re.compile(r"\bhypothetically\b", re.IGNORECASE),
)

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
    r"\bself[- ]harm\b",
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\blow mood\b",
    r"\b(anxious|anxiety)\b",
    r"\blonely\b",
]
_WITH_CARE_RE = re.compile("|".join(f"(?:{p})" for p in _WITH_CARE_PATTERNS), re.IGNORECASE)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROMPTS_PATH = _PROJECT_ROOT / "prompts" / "safety_prompts.txt"


def normalize_message(message: str) -> str:
    """NFKC unicode normalize, collapse whitespace, lowercase for matching."""
    text = unicodedata.normalize("NFKC", message)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _phrase_to_wb_pattern(phrase: str) -> str:
    parts = phrase.strip().split()
    if not parts:
        return ""
    escaped = r"\s+".join(re.escape(p) for p in parts)
    return rf"\b{escaped}\b"


@lru_cache(maxsize=1)
def get_crisis_keywords() -> tuple[str, ...]:
    """Load and merge crisis keywords once (code defaults + config file)."""
    cfg_path = _PROJECT_ROOT / "config" / "config.yaml"
    merged = list(CRISIS_KEYWORDS)
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        rel = ((cfg.get("safety") or {}).get("crisis_keywords_file")) or "config/crisis_keywords.txt"
        path = _PROJECT_ROOT / rel
        if path.is_file():
            extra = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            merged = list(dict.fromkeys([*merged, *extra]))
    return tuple(merged)


@lru_cache(maxsize=1)
def _get_crisis_patterns() -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(_phrase_to_wb_pattern(k), re.IGNORECASE) for k in get_crisis_keywords() if k.strip())


@lru_cache(maxsize=1)
def _load_safety_templates() -> dict[str, Any]:
    path = _DEFAULT_PROMPTS_PATH
    if not path.is_file():
        logger.warning("Missing %s — using built-in fallbacks", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def reload_safety_caches() -> None:
    """Clear caches (e.g. after config edits in tests)."""
    get_crisis_keywords.cache_clear()
    _get_crisis_patterns.cache_clear()
    _load_safety_templates.cache_clear()


def classify_tier(message: str, crisis_keywords: tuple[str, ...] | None = None) -> Tier:
    """
    Tier 4 = crisis, 3 = out-of-scope or jailbreak redirect, 2 = with care, 1 = default.
    Order: false-positive suppress → crisis (word boundaries) → jailbreak → out-of-scope → with-care.
    """
    norm = normalize_message(message)
    if not norm:
        return 1

    skip_crisis_due_to_false_positive = any(fp.search(norm) for fp in _CRISIS_FALSE_POSITIVE_RES)

    if crisis_keywords is None:
        crisis_patterns = _get_crisis_patterns()
    else:
        crisis_patterns = tuple(
            re.compile(_phrase_to_wb_pattern(k), re.IGNORECASE)
            for k in crisis_keywords
            if k.strip()
        )

    if not skip_crisis_due_to_false_positive:
        for pat in crisis_patterns:
            if pat.search(norm):
                return 4

    if any(j.search(norm) for j in _JAILBREAK_RES):
        return 3
    if _OUT_OF_SCOPE_RE.search(message):
        return 3
    if _WITH_CARE_RE.search(message):
        return 2
    return 1


def get_crisis_response() -> str:
    tmpl = _load_safety_templates().get("crisis_response")
    if not tmpl:
        tmpl = (
            "I'm really concerned about what you've shared. This chatbot is not able to assess risk or keep you safe.\n"
            "Please contact a trained responder or emergency service right now:\n"
            "- Estijaba (mental health support): {estijaba}\n"
            "- HOPE Line: {hope_line}\n"
            "- Emergency (immediate danger): {emergency}\n"
            "If you can, stay with someone you trust while you reach out."
        )
    return tmpl.format(
        estijaba=UAE_CRISIS_RESOURCES["Estijaba"],
        hope_line=UAE_CRISIS_RESOURCES["HOPE_line"],
        emergency=UAE_CRISIS_RESOURCES["Emergency"],
    ).strip()


def is_out_of_scope(message: str) -> bool:
    return classify_tier(message) == 3


def get_redirect_response() -> str:
    tmpl = _load_safety_templates().get("redirect_response")
    if not tmpl:
        tmpl = (
            "I can't provide therapy, diagnoses, or medical advice, and I won't tell you what you "
            "must do. I can still help with psychoeducation-style reflection: exploring how stress "
            "shows up in the body and mind, values, coping ideas, and questions that help you think "
            "clearly. What part of your situation feels safest to explore in that way?"
        )
    return tmpl.strip()


def get_tier2_system_addon() -> str:
    addon = _load_safety_templates().get("tier2_system_addon")
    if not addon:
        return (
            "The user may be distressed. Acknowledge briefly, ask about duration and intensity gently, "
            "then reflect or encourage professional support if appropriate. Stay Socratic; never diagnose."
        )
    return addon.strip()


class StickySession:
    """Tracks crisis mode; sticky blocks normal chat for the rest of the session."""

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
