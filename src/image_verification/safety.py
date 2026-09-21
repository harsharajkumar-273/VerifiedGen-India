"""Safety gate interfaces.

The default implementation consumes labels from an external safety classifier.
It is intentionally fail-closed when a label is explicitly unsafe. Replace
`labels_from_candidate` with a production NSFW/safety model adapter later.
"""
from dataclasses import dataclass
from typing import Iterable, List

BLOCKED_LABELS = {
    "nsfw", "sexual", "explicit", "pornography", "nudity", "graphic_violence",
    "violence", "gore", "self_harm", "hate", "extremism", "unsafe",
}

@dataclass
class SafetyResult:
    safe: bool
    score: float
    labels: List[str]
    reasons: List[str]

def labels_from_candidate(candidate) -> List[str]:
    metadata = getattr(candidate, "metadata", {}) or {}
    labels = getattr(candidate, "safety_labels", None) or metadata.get("safety_labels", [])
    if isinstance(labels, str):
        labels = [labels]
    return [str(label).strip().lower() for label in labels if str(label).strip()]

def evaluate_safety(labels: Iterable[str]) -> SafetyResult:
    normalized = [str(label).strip().lower() for label in labels]
    blocked = sorted(set(normalized) & BLOCKED_LABELS)
    return SafetyResult(
        safe=not blocked,
        score=0.0 if blocked else 1.0,
        labels=normalized,
        reasons=[f"Blocked safety label: {label}" for label in blocked],
    )

def safety_gate(candidate) -> SafetyResult:
    return evaluate_safety(labels_from_candidate(candidate))

