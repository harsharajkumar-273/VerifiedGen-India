"""Reward and constrained-decision utilities for verifier-guided generation."""
from dataclasses import dataclass
from typing import Dict

@dataclass
class RewardWeights:
    prompt_alignment: float = 0.30
    visual_quality: float = 0.20
    artifact_risk: float = 0.15
    text_rendering: float = 0.15
    cultural_context: float = 0.10
    safety: float = 0.10

def reward_from_report(report, weights: RewardWeights | None = None) -> float:
    weights = weights or RewardWeights()
    scores: Dict[str, float] = {axis.name: axis.score for axis in report.axis_scores}
    safety = scores.get("safety", 0.0)
    if safety < 1.0:
        return 0.0
    # Renormalise over the axes this prompt actually has, so prompts without
    # text or cultural requirements are not penalised for missing axes.
    present = {name: value for name, value in weights.__dict__.items() if name in scores}
    total = sum(present.values())
    if total <= 0:
        return 0.0
    reward = sum(scores[name] * value for name, value in present.items()) / total
    return round(max(0.0, min(1.0, reward)), 4)

def constrained_decision(report) -> str:
    """Safety is a hard constraint; quality/alignment can trigger retry."""
    safety = next((axis for axis in report.axis_scores if axis.name == "safety"), None)
    if safety is not None and not safety.passed:
        return "reject_safety"
    if report.decision == "pass":
        return "accept"
    return "regenerate"

