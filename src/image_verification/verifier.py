from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Tuple

from src.image_verification.image_features import color_coverage, compute_image_features
from src.image_verification.prompt_parser import COLOR_RGB, extract_prompt_requirements, normalize_label
from src.image_verification.schema import AxisScore, Detection, ImageCandidate, ImageVerificationReport
from src.image_verification.safety import safety_gate


@dataclass
class ThresholdConfig:
    pass_threshold: float = 0.72
    fail_threshold: float = 0.42
    min_quality: float = 0.45
    min_alignment: float = 0.55
    min_text_score: float = 0.70
    min_cultural_score: float = 0.60
    min_detection_confidence: float = 0.50
    color_presence_threshold: float = 0.12
    min_dimension: int = 256
    max_artifact_risk: float = 0.55


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _axis(name: str, score: float, passed: bool, notes: Iterable[str] = ()) -> AxisScore:
    return AxisScore(name=name, score=round(_clip(score), 4), passed=bool(passed), notes=list(notes))


def _aggregate_detections(
    detections: Iterable[Detection], minimum_confidence: float
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for detection in detections:
        if float(detection.confidence) < minimum_confidence:
            continue
        label = normalize_label(str(detection.label))
        count = int(detection.count or 1)
        counts[label] = max(counts.get(label, 0), count)
    return counts


def _score_quality(features: Any, config: ThresholdConfig) -> Tuple[AxisScore, List[str]]:
    notes: List[str] = []
    width = int(getattr(features, "width", 0))
    height = int(getattr(features, "height", 0))
    min_side = min(width, height) if width and height else 0
    resolution_score = _clip(min_side / max(config.min_dimension, 1))
    contrast = _clip(float(getattr(features, "contrast", 0.0)) / 0.25)
    edge_density = _clip(float(getattr(features, "edge_density", 0.0)) / 0.20)
    exposure = _clip(1.0 - float(getattr(features, "exposure_penalty", 0.0)))
    score = 0.35 * resolution_score + 0.25 * contrast + 0.20 * edge_density + 0.20 * exposure
    if min_side < config.min_dimension:
        notes.append(f"Resolution below target: {min_side}px < {config.min_dimension}px.")
    if contrast < 0.15:
        notes.append("Image has very low contrast.")
    passed = score >= config.min_quality
    return _axis("visual_quality", score, passed, notes), notes


def _score_artifacts(features: Any, config: ThresholdConfig) -> Tuple[AxisScore, List[str]]:
    notes: List[str] = []
    risk = 0.0
    if bool(getattr(features, "is_blank", False)):
        risk = max(risk, 1.0)
        notes.append("Image appears blank or near-blank.")
    risk = max(risk, _clip(float(getattr(features, "artifact_risk", 0.0))))
    if risk > config.max_artifact_risk:
        notes.append(f"Artifact risk is high ({risk:.2f}).")
    score = 1.0 - risk
    return _axis("artifact_risk", score, risk <= config.max_artifact_risk, notes), notes


def _score_safety(candidate: ImageCandidate) -> AxisScore:
    result = safety_gate(candidate)
    return _axis("safety", result.score, result.safe, result.reasons)


def _score_prompt_alignment(candidate: ImageCandidate, image: Any, config: ThresholdConfig) -> Tuple[AxisScore, List[str]]:
    requirements = extract_prompt_requirements(candidate.prompt, candidate.expected_objects or {})
    notes: List[str] = []
    scores: List[float] = []
    detected = _aggregate_detections(candidate.detections, config.min_detection_confidence)

    for label, expected_count in requirements.objects.items():
        normalized = normalize_label(label)
        actual = detected.get(normalized)
        if actual is None:
            notes.append(f"Needs detector evidence for object: {label}.")
            scores.append(0.5)
        else:
            scores.append(1.0 if actual >= expected_count else _clip(actual / expected_count))
            if actual < expected_count:
                notes.append(f"Expected {expected_count} {label}, observed {actual}.")

    for color, rgb in COLOR_RGB.items():
        if re.search(rf"\b{re.escape(color)}\b", candidate.prompt.lower()):
            coverage = color_coverage(image, rgb)
            scores.append(_clip(coverage / config.color_presence_threshold))
            if coverage < config.color_presence_threshold:
                notes.append(f"Expected visible {color} content, but coverage was low.")

    if not scores:
        scores.append(0.5)
        notes.append("No explicit color or object requirements were extracted from the prompt.")

    score = sum(scores) / len(scores)
    unresolved = any(note.startswith("Needs detector") for note in notes)
    passed = score >= config.min_alignment and not unresolved
    return _axis("prompt_alignment", score, passed, notes), notes


def _text_values(candidate: ImageCandidate) -> Tuple[List[str], str]:
    metadata = candidate.metadata or {}
    expected = getattr(candidate, "expected_text", None) or metadata.get("expected_text") or []
    observed = getattr(candidate, "ocr_text", None) or metadata.get("ocr_text") or ""
    if isinstance(expected, str):
        expected = [expected]
    return [str(value) for value in expected], str(observed)


def _score_text_rendering(candidate: ImageCandidate, config: ThresholdConfig) -> AxisScore | None:
    expected, observed = _text_values(candidate)
    if not expected:
        return None
    normalized_observed = observed.casefold()
    matches = sum(1 for value in expected if value.casefold() in normalized_observed)
    score = matches / len(expected)
    notes = [] if matches == len(expected) else ["Expected text was not fully recovered by OCR."]
    return _axis("text_rendering", score, score >= config.min_text_score, notes)


def _score_cultural_context(candidate: ImageCandidate, config: ThresholdConfig) -> AxisScore | None:
    metadata = candidate.metadata or {}
    expected = getattr(candidate, "cultural_markers", None) or metadata.get("cultural_markers") or []
    observed = getattr(candidate, "observed_cultural_markers", None) or metadata.get("observed_cultural_markers") or []
    if not expected:
        return None
    expected_norm = {normalize_label(str(value)) for value in expected}
    observed_norm = {normalize_label(str(value)) for value in observed}
    score = len(expected_norm & observed_norm) / len(expected_norm)
    notes = [] if score >= config.min_cultural_score else ["Expected cultural/context markers were not fully observed."]
    return _axis("cultural_context", score, score >= config.min_cultural_score, notes)


def verify_candidate(candidate: ImageCandidate, config: ThresholdConfig | None = None) -> ImageVerificationReport:
    config = config or ThresholdConfig()
    image, features = compute_image_features(candidate.image_path)
    quality_axis, quality_notes = _score_quality(features, config)
    artifact_axis, artifact_notes = _score_artifacts(features, config)
    alignment_axis, alignment_notes = _score_prompt_alignment(candidate, image, config)
    safety_axis = _score_safety(candidate)
    text_axis = _score_text_rendering(candidate, config)
    cultural_axis = _score_cultural_context(candidate, config)

    axes = [quality_axis, artifact_axis, alignment_axis, safety_axis]
    if text_axis is not None:
        axes.append(text_axis)
    if cultural_axis is not None:
        axes.append(cultural_axis)

    overall = sum(axis.score for axis in axes) / len(axes)
    failed = [axis for axis in axes if not axis.passed]
    severe = artifact_axis.score < 0.25 or quality_axis.score < 0.25 or alignment_axis.score < 0.25
    safety_blocked = not safety_axis.passed
    if safety_blocked:
        decision = "fail"
    elif overall >= config.pass_threshold and not failed and not severe:
        decision = "pass"
    elif overall <= config.fail_threshold or severe:
        decision = "fail"
    else:
        decision = "review"

    reasons: List[str] = []
    for axis in failed:
        reasons.extend(axis.notes or [f"{axis.name} failed its threshold."])
    reasons.extend(quality_notes if not quality_axis.passed else [])
    reasons.extend(artifact_notes if not artifact_axis.passed else [])
    reasons.extend(alignment_notes if not alignment_axis.passed else [])
    return ImageVerificationReport(
        candidate_id=candidate.candidate_id,
        prompt_id=candidate.prompt_id,
        decision=decision,
        overall_score=round(_clip(overall), 4),
        axis_scores=axes,
        failure_reasons=list(dict.fromkeys(reasons)),
        gold_label=candidate.gold_label,
        human_rank=candidate.human_rank,
    )


def rank_reports(reports: Iterable[ImageVerificationReport]) -> List[ImageVerificationReport]:
    return sorted(reports, key=lambda report: (-report.overall_score, report.candidate_id))


def select_best_report(reports: Iterable[ImageVerificationReport]) -> ImageVerificationReport:
    ranked = rank_reports(reports)
    if not ranked:
        raise ValueError("Cannot select a candidate from an empty report list.")
    return ranked[0]
