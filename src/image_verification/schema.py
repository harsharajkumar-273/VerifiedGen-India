from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Detection:
    label: str
    confidence: float = 1.0
    count: int = 1

@dataclass
class ImageCandidate:
    candidate_id: str
    prompt_id: str
    prompt: str
    image_path: str
    generator: str = "unknown"
    category: str = "unknown"
    expected_objects: Dict[str, int] = field(default_factory=dict)
    detections: List[Detection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    gold_label: Optional[str] = None
    human_rank: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageCandidate":
        copied = dict(data)
        copied["detections"] = [d if isinstance(d, Detection) else Detection(**d) for d in copied.get("detections", [])]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        metadata = dict(copied.get("metadata") or {})
        for key in list(copied):
            if key not in known:
                metadata[key] = copied.pop(key)
        copied["metadata"] = metadata
        return cls(**copied)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AxisScore:
    name: str
    score: float
    passed: bool
    notes: List[str] = field(default_factory=list)

@dataclass
class ImageVerificationReport:
    candidate_id: str
    prompt_id: str
    decision: str
    overall_score: float
    axis_scores: List[AxisScore]
    failure_reasons: List[str] = field(default_factory=list)
    gold_label: Optional[str] = None
    human_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

