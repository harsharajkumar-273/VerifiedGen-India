"""Automatic evaluators under study.

Each evaluator maps an image candidate to a scalar where higher means "more
acceptable for this prompt". The study only uses the ordering of scores, so
evaluators need not share a scale.

Heavy evaluators (CLIP, VLM judges) are run once with
``scripts/score_images.py`` and their outputs saved as score files::

    {"candidate_id": "...", "evaluator": "clipscore", "score": 0.31}

Scores produced anywhere else (a hosted VLM judge, VQAScore from its own
repository, etc.) can be added in the same format and analysed alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .language_metrics import character_error_rate
from .schema import ImageCandidate


class Evaluator:
    name = "evaluator"

    def score(self, candidate: ImageCandidate) -> Optional[float]:
        raise NotImplementedError

    def score_all(self, candidates: Iterable[ImageCandidate]) -> Dict[str, Optional[float]]:
        return {candidate.candidate_id: self.score(candidate) for candidate in candidates}


class RandomEvaluator(Evaluator):
    """Chance floor: a deterministic pseudo-random score per candidate."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, candidate: ImageCandidate) -> float:
        digest = hashlib.sha256(f"{self.seed}:{candidate.candidate_id}".encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(16 ** 12)


class VerifierEvaluator(Evaluator):
    """The multi-axis VerifiedGen verifier's overall score."""

    name = "verifiedgen"

    def __init__(self, config=None):
        from .verifier import ThresholdConfig

        self.config = config or ThresholdConfig()

    def score(self, candidate: ImageCandidate) -> float:
        from .verifier import verify_candidate

        return verify_candidate(candidate, self.config).overall_score


class ImageStatsEvaluator(Evaluator):
    """Prompt-blind baseline: pixel-level quality heuristics only."""

    name = "image_stats"

    def __init__(self, config=None):
        from .verifier import ThresholdConfig

        self.config = config or ThresholdConfig()

    def score(self, candidate: ImageCandidate) -> float:
        from .image_features import compute_image_features
        from .verifier import _score_artifacts, _score_quality

        _, features = compute_image_features(candidate.image_path)
        quality, _ = _score_quality(features, self.config)
        artifacts, _ = _score_artifacts(features, self.config)
        return (quality.score + artifacts.score) / 2.0


class OcrEvaluator(Evaluator):
    """1 - character error rate between the requested text and OCR output.

    Reads ``ocr_text`` from the manifest, so any OCR engine can be plugged in
    upstream. Returns ``None`` for prompts that request no text.
    """

    name = "ocr"

    def score(self, candidate: ImageCandidate) -> Optional[float]:
        metadata = candidate.metadata or {}
        expected = metadata.get("expected_text") or []
        if isinstance(expected, str):
            expected = [expected]
        if not expected or metadata.get("ocr_text") is None:
            return None
        return max(0.0, 1.0 - character_error_rate(" ".join(expected), str(metadata["ocr_text"])))


class PrecomputedEvaluator(Evaluator):
    """Reads ``metadata["scores"][name]`` from the manifest row."""

    def __init__(self, name: str):
        self.name = name

    def score(self, candidate: ImageCandidate) -> Optional[float]:
        value = ((candidate.metadata or {}).get("scores") or {}).get(self.name)
        return None if value is None else float(value)


class ClipScoreEvaluator(Evaluator):
    """Cosine similarity between CLIP image and text embeddings (CLIPScore without the 2.5 rescale).

    Works with any checkpoint loadable by ``transformers.CLIPModel``. Note that
    English CLIP tokenizers split Indic text into byte fragments and truncate
    at 77 tokens; that is part of what the study measures, not a bug to hide.
    """

    name = "clipscore"

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: Optional[str] = None, name: Optional[str] = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("ClipScoreEvaluator needs torch and transformers installed.") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        if name:
            self.name = name

    def score(self, candidate: ImageCandidate) -> float:
        from PIL import Image

        image = Image.open(candidate.image_path).convert("RGB")
        inputs = self.processor(text=[candidate.prompt], images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.no_grad():
            image_embeds = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_embeds = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return float((image_embeds * text_embeds).sum())


BUILTIN: Dict[str, Callable[..., Evaluator]] = {
    "random": RandomEvaluator,
    "verifiedgen": VerifierEvaluator,
    "image_stats": ImageStatsEvaluator,
    "ocr": OcrEvaluator,
    "clipscore": ClipScoreEvaluator,
}


def build_evaluator(name: str, **options) -> Evaluator:
    if name in BUILTIN:
        return BUILTIN[name](**options)
    return PrecomputedEvaluator(name)


def load_score_files(paths: Iterable[str | Path]) -> Dict[str, Dict[str, float]]:
    """Merge score files into ``{evaluator: {candidate_id: score}}``."""
    merged: Dict[str, Dict[str, float]] = {}
    for path in paths:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("score") is None:
                continue
            table = merged.setdefault(str(row["evaluator"]), {})
            if row["candidate_id"] in table:
                raise ValueError(f"{path}:{number}: duplicate score for {row['candidate_id']} / {row['evaluator']}.")
            table[row["candidate_id"]] = float(row["score"])
    return merged


def write_score_file(path: str | Path, evaluator_name: str, scores: Dict[str, Optional[float]]) -> int:
    rows = [{"candidate_id": cid, "evaluator": evaluator_name, "score": score} for cid, score in scores.items() if score is not None]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    from .cli import load_manifest

    parser = argparse.ArgumentParser(description="Score every candidate in a manifest with one or more evaluators.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluators", nargs="+", default=["random", "image_stats", "verifiedgen", "ocr"],
                        help=f"Built-ins: {', '.join(BUILTIN)}")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    candidates: List[ImageCandidate] = load_manifest(args.manifest)
    for name in args.evaluators:
        if name not in BUILTIN:
            parser.error(f"Unknown evaluator {name!r}; precomputed scores go straight to run_study.py --scores.")
        evaluator = build_evaluator(name, model_name=args.clip_model) if name == "clipscore" else build_evaluator(name)
        count = write_score_file(Path(args.output_dir) / f"{name}.scores.jsonl", evaluator.name, evaluator.score_all(candidates))
        print(f"[score] {name}: {count} scores")


if __name__ == "__main__":
    main()
