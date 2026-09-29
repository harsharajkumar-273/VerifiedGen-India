"""Human annotation: blinded task export, label aggregation and agreement.

Label file format (JSONL, one row per annotator per image)::

    {"task_id": "3f9c...", "annotator_id": "a07",
     "verdict": "accept" | "reject",
     "alignment": 1-5,
     "text_rendering": "correct" | "minor_error" | "wrong" | "no_text" | "na",
     "transcription": "text the annotator can read in the image",
     "cultural_accuracy": "accurate" | "inaccurate" | "unsure" | "na",
     "failure_types": ["garbled_text", "wrong_locale", ...]}

``verdict`` answers the primary protocol question: *would you show this
image to someone who typed this prompt?* It is the ground truth for the
reliability analysis. See docs/ANNOTATION_GUIDE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .schema import ImageCandidate
from .stats import auroc, krippendorff_alpha_nominal

VERDICTS = {"accept", "reject"}
FAILURE_TYPES = [
    "garbled_text", "wrong_text", "wrong_script", "missing_text",
    "wrong_locale", "cultural_error", "stereotype", "missing_object",
    "wrong_count", "visual_artifact", "low_quality", "unsafe",
]


@dataclass
class AggregatedLabel:
    candidate_id: str
    n_annotators: int
    accept_share: float
    human_good: Optional[bool]  # None when annotators are tied
    alignment_mean: Optional[float]
    text_correct_share: Optional[float]
    cultural_accurate_share: Optional[float]
    failure_types: Dict[str, int] = field(default_factory=dict)
    transcriptions: List[str] = field(default_factory=list)

    @property
    def human_score(self) -> float:
        """Continuous human judgement for rank correlations (alignment, else accept share)."""
        return self.alignment_mean if self.alignment_mean is not None else self.accept_share

    def to_dict(self) -> Dict:
        return {**asdict(self), "human_score": self.human_score}


def task_id_for(candidate_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{candidate_id}".encode("utf-8")).hexdigest()[:16]


def export_blinded_tasks(
    candidates: Iterable[ImageCandidate], salt: str, seed: int = 0, image_dir: Optional[str | Path] = None,
) -> tuple[List[Dict], Dict[str, str]]:
    """Build annotation tasks that hide generator, condition and any scores.

    Returns ``(tasks, key)`` where ``key`` maps task_id -> candidate_id and must
    be kept away from annotators. Tasks are shuffled so annotators do not see
    matched pairs or all candidates of one prompt back to back.

    Original image paths usually encode the prompt id (and so the condition
    and generator), so when ``image_dir`` is given each image is copied there
    as ``<task_id><ext>`` and tasks point at the copy. Always use it for real
    annotation rounds.
    """
    tasks, key = [], {}
    out_dir = Path(image_dir) if image_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        metadata = candidate.metadata or {}
        task_id = task_id_for(candidate.candidate_id, salt)
        key[task_id] = candidate.candidate_id
        image_path = candidate.image_path
        if out_dir is not None:
            source = Path(candidate.image_path)
            target = out_dir / f"{task_id}{source.suffix.lower()}"
            shutil.copyfile(source, target)
            image_path = str(target)
        tasks.append({
            "task_id": task_id,
            "prompt": candidate.prompt,
            "image_path": image_path,
            "expected_text": metadata.get("expected_text") or [],
            # Routing hint so text-bearing images go to people who read the script.
            "requires_reader_of": metadata.get("script") if metadata.get("expected_text") else None,
            "verdict": None, "alignment": None, "text_rendering": None,
            "transcription": None, "cultural_accuracy": None, "failure_types": [],
        })
    random.Random(seed).shuffle(tasks)
    return tasks, key


def load_label_rows(path: str | Path, key: Optional[Dict[str, str]] = None) -> List[Dict]:
    """Load annotator rows, mapping task_id back to candidate_id with ``key``."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "candidate_id" not in row:
            if key is None or row.get("task_id") not in key:
                raise ValueError(f"{path}:{number}: row has no candidate_id and its task_id is not in the key file.")
            row["candidate_id"] = key[row["task_id"]]
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"{path}:{number}: verdict must be one of {sorted(VERDICTS)}, got {verdict!r}.")
        rows.append(row)
    return rows


def aggregate_labels(rows: Iterable[Dict]) -> Dict[str, AggregatedLabel]:
    by_candidate: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate_id"]].append(row)
    result: Dict[str, AggregatedLabel] = {}
    for candidate_id, items in by_candidate.items():
        accepts = sum(1 for item in items if item["verdict"] == "accept")
        rejects = len(items) - accepts
        alignments = [float(item["alignment"]) for item in items if item.get("alignment") is not None]
        text = [item["text_rendering"] for item in items if item.get("text_rendering") not in (None, "na", "no_text")]
        culture = [item["cultural_accuracy"] for item in items if item.get("cultural_accuracy") not in (None, "na", "unsure")]
        failures: Dict[str, int] = defaultdict(int)
        for item in items:
            for failure in item.get("failure_types") or []:
                failures[failure] += 1
        result[candidate_id] = AggregatedLabel(
            candidate_id=candidate_id,
            n_annotators=len(items),
            accept_share=accepts / len(items),
            human_good=None if accepts == rejects else accepts > rejects,
            alignment_mean=sum(alignments) / len(alignments) if alignments else None,
            text_correct_share=sum(1 for value in text if value == "correct") / len(text) if text else None,
            cultural_accurate_share=sum(1 for value in culture if value == "accurate") / len(culture) if culture else None,
            failure_types=dict(failures),
            transcriptions=[str(item["transcription"]) for item in items if item.get("transcription")],
        )
    return result


def labels_from_gold(candidates: Iterable[ImageCandidate]) -> Dict[str, AggregatedLabel]:
    """Fallback for manifests that carry a single ``gold_label`` ("good"/"bad") per image."""
    result = {}
    for candidate in candidates:
        if candidate.gold_label is None:
            continue
        good = str(candidate.gold_label).lower() in {"good", "accept", "pass"}
        result[candidate.candidate_id] = AggregatedLabel(candidate.candidate_id, 1, float(good), good, None, None, None)
    return result


def verdict_agreement(rows: Iterable[Dict], candidate_ids: Optional[Iterable[str]] = None) -> Dict[str, Optional[float]]:
    """Inter-annotator agreement on the accept/reject verdict.

    * ``krippendorff_alpha``: chance-corrected agreement.
    * ``human_ceiling_auroc``: each annotator is held out in turn and their
      verdict is scored against the majority of the remaining annotators. A
      binary scorer's AUROC equals its balanced accuracy, so this number is
      directly comparable with evaluator AUROC: an evaluator can hardly be
      expected to agree with the panel better than a panel member does.
    """
    allowed = set(candidate_ids) if candidate_ids is not None else None
    by_candidate: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        if allowed is None or row["candidate_id"] in allowed:
            by_candidate[row["candidate_id"]].append(row)
    alpha = krippendorff_alpha_nominal([item["verdict"] for item in items] for items in by_candidate.values())
    held_out_scores: List[float] = []
    held_out_truth: List[bool] = []
    for items in by_candidate.values():
        if len(items) < 3:
            continue
        for index, item in enumerate(items):
            others = [other["verdict"] for j, other in enumerate(items) if j != index]
            accepts = others.count("accept")
            if accepts * 2 == len(others):
                continue
            held_out_truth.append(accepts * 2 > len(others))
            held_out_scores.append(1.0 if item["verdict"] == "accept" else 0.0)
    return {
        "items_with_multiple_ratings": float(sum(1 for items in by_candidate.values() if len(items) >= 2)),
        "krippendorff_alpha": alpha,
        "human_ceiling_auroc": auroc(held_out_scores, held_out_truth),
    }


def main() -> None:
    from .cli import load_manifest

    parser = argparse.ArgumentParser(description="Export blinded annotation tasks or aggregate returned labels.")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--manifest", required=True)
    export.add_argument("--output-jsonl", required=True)
    export.add_argument("--key-json", required=True, help="Private task_id -> candidate_id map. Do not share with annotators.")
    export.add_argument("--salt", required=True, help="Secret string so task ids cannot be reversed.")
    export.add_argument("--image-dir", required=True, help="Where to copy images under opaque task-id file names.")
    export.add_argument("--seed", type=int, default=0)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--labels-jsonl", required=True)
    agg.add_argument("--key-json")
    agg.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    if args.command == "export":
        tasks, key = export_blinded_tasks(load_manifest(args.manifest), args.salt, args.seed, args.image_dir)
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_jsonl).write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n", encoding="utf-8")
        Path(args.key_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.key_json).write_text(json.dumps(key, indent=2), encoding="utf-8")
        print(f"[annotation] wrote {len(tasks)} blinded tasks -> {args.output_jsonl}")
    else:
        key = json.loads(Path(args.key_json).read_text(encoding="utf-8")) if args.key_json else None
        labels = aggregate_labels(load_label_rows(args.labels_jsonl, key))
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_jsonl).write_text("\n".join(json.dumps(l.to_dict(), ensure_ascii=False) for l in labels.values()) + "\n", encoding="utf-8")
        print(f"[annotation] aggregated {len(labels)} images -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
