"""Evaluator-reliability study: are automatic evaluators worse on India-centric images?

The analysis follows docs/RESEARCH_PROTOCOL.md:

* **H1 (reliability gap).** AUROC of each evaluator against the human verdict
  is lower on India-centric prompts than on their matched controls.
  Gap = AUROC(control) - AUROC(India-centric); positive means worse on India.
* **H2 (false accepts).** With the accept threshold calibrated on control
  images (keeping ``target_true_accept_rate`` of human-accepted controls),
  more human-rejected India-centric images are accepted.
  Gap = FAR(India-centric) - FAR(control); positive means more slip through.
* **Decomposition (script track).** script effect = AUROC(india_latin) -
  AUROC(india_indic); locale effect = AUROC(global_latin) - AUROC(india_latin).
* **H3 (scripts).** Per-script reliability on Indic text prompts, related to a
  script-complexity index (exploratory).

Uncertainty comes from a bootstrap that resamples whole matched pairs, since
all images of a pair share a prompt template. Human inter-annotator
agreement per condition is reported as a ceiling so an evaluator gap is not
mistaken for a gap in label quality.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .human_labels import AggregatedLabel, aggregate_labels, labels_from_gold, load_label_rows, verdict_agreement
from .language_metrics import script_complexity
from .schema import ImageCandidate
from .selection import best_of_n_by_condition, selection_gap
from .stats import auroc, cluster_bootstrap, false_accept_rate, holm_adjust, spearman, threshold_at_true_accept_rate

# Which control condition calibrates the threshold for each track.
TRACK_CONTROL = {"script": "global_latin", "cultural": "global", "code_mixed": "global_latin"}
# (India-centric condition, control condition) for each paired track.
PRIMARY_CONTRASTS = {"script": ("india_indic", "global_latin"), "cultural": ("india", "global")}


@dataclass
class StudyConfig:
    bootstrap_samples: int = 2000
    seed: int = 0
    alpha: float = 0.05
    target_true_accept_rate: float = 0.8
    min_group_size: int = 10
    # Evaluators whose primary contrast forms the confirmatory family for the
    # Holm correction. Empty means every evaluator with scores except controls.
    primary_evaluators: tuple = ()
    control_evaluators: tuple = ("random", "image_stats")

    @classmethod
    def from_file(cls, path: Optional[str | Path]) -> "StudyConfig":
        if not path:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        study = dict(data.get("study", data))
        design = data.get("design", {})
        for key in ("primary_evaluators", "control_evaluators"):
            if key in design and key not in study:
                study[key] = tuple(design[key])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in study.items() if k in known})


@dataclass
class StudyRow:
    candidate_id: str
    prompt_id: str
    cluster: str
    condition: str
    track: str
    category: str
    script: str
    generator: str
    good: bool
    human_score: float
    complexity: Optional[float]


def build_rows(candidates: Iterable[ImageCandidate], labels: Dict[str, AggregatedLabel]) -> List[StudyRow]:
    rows = []
    for candidate in candidates:
        label = labels.get(candidate.candidate_id)
        if label is None or label.human_good is None:
            continue
        metadata = candidate.metadata or {}
        expected = metadata.get("expected_text") or []
        if isinstance(expected, str):
            expected = [expected]
        rows.append(StudyRow(
            candidate_id=candidate.candidate_id,
            prompt_id=candidate.prompt_id,
            cluster=str(metadata.get("pair_id") or candidate.prompt_id),
            condition=str(metadata.get("condition") or "unknown"),
            track=str(metadata.get("track") or "unknown"),
            category=candidate.category,
            script=str(metadata.get("script") or "none"),
            generator=candidate.generator,
            good=bool(label.human_good),
            human_score=label.human_score,
            complexity=script_complexity(" ".join(expected)) if expected else None,
        ))
    return rows


class _Analysis:
    """Metric helpers over a fixed row list and one evaluator's scores."""

    def __init__(self, rows: Sequence[StudyRow], scores: Dict[str, float], config: StudyConfig):
        self.rows = rows
        self.scores = scores
        self.config = config

    def _pick(self, indices: Iterable[int], condition: str):
        picked = [self.rows[i] for i in indices if self.rows[i].condition == condition and self.rows[i].candidate_id in self.scores]
        return [self.scores[r.candidate_id] for r in picked], [r.good for r in picked], [r.human_score for r in picked]

    def auroc(self, indices, condition) -> Optional[float]:
        scores, labels, _ = self._pick(indices, condition)
        return auroc(scores, labels)

    def spearman(self, indices, condition) -> Optional[float]:
        scores, _, human = self._pick(indices, condition)
        return spearman(scores, human) if len(scores) >= 3 else None

    def threshold(self, indices, control_condition) -> Optional[float]:
        scores, labels, _ = self._pick(indices, control_condition)
        return threshold_at_true_accept_rate(scores, labels, self.config.target_true_accept_rate)

    def far(self, indices, condition, control_condition) -> Optional[float]:
        indices = list(indices)
        cut = self.threshold(indices, control_condition)
        if cut is None:
            return None
        scores, labels, _ = self._pick(indices, condition)
        return false_accept_rate(scores, labels, cut)

    @staticmethod
    def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else a - b

    def bootstrap(self, statistic) -> Dict[str, Optional[float]]:
        return cluster_bootstrap([r.cluster for r in self.rows], statistic, self.config.bootstrap_samples, self.config.seed, self.config.alpha)


def _condition_table(analysis: _Analysis, rows: Sequence[StudyRow], label_rows: Optional[List[Dict]]) -> Dict[str, Dict]:
    everything = list(range(len(rows)))
    table: Dict[str, Dict] = {}
    for condition in sorted({r.condition for r in rows}):
        scores, labels, _ = analysis._pick(everything, condition)
        control = TRACK_CONTROL.get(next((r.track for r in rows if r.condition == condition), ""), None)
        entry = {
            "n": len(scores),
            "n_human_good": sum(labels),
            "human_good_rate": sum(labels) / len(labels) if labels else None,
            "auroc": analysis.auroc(everything, condition),
            "spearman": analysis.spearman(everything, condition),
            "false_accept_rate": analysis.far(everything, condition, control) if control else None,
            "threshold_calibrated_on": control,
        }
        if label_rows is not None:
            ids = {r.candidate_id for r in rows if r.condition == condition}
            ceiling = verdict_agreement(label_rows, ids)
            entry["human_ceiling_auroc"] = ceiling["human_ceiling_auroc"]
            entry["krippendorff_alpha"] = ceiling["krippendorff_alpha"]
            if entry["auroc"] is not None and ceiling["human_ceiling_auroc"]:
                entry["auroc_relative_to_ceiling"] = entry["auroc"] / ceiling["human_ceiling_auroc"]
        table[condition] = entry
    return table


def _contrasts(analysis: _Analysis, present: set) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    auroc_gaps = []
    for track, (india, control) in PRIMARY_CONTRASTS.items():
        if india not in present or control not in present:
            continue
        auroc_gaps.append((india, control))
        result[f"H1_auroc_gap_{track}"] = analysis.bootstrap(lambda idx, i=india, c=control: analysis.diff(analysis.auroc(idx, c), analysis.auroc(idx, i)))
        result[f"H2_far_gap_{track}"] = analysis.bootstrap(lambda idx, i=india, c=control: analysis.diff(analysis.far(idx, i, c), analysis.far(idx, c, c)))
        result[f"spearman_gap_{track}"] = analysis.bootstrap(lambda idx, i=india, c=control: analysis.diff(analysis.spearman(idx, c), analysis.spearman(idx, i)))
    if len(auroc_gaps) > 1:
        def pooled(idx):
            gaps = [analysis.diff(analysis.auroc(idx, c), analysis.auroc(idx, i)) for i, c in auroc_gaps]
            return None if any(g is None for g in gaps) else sum(gaps) / len(gaps)
        result["H1_auroc_gap_pooled"] = analysis.bootstrap(pooled)
    if {"india_indic", "india_latin", "global_latin"} <= present:
        result["script_effect_auroc"] = analysis.bootstrap(lambda idx: analysis.diff(analysis.auroc(idx, "india_latin"), analysis.auroc(idx, "india_indic")))
        result["locale_effect_auroc"] = analysis.bootstrap(lambda idx: analysis.diff(analysis.auroc(idx, "global_latin"), analysis.auroc(idx, "india_latin")))
    return result


def _by_script(analysis: _Analysis, rows: Sequence[StudyRow], config: StudyConfig) -> Dict:
    everything = list(range(len(rows)))
    threshold = analysis.threshold(everything, "global_latin")
    groups: Dict[str, List[StudyRow]] = defaultdict(list)
    for row in rows:
        if row.condition in {"india_indic", "india_code_mixed"}:
            groups[row.script].append(row)
    table = {}
    for script, members in sorted(groups.items()):
        scored = [r for r in members if r.candidate_id in analysis.scores]
        scores = [analysis.scores[r.candidate_id] for r in scored]
        labels = [r.good for r in scored]
        complexities = [r.complexity for r in members if r.complexity is not None]
        table[script] = {
            "n": len(scored),
            "human_good_rate": sum(labels) / len(labels) if labels else None,
            "auroc": auroc(scores, labels),
            "false_accept_rate": false_accept_rate(scores, labels, threshold) if threshold is not None else None,
            "script_complexity": sum(complexities) / len(complexities) if complexities else None,
            "underpowered": len(scored) < config.min_group_size,
        }
    usable = [v for v in table.values() if v["auroc"] is not None and v["script_complexity"] is not None]
    trend = spearman([v["script_complexity"] for v in usable], [v["auroc"] for v in usable]) if len(usable) >= 3 else None
    return {"scripts": table, "complexity_vs_auroc_spearman": trend, "n_scripts": len(usable)}


def _by_generator(rows: Sequence[StudyRow], scores: Dict[str, float], config: StudyConfig) -> Dict:
    table = {}
    for generator in sorted({r.generator for r in rows}):
        subset = [r for r in rows if r.generator == generator]
        analysis = _Analysis(subset, scores, config)
        everything = list(range(len(subset)))
        entry = {}
        for track, (india, control) in PRIMARY_CONTRASTS.items():
            entry[f"auroc_gap_{track}"] = analysis.diff(analysis.auroc(everything, control), analysis.auroc(everything, india))
        table[generator] = entry
    return table


def run_study(
    candidates: Sequence[ImageCandidate],
    labels: Dict[str, AggregatedLabel],
    scores: Dict[str, Dict[str, float]],
    config: Optional[StudyConfig] = None,
    label_rows: Optional[List[Dict]] = None,
) -> Dict:
    config = config or StudyConfig()
    rows = build_rows(candidates, labels)
    present = {r.condition for r in rows}
    result: Dict = {
        "config": config.__dict__,
        "n_labeled_images": len(rows),
        "n_clusters": len({r.cluster for r in rows}),
        "conditions": sorted(present),
        "human_agreement_overall": verdict_agreement(label_rows) if label_rows is not None else None,
        "evaluators": {},
    }
    for name, table in sorted(scores.items()):
        analysis = _Analysis(rows, table, config)
        result["evaluators"][name] = {
            "coverage": sum(1 for r in rows if r.candidate_id in table),
            "by_condition": _condition_table(analysis, rows, label_rows),
            "contrasts": _contrasts(analysis, present),
            "by_script": _by_script(analysis, rows, config),
            "by_generator": _by_generator(rows, table, config),
            "best_of_n": best_of_n_by_condition(rows, table),
            "best_of_n_capture_gap": {
                track: cluster_bootstrap(
                    [r.cluster for r in rows],
                    lambda idx, i=india, c=control: selection_gap([rows[k] for k in idx], table, c, i),
                    config.bootstrap_samples, config.seed, config.alpha,
                )
                for track, (india, control) in PRIMARY_CONTRASTS.items()
                if india in present and control in present
            },
        }
    _apply_holm(result, config)
    return result


def _apply_holm(result: Dict, config: StudyConfig) -> None:
    """Holm-correct the primary H1 contrast across the confirmatory evaluators.

    The primary contrast is the pooled AUROC gap when both tracks are present,
    otherwise the single available track's gap. Everything else in the report
    is secondary/exploratory and left unadjusted.
    """
    evaluators = result["evaluators"]
    family = [name for name in (config.primary_evaluators or evaluators) if name in evaluators and name not in config.control_evaluators]
    p_values = {}
    for name in family:
        contrasts = evaluators[name]["contrasts"]
        key = next((k for k in ("H1_auroc_gap_pooled", "H1_auroc_gap_script", "H1_auroc_gap_cultural") if k in contrasts), None)
        if key:
            p_values[(name, key)] = contrasts[key].get("p_two_sided")
    for (name, key), adjusted in holm_adjust(p_values).items():
        evaluators[name]["contrasts"][key]["p_holm"] = adjusted
    result["confirmatory_family"] = [f"{name}:{key}" for name, key in p_values]


# ---------------------------------------------------------------- reporting

def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_ci(entry: Dict) -> str:
    if entry.get("estimate") is None:
        return "–"
    if entry.get("ci_low") is None:
        return _fmt(entry["estimate"])
    return f"{entry['estimate']:+.3f} [{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}]"


def render_markdown(result: Dict) -> str:
    cfg = result["config"]
    lines = [
        "# Evaluator reliability on India-centric images",
        "",
        f"Labeled images: {result['n_labeled_images']} across {result['n_clusters']} prompt clusters. "
        f"Conditions: {', '.join(result['conditions'])}. "
        f"Bootstrap: {cfg['bootstrap_samples']} cluster resamples, {int((1 - cfg['alpha']) * 100)}% intervals. "
        f"False-accept thresholds keep {int(cfg['target_true_accept_rate'] * 100)}% of human-accepted control images.",
        "",
        f"Confirmatory family (Holm-corrected): {', '.join(result.get('confirmatory_family') or []) or 'none'}. "
        f"All other contrasts are exploratory. Bootstrap p-values cannot go below 1/{cfg['bootstrap_samples']}.",
        "",
    ]
    agreement = result.get("human_agreement_overall")
    if agreement:
        lines += [f"Human agreement (all images): Krippendorff's α = {_fmt(agreement['krippendorff_alpha'])}, "
                  f"held-out annotator AUROC = {_fmt(agreement['human_ceiling_auroc'])}.", ""]

    lines += ["## Headline contrasts", "",
              "Positive AUROC gap = evaluator is worse on India-centric prompts. Positive FAR gap = more human-rejected India-centric images slip through.",
              "", "| Evaluator | Contrast | Estimate [CI] | p (bootstrap) | p (Holm) |", "|---|---|---|---|---|"]
    for name, data in result["evaluators"].items():
        for contrast, entry in data["contrasts"].items():
            lines.append(f"| {name} | {contrast} | {_fmt_ci(entry)} | {_fmt(entry.get('p_two_sided'))} | {_fmt(entry.get('p_holm'))} |")

    for name, data in result["evaluators"].items():
        lines += ["", f"## {name}", "", f"Scored images: {data['coverage']}", "",
                  "| Condition | n | human good rate | AUROC | human ceiling | Spearman | FAR |", "|---|---|---|---|---|---|---|"]
        for condition, entry in data["by_condition"].items():
            lines.append(
                f"| {condition} | {entry['n']} | {_fmt(entry['human_good_rate'])} | {_fmt(entry['auroc'])} | "
                f"{_fmt(entry.get('human_ceiling_auroc'))} | {_fmt(entry['spearman'])} | {_fmt(entry['false_accept_rate'])} |"
            )
        scripts = data["by_script"]["scripts"]
        if scripts:
            lines += ["", "Indic scripts (India-centric text prompts):", "",
                      "| Script | n | complexity | human good rate | AUROC | FAR |", "|---|---|---|---|---|---|"]
            for script, entry in scripts.items():
                flag = " ⚠︎" if entry["underpowered"] else ""
                lines.append(f"| {script}{flag} | {entry['n']} | {_fmt(entry['script_complexity'])} | {_fmt(entry['human_good_rate'])} | {_fmt(entry['auroc'])} | {_fmt(entry['false_accept_rate'])} |")
            lines.append("")
            lines.append(f"Complexity vs AUROC across scripts (Spearman, exploratory): {_fmt(data['by_script']['complexity_vs_auroc_spearman'])}")
        bon = data["best_of_n"]
        if bon:
            lines += ["", "Best-of-N selection (mean human score of the pick):", "",
                      "| Condition | prompts | N | random | evaluator pick | oracle | capture |", "|---|---|---|---|---|---|---|"]
            for condition, entry in bon.items():
                lines.append(f"| {condition} | {entry['prompts']} | {entry['n']} | {_fmt(entry['random'])} | {_fmt(entry['selected'])} | {_fmt(entry['oracle'])} | {_fmt(entry['capture'])} |")
    lines += ["", "⚠︎ = fewer images than `min_group_size`; treat as descriptive only.", ""]
    return "\n".join(lines)


def main() -> None:
    from .cli import load_manifest
    from .evaluators import load_score_files

    parser = argparse.ArgumentParser(description="Run the pre-specified evaluator-reliability analysis.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels-jsonl", help="Raw per-annotator labels. Omit to fall back to manifest gold_label.")
    parser.add_argument("--key-json", help="task_id -> candidate_id map written at export time.")
    parser.add_argument("--scores", nargs="+", required=True, help="Score files from score_images.py or any external evaluator.")
    parser.add_argument("--config-file", default="configs/study.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    candidates = load_manifest(args.manifest)
    label_rows = None
    if args.labels_jsonl:
        key = json.loads(Path(args.key_json).read_text(encoding="utf-8")) if args.key_json else None
        label_rows = load_label_rows(args.labels_jsonl, key)
        labels = aggregate_labels(label_rows)
    else:
        labels = labels_from_gold(candidates)
    config = StudyConfig.from_file(args.config_file if Path(args.config_file).exists() else None)
    result = run_study(candidates, labels, load_score_files(args.scores), config, label_rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "study_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "study_report.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"[study] {result['n_labeled_images']} labeled images, {len(result['evaluators'])} evaluators -> {out}")


if __name__ == "__main__":
    main()
