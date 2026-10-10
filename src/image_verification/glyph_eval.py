"""Score evaluators on glyph minimal pairs.

Input: ``pairs.jsonl`` from ``glyph_pairs.build`` and one or more score files
(``{"candidate_id", "evaluator", "score"}``, candidate ids ``<pair_id>:clean`` /
``<pair_id>:corrupt``).

For each pair the outcome is 1 if the evaluator scores the image that matches
the prompt higher, 0 if lower and 0.5 on a tie. Three accuracies are reported:

* **forward** – the prompt asks for the correct word;
* **mirrored** – same images, the prompt asks for the corrupted string;
* **reading** – forward and mirrored pooled. Any prompt-independent preference
  (less ink, familiar-looking words, aesthetics) scores exactly 0.5 here, so
  reading accuracy above 0.5 means the evaluator compares the requested text
  with the rendered text. It is the primary metric.

``prior = forward - mirrored`` is the prompt-independent preference for the
correct-looking image (for VLM judges: autocorrection towards real words).

The headline contrast for each Indic script is

    gap = reading accuracy(Latin) - reading accuracy(Indic script)

on the error types both have (``SHARED_ERROR_TYPES``), with a bootstrap that
resamples whole concepts, so a concept's Latin and Indic pairs move together.
A positive gap means the evaluator catches Latin text errors more often than
Indic ones. Holm correction is applied across evaluators for the pooled gap.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from . import stats
from .evaluators import load_score_files
from .glyph_pairs import SCRIPTS, SHARED_ERROR_TYPES


def load_pairs(path: str | Path) -> List[Dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def outcomes(pairs: Sequence[Dict], scores: Dict[str, float]) -> List[Dict]:
    """One row per pair that has both scores."""
    rows = []
    for pair in pairs:
        clean = scores.get(f"{pair['pair_id']}:clean")
        corrupt = scores.get(f"{pair['pair_id']}:corrupt")
        if clean is None or corrupt is None:
            continue
        outcome = 1.0 if clean > corrupt else (0.0 if clean < corrupt else 0.5)
        rows.append({**pair, "outcome": outcome, "margin": clean - corrupt})
    return rows


def _accuracy(rows: Sequence[Dict], index: Iterable[int]) -> Optional[float]:
    values = [rows[i]["outcome"] for i in index]
    return sum(values) / len(values) if values else None


def _cell(rows: Sequence[Dict], samples: int, seed: int) -> Dict:
    if not rows:
        return {"n": 0, "accuracy": None}
    result = stats.cluster_bootstrap([r["concept"] for r in rows], lambda idx: _accuracy(rows, idx), samples, seed)
    return {"n": len(rows), "accuracy": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}


def _directions(rows: Sequence[Dict], samples: int, seed: int) -> Dict:
    forward = [r for r in rows if r.get("direction", "forward") == "forward"]
    mirrored = [r for r in rows if r.get("direction") == "mirrored"]
    cells = {"reading": _cell(rows, samples, seed), "forward": _cell(forward, samples, seed),
             "mirrored": _cell(mirrored, samples, seed)}
    if forward and mirrored:
        cells["prior"] = cells["forward"]["accuracy"] - cells["mirrored"]["accuracy"]
    return cells


def _gap(rows: Sequence[Dict], indic: Sequence[str], samples: int, seed: int) -> Dict:
    """Reading accuracy(latn) - reading accuracy(indic scripts), shared error types, concept-clustered bootstrap."""
    subset = [r for r in rows if r["error_type"] in SHARED_ERROR_TYPES and r["script"] in set(indic) | {"latn"}]

    def statistic(index: List[int]) -> Optional[float]:
        latin = _accuracy(subset, [i for i in index if subset[i]["script"] == "latn"])
        other = _accuracy(subset, [i for i in index if subset[i]["script"] != "latn"])
        return None if latin is None or other is None else latin - other

    if not subset:
        return {"estimate": None}
    result = stats.cluster_bootstrap([r["concept"] for r in subset], statistic, samples, seed)
    result["n_latin"] = sum(1 for r in subset if r["script"] == "latn")
    result["n_indic"] = len(subset) - result["n_latin"]
    return result


def analyse(pairs: Sequence[Dict], score_tables: Dict[str, Dict[str, float]], samples: int = 2000, seed: int = 0) -> Dict:
    scripts = sorted({p["script"] for p in pairs})
    indic = [s for s in scripts if SCRIPTS[s].is_indic]
    result: Dict = {"n_pairs": len(pairs), "scripts": scripts, "evaluators": {}}
    for name, scores in sorted(score_tables.items()):
        rows = outcomes(pairs, scores)
        errors = [r for r in rows if r["is_error"]]
        nulls = [r for r in rows if not r["is_error"]]
        spread = statistics.pstdev(scores.values()) if len(scores) > 1 else 0.0
        entry: Dict = {"n_scored_pairs": len(rows)}
        entry["overall"] = {s: _directions([r for r in errors if r["script"] == s and r["error_type"] in SHARED_ERROR_TYPES], samples, seed) for s in scripts}
        entry["by_error_type"] = {
            s: {e: _directions([r for r in errors if r["script"] == s and r["error_type"] == e], samples, seed)
                for e in sorted({r["error_type"] for r in errors if r["script"] == s})}
            for s in scripts
        }
        entry["by_length"] = {
            s: {b: _cell([r for r in errors if r["script"] == s and r["length_bin"] == b and r["error_type"] in SHARED_ERROR_TYPES], samples, seed)
                for b in ("1-2", "3-4", "5+")}
            for s in scripts
        }
        entry["gap_vs_latin"] = {s: _gap(errors, [s], samples, seed) for s in indic}
        if indic:
            entry["gap_vs_latin"]["pooled_indic"] = _gap(errors, indic, samples, seed)
        entry["null_font"] = {
            s: {
                "n": len(sub),
                "accuracy": (sum(r["outcome"] for r in sub) / len(sub)) if sub else None,
                "median_abs_margin_sd": (statistics.median(abs(r["margin"]) for r in sub) / spread) if sub and spread else None,
            }
            for s in scripts for sub in [[r for r in nulls if r["script"] == s]]
        }
        forward_errors = [r for r in errors if r.get("direction", "forward") == "forward"]
        entry["mean_margin_sd"] = {
            s: (statistics.mean(r["margin"] for r in forward_errors if r["script"] == s) / spread)
            if spread and any(r["script"] == s for r in forward_errors) else None
            for s in scripts
        }
        result["evaluators"][name] = entry
    pooled = {name: e["gap_vs_latin"].get("pooled_indic", {}).get("p_two_sided") for name, e in result["evaluators"].items()}
    for name, adjusted in stats.holm_adjust(pooled).items():
        if "pooled_indic" in result["evaluators"][name]["gap_vs_latin"]:
            result["evaluators"][name]["gap_vs_latin"]["pooled_indic"]["p_holm"] = adjusted
    return result


def _f(value, digits: int = 2) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def _ci(cell: Dict) -> str:
    if cell.get("accuracy") is None:
        return "–"
    if cell.get("ci_low") is None:
        return _f(cell["accuracy"])
    return f"{_f(cell['accuracy'])} [{_f(cell['ci_low'])}, {_f(cell['ci_high'])}]"


def render_markdown(result: Dict) -> str:
    scripts = result["scripts"]
    names = [SCRIPTS[s].name for s in scripts]
    head = "| " + " | ".join(names) + " |"
    rule = "|---|" + "---|" * len(scripts)
    lines = ["# Glyph minimal pairs: evaluator results", "",
             f"{result['n_pairs']} pairs (forward + mirrored). Accuracy = P(image matching the prompt scored higher); 0.5 is chance. "
             "**Reading** accuracy pools forward and mirrored pairs and is immune to prompt-independent preferences. "
             "Intervals are 95% bootstrap intervals resampling whole concepts.", ""]
    lines += ["## Reading accuracy, shared error types", "", "| Evaluator " + head, rule]
    for name, e in result["evaluators"].items():
        lines.append(f"| {name} | " + " | ".join(_ci(e["overall"][s]["reading"]) for s in scripts) + " |")
    lines += ["", "## Forward / mirrored accuracy and prompt-independent prior (forward − mirrored)", "",
              "| Evaluator | Script | Forward | Mirrored | Prior |", "|---|---|---|---|---|"]
    for name, e in result["evaluators"].items():
        for s in scripts:
            c = e["overall"][s]
            lines.append(f"| {name} | {SCRIPTS[s].name} | {_ci(c['forward'])} | {_ci(c['mirrored'])} | {_f(c.get('prior'))} |")
    lines += ["", "## Gap: Latin minus Indic reading accuracy (positive = worse on Indic)", "",
              "| Evaluator | Script | Gap [95% CI] | p (two-sided) | p (Holm, pooled only) |", "|---|---|---|---|---|"]
    for name, e in result["evaluators"].items():
        for script, g in e["gap_vs_latin"].items():
            label = SCRIPTS[script].name if script in SCRIPTS else "Pooled Indic"
            interval = f"{_f(g.get('estimate'))} [{_f(g.get('ci_low'))}, {_f(g.get('ci_high'))}]"
            lines.append(f"| {name} | {label} | {interval} | {_f(g.get('p_two_sided'), 3)} | {_f(g.get('p_holm'), 3)} |")
    lines += ["", "## By error type (reading accuracy; forward only for unshaped and null_font)", ""]
    for name, e in result["evaluators"].items():
        types = sorted({t for s in scripts for t in e["by_error_type"][s]})
        lines += [f"**{name}**", "", "| Error type " + head, rule]
        for t in types:
            lines.append(f"| {t} | " + " | ".join(_ci(e["by_error_type"][s].get(t, {}).get("reading", {})) for s in scripts) + " |")
        lines.append("")
    lines += ["## Reading accuracy by word length (syllables; shared error types)", ""]
    for name, e in result["evaluators"].items():
        lines += [f"**{name}**", "", "| Length " + head, rule]
        for b in ("1-2", "3-4", "5+"):
            lines.append(f"| {b} | " + " | ".join(_ci(e["by_length"][s][b]) for s in scripts) + " |")
        lines.append("")
    lines += ["## Null control (same correct text, different font)", "",
              "Accuracy should be near 0.5. The median |score difference| in units of the evaluator's score SD is its noise floor.", "",
              "| Evaluator | Script | n | Accuracy | Median abs. margin (SD) |", "|---|---|---|---|---|"]
    for name, e in result["evaluators"].items():
        for s, cell in e["null_font"].items():
            if cell["n"]:
                lines.append(f"| {name} | {SCRIPTS[s].name} | {cell['n']} | {_f(cell['accuracy'])} | {_f(cell['median_abs_margin_sd'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise accuracy of evaluators on glyph minimal pairs.")
    parser.add_argument("--pairs", required=True, help="pairs.jsonl from build_glyph_pairs.py")
    parser.add_argument("--scores", nargs="+", required=True, help="Score files (JSONL).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = analyse(load_pairs(args.pairs), load_score_files(args.scores), args.bootstrap, args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "glyph_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "glyph_results.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"[glyph] wrote {out / 'glyph_results.md'}")


if __name__ == "__main__":
    main()
