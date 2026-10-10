"""Native-reader validation of the glyph pairs.

Ground truth comes from how each pair is built, but only if the "corrupt"
text really reads as wrong, and the "clean" text really reads as right, to a
native reader. This module samples rendered images, exports a blinded sheet
(one image per row, the clean/corrupt role hidden, order shuffled), and turns
the returned answers into an exclusions file for ``glyph_eval``.

Readers answer three questions per image:

1. ``transcription`` – type exactly what the sign says.
2. ``matches_request`` – yes / no / unsure: does the sign say the requested
   word, spelled correctly in the requested script?
3. ``legible`` – yes / no.

Rules (pre-specified):

* A **clean image fails** when the majority says it does not match the request.
  Its pair is excluded.
* An **error type fails for a script** when fewer than 80% of its corrupt
  images are judged "does not match". That error type is excluded for that
  script, because readers do not see it as an error.
* Illegible images (majority "no") are excluded with their pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from .glyph_pairs import SCRIPTS

FIELDS = ["item_id", "image_file", "requested_text", "language", "transcription", "matches_request", "legible", "comment"]


def export(pairs_path: Path, image_dir: Path, out_dir: Path, per_cell: int = 6, seed: int = 0, salt: str = "") -> Dict:
    pairs = [json.loads(line) for line in Path(pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    forward = [p for p in pairs if p.get("direction", "forward") == "forward"]
    rng = random.Random(seed)
    cells: Dict[tuple, List[Dict]] = defaultdict(list)
    for pair in forward:
        cells[(pair["script"], pair["error_type"])].append(pair)
    chosen = [p for key in sorted(cells) for p in rng.sample(cells[key], min(per_cell, len(cells[key])))]
    items, key_rows = [], []
    for pair in chosen:
        for role in ("clean", "corrupt"):
            image = pair[f"{role}_image"]
            item_id = hashlib.sha256(f"{salt}:{pair['pair_id']}:{role}".encode("utf-8")).hexdigest()[:10]
            items.append({"item_id": item_id, "image_file": f"images/{item_id}.png", "requested_text": pair["target_text"],
                          "language": SCRIPTS[pair["script"]].language})
            key_rows.append({"item_id": item_id, "pair_id": pair["pair_id"], "role": role, "script": pair["script"],
                             "error_type": pair["error_type"], "source_image": image})
    rng.shuffle(items)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    source = {row["item_id"]: row["source_image"] for row in key_rows}
    for item in items:
        shutil.copyfile(Path(image_dir) / source[item["item_id"]], out_dir / item["image_file"])
    # One sheet per language, so each goes to readers of that language.
    by_language: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        by_language[item["language"]].append(item)
    for language, group in by_language.items():
        tag = language.lower()
        with open(out_dir / f"validation_sheet_{tag}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for item in group:
                writer.writerow({**item, "transcription": "", "matches_request": "", "legible": "", "comment": ""})
        (out_dir / f"index_{tag}.html").write_text(_html(group, f"validation_sheet_{tag}.csv"), encoding="utf-8")
    (out_dir / "validation.key.json").write_text(json.dumps(key_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"items": len(items), "pairs": len(chosen), "cells": len(cells),
            "per_language": {k: len(v) for k, v in by_language.items()}}


def _html(items: Sequence[Dict], sheet_name: str) -> str:
    rows = "\n".join(
        f'<tr><td>{html.escape(i["item_id"])}</td><td><img src="{html.escape(i["image_file"])}" width="256"></td>'
        f'<td>{html.escape(i["language"])}: <b>{html.escape(i["requested_text"])}</b></td></tr>' for i in items)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Sign validation</title>
<style>body{{font-family:sans-serif;margin:16px}}td{{padding:6px;border-bottom:1px solid #ccc;vertical-align:middle}}</style></head>
<body><h1>Sign validation</h1>
<p>For each image, fill in its row of <code>{html.escape(sheet_name)}</code> (matched by item id):</p>
<ol><li><b>transcription</b>: type exactly what the sign says, including any mistakes.</li>
<li><b>matches_request</b>: <i>yes</i> if the sign says the requested word, spelled correctly in the requested script; otherwise <i>no</i> (or <i>unsure</i>).</li>
<li><b>legible</b>: <i>yes</i> or <i>no</i>.</li></ol>
<p>Some signs are deliberately wrong. Please judge each one on its own.</p>
<table><tr><th>Item</th><th>Image</th><th>Requested</th></tr>
{rows}
</table></body></html>
"""


def _majority(values: Sequence[str]) -> str:
    votes = Counter(v.strip().lower() for v in values if v and v.strip())
    if not votes:
        return ""
    (top, count), *rest = votes.most_common()
    return "" if rest and rest[0][1] == count else top


def aggregate(sheets: Sequence[Path], key_path: Path, threshold: float = 0.8) -> Dict:
    key = {row["item_id"]: row for row in json.loads(Path(key_path).read_text(encoding="utf-8"))}
    answers: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for sheet in sheets:
        with open(sheet, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["item_id"] in key:
                    for field in ("matches_request", "legible"):
                        answers[row["item_id"]][field].append(row.get(field, ""))
    excluded_pairs, detection = set(), defaultdict(list)
    clean_fail, illegible = [], []
    for item_id, fields in answers.items():
        info = key[item_id]
        verdict = _majority(fields["matches_request"])
        if _majority(fields["legible"]) == "no":
            illegible.append(info["pair_id"])
            excluded_pairs.add(info["pair_id"])
        if info["role"] == "clean" and verdict == "no":
            clean_fail.append(info["pair_id"])
            excluded_pairs.add(info["pair_id"])
        if info["role"] == "corrupt" and info["error_type"] != "null_font" and verdict in ("yes", "no"):
            detection[(info["script"], info["error_type"])].append(verdict == "no")
    rates = {f"{s}/{e}": sum(v) / len(v) for (s, e), v in detection.items()}
    failed_types: Dict[str, List[str]] = defaultdict(list)
    for (script, error_type), values in detection.items():
        if sum(values) / len(values) < threshold:
            failed_types[script].append(error_type)
    return {"pairs": sorted(excluded_pairs), "error_types": {k: sorted(v) for k, v in failed_types.items()},
            "detection_rate": rates, "clean_failures": sorted(clean_fail), "illegible": sorted(illegible),
            "threshold": threshold, "items_answered": len(answers)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Native-reader validation of glyph pairs.")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--pairs", required=True)
    ex.add_argument("--image-dir", required=True)
    ex.add_argument("--out-dir", required=True)
    ex.add_argument("--per-cell", type=int, default=6, help="Pairs per (script, error type).")
    ex.add_argument("--salt", default="")
    ex.add_argument("--seed", type=int, default=0)
    ag = sub.add_parser("aggregate")
    ag.add_argument("--sheets", nargs="+", required=True)
    ag.add_argument("--key", required=True)
    ag.add_argument("--out", required=True, help="exclusions.json for analyze_glyph_pairs.py --exclusions")
    ag.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export(Path(args.pairs), Path(args.image_dir), Path(args.out_dir), args.per_cell, args.seed, args.salt)))
    else:
        result = aggregate([Path(s) for s in args.sheets], Path(args.key), args.threshold)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[validation] excluded {len(result['pairs'])} pairs; failed error types: {result['error_types']}")


if __name__ == "__main__":
    main()
