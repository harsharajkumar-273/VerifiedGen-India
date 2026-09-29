"""Script-aware OCR metrics for manifest rows with expected_text and ocr_text."""
import re
import unicodedata
from collections import defaultdict


def normalize_text(text: str) -> str:
    """NFC-normalise, casefold and collapse whitespace.

    NFC matters for Indic scripts: the same syllable can be encoded as a
    precomposed character or as base + combining sign, and OCR engines differ.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text)).strip().casefold())


def script_complexity(text: str) -> float:
    """Share of non-space code points that are combining marks (Unicode Mn/Mc).

    A crude, reproducible proxy for how much a script relies on vowel signs,
    viramas and other stacked marks: 0 for plain Latin, higher for scripts
    whose syllables are built from a base letter plus marks.
    """
    chars = [c for c in unicodedata.normalize("NFC", str(text)) if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if unicodedata.category(c) in {"Mn", "Mc"}) / len(chars)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = list(normalize_text(reference)), list(normalize_text(hypothesis))
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def exact_match(reference: str, hypothesis: str) -> float:
    return float(normalize_text(reference) == normalize_text(hypothesis))


def evaluate_ocr_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        expected = row.get("expected_text") or row.get("metadata", {}).get("expected_text") or []
        observed = row.get("ocr_text") or row.get("metadata", {}).get("ocr_text") or ""
        if isinstance(expected, str): expected = [expected]
        reference = " ".join(expected)
        language = row.get("language") or row.get("metadata", {}).get("language") or "unknown"
        script = row.get("script") or row.get("metadata", {}).get("script") or "unknown"
        grouped[(language, script)].append({"exact_match": exact_match(reference, observed), "cer": character_error_rate(reference, observed)})
    result = {}
    for (language, script), values in grouped.items():
        result[f"{language}|{script}"] = {"count": len(values), "exact_match": sum(v["exact_match"] for v in values)/len(values), "cer": sum(v["cer"] for v in values)/len(values)}
    return result
