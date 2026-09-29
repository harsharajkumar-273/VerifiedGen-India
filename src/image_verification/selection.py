"""Best-of-N selection: does picking the evaluator's top image help humans?

For each prompt with N labeled candidates we compare the human score of

* a random candidate (the mean over candidates),
* the evaluator's top-scored candidate (ties averaged), and
* the best candidate by human score (the oracle).

``capture`` = (evaluator pick - random) / (oracle - random), pooled over
prompts: 1.0 means the evaluator always finds the human favourite, 0.0 means
it is no better than chance, negative means it actively picks worse images.
The ``curve`` gives the expected human score of the evaluator's pick from
random subsets of size n = 1..N; a curve that flattens or drops while the
evaluator's own score keeps rising is the over-optimisation signature.
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

MAX_SUBSETS = 500


def _groups(rows: Sequence, scores: Dict[str, float], condition: str) -> List[tuple]:
    """Return ``[(multiplicity, [(evaluator_score, human_score), ...]), ...]`` per prompt.

    Bootstrap resamples can contain the same prompt more than once; the
    multiplicity keeps those draws weighted correctly.
    """
    by_prompt: Dict[str, List] = defaultdict(list)
    for row in rows:
        if row.condition == condition and row.candidate_id in scores:
            by_prompt[row.prompt_id].append(row)
    groups = []
    for members in by_prompt.values():
        unique = {row.candidate_id: row for row in members}
        if len(unique) < 2:
            continue
        pairs = [(scores[cid], row.human_score) for cid, row in unique.items()]
        groups.append((len(members) / len(unique), pairs))
    return groups


def _pick(pairs: Sequence[tuple]) -> float:
    best = max(score for score, _ in pairs)
    tied = [human for score, human in pairs if score == best]
    return sum(tied) / len(tied)


def _expected_pick(pairs: Sequence[tuple], n: int, rng: random.Random) -> float:
    combos = list(itertools.combinations(range(len(pairs)), n)) if n < len(pairs) else [tuple(range(len(pairs)))]
    if len(combos) > MAX_SUBSETS:
        combos = rng.sample(combos, MAX_SUBSETS)
    return sum(_pick([pairs[i] for i in combo]) for combo in combos) / len(combos)


def capture(rows: Sequence, scores: Dict[str, float], condition: str) -> Optional[float]:
    gain = headroom = 0.0
    for weight, pairs in _groups(rows, scores, condition):
        humans = [human for _, human in pairs]
        mean = sum(humans) / len(humans)
        gain += weight * (_pick(pairs) - mean)
        headroom += weight * (max(humans) - mean)
    return gain / headroom if headroom > 0 else None


def selection_gap(rows: Sequence, scores: Dict[str, float], control: str, india: str) -> Optional[float]:
    """capture(control) - capture(India-centric); positive = selection helps less on India."""
    a, b = capture(rows, scores, control), capture(rows, scores, india)
    return None if a is None or b is None else a - b


def best_of_n_by_condition(rows: Sequence, scores: Dict[str, float], seed: int = 0) -> Dict[str, Dict]:
    rng = random.Random(seed)
    result = {}
    for condition in sorted({row.condition for row in rows}):
        groups = _groups(rows, scores, condition)
        if not groups:
            continue
        n_max = min(len(pairs) for _, pairs in groups)
        random_mean = sum(sum(h for _, h in pairs) / len(pairs) for _, pairs in groups) / len(groups)
        result[condition] = {
            "prompts": len(groups),
            "n": n_max,
            "random": random_mean,
            "selected": sum(_pick(pairs) for _, pairs in groups) / len(groups),
            "oracle": sum(max(h for _, h in pairs) for _, pairs in groups) / len(groups),
            "capture": capture(rows, scores, condition),
            "curve": {n: sum(_expected_pick(pairs, n, rng) for _, pairs in groups) / len(groups) for n in range(1, n_max + 1)},
        }
    return result
