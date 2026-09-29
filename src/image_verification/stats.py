"""Dependency-free statistics for the evaluator-reliability study.

Everything here is pure Python so the analysis can run on a laptop without
torch or scipy. Conventions:

- ``labels`` are booleans where ``True`` means the human panel judged the image
  acceptable ("good") and ``False`` means it should have been rejected ("bad").
- ``scores`` are evaluator outputs where higher means "more acceptable". Only
  their order matters for AUROC and Spearman.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


def rankdata(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based), matching scipy.stats.rankdata(method="average")."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Probability that a random good image outscores a random bad one (ties count 1/2).

    Returns ``None`` when either class is empty, because AUROC is undefined.
    """
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mean_x, mean_y = sum(x) / n, sum(y) / n
    sxx = sum((a - mean_x) ** 2 for a in x)
    syy = sum((b - mean_y) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman's rho with average ranks for ties."""
    if len(x) != len(y):
        raise ValueError("spearman() needs sequences of equal length.")
    return pearson(rankdata(x), rankdata(y))


def threshold_at_true_accept_rate(scores: Sequence[float], labels: Sequence[bool], target: float) -> Optional[float]:
    """Smallest threshold that accepts at least ``target`` of the good images.

    An image is accepted when ``score >= threshold``. This mirrors how a
    practitioner calibrates an evaluator: pick the cut-off that keeps, say, 80%
    of good images, then see how many bad images sneak through.
    """
    good = sorted((s for s, label in zip(scores, labels) if label), reverse=True)
    if not good:
        return None
    keep = max(1, math.ceil(target * len(good)))
    return good[keep - 1]


def false_accept_rate(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> Optional[float]:
    """Share of human-rejected images that the evaluator accepts at ``threshold``."""
    bad = [s for s, label in zip(scores, labels) if not label]
    if not bad:
        return None
    return sum(1 for s in bad if s >= threshold) / len(bad)


def false_reject_rate(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> Optional[float]:
    """Share of human-accepted images that the evaluator rejects at ``threshold``."""
    good = [s for s, label in zip(scores, labels) if label]
    if not good:
        return None
    return sum(1 for s in good if s < threshold) / len(good)


def krippendorff_alpha_nominal(units: Iterable[Sequence[Hashable]]) -> Optional[float]:
    """Krippendorff's alpha for nominal data.

    ``units`` holds, for each rated item, the labels given by its annotators
    (missing ratings simply omitted). Items with fewer than two ratings are
    ignored, as in the standard coincidence-matrix formulation.
    """
    coincidence: Dict[Tuple[Hashable, Hashable], float] = defaultdict(float)
    for values in units:
        values = list(values)
        m = len(values)
        if m < 2:
            continue
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                if i != j:
                    coincidence[(a, b)] += 1.0 / (m - 1)
    if not coincidence:
        return None
    marginals: Dict[Hashable, float] = defaultdict(float)
    for (a, _), weight in coincidence.items():
        marginals[a] += weight
    total = sum(marginals.values())
    if total <= 1:
        return None
    observed = sum(w for (a, b), w in coincidence.items() if a != b) / total
    expected = sum(marginals[a] * marginals[b] for a in marginals for b in marginals if a != b) / (total * (total - 1))
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, ``q`` in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile() of empty sequence")
    position = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_bootstrap(
    clusters: Sequence[Hashable],
    statistic: Callable[[List[int]], Optional[float]],
    samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Optional[float]]:
    """Percentile bootstrap that resamples whole clusters with replacement.

    Images generated from the same prompt (or the same matched prompt pair)
    are correlated, so resampling individual images would understate the
    uncertainty. ``statistic`` receives the list of row indices in the
    resample and returns a number (or ``None`` when undefined for that draw).

    The returned ``p_two_sided`` is the bootstrap sign test for the statistic
    being different from zero; it is only meaningful for difference statistics.
    """
    index_by_cluster: Dict[Hashable, List[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        index_by_cluster[cluster].append(index)
    keys = list(index_by_cluster)
    point = statistic(list(range(len(clusters))))
    if samples <= 0 or not keys:
        return {"estimate": point, "ci_low": None, "ci_high": None, "p_two_sided": None, "valid_samples": 0}
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(samples):
        rows: List[int] = []
        for _ in keys:
            rows.extend(index_by_cluster[keys[rng.randrange(len(keys))]])
        value = statistic(rows)
        if value is not None and not math.isnan(value):
            draws.append(value)
    if not draws:
        return {"estimate": point, "ci_low": None, "ci_high": None, "p_two_sided": None, "valid_samples": 0}
    below = sum(1 for value in draws if value <= 0) / len(draws)
    above = sum(1 for value in draws if value >= 0) / len(draws)
    return {
        "estimate": point,
        "ci_low": percentile(draws, 100 * alpha / 2),
        "ci_high": percentile(draws, 100 * (1 - alpha / 2)),
        "p_two_sided": min(1.0, 2 * min(below, above)),
        "valid_samples": len(draws),
    }


def holm_adjust(p_values: Dict[Hashable, Optional[float]]) -> Dict[Hashable, Optional[float]]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    present = sorted(((p, key) for key, p in p_values.items() if p is not None), key=lambda item: item[0])
    adjusted: Dict[Hashable, Optional[float]] = {key: None for key in p_values}
    running = 0.0
    m = len(present)
    for rank, (p, key) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted
