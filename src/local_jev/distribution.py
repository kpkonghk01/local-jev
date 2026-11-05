"""Distributions: a probability for every Label in a Label set, summing to 1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from local_jev.decider import Label


class NotADistribution(ValueError):
    """A Decider answered with something that is not a Distribution over the Question's Labels."""


def over_labels(raw: Mapping[Label, float], labels: Sequence[Label]) -> dict[Label, float]:
    """`raw` rescaled into a Distribution over exactly `labels`, in Label order."""
    if set(raw) != set(labels):
        raise NotADistribution(f"it covers {sorted(raw)}, not the Label set {sorted(labels)}")
    bad = {label: p for label, p in raw.items() if not (math.isfinite(p) and p >= 0)}
    if bad:
        raise NotADistribution(f"these values are not probabilities: {bad}")
    total = sum(raw[label] for label in labels)
    if total <= 0:
        raise NotADistribution("every probability is 0")
    return {label: raw[label] / total for label in labels}
