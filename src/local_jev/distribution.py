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
    if sum(raw.values()) <= 0:
        raise NotADistribution("every probability is 0")
    return normalised({label: raw[label] for label in labels})


def normalised(weights: Mapping[Label, float]) -> dict[Label, float]:
    """Non-negative `weights`, not all 0, rescaled to sum to 1."""
    total = sum(weights.values())
    return {label: w / total for label, w in weights.items()}


def softmax(logits: Mapping[Label, float]) -> dict[Label, float]:
    """The Distribution with these log-probabilities up to a constant; -inf is probability 0."""
    top = max(logits.values())
    return normalised({label: math.exp(z - top) for label, z in logits.items()})


def argmax(distribution: Mapping[Label, float]) -> Label:
    """The most likely Label; a tie goes to the alphabetically first, as jevbench breaks it."""
    return min(distribution, key=lambda label: (-distribution[label], label))
