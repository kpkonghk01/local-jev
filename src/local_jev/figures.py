"""How the command line and the comparison report write a figure; `n/a` when it is missing."""

from __future__ import annotations


def percent(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def number(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"
