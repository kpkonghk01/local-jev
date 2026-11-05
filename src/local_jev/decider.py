"""The Decider seam (ADR-0003): a State and a Question in, a Decision out.

The Wire API depends only on this module. Each Decider (d1's native head, Option-ID scoring over a
Chat model) is one implementation of `Decider`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

type State = Any
"""The situation a Decision is about: plain text, or any JSON value."""

type Label = str


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick exactly one of several named options; `criteria` says what each Label means."""

    type: ClassVar[Literal["choice"]] = "choice"
    instructions: str
    criteria: Mapping[Label, str | None]

    @property
    def labels(self) -> tuple[Label, ...]:
        return tuple(self.criteria)


type Question = ChoiceQuestion


@dataclass(frozen=True)
class Usage:
    """Tokens a Decider read and generated for a Decision; output is 0 when nothing is generated."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class Decision:
    """One answer to one Question: the Decider's raw Distribution over the Question's Label set."""

    distribution: Mapping[Label, float]
    usage: Usage = Usage()


@dataclass(frozen=True)
class Runtime:
    """What a Decider runs: its name, its model at a pinned revision, its device (cuda/mps/cpu)."""

    decider: str
    model: str
    revision: str
    device: str


class Decider(Protocol):
    @property
    def runtime(self) -> Runtime: ...

    def decide(self, state: State, question: Question) -> Decision: ...
