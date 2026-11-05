"""A fake d1 model: the same `system_one` call as `LiquidAI/d1-2B`, answering with canned answers.

Answers use d1's own shape, as observed from the real model (MPS bf16 and CPU fp32):

    choice: {"type": "choice", "choice": "billing", "confidence": 0.7729,
             "probabilities": {"billing": 0.7729, "shipping": 0.1732, "technical": 0.0539}}

Like the real model, the fake refuses questions d1 itself would refuse, with d1's messages.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

Answer = Callable[[Any, dict], dict]


class FakeD1Model:
    def __init__(self, answer: Answer, input_tokens: int = 73):
        self._answer = answer
        self._input_tokens = input_tokens

    @classmethod
    def answering(cls, answer: dict) -> FakeD1Model:
        """A model that gives the same answer to every question."""
        return cls(lambda state, question: answer)

    def system_one(self, state, questions, media=None, score_readout="median", set_size=None):
        answers = {}
        for name, question in questions.items():
            _check_question(name, question)
            answers[name] = copy.deepcopy(self._answer(state, question))
        usage = {"input_tokens": self._input_tokens, "output_tokens": 0}
        return {"answers": answers, "usage": usage}


def d1_choice(probabilities: Mapping[str, float]) -> dict:
    """A d1 `choice` answer over `probabilities`, with d1's `choice` and `confidence` fields."""
    choice = max(probabilities, key=probabilities.__getitem__)
    return {
        "type": "choice",
        "choice": choice,
        "confidence": probabilities[choice],
        "probabilities": dict(probabilities),
    }


def _check_question(name: str, q: dict) -> None:
    """The subset of d1's own question checks (prompt.check_question) for the types in use."""
    if q.get("type") != "choice":
        raise ValueError(f"question {name!r}: the fake d1 model only answers choice, not {q!r}")
    instructions = q.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f'question {name!r}: "instructions" is the question as text')
    criteria = q.get("criteria")
    names_options = isinstance(criteria, Mapping) and all(isinstance(k, str) for k in criteria)
    if not (criteria and names_options):
        raise ValueError(f'question {name!r}: a choice\'s "criteria" names the options')
