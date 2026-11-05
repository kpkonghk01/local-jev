"""A fake d1 model: the same `system_one` call as `LiquidAI/d1-2B`, answering with canned answers.

Answers use d1's own shape, as observed from the real model (MPS bf16 and CPU fp32):

    choice: {"type": "choice", "choice": "billing", "confidence": 0.7729,
             "probabilities": {"billing": 0.7729, "shipping": 0.1732, "technical": 0.0539}}

Like the real model, the fake refuses questions d1 itself would refuse, with d1's messages. Also
like the real model, questions packed into one call are not answered independently: d1 reads them
in one sequence after the State, so each also hears the earlier ones (observed: 0.7729 alone,
0.8059 packed behind other questions).
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

Answer = Callable[[Any, dict], dict]


class FakeD1Model:
    def __init__(
        self, answer: Answer, input_tokens: int | None = 73, seconds_per_call: float = 0.0
    ):
        """`input_tokens=None` leaves `usage` out of the model's output altogether."""
        self._answer = answer
        self._input_tokens = input_tokens
        self._seconds_per_call = seconds_per_call
        self._running = threading.Lock()

    @classmethod
    def answering(cls, answer: dict, **kwargs) -> FakeD1Model:
        """A model that gives the same answer to every question."""
        return cls(lambda state, question: answer, **kwargs)

    def system_one(self, state, questions, media=None, score_readout="median", set_size=None):
        """Like any model holding one set of weights and caches, it cannot run two calls at once."""
        if not self._running.acquire(blocking=False):
            raise RuntimeError("the fake d1 model was called while another call was running")
        try:
            return self._system_one(state, questions)
        finally:
            self._running.release()

    def _system_one(self, state, questions) -> dict:
        time.sleep(self._seconds_per_call)
        answers = {}
        for earlier, (name, question) in enumerate(questions.items()):
            _check_question(name, question)
            answer = copy.deepcopy(self._answer(state, question))
            answers[name] = _heard_after(earlier, answer)
        if self._input_tokens is None:
            return {"answers": answers}
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


def _heard_after(earlier: int, answer: dict) -> dict:
    """`answer` as d1 gives it when `earlier` questions precede it in the same call."""
    shift = 0.03 * earlier
    if probabilities := answer.get("probabilities"):
        top = max(probabilities, key=probabilities.__getitem__)
        probabilities[top] += shift
        total = sum(probabilities.values())
        answer["probabilities"] = {label: p / total for label, p in probabilities.items()}
    if "noul" in answer:
        answer["noul"] = min(1.0, answer["noul"] + shift)
    return answer


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
