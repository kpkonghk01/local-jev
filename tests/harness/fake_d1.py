"""A fake d1 model: the same `system_one` call as `LiquidAI/d1-2B`, answering with canned answers.

Answers use d1's own shape, as observed from the real model (MPS bf16 and CPU fp32):

    choice: {"type": "choice", "choice": "billing", "confidence": 0.7729,
             "probabilities": {"billing": 0.7729, "shipping": 0.1732, "technical": 0.0539}}
    noul:   {"type": "noul", "noul": 0.99936}
    score:  {"type": "score", "score": 1.0, "confidence": 0.6718,
             "probabilities": {"0": 0.2293, "1": 0.6718, "2": 0.0989},
             "legend": {"0": "low", "1": "medium", "2": "high"}}

d1's `score` is a float and the median level (the first whose cumulative probability reaches 0.5),
not the most likely one.

Like the real model, the fake refuses questions d1 itself would refuse, with d1's messages. Also
like the real model, questions packed into one call are not answered independently: d1 reads them
in one sequence after the State, so each also hears the earlier ones (observed: 0.7729 alone,
0.8059 packed behind other questions).

The real model's prompt code defuses its tokenizer's special tokens (`<|im_end|>`, `<|im_start|>`)
in a State or question, but not `<think>` or `</think>`, which its tokenizer reads as control tokens
without marking them special (observed: a State spelling `<think></think>` put two of each in d1's
prompt). The fake refuses any text holding one of `CONTROL_TOKENS`, which no caller's text may
become.

The fake keeps every question it was asked, in d1's own question format, in `asked`.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

Answer = Callable[[Any, dict], dict]

CONTROL_TOKENS = ("<think>", "</think>")
"""The pinned d1 tokenizer's control tokens that its own prompt code leaves as they are."""


class FakeD1Model:
    def __init__(
        self, answer: Answer, input_tokens: int | None = 73, seconds_per_call: float = 0.0
    ):
        """`input_tokens=None` leaves `usage` out of the model's output altogether."""
        self._answer = answer
        self._input_tokens = input_tokens
        self._seconds_per_call = seconds_per_call
        self._running = threading.Lock()
        self.asked: list[dict] = []

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
        _check_text("the State", state)
        time.sleep(self._seconds_per_call)
        answers = {}
        for earlier, (name, question) in enumerate(questions.items()):
            _check_question(name, question)
            self.asked.append(copy.deepcopy(question))
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


def d1_noul(p_yes: float) -> dict:
    """A d1 `noul` answer: P(yes) alone, with no probabilities dict."""
    return {"type": "noul", "noul": p_yes}


def d1_score(probabilities: list[float], levels: list[str] | None = None) -> dict:
    """A d1 `score` answer over levels 0..N-1, with d1's median `score`, `confidence`, `legend`."""
    cumulative = 0.0
    median = len(probabilities) - 1
    for level, p in enumerate(probabilities):
        cumulative += p
        if cumulative >= 0.5:
            median = level
            break
    levels = levels or [f"level {i}" for i in range(len(probabilities))]
    return {
        "type": "score",
        "score": float(median),
        "confidence": max(probabilities),
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        "legend": {str(i): text for i, text in enumerate(levels)},
    }


def _heard_after(earlier: int, answer: dict) -> dict:
    """`answer` as d1 gives it when `earlier` questions precede it in the same call."""
    if not earlier:
        return answer  # exactly the canned answer, even one that is not a Distribution
    shift = 0.03 * earlier
    if probabilities := answer.get("probabilities"):
        top = max(probabilities, key=probabilities.__getitem__)
        probabilities[top] += shift
        total = sum(probabilities.values())
        answer["probabilities"] = {label: p / total for label, p in probabilities.items()}
    if "noul" in answer:
        answer["noul"] = min(1.0, answer["noul"] + shift)
    return answer


_TYPES = ("noul", "choice", "score", "text", "number", "date", "box", "boxes")


def _check_question(name: str, q: dict) -> None:
    """d1's own question checks (prompt.check_question) for the noul, choice and score types."""
    kind, criteria = q.get("type"), q.get("criteria")
    if kind not in _TYPES:
        raise ValueError(f'question {name!r}: "type" is one of {", ".join(_TYPES)}, not {kind!r}')
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"question {name!r}: the fake d1 model does not answer {kind!r}")
    instructions = q.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f'question {name!r}: "instructions" is the question as text')
    if kind == "choice":
        names_options = isinstance(criteria, Mapping) and all(isinstance(k, str) for k in criteria)
        if not (criteria and names_options):
            raise ValueError(f'question {name!r}: a choice\'s "criteria" names the options')
    if kind == "score":
        levels = isinstance(criteria, list | tuple) and all(isinstance(c, str) for c in criteria)
        if not (levels and 2 <= len(criteria) <= 10):
            raise ValueError(f'question {name!r}: a score\'s "criteria" lists 2 to 10 levels')
    if kind == "noul" and criteria is not None and not isinstance(criteria, Mapping):
        raise ValueError(f'question {name!r}: a noul\'s "criteria", when given, is a mapping')
    if q.get("multiple"):
        raise ValueError(
            f"question {name!r}: the fake d1 model does not answer multi-select choice"
        )
    _check_text(f"question {name!r}", q)


def _check_text(where: str, value: Any) -> None:
    """Refuses `value` (a JSON value) if any string in it would be read as a control token."""
    text = json.dumps(value, ensure_ascii=False)
    if read := [token for token in CONTROL_TOKENS if token in text]:
        raise ValueError(f"{where} holds {read}, which d1 would read as control tokens")
