"""The LFM3 Option-ID Decider: a Chat model answers through Option-ID scoring (ADR-0002)."""

from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping, Sequence

from local_jev.decider import (
    ChoiceQuestion,
    Decision,
    Label,
    NoulQuestion,
    Question,
    Runtime,
    ScoreQuestion,
    State,
    Usage,
)
from local_jev.lm_backend import LMBackend, Messages

LFM3_MODEL_ID = "LiquidAI/LFM3-1.7B"
LFM3_REVISION = "0dc4268b561499c648c7ba7c9466bf3d8d01517a"
"""Pinning the revision also pins the remote code that runs."""

_LETTERS = string.ascii_uppercase
_YES_NO = {"yes": ("yes", "Yes"), "no": ("no", "No")}
"""A Chat model may say either; most of its mass is on "Yes", so each Label pools both."""


class OptionIDDecider:
    def __init__(
        self,
        backend: LMBackend,
        *,
        device: str,
        model_id: str = LFM3_MODEL_ID,
        revision: str = LFM3_REVISION,
    ):
        """The Option-ID Decider over `backend`, which runs `model_id` at `revision` on `device`."""
        self._backend = backend
        self.runtime = Runtime(decider="lfm3", model=model_id, revision=revision, device=device)

    def decide(self, state: State, question: Question) -> Decision:
        forms = _option_id_forms(question)
        candidates = [form for label_forms in forms.values() for form in label_forms]
        # The one forward pass; everything after it is arithmetic over the Label set.
        scored = self._backend.next_token(_messages(state, question, forms), candidates)
        pooled = {
            label: _logsumexp([scored.logprobs[form] for form in label_forms])
            for label, label_forms in forms.items()
        }
        return Decision(
            distribution=_softmax(pooled), usage=Usage(input_tokens=scored.prompt_tokens)
        )


def _option_id_forms(question: Question) -> dict[Label, tuple[str, ...]]:
    """Each Label and the forms of the Option ID that stands for it, the shown form first."""
    match question:
        case ChoiceQuestion(labels=labels):
            return {label: (letter,) for letter, label in zip(_LETTERS, labels, strict=False)}
        case NoulQuestion():
            return dict(_YES_NO)
        case ScoreQuestion(labels=labels):
            return {label: (label,) for label in labels}  # "0".."9", each one digit


def _messages(state: State, question: Question, forms: Mapping[Label, Sequence[str]]) -> Messages:
    """One user turn; the backend's chat template turns it into the prompt, ending where the
    assistant's answer starts, so the Option ID is the very next token."""
    lines = ["Situation:", _as_text(state), "", f"Question: {question.instructions}"]
    match question:
        case ChoiceQuestion(criteria=criteria):
            lines += ["", "Options:"]
            for label, description in criteria.items():
                shown = f"{label}: {description}" if description else label
                lines.append(f"{forms[label][0]}. {shown}")
            lines += ["", "Reply with the letter of the one best option, and nothing else."]
        case NoulQuestion(criteria=criteria):
            if criteria:
                lines += [f"{label}: {criteria[label]}" for label in ("yes", "no")]
            lines += ["", "Reply with yes or no, and nothing else."]
        case ScoreQuestion(criteria=levels):
            lines += ["", "Levels, lowest first:"]
            lines += [f"{level}: {text}" for level, text in enumerate(levels)]
            lines += ["", "Reply with the number of the one best level, and nothing else."]
    return [{"role": "user", "content": "\n".join(lines)}]


def _as_text(state: State) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def _logsumexp(logprobs: Sequence[float]) -> float:
    top = max(logprobs)
    if top == -math.inf:
        return top
    return top + math.log(sum(math.exp(lp - top) for lp in logprobs))


def _softmax(logprobs: Mapping[Label, float]) -> dict[Label, float]:
    top = max(logprobs.values())
    weights = {label: math.exp(lp - top) for label, lp in logprobs.items()}
    total = sum(weights.values())
    return {label: w / total for label, w in weights.items()}
