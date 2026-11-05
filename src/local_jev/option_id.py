"""The LFM3 Option-ID Decider: a Chat model answers through Option-ID scoring (ADR-0002).

Each Label stands for a one-token Option ID: the letters A to Z for `choice` options, in the order
the Criteria arrive; yes/no for `noul`, pooling each with its capitalised form (LFM3 puts most of
its mass on "Yes"); the level digits 0 to 9 for `score`. One forward pass reads the next-token
likelihood of every Option ID right where the assistant's answer starts (after LFM3's empty think
block, with no leading space), and the Distribution is that likelihood normalised over the Labels.

Known risk, not mitigated here: on a 3-option probe LFM3-1.7B picked whichever of two plausible
options was listed later, at 0.85-1.00, in every one of the 6 orders (a recency bias). Averaging
over option orders would fix the order, not the model, and would cost more than one pass.
"""

from __future__ import annotations

import json
import logging
import math
import string
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

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
from local_jev.device import pick_device
from local_jev.lm_backend import LMBackend, Messages

LFM3_MODEL_ID = "LiquidAI/LFM3-1.7B"
LFM3_REVISION = "0dc4268b561499c648c7ba7c9466bf3d8d01517a"
"""Pinning the revision also pins the remote code that runs."""
LFM3_CHAT_TEMPLATE_OPTIONS: Mapping[str, Any] = MappingProxyType({"enable_thinking": False})
"""Thinking off: the template renders an empty think block and the answer starts right after it."""

log = logging.getLogger(__name__)

_LETTERS = string.ascii_uppercase
_YES_NO = {"yes": ("yes", "Yes"), "no": ("no", "No")}
"""A Chat model may say either; most of its mass is on "Yes", so each Label pools both."""

OPTION_IDS = (*_LETTERS, *(form for forms in _YES_NO.values() for form in forms), *string.digits)
"""Every Option ID form this Decider scores; each must be one token where it is scored."""


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


def load_lfm3(
    device: str = "auto",
    dtype: str = "auto",
    model_id: str = LFM3_MODEL_ID,
    revision: str = LFM3_REVISION,
    chat_template_options: Mapping[str, Any] = LFM3_CHAT_TEMPLATE_OPTIONS,
) -> OptionIDDecider:
    """The Option-ID Decider over a real Chat model (LFM3-1.7B unless told otherwise) on
    transformers, downloaded on first use, checked and warmed up."""
    from local_jev.hf_backend import load_transformers_backend

    device = pick_device(device)
    started = time.perf_counter()
    backend = load_transformers_backend(
        model_id,
        revision,
        device=device,
        dtype=dtype,
        chat_template_options=chat_template_options,
    )
    # Fail at start, not on the first request, if any Option ID is not one token where it is
    # scored (a swapped-in model's tokenizer may split one); also pays for the first pass.
    backend.next_token([{"role": "user", "content": "Warm-up."}], OPTION_IDS)
    log.info("%s ready in %.1f s", model_id, time.perf_counter() - started)
    return OptionIDDecider(backend, device=device, model_id=model_id, revision=revision)


def _option_id_forms(question: Question) -> dict[Label, tuple[str, ...]]:
    """Each Label and the forms of the Option ID that stands for it, the shown form first."""
    match question:
        case ChoiceQuestion(labels=labels):
            if len(labels) > len(_LETTERS):  # the Wire API refuses these before they get here
                raise ValueError(f"a choice of {len(labels)} options has no letter for each")
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
