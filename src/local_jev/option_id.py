"""The LFM3 Option-ID Decider: a Chat model answers through Option-ID scoring (ADR-0002).

Each Label stands for a one-token Option ID: the letters A to Z for `choice` options, in the order
the Criteria arrive; yes/no for `noul`, pooling each with its capitalised form (LFM3 puts most of
its mass on "Yes"); the level digits 0 to 9 for `score`. One forward pass reads the next-token
likelihood of every Option ID right where the assistant's answer starts (after LFM3's empty think
block, with no leading space), and the Distribution is that likelihood normalised over the Labels.

With `explain`, a Reason is generated afterwards, from the same Situation and Question with the
most likely Label given as the answer, and asked for in the State's language (Traditional Chinese
or English), so it explains the Decision and cannot change it.

Known risk, not mitigated here: the letter position sways LFM3-1.7B. On a 3-option probe it picked
whichever of two plausible options was listed later, at 0.85-1.00, in every one of the 6 orders;
on the 231 public Tasks it leaned the other way, picking A in 57 of 139 choices where A was right in
30 (docs/feasibility-report.md). Averaging over option orders would fix the order, not the model,
and would cost more than one pass.
"""

from __future__ import annotations

import json
import logging
import math
import re
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

REASON_MAX_TOKENS = 64
"""A Reason is short: generation stops at the model's end of turn or at this many tokens.
Asked for one sentence, LFM3-1.7B wrote 27-35 tokens in English and in Chinese (about one token
per Han character), in about 1.5 s on MPS."""

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

    def decide(self, state: State, question: Question, explain: bool = False) -> Decision:
        forms = _option_id_forms(question)
        candidates = [form for label_forms in forms.values() for form in label_forms]
        # The one forward pass; everything after it is arithmetic over the Label set.
        scored = self._backend.next_token(_messages(state, question, forms), candidates)
        pooled = {
            label: _logsumexp([scored.logprobs[form] for form in label_forms])
            for label, label_forms in forms.items()
        }
        distribution = _softmax(pooled)
        usage = Usage(input_tokens=scored.prompt_tokens)
        if not explain:
            return Decision(distribution, usage)
        # The Distribution is fixed by now: the Reason explains the Decision, it cannot change it.
        most_likely = max(distribution, key=distribution.__getitem__)
        written = self._backend.generate(
            _reason_messages(state, question, forms, most_likely), REASON_MAX_TOKENS
        )
        # The Reason's prompt is read in a pass of its own, so its tokens count as read too.
        usage += Usage(written.prompt_tokens, written.output_tokens)
        return Decision(distribution, usage, reason=written.text)


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
    match question:
        case ChoiceQuestion():
            reply = "Reply with the letter of the one best option, and nothing else."
        case NoulQuestion():
            reply = "Reply with yes or no, and nothing else."
        case ScoreQuestion():
            reply = "Reply with the number of the one best level, and nothing else."
    return [{"role": "user", "content": "\n".join([*_asking(state, question, forms), "", reply])}]


def _reason_messages(
    state: State, question: Question, forms: Mapping[Label, Sequence[str]], answer: Label
) -> Messages:
    """One user turn: the same Situation and Question, the answer already chosen, and a request
    to say why, asked in the State's language."""
    lines = [*_asking(state, question, forms), "", f"The answer is: {answer}", ""]
    return [{"role": "user", "content": "\n".join([*lines, _ASK_WHY[_language(state)]])}]


def _asking(state: State, question: Question, forms: Mapping[Label, Sequence[str]]) -> list[str]:
    """The Situation, the Question and what each Label means, as lines of the prompt."""
    lines = ["Situation:", _as_text(state), "", f"Question: {question.instructions}"]
    match question:
        case ChoiceQuestion(criteria=criteria):
            lines += ["", "Options:"]
            for label, description in criteria.items():
                lines.append(f"{forms[label][0]}. {label}: {description}")
        case NoulQuestion(criteria=criteria):
            if criteria:
                lines += [f"{label}: {criteria[label]}" for label in ("yes", "no")]
        case ScoreQuestion(criteria=levels):
            lines += ["", "Levels, lowest first:"]
            lines += [f"{level}: {text}" for level, text in enumerate(levels)]
    return lines


_ASK_WHY = {
    "en": "In one short sentence, explain why this is the answer.",
    "zh-TW": "請用繁體中文（台灣用語），以一句話簡短說明為什麼是這個答案。",
}
"""Asked in the language the Reason must be in: on a Chinese State with English instructions,
LFM3-1.7B wrote English when only asked to explain."""


def _language(state: State) -> str:
    """Traditional Chinese when Han characters carry the State, English otherwise.

    Only the State's own text counts, not a JSON State's keys. A Chinese State often keeps names,
    codes, tables or program code in Latin script, which can outnumber its Chinese words; so it is
    Chinese when it has at least half as many Han characters as Latin words. An English State
    quoting a Chinese name has far fewer. (Over the 231 public Tasks and their Traditional-Chinese
    copy, this tells every State's language apart; the closest Chinese State has 0.73 Han
    characters per Latin word.) Traditional Chinese is the only Chinese local-jev serves, so any
    Han text gets a Reason in it.
    """
    text = "\n".join(_strings(state))
    han = len(_HAN.findall(text))
    return "zh-TW" if han and 2 * han >= len(_LATIN_WORD.findall(text)) else "en"


def _strings(value: Any) -> list[str]:
    """Every string value in a JSON value; a text State is one."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


_HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
"""CJK Unified Ideographs, Extension A and Compatibility Ideographs."""
_LATIN_WORD = re.compile("[A-Za-z]+")


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
