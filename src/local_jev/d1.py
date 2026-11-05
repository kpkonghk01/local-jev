"""The d1 Decider: `LiquidAI/d1-2B` answers natively through its own `system_one` head."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from local_jev.control_tokens import NO_CONTROL_TOKENS, ControlTokens
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
from local_jev.device import pick_device, pick_dtype

D1_MODEL_ID = "LiquidAI/d1-2B"
D1_REVISION = "652f920e544ea37f81e4b0ce89b81d4ed0da90b1"
"""Pinning the revision also pins the remote code that runs."""

log = logging.getLogger(__name__)


class D1Model(Protocol):
    """The part of the d1 model object the Decider uses (the remote code's `D1Model`)."""

    def system_one(self, state: Any, questions: dict[str, dict]) -> dict: ...


_QUESTION_NAME = "decision"
"""The name each Question is passed to d1 under, one per call."""


class D1Decider:
    def __init__(
        self,
        model: D1Model,
        *,
        device: str,
        model_id: str = D1_MODEL_ID,
        revision: str = D1_REVISION,
        control_tokens: ControlTokens = NO_CONTROL_TOKENS,
    ):
        """The d1 Decider over `model`, which is `model_id` at `revision`, loaded onto `device`.

        `control_tokens` are its tokenizer's: a State or Question spelling one reaches d1 as text.
        """
        self._model = model
        self._control_tokens = control_tokens
        self.runtime = Runtime(decider="d1", model=model_id, revision=revision, device=device)

    def decide(self, state: State, question: Question, explain: bool = False) -> Decision:
        """d1 cannot write text (ADR-0003), so `explain` is ignored: no Reason, and no error."""
        # d1 defuses its special tokens itself, but not `<think>` or `</think>` (not special).
        state = self._control_tokens.defuse_json(state)
        d1_question = _to_d1(question, self._control_tokens.defuse)
        # One Question per call: d1 packs a call's questions into one sequence, so each answer would
        # shift with the other Questions in the request.
        out = self._model.system_one(state, {_QUESTION_NAME: d1_question})
        answer = out["answers"][_QUESTION_NAME]
        return Decision(distribution=_distribution(question, answer), usage=_usage(out))


def _to_d1(question: Question, as_text: Callable[[str], str]) -> dict:
    """`question` in d1's own question format, each of its texts passed through `as_text`.

    The Labels are left as they are: d1 answers with them.
    """
    d1_question = {"type": question.type, "instructions": as_text(question.instructions)}
    match question:
        case ChoiceQuestion(criteria=criteria):
            d1_question["criteria"] = {label: as_text(text) for label, text in criteria.items()}
        case NoulQuestion(criteria=None):
            pass
        case NoulQuestion(criteria=criteria):
            d1_question["criteria"] = {
                "true": as_text(criteria["yes"]),
                "false": as_text(criteria["no"]),
            }
        case ScoreQuestion(criteria=criteria):
            d1_question["criteria"] = [as_text(level) for level in criteria]
    return d1_question


def _distribution(question: Question, answer: dict) -> Mapping[Label, float]:
    """d1's `answer` to `question` as a raw Distribution over the Question's Labels."""
    if isinstance(question, NoulQuestion):
        return {"yes": answer["noul"], "no": 1 - answer["noul"]}
    # choice: keyed by Label; score: keyed by level index strings, "0" to "N-1". d1's own `score`
    # field is its median level, not the most likely one, so it is not used.
    return answer["probabilities"]


def _usage(out: dict) -> Usage:
    """d1's own token counts; it reads the prompt in one pass and never generates."""
    usage = out.get("usage") or {}
    return Usage(input_tokens=int(usage.get("input_tokens") or 0))


def load_d1(
    device: str = "auto",
    dtype: str = "auto",
    model_id: str = D1_MODEL_ID,
    revision: str = D1_REVISION,
) -> D1Decider:
    """The d1 Decider over the real model, downloaded on first use and warmed up."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device(device)
    dtype = pick_dtype(device, dtype)
    log.info("loading %s@%s on %s (%s)", model_id, revision, device, dtype)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, dtype=getattr(torch, dtype)
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    decider = D1Decider(
        model,
        device=device,
        model_id=model_id,
        revision=revision,
        control_tokens=ControlTokens.of(tokenizer),
    )
    # The first call builds d1's runner (tokenizer, calibration, fused projections); pay for it now.
    decider.decide(
        "Warm-up.", ChoiceQuestion("Is this a warm-up?", {"yes": "It is", "no": "It is not"})
    )
    log.info("d1 ready in %.1f s", time.perf_counter() - started)
    return decider
