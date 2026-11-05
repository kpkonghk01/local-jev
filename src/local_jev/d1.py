"""The d1 Decider: `LiquidAI/d1-2B` answers natively through its own `system_one` head."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from local_jev.decider import ChoiceQuestion, Decision, Question, State
from local_jev.device import pick_device, pick_dtype

D1_MODEL_ID = "LiquidAI/d1-2B"
D1_REVISION = "652f920e544ea37f81e4b0ce89b81d4ed0da90b1"
"""Pinning the revision also pins the remote code that runs."""

log = logging.getLogger(__name__)


class D1Model(Protocol):
    """The part of the d1 model object the Decider uses (the remote code's `D1Model`)."""

    def system_one(self, state: Any, questions: dict[str, dict]) -> dict: ...


_NAME = "decision"


class D1Decider:
    def __init__(self, model: D1Model):
        self._model = model

    def decide(self, state: State, question: Question) -> Decision:
        out = self._model.system_one(state, {_NAME: _to_d1(question)})
        answer = out["answers"][_NAME]
        return Decision(distribution=answer["probabilities"])


def _to_d1(question: Question) -> dict:
    return {
        "type": question.type,
        "instructions": question.instructions,
        "criteria": dict(question.criteria),
    }


def load_d1(
    device: str = "auto",
    dtype: str = "auto",
    model_id: str = D1_MODEL_ID,
    revision: str = D1_REVISION,
) -> D1Decider:
    """The d1 Decider over the real model, downloaded on first use and warmed up."""
    import torch
    from transformers import AutoModel

    device = pick_device(device)
    dtype = pick_dtype(device, dtype)
    log.info("loading %s@%s on %s (%s)", model_id, revision, device, dtype)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, dtype=getattr(torch, dtype)
    ).to(device)
    decider = D1Decider(model)
    # The first call builds d1's runner (tokenizer, calibration, fused projections); pay for it now.
    decider.decide("Warm-up.", ChoiceQuestion("Is this a warm-up?", {"yes": None, "no": None}))
    log.info("d1 ready in %.1f s", time.perf_counter() - started)
    return decider
