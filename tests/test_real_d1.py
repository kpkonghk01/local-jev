"""The real `LiquidAI/d1-2B` behind `/v1/systemone` (run with `uv run pytest -m real_model`).

Downloads the model at its pinned revision on first use (about 3.6 GB into the Hugging Face cache).

Observed 2026-11-05 (MPS bf16, transformers 5.22.1, torch 2.11.0), matching d1's model card:

- noul (easy-fact-00): `{"type": "noul", "noul": 0.99965}`, a float P(yes), no probabilities dict.
- score (original-ordinal-01-0): `{"type": "score", "score": 0.0, "confidence": 0.998,
  "probabilities": {"0": 0.998, "1": 0.00084, "2": 0.00073, "3": 0.00045}, "legend": {...}}`.
  Probability keys are the level index strings "0".."N-1"; `score` is a float median level, so the
  wire `score` is taken from the most likely level instead.
"""

import pytest
from harness import jevbench
from harness.server import serving

from local_jev.d1 import load_d1

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def d1_server():
    with serving(load_d1, ready_within=900) as client:
        yield client


@pytest.mark.parametrize(
    "task_id",
    [
        "easy-intent-00",
        "hard-opus-b-ambiguous-10",
        "easy-fact-00",  # noul
        "original-ordinal-01-0",  # score, 4 levels
    ],
)
def test_real_d1_answers_a_public_task_validly(d1_server, task_id):
    task = jevbench.task(task_id)
    response = d1_server.post("/v1/systemone", json=jevbench.request_for(task, "d1-2B"))

    print(task_id, "expected", task["expected"], "got", response.json())
    assert jevbench.rejections(task, response.status_code, response.text) == []
