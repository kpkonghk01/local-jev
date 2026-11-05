"""The real `LiquidAI/LFM3-1.7B`, behind the LM backend seam and `/v1/systemone`.

Run with `uv run pytest -m real_model`. Downloads the model at its pinned revision on first use
(about 3.2 GB into the Hugging Face cache).

Observed 2026-11-05 (MPS bf16, transformers 5.22.1, torch 2.11.0):

- Every Option ID (A-Z, yes/Yes/no/No, 0-9) is one token right after `<think></think>`;
  `" 0"` and `"YES"` are two.
- easy-intent-00: track_order 0.99999 (right). hard-opus-b-ambiguous-10: ask_user_which_alex 0.81
  (right). easy-fact-00: P(yes) 0.99999 (right). original-ordinal-01-0: level 1 at 0.68, level 0
  at 0.28 (expected 0). Near-certain answers are typical: calibration has work to do.
- About 115-240 ms per Decision for 80-530 prompt tokens; loading takes about 8 s from the cache.
"""

import math

import pytest
from harness import jevbench
from harness.server import serving_option_id

from local_jev.device import pick_device
from local_jev.hf_backend import load_transformers_backend
from local_jev.lm_backend import NotOneToken
from local_jev.option_id import (
    LFM3_CHAT_TEMPLATE_OPTIONS,
    LFM3_MODEL_ID,
    LFM3_REVISION,
    OPTION_IDS,
)

pytestmark = pytest.mark.real_model

DEVICE = pick_device()
ASK = [{"role": "user", "content": "Is the sky blue on a clear day? Reply with yes or no."}]


@pytest.fixture(scope="module")
def backend():
    return load_transformers_backend(
        LFM3_MODEL_ID,
        LFM3_REVISION,
        device=DEVICE,
        chat_template_options=LFM3_CHAT_TEMPLATE_OPTIONS,
    )


@pytest.fixture(scope="module")
def lfm3_server(backend):
    with serving_option_id(backend, device=DEVICE) as client:
        yield client


def test_every_option_id_is_one_token_at_the_scored_position(backend):
    scored = backend.next_token(ASK, OPTION_IDS)

    assert set(scored.logprobs) == set(OPTION_IDS)
    assert all(math.isfinite(lp) and lp <= 0 for lp in scored.logprobs.values())
    assert sum(math.exp(lp) for lp in scored.logprobs.values()) <= 1 + 1e-3
    assert scored.logprobs["Yes"] > scored.logprobs["No"]


@pytest.mark.parametrize("candidate", [" 0", "YES"])  # two tokens each in LFM3's tokenizer
def test_a_candidate_of_more_than_one_token_is_refused(backend, candidate):
    with pytest.raises(NotOneToken):
        backend.next_token(ASK, ["Yes", candidate])


def test_generation_stops_within_its_bound(backend):
    count = [{"role": "user", "content": "Count from 1 to 100, separated by commas."}]

    generated = backend.generate(count, max_new_tokens=8)

    assert 0 < generated.output_tokens <= 8
    assert generated.text.startswith("1")
    assert generated.prompt_tokens > 0


def test_generation_ends_with_the_models_turn_as_plain_text(backend):
    ok = [{"role": "user", "content": "Reply with the single word OK and nothing else."}]

    generated = backend.generate(ok, max_new_tokens=32)

    assert generated.output_tokens < 32
    assert "OK" in generated.text
    assert "<|" not in generated.text


@pytest.mark.parametrize(
    "task_id",
    [
        "easy-intent-00",  # choice, 5 options
        "hard-opus-b-ambiguous-10",  # choice, 3 options, long State
        "easy-fact-00",  # noul
        "original-ordinal-01-0",  # score, 4 levels
    ],
)
def test_real_lfm3_answers_a_public_task_validly(lfm3_server, task_id):
    task = jevbench.task(task_id)
    response = lfm3_server.post("/v1/systemone", json=jevbench.request_for(task, "LFM3-1.7B"))

    print(task_id, "expected", task["expected"], "got", response.json())
    assert jevbench.rejections(task, response.status_code, response.text) == []


def test_real_lfm3_answers_26_options_and_10_levels_validly(lfm3_server):
    labels = [f"product_{i:02}" for i in range(1, 27)]
    choice = {
        "id": "26-options",
        "labels": labels,
        "question": {
            "type": "choice",
            "instructions": "Which product is the message about?",
            "criteria": {label: f"Product number {label[-2:]}" for label in labels},
        },
        "state": "Product number 26 stopped charging after a week.",
    }
    score = {
        "id": "10-levels",
        "labels": [str(i) for i in range(10)],
        "question": {
            "type": "score",
            "instructions": "How satisfied is the customer, from 0 (not at all) to 9 (fully)?",
            "criteria": [f"Satisfaction {i} of 9" for i in range(10)],
        },
        "state": "Absolutely perfect, everything I hoped for!",
    }

    for task in (choice, score):
        response = lfm3_server.post("/v1/systemone", json=jevbench.request_for(task))
        print(task["id"], response.json())
        assert jevbench.rejections(task, response.status_code, response.text) == []
