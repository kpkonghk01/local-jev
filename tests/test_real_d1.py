"""The real `LiquidAI/d1-2B` behind `/v1/systemone` (run with `uv run pytest -m real_model`).

Downloads the model at its pinned revision on first use (about 3.6 GB into the Hugging Face cache).
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


@pytest.mark.parametrize("task_id", ["easy-intent-00", "hard-opus-b-ambiguous-10"])
def test_real_d1_answers_a_public_choice_task_validly(d1_server, task_id):
    task = jevbench.task(task_id)
    response = d1_server.post("/v1/systemone", json=jevbench.request_for(task, "d1-2B"))

    print(task_id, "expected", task["expected"], "got", response.json())
    assert jevbench.rejections(task, response.status_code, response.text) == []
