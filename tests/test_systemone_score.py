"""`/v1/systemone` answers a `score` Question through the d1 Decider (fake d1 model)."""

import pytest
from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_score
from harness.server import serving_d1

ORIGINAL_ORDINAL = jevbench.task("original-ordinal-01-0")  # 4 levels


def test_score_task_gets_an_answer_jevbench_accepts():
    model = FakeD1Model.answering(d1_score([0.6718, 0.2293, 0.0700, 0.0289]))
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(ORIGINAL_ORDINAL))

    assert jevbench.rejections(ORIGINAL_ORDINAL, response.status_code, response.text) == []


def test_score_answer_is_the_most_likely_level_not_d1s_median_level():
    # Cumulative 0.30, 0.45, 0.60, 1.00: d1's median level is 2, the most likely level is 3.
    model = FakeD1Model.answering(d1_score([0.30, 0.15, 0.15, 0.40]))
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(ORIGINAL_ORDINAL))

    assert response.json()["answers"]["decision"] == {
        "type": "score",
        "score": 3,
        "probabilities": pytest.approx({"0": 0.30, "1": 0.15, "2": 0.15, "3": 0.40}),
    }


def test_score_levels_reach_d1_lowest_first():
    model = FakeD1Model.answering(d1_score([0.7, 0.1, 0.1, 0.1]))
    with serving_d1(model) as client:
        client.post("/v1/systemone", json=jevbench.request_for(ORIGINAL_ORDINAL))

    [asked] = model.asked
    assert asked["criteria"] == [
        "No function impaired; cosmetic only",
        "One user or a nonessential function impaired, with a workaround",
        "Many users blocked from a core function, no data loss",
        "Confirmed irreversible data loss or physical harm",
    ]


@pytest.mark.parametrize("levels", [2, 10])
def test_score_questions_with_2_to_10_levels_are_answered(levels):
    task = {
        **ORIGINAL_ORDINAL,
        "labels": [str(i) for i in range(levels)],
        "question": {
            **ORIGINAL_ORDINAL["question"],
            "criteria": [f"level {i}" for i in range(levels)],
        },
    }
    model = FakeD1Model.answering(d1_score([1 / levels] * levels))
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(task))

    assert jevbench.rejections(task, response.status_code, response.text) == []
