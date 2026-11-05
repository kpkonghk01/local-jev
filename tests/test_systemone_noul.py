"""`/v1/systemone` answers a `noul` Question through the d1 Decider (fake d1 model)."""

import pytest
from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_noul
from harness.server import serving_d1

EASY_FACT = jevbench.task("easy-fact-00")


def test_noul_task_gets_an_answer_jevbench_accepts():
    with serving_d1(FakeD1Model.answering(d1_noul(0.99936))) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_FACT))

    assert jevbench.rejections(EASY_FACT, response.status_code, response.text) == []


def test_noul_answer_is_d1s_probability_of_yes():
    with serving_d1(FakeD1Model.answering(d1_noul(0.0731))) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_FACT))

    assert response.json()["answers"]["decision"] == {"type": "noul", "noul": pytest.approx(0.0731)}


def test_noul_criteria_tell_d1_what_yes_and_no_mean():
    model = FakeD1Model.answering(d1_noul(0.9))
    with serving_d1(model) as client:
        client.post("/v1/systemone", json=jevbench.request_for(EASY_FACT))

    [asked] = model.asked
    assert asked["criteria"] == {
        "true": "The text states that this is so",
        "false": "The text states that this is not so",
    }


@pytest.mark.parametrize("p_yes", [1.2, -0.1, float("nan")], ids=["above-1", "below-0", "nan"])
def test_a_d1_noul_that_is_not_a_probability_is_a_server_error(p_yes):
    with serving_d1(FakeD1Model.answering(d1_noul(p_yes))) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_FACT))

    assert response.status_code == 500
    assert "answers" not in response.json()


def test_noul_question_without_criteria_is_answered():
    request = jevbench.request_for(EASY_FACT)
    del request["questions"]["decision"]["criteria"]
    model = FakeD1Model.answering(d1_noul(0.8))
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=request)

    assert jevbench.rejections(EASY_FACT, response.status_code, response.text) == []
    assert "criteria" not in model.asked[0]
