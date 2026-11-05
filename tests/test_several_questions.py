"""One `/v1/systemone` request can carry several named Questions about the same State."""

import pytest
from harness.fake_d1 import FakeD1Model, d1_choice
from harness.server import serving_d1

STATE = "Order #1182 arrived crushed and I need a replacement before Friday."
INTENT = {
    "type": "choice",
    "instructions": "Which intent does the message express?",
    "criteria": {
        "report_damage": "Says an item arrived damaged",
        "track_order": "Asks where an order is",
    },
}
URGENCY = {
    "type": "choice",
    "instructions": "How urgent is the message?",
    "criteria": {"low": "Can wait a week", "high": "Needs action within days"},
}
D1_PROBABILITIES = {
    INTENT["instructions"]: {"report_damage": 0.8, "track_order": 0.2},
    URGENCY["instructions"]: {"low": 0.3, "high": 0.7},
}


def _d1_answering_each_question():
    return FakeD1Model(lambda state, q: d1_choice(D1_PROBABILITIES[q["instructions"]]))


def _ask(client, questions: dict) -> dict:
    response = client.post(
        "/v1/systemone", json={"state": STATE, "model": "d1-2B", "questions": questions}
    )
    assert response.status_code == 200, response.text
    return response.json()["answers"]


def test_every_named_question_is_answered_under_its_own_name():
    with serving_d1(_d1_answering_each_question()) as client:
        answers = _ask(client, {"intent": INTENT, "urgency": URGENCY})

    assert answers == {
        "intent": {
            "type": "choice",
            "choice": "report_damage",
            "probabilities": pytest.approx({"report_damage": 0.8, "track_order": 0.2}),
        },
        "urgency": {
            "type": "choice",
            "choice": "high",
            "probabilities": pytest.approx({"low": 0.3, "high": 0.7}),
        },
    }


def test_an_answer_does_not_depend_on_the_other_questions_in_the_request():
    with serving_d1(_d1_answering_each_question()) as client:
        alone = _ask(client, {"urgency": URGENCY})["urgency"]
        together = _ask(client, {"intent": INTENT, "urgency": URGENCY})["urgency"]

    assert together == alone
