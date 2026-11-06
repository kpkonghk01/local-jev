"""`/v1/systemone` refuses unsupported and malformed requests with a 4xx and a clear message.

Refusals are 422: jevbench counts a 422 as a wrong answer but, unlike other failures, not towards
the three consecutive failures that stop its run (jevbench/runner.py).
"""

import pytest
from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_choice
from harness.server import serving_d1

EASY_INTENT = jevbench.task("easy-intent-00")


def _refusal(request: dict) -> str:
    """The message the server refuses `request` with; the request never reaches the model."""
    model = FakeD1Model.answering(d1_choice({"a": 0.5, "b": 0.5}))
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=request)

    assert response.status_code == 422, response.text
    assert model.asked == []
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    return detail


def _asking(question: dict) -> dict:
    return {"state": "Where is my package?", "questions": {"decision": question}}


def test_one_bad_question_refuses_the_whole_request():
    request = jevbench.request_for(EASY_INTENT)
    request["questions"]["order_number"] = {"type": "text", "instructions": "The order number?"}

    detail = _refusal(request)

    assert detail.startswith("questions.order_number: ")


@pytest.mark.parametrize("question_type", ["text", "number", "date", "box", "boxes"])
def test_unsupported_question_types_are_refused_by_name(question_type):
    detail = _refusal(_asking({"type": question_type, "instructions": "What is the order number?"}))

    assert f"'{question_type}'" in detail
    assert "not supported" in detail


def test_multi_select_choice_is_refused():
    question = {**jevbench.request_for(EASY_INTENT)["questions"]["decision"], "multiple": True}

    detail = _refusal(_asking(question))

    assert "multi-select choice" in detail


def test_a_body_that_is_not_json_is_refused():
    with serving_d1(FakeD1Model.answering(d1_choice({"a": 0.5, "b": 0.5}))) as client:
        response = client.post(
            "/v1/systemone",
            content=b'{"state": "x", ',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("the request body is not valid JSON")


@pytest.mark.parametrize("state", [None, "", "   "], ids=["null", "empty", "blank"])
def test_a_request_without_a_state_is_refused(state):
    request = jevbench.request_for(EASY_INTENT)

    missing = _refusal({k: v for k, v in request.items() if k != "state"})
    empty = _refusal({**request, "state": state})

    assert missing.startswith("state: ")
    assert empty.startswith("state: ")


@pytest.mark.parametrize("questions", [{}, None, ["decision"]], ids=["empty", "null", "list"])
def test_a_request_without_questions_is_refused(questions):
    request = jevbench.request_for(EASY_INTENT)

    missing = _refusal({k: v for k, v in request.items() if k != "questions"})
    empty = _refusal({**request, "questions": questions})

    assert missing.startswith("questions: ")
    assert empty.startswith("questions: ")


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param(None, id="null"),
        pytest.param({}, id="empty"),
        pytest.param(["Wants to track an order", "Wants to cancel it"], id="levels"),
        pytest.param({"track_order": 3}, id="non-text-description"),
        pytest.param({"track_order": None, "cancel_order": None}, id="labels-without-criteria"),
        pytest.param(
            {"track_order": "Asks where it is", "cancel_order": " "}, id="blank-criterion"
        ),
    ],
)
def test_a_choice_without_criteria_naming_its_options_is_refused(criteria):
    question = jevbench.request_for(EASY_INTENT)["questions"]["decision"]

    missing = _refusal(_asking({k: v for k, v in question.items() if k != "criteria"}))
    unusable = _refusal(_asking({**question, "criteria": criteria}))

    assert missing.startswith("questions.decision: ")
    assert unusable.startswith("questions.decision: ")
    assert "criteria" in missing and "criteria" in unusable


def test_a_choice_of_more_than_26_options_is_refused_whatever_the_decider():
    question = jevbench.request_for(EASY_INTENT)["questions"]["decision"]
    criteria = {f"option_{i}": f"Option {i}" for i in range(27)}

    detail = _refusal(_asking({**question, "criteria": criteria}))

    assert detail.startswith("questions.decision: ")
    assert "at most 26 options" in detail


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param([], id="0-levels"),
        pytest.param(["Only level"], id="1-level"),
        pytest.param([f"level {i}" for i in range(11)], id="11-levels"),
        pytest.param({"0": "low", "1": "high"}, id="mapping"),
        pytest.param(["low", 2], id="non-text-level"),
        pytest.param(None, id="null"),
    ],
)
def test_a_score_without_2_to_10_levels_is_refused(criteria):
    question = jevbench.request_for(jevbench.task("original-ordinal-01-0"))["questions"]["decision"]

    detail = _refusal(_asking({**question, "criteria": criteria}))

    assert detail.startswith("questions.decision: ")
    assert "2 to 10 levels" in detail


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param({"yes": "It has shipped", "no": "It has not"}, id="keyed-by-label"),
        pytest.param({"true": "It has shipped"}, id="only-true"),
        pytest.param({"true": "It has shipped", "false": "It has not", "maybe": "?"}, id="extra"),
        pytest.param({"true": "It has shipped", "false": None}, id="non-text"),
        pytest.param(["It has not", "It has shipped"], id="list"),
    ],
)
def test_a_noul_whose_criteria_are_not_keyed_true_and_false_is_refused(criteria):
    question = jevbench.request_for(jevbench.task("easy-fact-00"))["questions"]["decision"]

    detail = _refusal(_asking({**question, "criteria": criteria}))

    assert detail.startswith("questions.decision: ")
    assert '"true"' in detail and '"false"' in detail


@pytest.mark.parametrize(
    "question", ["Which intent?", None, ["choice"]], ids=["text", "null", "list"]
)
def test_a_question_that_is_not_an_object_is_refused(question):
    detail = _refusal(_asking(question))

    assert detail.startswith("questions.decision: ")


def test_a_question_without_a_type_is_refused():
    question = jevbench.request_for(EASY_INTENT)["questions"]["decision"]

    detail = _refusal(_asking({k: v for k, v in question.items() if k != "type"}))

    assert detail.startswith("questions.decision: ")
    assert '"type"' in detail


def test_a_question_of_an_unknown_type_is_refused_by_name():
    question = jevbench.request_for(EASY_INTENT)["questions"]["decision"]

    detail = _refusal(_asking({**question, "type": "ranking"}))

    assert "'ranking'" in detail


@pytest.mark.parametrize(
    "instructions", [None, "", "  \n", 42], ids=["null", "empty", "blank", "number"]
)
def test_a_question_without_instructions_is_refused(instructions):
    question = jevbench.request_for(EASY_INTENT)["questions"]["decision"]

    missing = _refusal(_asking({k: v for k, v in question.items() if k != "instructions"}))
    unusable = _refusal(_asking({**question, "instructions": instructions}))

    assert missing.startswith("questions.decision: ")
    assert unusable.startswith("questions.decision: ")
    assert "instructions" in missing and "instructions" in unusable
