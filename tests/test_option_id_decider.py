"""`/v1/systemone` answers through the LFM3 Option-ID Decider (fake LM backend, fixed logits).

The fake's next-token probabilities stand in for the model's; the Decider's Distribution must be
exactly those probabilities, restricted to the Question's Option IDs and normalised.
"""

import pytest
from harness import jevbench
from harness.fake_lm import FakeLMBackend
from harness.server import serving_option_id

EASY_INTENT = jevbench.task("easy-intent-00")  # choice, 5 options


def test_choice_task_gets_an_answer_jevbench_accepts():
    backend = FakeLMBackend({"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.6})
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT))

    assert jevbench.rejections(EASY_INTENT, response.status_code, response.text) == []


def _ask(client, question: dict, state="Where is my package? It has not arrived.") -> dict:
    response = client.post("/v1/systemone", json={"state": state, "questions": {"q": question}})
    assert response.status_code == 200, response.text
    return response.json()["answers"]["q"]


TEAM = {
    "type": "choice",
    "instructions": "Which team should handle the message?",
    "criteria": {  # deliberately not in alphabetical order: A, B, C follow the Criteria as sent
        "shipping": "Delivery and parcels",
        "technical": "App or site faults",
        "billing": "Charges, refunds, invoices",
    },
}


def test_choice_answer_is_the_label_of_the_most_likely_letter_in_criteria_order():
    backend = FakeLMBackend({"A": 0.25, "B": 0.15, "C": 0.6})
    with serving_option_id(backend) as client:
        answer = _ask(client, TEAM)

    assert answer == {
        "type": "choice",
        "choice": "billing",
        "probabilities": pytest.approx({"shipping": 0.25, "technical": 0.15, "billing": 0.6}),
    }


def test_choice_distribution_is_normalised_over_the_label_set_alone():
    # Half the model's mass is on "The", and "D" is not an option of this Question.
    backend = FakeLMBackend({"The": 0.5, "D": 0.2, "A": 0.06, "B": 0.09, "C": 0.15})
    with serving_option_id(backend) as client:
        probabilities = _ask(client, TEAM)["probabilities"]

    assert probabilities == pytest.approx({"shipping": 0.2, "technical": 0.3, "billing": 0.5})
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)


EASY_FACT = jevbench.task("easy-fact-00")  # noul


def test_noul_task_gets_an_answer_jevbench_accepts():
    backend = FakeLMBackend({"Yes": 0.7, "yes": 0.1, "No": 0.1, "no": 0.1})
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_FACT))

    assert jevbench.rejections(EASY_FACT, response.status_code, response.text) == []


def test_noul_answer_pools_each_labels_case_variants():
    # A Chat model puts most of its mass on "Yes", not "yes": lowercase alone would say no (0.25).
    backend = FakeLMBackend({"Yes": 0.7, "yes": 0.02, "no": 0.06, "No": 0.02, "The": 0.2})
    with serving_option_id(backend) as client:
        answer = _ask(client, jevbench.request_for(EASY_FACT)["questions"]["decision"])

    assert answer == {"type": "noul", "noul": pytest.approx(0.9)}


ORIGINAL_ORDINAL = jevbench.task("original-ordinal-01-0")  # score, 4 levels


def test_score_task_gets_an_answer_jevbench_accepts():
    backend = FakeLMBackend({"0": 0.7, "1": 0.2, "2": 0.05, "3": 0.05})
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(ORIGINAL_ORDINAL))

    assert jevbench.rejections(ORIGINAL_ORDINAL, response.status_code, response.text) == []


def test_score_answer_is_the_most_likely_level_number_over_the_scales_levels_alone():
    # "7" is not a level of this 4-level scale.
    backend = FakeLMBackend({"0": 0.05, "1": 0.1, "2": 0.25, "3": 0.1, "7": 0.5})
    with serving_option_id(backend) as client:
        answer = _ask(client, jevbench.request_for(ORIGINAL_ORDINAL)["questions"]["decision"])

    assert answer == {
        "type": "score",
        "score": 2,
        "probabilities": pytest.approx({"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.2}),
    }


def _choice_of(options: int) -> dict:
    return {
        "type": "choice",
        "instructions": "Which product is the message about?",
        "criteria": {f"product_{i:02}": f"Product number {i}" for i in range(1, options + 1)},
    }


def test_a_choice_of_26_options_is_answered_with_letters_a_to_z():
    backend = FakeLMBackend({"Z": 0.9})
    with serving_option_id(backend) as client:
        answer = _ask(client, _choice_of(26))

    assert answer["choice"] == "product_26"
    assert len(answer["probabilities"]) == 26


def test_a_choice_of_more_than_26_options_is_refused_before_it_reaches_the_model():
    backend = FakeLMBackend({"Z": 0.9})
    with serving_option_id(backend) as client:
        response = client.post(
            "/v1/systemone", json={"state": "It broke.", "questions": {"q": _choice_of(27)}}
        )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("questions.q: ")
    assert "at most 26 options" in response.json()["detail"]
    assert backend.asked == []


def _prompt_text(backend: FakeLMBackend) -> str:
    [messages] = backend.asked
    return "\n".join(message["content"] for message in messages)


@pytest.mark.parametrize("task", [EASY_INTENT, EASY_FACT, ORIGINAL_ORDINAL], ids=lambda t: t["id"])
def test_the_model_is_shown_the_state_the_instructions_and_every_criterion(task):
    backend = FakeLMBackend({"A": 0.5, "Yes": 0.2, "0": 0.3})
    with serving_option_id(backend) as client:
        client.post("/v1/systemone", json=jevbench.request_for(task))

    shown = _prompt_text(backend)
    question = task["question"]
    criteria = question["criteria"]
    descriptions = criteria.values() if isinstance(criteria, dict) else criteria
    assert task["state"] in shown
    assert question["instructions"] in shown
    assert all(description in shown for description in descriptions)


def test_a_json_state_is_shown_to_the_model_as_json():
    backend = FakeLMBackend({"A": 0.6})
    state = {"order": 1182, "status": "shipped", "carrier": None}
    with serving_option_id(backend) as client:
        _ask(client, TEAM, state=state)

    assert '"status": "shipped"' in _prompt_text(backend)
    assert '"carrier": null' in _prompt_text(backend)


@pytest.mark.parametrize("task", [EASY_INTENT, EASY_FACT, ORIGINAL_ORDINAL], ids=lambda t: t["id"])
def test_each_decision_costs_one_forward_pass(task):
    backend = FakeLMBackend({"A": 0.5, "Yes": 0.2, "0": 0.3})
    request = jevbench.request_for(task)
    question = request["questions"]["decision"]
    request["questions"] = {"first": question, "second": question}
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=request)

    assert response.status_code == 200
    assert len(backend.asked) == 2


def test_usage_reports_the_prompt_tokens_read_for_every_question_and_no_output_tokens():
    request = jevbench.request_for(EASY_INTENT)
    question = request["questions"]["decision"]
    request["questions"] = {"first": question, "second": question}
    backend = FakeLMBackend({"E": 0.6}, prompt_tokens=103)
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=request)

    assert response.json()["usage"] == {"input_tokens": 206, "output_tokens": 0}


def test_runtime_says_the_decider_runs_lfm3_at_its_pinned_revision():
    with serving_option_id(FakeLMBackend({"E": 0.6}), device="mps") as client:
        body = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT)).json()

    assert {k: v for k, v in body["runtime"].items() if not k.endswith("_ms")} == {
        "decider": "lfm3",
        "model": "LiquidAI/LFM3-1.7B",
        "revision": "0dc4268b561499c648c7ba7c9466bf3d8d01517a",
        "device": "mps",
    }
