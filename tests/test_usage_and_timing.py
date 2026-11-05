"""Responses report token usage and the server's own timing (fake d1 model)."""

from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_choice
from harness.server import serving_d1

EASY_INTENT = jevbench.task("easy-intent-00")
D1_ANSWER = d1_choice(
    {
        "billing_question": 0.0732,
        "cancel_order": 0.0539,
        "change_address": 0.0412,
        "report_damage": 0.0588,
        "track_order": 0.7729,
    }
)


def test_usage_reports_the_tokens_d1_read_and_no_output_tokens():
    model = FakeD1Model.answering(D1_ANSWER, input_tokens=73)
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT))

    assert response.json()["usage"] == {"input_tokens": 73, "output_tokens": 0}


def test_usage_counts_the_tokens_read_for_every_question_in_the_request():
    request = jevbench.request_for(EASY_INTENT)
    question = request["questions"]["decision"]
    request["questions"] = {"first": question, "second": question}
    model = FakeD1Model.answering(D1_ANSWER, input_tokens=73)
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=request)

    assert response.json()["usage"] == {"input_tokens": 146, "output_tokens": 0}


def test_timing_reports_the_server_total_and_the_time_spent_in_the_decider():
    request = jevbench.request_for(EASY_INTENT)
    question = request["questions"]["decision"]
    request["questions"] = {"first": question, "second": question}
    model = FakeD1Model.answering(D1_ANSWER, seconds_per_call=0.02)
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=request)

    timing = response.json()["timing"]
    assert timing["decider_ms"] >= 40
    assert timing["total_ms"] >= timing["decider_ms"]


def test_runtime_carries_the_timing_and_what_the_decider_runs_on():
    """jevbench copies a response's top-level `runtime` into every results record."""
    model = FakeD1Model.answering(D1_ANSWER)
    with serving_d1(model, device="mps") as client:
        body = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT)).json()

    assert body["runtime"] == {
        "decider": "d1",
        "model": "LiquidAI/d1-2B",
        "revision": "652f920e544ea37f81e4b0ce89b81d4ed0da90b1",
        "device": "mps",
        "total_ms": body["timing"]["total_ms"],
        "decider_ms": body["timing"]["decider_ms"],
    }


def test_usage_is_zero_when_d1_does_not_report_it():
    model = FakeD1Model.answering(D1_ANSWER, input_tokens=None)
    with serving_d1(model) as client:
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT))

    assert response.json()["usage"] == {"input_tokens": 0, "output_tokens": 0}
