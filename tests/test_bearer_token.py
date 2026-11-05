"""An optional bearer token protects `/v1/systemone`; without one, any token or none is accepted."""

import pytest
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


def _ask(client, headers: dict):
    return client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT), headers=headers)


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-token"),
        pytest.param({"Authorization": "Bearer guess"}, id="wrong-token"),
        pytest.param({"Authorization": "Basic czNjcmV0"}, id="not-bearer"),
    ],
)
def test_with_a_token_configured_a_missing_or_wrong_token_gets_401(headers):
    with serving_d1(FakeD1Model.answering(D1_ANSWER), token="s3cret") as client:
        response = _ask(client, headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_with_a_token_configured_the_right_token_gets_an_answer_jevbench_accepts():
    with serving_d1(FakeD1Model.answering(D1_ANSWER), token="s3cret") as client:
        response = _ask(client, {"Authorization": "Bearer s3cret"})

    assert jevbench.rejections(EASY_INTENT, response.status_code, response.text) == []


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"Authorization": "Bearer local"}, id="jevbench-dummy-token"),
        pytest.param({}, id="no-token"),
    ],
)
def test_without_a_token_configured_any_token_or_none_is_accepted(headers):
    with serving_d1(FakeD1Model.answering(D1_ANSWER)) as client:
        response = _ask(client, headers)

    assert jevbench.rejections(EASY_INTENT, response.status_code, response.text) == []


def test_health_needs_no_token():
    with serving_d1(FakeD1Model.answering(D1_ANSWER), token="s3cret") as client:
        response = client.get("/health")

    assert (response.status_code, response.json()) == (200, {"status": "ready"})
