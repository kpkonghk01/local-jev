"""`local-jev serve` picks d1 unless told otherwise: on the 231 public Tasks d1 was the more
accurate Decider (docs/feasibility-report.md).

Each Decider's loader is swapped for one over a fake model (the model boundary), and the app
`serve` builds is driven in-process instead of being handed to uvicorn.
"""

import pytest
from fastapi.testclient import TestClient
from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_choice
from harness.fake_lm import FakeLMBackend
from harness.server import wait_until_ready

from local_jev import cli
from local_jev.d1 import D1Decider
from local_jev.option_id import OptionIDDecider

EASY_INTENT = jevbench.task("easy-intent-00")


@pytest.fixture
def served(monkeypatch):
    """The app each `serve` built, in order, with both Deciders over fake models."""
    answer = d1_choice({label: 1 / len(EASY_INTENT["labels"]) for label in EASY_INTENT["labels"]})
    monkeypatch.setitem(
        cli.DECIDERS, "d1", lambda args: D1Decider(FakeD1Model.answering(answer), device="cpu")
    )
    monkeypatch.setitem(
        cli.DECIDERS, "lfm3", lambda args: OptionIDDecider(FakeLMBackend({"A": 1.0}), device="cpu")
    )
    apps = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: apps.append(app))
    return apps


def _decider_answering(app) -> str:
    with TestClient(app) as client:
        wait_until_ready(client, within=5)
        response = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT))
    assert response.status_code == 200
    return response.json()["runtime"]["decider"]


def test_serve_without_a_decider_serves_d1(served):
    cli.main(["serve", "--no-calibration"])

    assert _decider_answering(served[-1]) == "d1"


def test_serve_still_serves_the_decider_it_is_given(served):
    cli.main(["serve", "--decider", "lfm3", "--no-calibration"])

    assert _decider_answering(served[-1]) == "lfm3"
