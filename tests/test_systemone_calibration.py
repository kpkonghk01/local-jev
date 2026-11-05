"""`/v1/systemone` serves Distributions calibrated with the active Decider's own temperatures."""

import json
import time
from pathlib import Path

import pytest
from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_choice, d1_noul, d1_score
from harness.fake_lm import FakeLMBackend
from harness.server import serving_d1, serving_option_id, starting

from local_jev.d1 import D1_MODEL_ID, D1_REVISION, D1Decider

EASY_INTENT = jevbench.task("easy-intent-00")
EASY_FACT = jevbench.task("easy-fact-00")
ORDINAL = jevbench.task("original-ordinal-01-0")

# Squares of 0.8, 0.4, 0.3, 0.2 and 0.1, so that T = 2 (square roots) gives round numbers.
SQUARES = {
    "track_order": 0.64,
    "cancel_order": 0.16,
    "billing_question": 0.09,
    "change_address": 0.04,
    "report_damage": 0.01,
}


def _calibrations(
    directory: Path,
    temperatures: dict,
    decider: str = "d1",
    model: str = D1_MODEL_ID,
    revision: str = D1_REVISION,
) -> Path:
    """A calibrations directory holding one calibration for `decider`, fitted for `model`."""
    directory.mkdir(parents=True, exist_ok=True)
    calibration = {
        "decider": decider,
        "model": model,
        "revision": revision,
        "temperatures": temperatures,
        "fitted_from": {
            "run": "20261105T085252Z-public",
            "task_set": "public",
            "n_calibration": 231,
        },
    }
    (directory / f"{decider}.json").write_text(json.dumps(calibration), encoding="utf-8")
    return directory


def _decision(client, task) -> dict:
    response = client.post("/v1/systemone", json=jevbench.request_for(task))
    assert jevbench.rejections(task, response.status_code, response.text) == []
    return response.json()["answers"]["decision"]


def test_a_choice_distribution_is_served_with_the_choice_temperature(tmp_path):
    calibrations = _calibrations(tmp_path, {"choice": 2.0})
    model = FakeD1Model.answering(d1_choice(SQUARES))
    with serving_d1(model, calibrations=calibrations) as client:
        answer = _decision(client, EASY_INTENT)

    # Log-probabilities halved and renormalised: each p becomes √p / Σ√p, and Σ√p = 1.8.
    assert answer == {
        "type": "choice",
        "choice": "track_order",
        "probabilities": pytest.approx(
            {
                "track_order": 0.8 / 1.8,
                "cancel_order": 0.4 / 1.8,
                "billing_question": 0.3 / 1.8,
                "change_address": 0.2 / 1.8,
                "report_damage": 0.1 / 1.8,
            }
        ),
    }


def test_noul_and_score_distributions_are_served_with_their_own_temperatures(tmp_path):
    calibrations = _calibrations(tmp_path, {"choice": 2.0, "noul": 0.5, "score": 0.5})

    def answer(state, question):
        return d1_noul(0.75) if question["type"] == "noul" else d1_score([0.1, 0.2, 0.3, 0.4])

    with serving_d1(FakeD1Model(answer), calibrations=calibrations) as client:
        noul, score = _decision(client, EASY_FACT), _decision(client, ORDINAL)

    # T = 0.5 squares each p: 0.75² / (0.75² + 0.25²) = 0.9; 0.1², 0.2², 0.3², 0.4² over 0.3.
    assert noul == {"type": "noul", "noul": pytest.approx(0.9)}
    assert score == {
        "type": "score",
        "score": 3,
        "probabilities": pytest.approx({"0": 1 / 30, "1": 4 / 30, "2": 9 / 30, "3": 16 / 30}),
    }


def test_a_question_type_without_a_temperature_is_served_raw(tmp_path):
    calibrations = _calibrations(tmp_path, {"noul": 0.5})
    model = FakeD1Model.answering(d1_choice(SQUARES))
    with serving_d1(model, calibrations=calibrations) as client:
        answer = _decision(client, EASY_INTENT)

    assert answer["probabilities"] == pytest.approx({k: p / 0.94 for k, p in SQUARES.items()})


def test_the_runtime_says_which_calibration_was_applied(tmp_path):
    calibrations = _calibrations(tmp_path, {"choice": 1.25, "noul": 0.8})
    with serving_d1(FakeD1Model.answering(d1_choice(SQUARES)), calibrations=calibrations) as c:
        body = c.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT)).json()

    assert body["runtime"]["calibration"] == {
        "temperatures": {"choice": 1.25, "noul": 0.8},
        "fitted_from": "20261105T085252Z-public",
    }


def test_without_a_calibration_for_the_decider_distributions_are_served_raw(tmp_path):
    model = FakeD1Model.answering(d1_choice(SQUARES))
    with serving_d1(model, calibrations=tmp_path / "empty") as client:
        body = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT)).json()

    assert body["runtime"]["calibration"] is None
    assert body["answers"]["decision"]["probabilities"] == pytest.approx(
        {k: p / 0.94 for k, p in SQUARES.items()}
    )


def test_a_calibration_fitted_for_another_revision_is_not_applied(tmp_path):
    calibrations = _calibrations(tmp_path, {"choice": 2.0}, revision="0" * 40)
    model = FakeD1Model.answering(d1_choice(SQUARES))
    with serving_d1(model, calibrations=calibrations) as client:
        body = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT)).json()

    assert body["runtime"]["calibration"] is None
    assert body["answers"]["decision"]["probabilities"] == pytest.approx(
        {k: p / 0.94 for k, p in SQUARES.items()}
    )


def test_deciders_never_share_a_calibration(tmp_path):
    """An lfm3 server beside d1's calibration serves exactly what it serves with none at all."""
    calibrations = _calibrations(tmp_path, {"choice": 2.0, "noul": 2.0, "score": 2.0})
    backend = FakeLMBackend({"E": 0.6, "A": 0.2})
    with serving_option_id(backend) as client:
        uncalibrated = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT))
    with serving_option_id(backend, calibrations=calibrations) as client:
        beside_d1s = client.post("/v1/systemone", json=jevbench.request_for(EASY_INTENT))

    assert beside_d1s.json()["runtime"]["calibration"] is None
    assert beside_d1s.json()["answers"] == uncalibrated.json()["answers"]


@pytest.mark.parametrize(
    "temperatures",
    [
        pytest.param({"choice": 0}, id="zero"),
        pytest.param({"choice": -1.5}, id="negative"),
        pytest.param({"choice": "2"}, id="not-a-number"),
        pytest.param({"text": 1.5}, id="unknown-question-type"),
    ],
)
def test_a_broken_calibration_fails_the_server_at_start(tmp_path, temperatures):
    calibrations = _calibrations(tmp_path, temperatures)
    model = FakeD1Model.answering(d1_choice(SQUARES))
    with starting(lambda: D1Decider(model, device="cpu"), calibrations=calibrations) as client:
        health = _settled_health(client)

    assert health.status_code == 503
    assert health.json()["status"] == "failed"
    assert "d1.json" in health.json()["error"]


def _settled_health(client):
    deadline = time.monotonic() + 5
    while (health := client.get("/health")).json()["status"] == "loading":
        assert time.monotonic() < deadline, "still loading after 5 s"
        time.sleep(0.01)
    return health
