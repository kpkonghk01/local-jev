"""The health endpoint reports ready only once the Decider has loaded."""

import threading
import time

from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_choice
from harness.server import starting, wait_until_ready

from local_jev.d1 import D1Decider


def _d1_loading_until(released: threading.Event):
    def load():
        released.wait(timeout=5)
        return D1Decider(FakeD1Model.answering(d1_choice({"a": 0.6, "b": 0.4})), device="cpu")

    return load


def test_health_is_not_ready_until_the_decider_has_loaded():
    released = threading.Event()
    with starting(_d1_loading_until(released)) as client:
        while_loading = client.get("/health")
        released.set()
        wait_until_ready(client, within=5)
        once_loaded = client.get("/health")

    assert (while_loading.status_code, while_loading.json()["status"]) == (503, "loading")
    assert (once_loaded.status_code, once_loaded.json()["status"]) == (200, "ready")


def test_health_reports_a_decider_that_failed_to_load():
    def load():
        raise OSError("LiquidAI/d1-2B: connection to huggingface.co refused")

    with starting(load) as client:
        deadline = time.monotonic() + 5
        while (health := client.get("/health")).json()["status"] == "loading":
            assert time.monotonic() < deadline, "still loading after 5 s"
            time.sleep(0.01)

    assert health.status_code == 503
    assert health.json() == {
        "status": "failed",
        "error": "OSError: LiquidAI/d1-2B: connection to huggingface.co refused",
    }


def test_questions_asked_while_the_decider_is_loading_get_503():
    released = threading.Event()
    with starting(_d1_loading_until(released)) as client:
        response = client.post(
            "/v1/systemone", json=jevbench.request_for(jevbench.task("easy-intent-00"))
        )
        released.set()

    assert response.status_code == 503
