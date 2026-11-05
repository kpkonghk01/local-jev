"""Requests are processed one at a time: concurrent requests never interleave model access."""

import itertools
import time
from concurrent.futures import ThreadPoolExecutor

from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_choice
from harness.server import serving_d1

EASY_INTENT = jevbench.task("easy-intent-00")
D1_ANSWER = d1_choice({"track_order": 0.8, "cancel_order": 0.2})
QUESTION = {
    "type": "choice",
    "instructions": "Which intent does the message express?",
    "criteria": {"track_order": "Asks where an order is", "cancel_order": "Wants to cancel"},
}


def _post_concurrently(client, bodies: list[dict]) -> list:
    with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        return list(pool.map(lambda body: client.post("/v1/systemone", json=body), bodies))


def test_concurrent_requests_never_reach_the_model_at_the_same_time():
    model = FakeD1Model.answering(D1_ANSWER, seconds_per_call=0.02)
    bodies = [{"state": f"Message {i}", "questions": {"decision": QUESTION}} for i in range(6)]
    with serving_d1(model) as client:
        responses = _post_concurrently(client, bodies)

    assert [r.status_code for r in responses] == [200] * 6


def test_concurrent_requests_never_interleave_their_questions():
    """Each request holds the model from its first Question to its last: no two requests' spans
    on the model overlap, whichever request goes first."""
    heard_at: dict[str, list[float]] = {}

    def answer(state, question):
        heard_at.setdefault(state, []).append(time.monotonic())
        return D1_ANSWER

    model = FakeD1Model(answer, seconds_per_call=0.02)
    questions = {"first": QUESTION, "second": QUESTION, "third": QUESTION}
    bodies = [{"state": f"Message {i}", "questions": questions} for i in range(4)]
    with serving_d1(model) as client:
        responses = _post_concurrently(client, bodies)

    assert [r.status_code for r in responses] == [200] * 4
    spans = sorted((min(times), max(times)) for times in heard_at.values())
    overlapping = [(a, b) for a, b in itertools.pairwise(spans) if b[0] < a[1]]
    assert len(spans) == 4
    assert overlapping == []
