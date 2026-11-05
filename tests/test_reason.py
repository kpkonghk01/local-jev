"""`explain: true` asks for a Reason: the LFM3 Option-ID Decider writes one, d1 answers without.

The fake LM backend writes whatever text the test gives it, one token per word, cut off at the
bound it is asked for, as the real backend is.
"""

import json
import re
from pathlib import Path

import pytest
from harness import jevbench
from harness.fake_d1 import FakeD1Model, d1_choice, d1_noul, d1_score
from harness.fake_lm import FakeLMBackend
from harness.server import serving_d1, serving_option_id

from local_jev.bench import TASK_SETS
from local_jev.option_id import REASON_MAX_TOKENS

EASY_INTENT = jevbench.task("easy-intent-00")  # choice, 5 options; E is track_order
LIKELY_TRACK_ORDER = {"A": 0.05, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.8}
REASON = "The customer asks where a package they ordered last week is."


def _explained(task: dict) -> dict:
    return {**jevbench.request_for(task), "explain": True}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_with_explain_the_lfm3_answer_carries_the_reason_the_model_wrote():
    backend = FakeLMBackend(LIKELY_TRACK_ORDER, writes=REASON)
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=_explained(EASY_INTENT))

    assert response.status_code == 200, response.text
    assert response.json()["answers"]["decision"]["reason"] == REASON


ORIGINAL_ORDINAL = jevbench.task("original-ordinal-01-0")  # score, 4 levels
EASY_FACT = jevbench.task("easy-fact-00")  # noul


@pytest.mark.parametrize("task", [EASY_INTENT, EASY_FACT, ORIGINAL_ORDINAL], ids=lambda t: t["id"])
def test_asking_for_a_reason_leaves_the_answer_otherwise_unchanged(task):
    backend = FakeLMBackend({"A": 0.2, "E": 0.5, "Yes": 0.15, "1": 0.1, "2": 0.05}, writes=REASON)
    with serving_option_id(backend) as client:
        plain = client.post("/v1/systemone", json=jevbench.request_for(task)).json()
        explained = client.post("/v1/systemone", json=_explained(task)).json()

    answer = explained["answers"]["decision"]
    assert answer.pop("reason") == REASON
    assert answer == plain["answers"]["decision"]


def test_usage_counts_the_tokens_read_and_written_for_the_reason():
    # The fake reads 100 tokens for each prompt: one to score, one to write the Reason from.
    backend = FakeLMBackend(LIKELY_TRACK_ORDER, prompt_tokens=100, writes="Asks where it is.")
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=_explained(EASY_INTENT))

    assert response.json()["usage"] == {"input_tokens": 200, "output_tokens": 4}


def test_a_reason_that_runs_on_is_cut_off_at_the_length_cap():
    backend = FakeLMBackend(LIKELY_TRACK_ORDER, writes=" ".join(["and"] * 1000))
    with serving_option_id(backend) as client:
        body = client.post("/v1/systemone", json=_explained(EASY_INTENT)).json()

    assert len(body["answers"]["decision"]["reason"].split()) == REASON_MAX_TOKENS
    assert body["usage"]["output_tokens"] == REASON_MAX_TOKENS


def test_a_reason_is_escaped_into_an_answer_jevbench_still_accepts():
    awkward = 'They wrote "where is it?" \\ twice,\nthen }]} and a tab\tand ✓.'
    backend = FakeLMBackend(LIKELY_TRACK_ORDER, writes=awkward)
    with serving_option_id(backend) as client:
        response = client.post("/v1/systemone", json=_explained(EASY_INTENT))

    assert jevbench.rejections(EASY_INTENT, response.status_code, response.text) == []
    assert response.json()["answers"]["decision"]["reason"] == awkward


@pytest.mark.parametrize("explain", [None, False], ids=["left-out", "false"])
def test_without_explain_no_reason_is_written(explain):
    request = jevbench.request_for(EASY_INTENT)
    if explain is not None:
        request["explain"] = explain
    backend = FakeLMBackend(LIKELY_TRACK_ORDER, writes=REASON)
    with serving_option_id(backend) as client:
        body = client.post("/v1/systemone", json=request).json()

    assert "reason" not in body["answers"]["decision"]
    assert body["usage"]["output_tokens"] == 0


@pytest.mark.parametrize("explain", ["yes", 1, None])
def test_explain_is_true_or_false_and_nothing_else(explain):
    request = {**jevbench.request_for(EASY_INTENT), "explain": explain}
    with serving_option_id(FakeLMBackend(LIKELY_TRACK_ORDER, writes=REASON)) as client:
        response = client.post("/v1/systemone", json=request)

    assert response.status_code == 422
    assert response.json()["detail"].startswith("explain: ")


HAN = re.compile("[一-鿿]")
CHINESE_REASON = "因為訊息在問包裹在哪裡。"
ENGLISH_REASON = "Because the message asks where the parcel is."


def _in_the_language_asked(shown: str) -> str:
    """As LFM3-1.7B does: a Reason in the language of the request it answers, the last thing it
    is asked, whatever the language of the State before it."""
    asked = shown.rstrip().splitlines()[-1]
    return CHINESE_REASON if HAN.search(asked) else ENGLISH_REASON


ZH_TW_TASKS = [t for _, path in TASK_SETS["zh-tw"](Path("unused")) for t in _jsonl(path)]


def test_every_traditional_chinese_state_gets_a_chinese_reason():
    """Even where a State keeps English names, codes, tables or program code."""
    backend = FakeLMBackend({"A": 0.4, "Yes": 0.3, "0": 0.3}, writes=_in_the_language_asked)
    with serving_option_id(backend) as client:
        reasons = {
            t["id"]: client.post("/v1/systemone", json=_explained(t)).json()["answers"]["decision"]
            for t in ZH_TW_TASKS
        }

    assert len(reasons) == 231
    assert [i for i, answer in reasons.items() if answer["reason"] != CHINESE_REASON] == []


@pytest.mark.parametrize(
    "state",
    [
        EASY_INTENT["state"],
        jevbench.task("hard-opus-b-ambiguous-10")["state"],
        "The customer, 王小明, asks where the parcel he ordered last week is.",
        {"customer": "王小明", "message": "Where is the parcel I ordered last week?"},
    ],
    ids=["text", "long", "quoting-a-chinese-name", "json-quoting-a-chinese-name"],
)
def test_an_english_state_gets_an_english_reason(state):
    backend = FakeLMBackend(LIKELY_TRACK_ORDER, writes=_in_the_language_asked)
    with serving_option_id(backend) as client:
        request = {**_explained(EASY_INTENT), "state": state}
        answer = client.post("/v1/systemone", json=request).json()["answers"]["decision"]

    assert answer["reason"] == ENGLISH_REASON


D1_TRACK_ORDER = d1_choice(
    {
        "billing_question": 0.0732,
        "cancel_order": 0.0539,
        "change_address": 0.0412,
        "report_damage": 0.0588,
        "track_order": 0.7729,
    }
)


@pytest.mark.parametrize("task", [EASY_INTENT, EASY_FACT, ORIGINAL_ORDINAL], ids=lambda t: t["id"])
def test_d1_answers_explain_normally_without_a_reason(task):
    canned = {"choice": D1_TRACK_ORDER, "noul": d1_noul(0.9), "score": d1_score([0.7, 0.2, 0.1, 0])}
    model = FakeD1Model.answering(canned[task["question"]["type"]])
    with serving_d1(model) as client:
        plain = client.post("/v1/systemone", json=jevbench.request_for(task))
        explained = client.post("/v1/systemone", json=_explained(task))

    assert explained.status_code == 200, explained.text
    assert jevbench.rejections(task, explained.status_code, explained.text) == []
    assert explained.json()["answers"] == plain.json()["answers"]
    assert "reason" not in explained.json()["answers"]["decision"]
