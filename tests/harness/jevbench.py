"""jevbench's view of a `/v1/systemone` server: the requests it sends and the answers it accepts.

Mirrors jevbench at commit 609a664e81f46e03731600bd7bd9f1b7a61257c5
(https://github.com/fstandhartinger/jevbench):

- `request_for` builds the body exactly as `TypeSafeAdapter.build_request` does
  (jevbench/adapters/typesafe.py, `build_question` in jevbench/adapters/base.py).
- `rejections` applies the adapter's acceptance rules (typesafe.py, `run`) and then the scorer's
  strict Distribution rules (jevbench/scoring.py, `validate_probs` with `SUM_TOL`).

The sample Tasks in `jevbench_tasks.jsonl` are verbatim lines from jevbench's public datasets
(easy.jsonl line 1, hard.jsonl line 22). Copyright (c) 2026 Florian Standhartinger and
contributors, MIT License.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

SUM_TOL = 1e-3  # jevbench/scoring.py: the strict, pre-registered sum tolerance

_TASKS_FILE = Path(__file__).with_name("jevbench_tasks.jsonl")


@cache
def _tasks() -> dict[str, dict]:
    lines = _TASKS_FILE.read_text(encoding="utf-8").splitlines()
    return {t["id"]: t for t in map(json.loads, lines)}


def task(task_id: str) -> dict:
    """A public jevbench Task, as jevbench loads it from its dataset files."""
    return _tasks()[task_id]


def request_for(task: dict, model: str = "jev-latest") -> dict:
    """The `/v1/systemone` body jevbench's `typesafe` adapter sends for `task`."""
    q = {"type": task["question"]["type"], "instructions": task["question"]["instructions"]}
    if task["question"].get("criteria") is not None:
        q["criteria"] = task["question"]["criteria"]
    return {"state": task["state"], "model": model, "questions": {"decision": q}}


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} in the response body (jevbench's raw-evidence write crashes on it)")


def rejections(task: dict, status: int, text: str) -> list[str]:
    """Why jevbench would not count this response as a valid answer to `task`; empty if it would."""
    try:
        body = json.loads(text, parse_constant=_refuse_constant)
    except ValueError as e:
        return [f"body is not strict JSON: {e}"]
    if status != 200 or not isinstance(body, dict):
        return [f"HTTP {status}: {str(body)[:300]}"]
    ans = (body.get("answers") or {}).get("decision")
    if not isinstance(ans, dict):
        return ["missing 'answers.decision'"]
    qtype = task["question"]["type"]
    if ans.get("type") != qtype:
        return ["native answer type mismatch"]
    if qtype == "choice":
        if ans.get("choice") not in task["labels"]:
            return [f"choice {ans.get('choice')!r} is not one of the Labels"]
        probs = ans.get("probabilities")
        if not isinstance(probs, dict):
            return ["choice answer missing probabilities"]
    else:
        raise NotImplementedError(f"acceptance rules for {qtype!r} are not mirrored yet")
    return _distribution_problems(probs, task["labels"])


def _distribution_problems(probs: dict, labels: list[str]) -> list[str]:
    want, got = set(labels), set(probs)
    if got != want:
        return [f"label keys mismatch: missing={sorted(want - got)} extra={sorted(got - want)}"]
    problems = []
    for k, v in probs.items():
        if isinstance(v, bool) or not isinstance(v, int | float):
            problems.append(f"prob[{k!r}] is not a number: {v!r}")
        elif not math.isfinite(v) or not 0.0 <= v <= 1.0:
            problems.append(f"prob[{k!r}] out of [0,1]: {v}")
    if not problems and abs(sum(probs.values()) - 1.0) > SUM_TOL:
        problems.append(f"probs sum to {sum(probs.values())}, tolerance {SUM_TOL}")
    return problems
