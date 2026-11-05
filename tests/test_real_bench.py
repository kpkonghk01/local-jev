"""jevbench's `typesafe` adapter against a real d1 server (run with `uv run pytest -m real_model`).

Downloads d1 at its pinned revision on first use, like `test_real_d1.py`, and fetches jevbench into
the project cache like `test_bench.py`.
"""

import json
from pathlib import Path

import pytest
from harness.server import listening

from local_jev.cli import main
from local_jev.d1 import load_d1

pytestmark = pytest.mark.real_model

PROJECT_CACHE = Path(__file__).parents[1] / ".cache"


def test_jevbench_gets_10_valid_answers_from_a_real_d1_server(tmp_path, capsys):
    with listening(load_d1) as endpoint:
        main(
            ["bench", "--endpoint", endpoint, "--task-set", "public", "--limit", "10"]
            + ["--runs-dir", str(tmp_path), "--cache-dir", str(PROJECT_CACHE)]
        )
    print(capsys.readouterr().out)
    (run,) = tmp_path.iterdir()
    decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]

    assert {d["question_type"] for d in decisions} == {"choice", "noul", "score"}
    assert [(d["task_id"], d["valid"]) for d in decisions] == [
        (d["task_id"], True) for d in decisions
    ]
    assert len(decisions) == 10
