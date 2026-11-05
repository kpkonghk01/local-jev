"""`local-jev bench` runs jevbench's `typesafe` adapter against a running server (fake d1 model).

jevbench itself is real: the runner fetches it at the pinned commit into the project's git-ignored
cache (network on the first run only) and runs it as a subprocess, as the developer would.
"""

import json
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

import pytest
from harness.fake_d1 import FakeD1Model, d1_choice, d1_noul, d1_score
from harness.fake_lm import FakeLMBackend
from harness.server import listening

from local_jev.cli import main
from local_jev.d1 import D1Decider
from local_jev.option_id import OptionIDDecider

PROJECT_CACHE = Path(__file__).parents[1] / ".cache"


def _plausible(state, question: dict) -> dict:
    """A d1 answer to any public Task: most of the mass on the first option or level."""
    match question["type"]:
        case "choice":
            first, *rest = question["criteria"]
            return d1_choice({first: 0.6, **{label: 0.4 / len(rest) for label in rest}})
        case "noul":
            return d1_noul(0.8)
        case "score":
            n = len(question["criteria"])
            return d1_score([0.7] + [0.3 / (n - 1)] * (n - 1))


def _fake_d1(answer=_plausible, device="cpu"):
    return lambda: D1Decider(FakeD1Model(answer), device=device)


def _bench(endpoint: str, runs_dir: Path, *args: str) -> Path:
    """Runs `local-jev bench` and returns the run directory it made."""
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    main(
        ["bench", "--endpoint", endpoint, "--runs-dir", str(runs_dir)]
        + ["--cache-dir", str(PROJECT_CACHE), *args]
    )
    (run,) = set(runs_dir.iterdir()) - before
    return run


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory) -> Path:
    """One 10-Task run against a server with no token, as a fresh shell would run it."""
    runs = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as env:
        env.delenv("TYPESAFE_API_KEY", raising=False)
        env.delenv("LOCAL_JEV_TOKEN", raising=False)
        with listening(_fake_d1()) as endpoint:
            return _bench(endpoint, runs, "--task-set", "public", "--limit", "10")


def test_a_run_keeps_jevbench_results_raw_evidence_ledger_and_summary(smoke_run):
    results = _jsonl(smoke_run / "results.jsonl")
    summary = json.loads((smoke_run / "summary.json").read_text())

    assert len(results) == 10
    assert all(r["ok"] and r["valid"] for r in results)
    assert len(list((smoke_run / "raw").iterdir())) == 10
    assert (smoke_run / "ledger.jsonl").stat().st_size > 0
    assert (summary["n_attempted"], summary["schema_validity"], summary["complete"]) == (
        10,
        1.0,
        True,
    )


def test_a_run_against_a_local_server_reserves_and_charges_nothing(smoke_run):
    """Unpriced, jevbench would hold $0.02 per Task against its $15 cap, and cost would be null."""
    summary = json.loads((smoke_run / "summary.json").read_text())
    ledger = _jsonl(smoke_run / "ledger.jsonl")

    assert summary["ledger_charged_usd"] == 0.0
    assert summary["price_per_1000_decisions_usd"] == 0.0
    assert {row["reserved_usd"] for row in ledger if row["event"] == "reserve"} == {0.0}


@pytest.mark.parametrize(
    "task_set, tasks_per_part",
    [
        ("easy", {"easy": 48}),
        ("original", {"original": 72}),
        ("hard", {"hard": 111}),
        ("public", {"easy": 48, "original": 72, "hard": 111}),
    ],
)
def test_a_task_set_runs_every_task_in_it(tmp_path, task_set, tasks_per_part):
    with listening(_fake_d1()) as endpoint:
        run = _bench(endpoint, tmp_path, "--task-set", task_set)
    decisions = _jsonl(run / "decisions.jsonl")

    assert Counter(d["task_set"] for d in decisions) == tasks_per_part
    assert all(d["valid"] for d in decisions)


def test_a_limited_run_takes_tasks_from_every_part_of_the_set_and_every_question_type(smoke_run):
    tasks = _jsonl(smoke_run / "tasks.jsonl")

    assert len(tasks) == 10
    assert {t["question"]["type"] for t in tasks} == {"choice", "noul", "score"}
    assert {t["id"].split("-")[0] for t in tasks} == {"easy", "original", "hard"}


def test_a_run_keeps_each_tasks_served_distribution_beside_its_expected_label(smoke_run):
    decisions = {d["task_id"]: d for d in _jsonl(smoke_run / "decisions.jsonl")}
    noul, score, choice = (
        decisions[i] for i in ("easy-fact-00", "original-ordinal-01-0", "easy-intent-00")
    )

    assert len(decisions) == 10
    assert (noul["task_set"], noul["question_type"], noul["expected"]) == ("easy", "noul", "yes")
    assert (noul["labels"], noul["valid"], noul["correct"]) == (["no", "yes"], True, True)
    assert noul["distribution"] == {"yes": 0.8, "no": pytest.approx(0.2)}
    assert (score["task_set"], score["question_type"], score["expected"]) == (
        "original",
        "score",
        "0",
    )
    assert score["labels"] == ["0", "1", "2", "3"]
    assert score["distribution"] == pytest.approx({"0": 0.7, "1": 0.1, "2": 0.1, "3": 0.1})
    # The fake puts 0.6 on the first option, and jevbench sends a choice's criteria alphabetically.
    assert (choice["expected"], choice["valid"], choice["correct"]) == ("track_order", True, False)
    assert max(choice["distribution"], key=choice["distribution"].get) == "billing_question"


def test_each_kept_decision_carries_the_servers_runtime(smoke_run):
    (decision, *_) = _jsonl(smoke_run / "decisions.jsonl")

    assert decision["runtime"]["device"] == "cpu"
    assert decision["runtime"]["decider_ms"] >= 0
    assert decision["latency_s"] > 0


def test_a_run_records_the_decider_model_revision_and_hardware_the_server_reports(smoke_run):
    record = json.loads((smoke_run / "run.json").read_text())

    assert record["server"] == {
        "decider": "d1",
        "model": "LiquidAI/d1-2B",
        "revision": "652f920e544ea37f81e4b0ce89b81d4ed0da90b1",
        "device": "cpu",
    }
    assert record["jevbench"] == {
        "url": "https://github.com/fstandhartinger/jevbench.git",
        "commit": "609a664e81f46e03731600bd7bd9f1b7a61257c5",
    }


def test_a_token_protected_server_gets_the_token_from_local_jev_token(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    with listening(_fake_d1(), token="s3cret") as endpoint:
        run = _bench(endpoint, tmp_path, "--limit", "3")

    assert [d["valid"] for d in _jsonl(run / "decisions.jsonl")] == [True, True, True]


def test_every_run_gets_a_new_directory_and_earlier_runs_are_kept(tmp_path):
    with listening(_fake_d1()) as endpoint:
        first = _bench(endpoint, tmp_path, "--limit", "2")
        kept = (first / "results.jsonl").read_bytes()
        second = _bench(endpoint, tmp_path, "--limit", "2")

    assert second != first
    assert (first / "results.jsonl").read_bytes() == kept
    assert len(_jsonl(second / "results.jsonl")) == 2


def test_a_run_ends_by_saying_where_it_is_and_what_it_scored(tmp_path, capsys):
    with listening(_fake_d1()) as endpoint:
        run = _bench(endpoint, tmp_path, "--task-set", "easy", "--limit", "4")
    said = capsys.readouterr().out

    assert str(run) in said
    assert "d1 LiquidAI/d1-2B@652f920 on cpu" in said
    # easy-intent-00 and -01 expect track_order and cancel_order, not the fake's (alphabetically
    # first) billing_question; of easy-fact-00 and -01, only the first expects the fake's yes.
    assert "4 Tasks: Valid 100.0%, Accuracy 25.0%" in said


JEVBENCH_COMMIT = "609a664e81f46e03731600bd7bd9f1b7a61257c5"


@pytest.fixture
def jevbench_mirror(smoke_run) -> str:
    """The project cache's jevbench checkout (fetched by `smoke_run`), as a URL to fetch from."""
    return (PROJECT_CACHE / "jevbench" / JEVBENCH_COMMIT).as_uri()


def _head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def _bench_with_cache(endpoint: str, tmp_path: Path, cache: Path) -> Path:
    return _bench(endpoint, tmp_path / "runs", "--limit", "2", "--cache-dir", str(cache))


def test_jevbench_is_fetched_at_the_pinned_commit_and_then_reused(
    tmp_path, monkeypatch, jevbench_mirror
):
    cache = tmp_path / "cache"
    with listening(_fake_d1()) as endpoint:
        monkeypatch.setenv("LOCAL_JEV_JEVBENCH_URL", jevbench_mirror)
        _bench_with_cache(endpoint, tmp_path, cache)
        fetched = _head(cache / "jevbench" / JEVBENCH_COMMIT)
        monkeypatch.setenv("LOCAL_JEV_JEVBENCH_URL", (tmp_path / "nowhere").as_uri())
        run = _bench_with_cache(endpoint, tmp_path, cache)

    assert fetched == JEVBENCH_COMMIT
    assert len(_jsonl(run / "results.jsonl")) == 2


def test_a_cached_checkout_at_another_commit_is_fetched_again(
    tmp_path, monkeypatch, jevbench_mirror
):
    stale = tmp_path / "cache" / "jevbench" / JEVBENCH_COMMIT
    stale.mkdir(parents=True)
    for git in (["init", "--quiet"], ["commit", "--quiet", "--allow-empty", "-m", "not jevbench"]):
        subprocess.run(["git", "-C", str(stale), *git], check=True, capture_output=True)
    monkeypatch.setenv("LOCAL_JEV_JEVBENCH_URL", jevbench_mirror)
    with listening(_fake_d1()) as endpoint:
        run = _bench_with_cache(endpoint, tmp_path, tmp_path / "cache")

    assert _head(stale) == JEVBENCH_COMMIT
    assert len(_jsonl(run / "results.jsonl")) == 2


def test_a_failed_fetch_says_so_and_starts_no_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_JEVBENCH_URL", (tmp_path / "nowhere").as_uri())
    with listening(_fake_d1()) as endpoint, pytest.raises(SystemExit, match="fetch"):
        main(
            ["bench", "--endpoint", endpoint, "--runs-dir", str(tmp_path / "runs")]
            + ["--cache-dir", str(tmp_path / "cache")]
        )

    assert not (tmp_path / "runs").exists()


def test_by_default_runs_go_under_runs_and_jevbench_under_cache_in_the_current_directory(
    tmp_path, monkeypatch, jevbench_mirror
):
    monkeypatch.setenv("LOCAL_JEV_JEVBENCH_URL", jevbench_mirror)
    monkeypatch.chdir(tmp_path)
    with listening(_fake_d1()) as endpoint:
        main(["bench", "--endpoint", endpoint, "--limit", "2"])

    (run,) = (tmp_path / "runs").iterdir()
    assert len(_jsonl(run / "results.jsonl")) == 2
    assert _head(tmp_path / ".cache" / "jevbench" / JEVBENCH_COMMIT) == JEVBENCH_COMMIT


def test_a_run_waits_for_a_server_whose_decider_is_still_loading(tmp_path):
    """jevbench stops after three failures in a row, so it must never meet a cold server's 503s."""

    def slow_load():
        time.sleep(1.5)
        return _fake_d1()()

    with listening(slow_load) as endpoint:
        run = _bench(endpoint, tmp_path, "--limit", "3")

    assert [d["valid"] for d in _jsonl(run / "decisions.jsonl")] == [True, True, True]


def test_no_run_starts_when_no_server_answers(tmp_path):
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{unused.getsockname()[1]}"
    with pytest.raises(SystemExit, match="no server answers"):
        _bench(endpoint, tmp_path / "runs")

    assert not (tmp_path / "runs").exists()


def test_no_run_starts_when_the_servers_decider_failed_to_load(tmp_path):
    def broken_load():
        raise OSError("LiquidAI/d1-2B: connection to huggingface.co refused")

    with listening(broken_load) as endpoint, pytest.raises(SystemExit, match="huggingface.co"):
        _bench(endpoint, tmp_path / "runs")

    assert not (tmp_path / "runs").exists()


def test_a_run_jevbench_stops_early_fails_and_says_how_far_it_got(tmp_path):
    def crash(state, question):
        raise RuntimeError("the model crashed")

    with listening(_fake_d1(crash)) as endpoint:
        with pytest.raises(SystemExit, match=r"stopped after 3 of 10 Tasks") as stopped:
            _bench(endpoint, tmp_path, "--limit", "10")

    (run,) = tmp_path.iterdir()
    assert str(run) in str(stopped.value)
    assert len(_jsonl(run / "results.jsonl")) == 3


def test_a_run_never_asks_for_a_reason(tmp_path):
    """So a benchmark's latency never includes writing one, even on a Decider that can."""
    backend = FakeLMBackend({"A": 0.5, "Yes": 0.3, "0": 0.2}, writes="Because it fits best.")
    with listening(lambda: OptionIDDecider(backend, device="cpu")) as endpoint:
        run = _bench(endpoint, tmp_path, "--limit", "10")
    # jevbench keeps each Task's request and response as raw evidence.
    exchanges = [json.loads(raw.read_text()) for raw in (run / "raw").iterdir()]
    results = _jsonl(run / "results.jsonl")

    assert len(exchanges) == 10
    assert not any(e["request"].get("explain") for e in exchanges)
    assert not any("reason" in e["response"]["answers"]["decision"] for e in exchanges)
    assert {r["usage"]["output_tokens"] for r in results} == {0}
