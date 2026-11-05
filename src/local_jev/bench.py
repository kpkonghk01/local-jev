"""The bench runner: jevbench's own `typesafe` adapter against a running server.

jevbench is fetched at a pinned commit into a git-ignored cache and run from there as a
subprocess (it needs only the standard library). Every run gets a new directory of its own.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import fields
from datetime import UTC, datetime
from itertools import count, zip_longest
from pathlib import Path
from typing import NamedTuple

from local_jev.decider import Runtime

JEVBENCH_URL = "https://github.com/fstandhartinger/jevbench.git"
JEVBENCH_COMMIT = "609a664e81f46e03731600bd7bd9f1b7a61257c5"
JEVBENCH_URL_ENV = "LOCAL_JEV_JEVBENCH_URL"
"""When set, the git URL jevbench is fetched from instead (a mirror holding the pinned commit)."""

TaskFiles = Callable[[Path], list[tuple[str, Path]]]
"""Given jevbench's checkout, a Task set's files, each named for the part of the set it holds."""


def _public(*parts: str) -> TaskFiles:
    return lambda jevbench: [(p, jevbench / "datasets" / "public" / f"{p}.jsonl") for p in parts]


ZH_TW_TASKS = Path(__file__).with_name("zh_tw")
"""The Traditional-Chinese copy of the public Tasks, committed with local-jev (see its README)."""


def _zh_tw(*parts: str) -> TaskFiles:
    return lambda _jevbench: [(f"zh-tw-{p}", ZH_TW_TASKS / f"{p}.jsonl") for p in parts]


TASK_SETS: dict[str, TaskFiles] = {
    "easy": _public("easy"),
    "original": _public("original"),
    "hard": _public("hard"),
    "public": _public("easy", "original", "hard"),
    "zh-tw-easy": _zh_tw("easy"),
    "zh-tw-original": _zh_tw("original"),
    "zh-tw-hard": _zh_tw("hard"),
    "zh-tw": _zh_tw("easy", "original", "hard"),
}
"""The Task sets `bench` can run; `public` is all 231 public Tasks, `zh-tw` their copy with each
State in Traditional Chinese.

Another set is one more entry; its files need not be in jevbench's checkout. Its part names go into
every kept Decision's `task_set`, so a translated copy names its parts apart (`zh-tw-easy`).
"""

_KEY_ENV = "TYPESAFE_API_KEY"


class BenchFailed(Exception):
    pass


def fetch_jevbench(cache_dir: Path) -> Path:
    """jevbench's checkout at the pinned commit under `cache_dir`, fetched unless already there."""
    checkout = cache_dir / "jevbench" / JEVBENCH_COMMIT
    if _head(checkout) == JEVBENCH_COMMIT:
        return checkout
    url = _jevbench_url()
    print(f"[local-jev] fetching jevbench {JEVBENCH_COMMIT[:7]} from {url}", flush=True)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    # Fetched beside the cache and moved in whole, so an interrupted fetch leaves no checkout.
    staging = Path(tempfile.mkdtemp(prefix=".fetching-", dir=checkout.parent))
    try:
        _git(staging, "init", "--quiet")
        try:
            _git(staging, "fetch", "--quiet", "--depth", "1", url, JEVBENCH_COMMIT)
        except BenchFailed as e:
            raise BenchFailed(f"could not fetch jevbench {JEVBENCH_COMMIT} from {url}: {e}") from e
        _git(staging, "-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD")
        shutil.rmtree(checkout, ignore_errors=True)
        staging.rename(checkout)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return checkout


def _jevbench_url() -> str:
    """Where jevbench is fetched from: the mirror in `LOCAL_JEV_JEVBENCH_URL`, if set."""
    return os.environ.get(JEVBENCH_URL_ENV) or JEVBENCH_URL


def _head(checkout: Path) -> str | None:
    """The commit `checkout` has checked out, if it is a checkout of its own."""
    if not (checkout / ".git").is_dir():
        return None  # git would answer for whatever repository encloses it
    try:
        return _git(checkout, "rev-parse", "HEAD")
    except BenchFailed:
        return None


def _git(where: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(where), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise BenchFailed(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def bench(
    endpoint: str,
    task_set: str = "public",
    *,
    limit: int | None = None,
    runs_dir: Path = Path("runs"),
    cache_dir: Path = Path(".cache"),
    token: str | None = None,
    ready_timeout_s: float = 600.0,
) -> Path:
    """Runs jevbench on `task_set` against the server at `endpoint`; returns the run directory.

    `token` is the server's bearer token, if it has one. A server still loading its Decider is
    waited for, up to `ready_timeout_s`.
    """
    started = _now()
    # jevbench runs from its own checkout, so it is given absolute paths only.
    runs_dir, cache_dir = runs_dir.resolve(), cache_dir.resolve()
    jevbench = fetch_jevbench(cache_dir)
    _wait_until_ready(endpoint, ready_timeout_s)
    tasks = _tasks(TASK_SETS[task_set](jevbench))
    if limit is not None:
        tasks = _spread(tasks, limit)
    run = _new_run_dir(runs_dir, task_set, limit)
    tasks_file = run / "tasks.jsonl"
    tasks_file.write_text("".join(t.line for t in tasks), encoding="utf-8")

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",  # no __pycache__ in the cached checkout
        "NO_PROXY": ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1,localhost"])),
        # The `typesafe` adapter sends nothing while its key is empty, and a server without a
        # token ignores whichever one it is sent.
        _KEY_ENV: token or "local-dummy",
    }
    ran = _jevbench(
        jevbench,
        env,
        "run",
        "--tasks", tasks_file,
        "--adapter", "typesafe",
        "--endpoint", endpoint,
        # Only a label: the server echoes it; what it runs is in each answer's `runtime`.
        "--model", "local-jev",
        "--key-env", _KEY_ENV,
        # Unpriced, jevbench holds $0.02 per Task against its $15 cap and leaves cost null.
        "--price-in-per-m", "0",
        "--price-out-per-m", "0",
        "--reserve-usd", "0",
        "--cost-basis", "local_self_hosted_no_tariff",
        "--results", run / "results.jsonl",
        "--raw-dir", run / "raw",
        "--ledger", run / "ledger.jsonl",
        "--manifest", run / "manifest.json",
        "--run-label", run.name,
    )  # fmt: skip
    results = _jsonl(run / "results.jsonl")
    if ran.returncode != 0 or len(results) != len(tasks):
        # It stops at once on 401, 403 or 429, and after three failures in a row.
        raise BenchFailed(
            f"jevbench stopped after {len(results)} of {len(tasks)} Tasks "
            f"(exit {ran.returncode}); what it did is in {run}"
        )
    summarized = _jevbench(
        jevbench,
        env,
        "summarize",
        "--tasks", tasks_file,
        "--results", run / "results.jsonl",
        "--ledger", run / "ledger.jsonl",
        "--public-export", run / "public-export.json",
    )  # fmt: skip
    if summarized.returncode != 0:
        raise BenchFailed(f"jevbench summarize exited with {summarized.returncode}; see {run}")
    (run / "summary.json").write_text(summarized.stdout, encoding="utf-8")
    _write_decisions(run / "decisions.jsonl", tasks, results)
    record = {
        "jevbench": {"url": _jevbench_url(), "commit": JEVBENCH_COMMIT},
        "endpoint": endpoint,
        "task_set": task_set,
        "limit": limit,
        "n_tasks": len(tasks),
        "started_utc": started,
        "finished_utc": _now(),
        "server": _server(results),
    }
    (run / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return run


def headline(run: Path) -> str:
    """One line saying what a finished run measured, and on what."""
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    server = record["server"]
    who = (
        f"{server['decider']} {server['model']}@{server['revision'][:7]} on {server['device']}, "
        + ("calibrated" if server.get("calibration") else "raw")
        if server
        else "an unidentified server"
    )
    ece = (summary.get("ece") or {}).get("ece")
    latency = summary.get("latency") or {}
    return (
        f"{who}, Task set {record['task_set']}, {summary['n_attempted']} Tasks: "
        f"Valid {_percent(summary.get('schema_validity'))}, "
        f"Accuracy {_percent(summary.get('accuracy'))}, "
        f"Brier {_number(summary.get('brier_mean'))}, ECE {_number(ece)}, "
        f"client p50 {_number(latency.get('p50_s'))} s, p95 {_number(latency.get('p95_s'))} s"
    )


def _percent(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _number(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _write_decisions(path: Path, tasks: list[_Task], results: list[dict]) -> None:
    """One line per Task jevbench attempted: its served Distribution beside its expected Label.

    jevbench's results carry neither the expected Label nor the Question type, so they are joined
    here with the Tasks on `task_id`. The calibration fitter reads this file. Each line holds:
    `task_id`; `task_set` (the part: easy, original, hard); `question_type`; `labels` (the Label
    set); `expected` (a Label, always a string); `distribution` (as served, keyed by Label, or
    null); jevbench's `valid`, `predicted` and `correct`; `latency_s` (client); and the answer's
    `runtime` (Decider, model, revision, device, the `calibration` applied or null, server
    `total_ms` and `decider_ms`).
    """
    by_id = {t.task["id"]: t for t in tasks}
    with path.open("x", encoding="utf-8") as out:
        for result in results:
            t = by_id[result["task_id"]]
            expected = t.task["expected"]
            decision = {
                "task_id": result["task_id"],
                "task_set": t.part,
                "question_type": t.task["question"]["type"],
                "labels": t.task["labels"],
                "expected": None if expected is None else str(expected),
                "distribution": result["probs_as_returned"],
                "valid": result["valid"],
                "predicted": result["predicted"],
                "correct": result["correct"],
                "latency_s": result["latency_s"],
                "runtime": result["runtime"],
            }
            out.write(json.dumps(decision, ensure_ascii=False) + "\n")


def _wait_until_ready(endpoint: str, within_s: float) -> None:
    """Returns once the server's health reports ready; jevbench stops after 3 failures in a row."""
    health = endpoint.rstrip("/") + "/health"
    deadline = time.monotonic() + within_s
    while True:
        try:
            with _NO_PROXY.open(health, timeout=10):
                return
        except urllib.error.HTTPError as e:
            said = e.read().decode(errors="replace")
            if e.code != 503 or _status(said) != "loading":
                raise BenchFailed(
                    f"the server at {endpoint} is not ready: HTTP {e.code} {said}"
                ) from e
        except OSError as e:  # refused, unreachable, timed out
            raise BenchFailed(f"no server answers at {endpoint} ({e})") from e
        if time.monotonic() > deadline:
            raise BenchFailed(f"the server at {endpoint} is still loading after {within_s:g} s")
        time.sleep(0.5)


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _status(health: str) -> str | None:
    try:
        return json.loads(health).get("status")
    except (ValueError, AttributeError):
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _server(results: list[dict]) -> dict | None:
    """What the server ran, from the `runtime` its answers carry; None if no answer carried one.

    `calibration` is the calibration the server applied, null when it served raw Distributions.
    """
    for result in results:
        if runtime := result.get("runtime"):
            ran = {field.name: runtime.get(field.name) for field in fields(Runtime)}
            return {**ran, "calibration": runtime.get("calibration")}
    return None


class _Task(NamedTuple):
    part: str
    """The part of the Task set it comes from, as `easy`."""
    line: str
    """Its JSONL line, verbatim."""
    task: dict


def _tasks(files: list[tuple[str, Path]]) -> list[_Task]:
    """Every Task in `files`, in file order."""
    return [
        _Task(part, line + "\n", json.loads(line))
        for part, path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _spread(tasks: list[_Task], n: int) -> list[_Task]:
    """`n` of `tasks`, taken in turn from each (part, Question type) group, kept in file order.

    So even a small smoke run asks every Question type the Task set has.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault((t.part, t.task["question"]["type"]), []).append(i)
    in_turn = [i for round_ in zip_longest(*groups.values()) for i in round_ if i is not None]
    return [tasks[i] for i in sorted(in_turn[:n])]


def _new_run_dir(runs_dir: Path, task_set: str, limit: int | None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}-{task_set}" + (f"-limit{limit}" if limit is not None else "")
    runs_dir.mkdir(parents=True, exist_ok=True)
    for attempt in count(1):
        run = runs_dir / (name if attempt == 1 else f"{name}-{attempt}")
        try:
            run.mkdir()  # never an existing directory: jevbench's files there are create-only
        except FileExistsError:
            continue
        return run


def _jevbench(
    checkout: Path, env: dict[str, str], *args: str | Path
) -> subprocess.CompletedProcess[str]:
    """Runs jevbench's CLI from its checkout (never `uv run` there: it has its own pyproject).

    `run` reports its progress on the console; `summarize`'s stdout is the summary, kept.
    """
    command = [sys.executable, "-m", "jevbench.cli", *map(str, args)]
    stdout = subprocess.PIPE if args[0] == "summarize" else None
    return subprocess.run(command, cwd=checkout, env=env, stdout=stdout, text=True)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
