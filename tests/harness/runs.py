"""Finished `local-jev bench` run directories, written by hand with figures a test chooses.

Only the files and fields the comparison report reads are written, in the shapes `bench` and
`calibrate` write them (src/local_jev/bench.py, src/local_jev/calibration.py).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

D1 = {
    "decider": "d1",
    "model": "LiquidAI/d1-2B",
    "revision": "652f920e544ea37f81e4b0ce89b81d4ed0da90b1",
}
LFM3 = {
    "decider": "lfm3",
    "model": "LiquidAI/LFM3-1.7B",
    "revision": "0dc4268b561499c648c7ba7c9466bf3d8d01517a",
}
PUBLIC_PARTS = ("easy", "original", "hard")
ZH_TW_PARTS = ("zh-tw-easy", "zh-tw-original", "zh-tw-hard")


def bench_run(
    runs_dir: Path,
    name: str,
    decider: dict = D1,
    *,
    device: str | Sequence[str] = "mps",
    task_set: str = "public",
    parts: Sequence[str] = PUBLIC_PARTS,
    limit: int | None = None,
    n_tasks: int = 231,
    server_ms: float | Sequence[float] = 150.0,
    calibration: dict | None = None,
    accuracy: float = 0.8,
    brier: float = 0.15,
    ece: float = 0.05,
    valid: float = 1.0,
    client_p50_s: float = 0.2,
    client_p95_s: float = 0.4,
) -> Path:
    """A finished run of `n_tasks` Tasks; summary.json carries the jevbench figures given.

    `server_ms` is every answer's server `total_ms`, or one per Task; `device` likewise.
    `calibration` is the calibration the server applied (None: raw).
    """
    run = runs_dir / name
    run.mkdir(parents=True)
    each_ms = [server_ms] * n_tasks if isinstance(server_ms, int | float) else list(server_ms)
    devices = [device] * n_tasks if isinstance(device, str) else list(device)
    assert len(each_ms) == len(devices) == n_tasks
    with (run / "decisions.jsonl").open("w", encoding="utf-8") as out:
        for i, (ms, on) in enumerate(zip(each_ms, devices, strict=True)):
            runtime = {**decider, "device": on, "calibration": calibration}
            runtime |= {"total_ms": ms, "decider_ms": ms - 0.5}
            line = {
                "task_id": f"task-{i:03d}",
                "task_set": parts[i % len(parts)],
                "question_type": "noul",
                "labels": ["no", "yes"],
                "expected": "yes",
                "distribution": {"yes": 0.9, "no": 0.1},
                "valid": True,
                "predicted": "yes",
                "correct": True,
                "latency_s": ms / 1000 + 0.01,
                "runtime": runtime,
            }
            out.write(json.dumps(line) + "\n")
    summary = {
        "n_planned": n_tasks,
        "n_attempted": n_tasks,
        "n_scorable": n_tasks,
        "complete": True,
        "accuracy": accuracy,
        "brier_mean": brier,
        "ece": {"ece": ece, "n": n_tasks, "bins": []},
        "schema_validity": valid,
        "latency": {"n": n_tasks, "p50_s": client_p50_s, "p95_s": client_p95_s},
    }
    _write(run / "summary.json", summary)
    first_device = devices[0]
    _write(
        run / "run.json",
        {
            "jevbench": {
                "url": "https://github.com/fstandhartinger/jevbench.git",
                "commit": "609a664e81f46e03731600bd7bd9f1b7a61257c5",
            },
            "endpoint": "http://127.0.0.1:8000",
            "task_set": task_set,
            "limit": limit,
            "n_tasks": n_tasks,
            "started_utc": "2026-11-05T08:52:52+00:00",
            "finished_utc": "2026-11-05T08:55:14+00:00",
            "server": {**decider, "device": first_device, "calibration": calibration},
        },
    )
    return run


def calibration_report(
    run: Path, *, accuracy: float, brier: float, ece: float, raw_brier: float = 0.3
) -> Path:
    """Writes a `calibrate` report into `run`, with the cross-validated calibrated figures given."""
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    n = record["n_tasks"]
    _write(
        run / "calibration-report.json",
        {
            "run": run.name,
            "task_set": record["task_set"],
            **record["server"],
            "temperatures": {"choice": 1.1, "noul": 2.0, "score": 1.2},
            "calibration_file": "src/local_jev/calibrations/d1.json",
            "cross_validation": {
                "folds": 5,
                "seed": 0,
                "fold_temperatures": [],
                "raw": {
                    "n": n,
                    "n_calibration": n,
                    "accuracy": accuracy,
                    "brier": raw_brier,
                    "ece": 0.1,
                },
                "calibrated": {
                    "n": n,
                    "n_calibration": n,
                    "accuracy": accuracy,
                    "brier": brier,
                    "ece": ece,
                },
                "by_question_type": {},
            },
        },
    )
    return run


def report(directory: Path, *runs: Path) -> str:
    """The Markdown `local-jev report` writes on `runs`, written into `directory`."""
    from local_jev.cli import main

    output = directory / "report.md"
    main(["report", *map(str, runs), "--output", str(output)])
    return output.read_text(encoding="utf-8")


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
