"""The calibration fitter: a run's raw Distributions and expected Labels in, temperatures and a
cross-validated report out (synthetic data with a known truth)."""

import json
import math
import random
from pathlib import Path

import pytest

from local_jev.calibration import Calibration, TaskDecision, fit
from local_jev.cli import main

D1_RUNTIME = {
    "decider": "d1",
    "model": "LiquidAI/d1-2B",
    "revision": "652f920e544ea37f81e4b0ce89b81d4ed0da90b1",
    "device": "mps",
    "calibration": None,
    "total_ms": 98.5,
    "decider_ms": 97.9,
}


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    weights = [math.exp(z - top) for z in logits]
    return [w / sum(weights) for w in weights]


def _synthetic(
    n: int, question_type: str = "choice", sharpened_by: float = 1.0, seed: int = 1
) -> list[TaskDecision]:
    """`n` Decisions whose expected Labels are drawn from a true Distribution, reported with its
    logits multiplied by `sharpened_by`: 1 is calibrated, 2 over-confident (true T = 2)."""
    rng = random.Random(seed)
    decisions = []
    for _ in range(n):
        if question_type == "noul":
            labels = ["yes", "no"]
        elif question_type == "score":
            labels = [str(level) for level in range(4)]
        else:
            labels = [f"option_{i}" for i in range(rng.choice([3, 4, 5, 6]))]
        logits = [rng.gauss(0, 1.5) for _ in labels]
        truth = _softmax(logits)
        (expected,) = rng.choices(labels, weights=truth)
        reported = _softmax([sharpened_by * z for z in logits])
        decisions.append(
            TaskDecision(question_type, expected, dict(zip(labels, reported, strict=True)))
        )
    return decisions


def test_over_confident_decisions_get_a_temperature_above_one_and_a_lower_brier():
    fitted = fit(_synthetic(2000, sharpened_by=2.0))
    cv = fitted.report["cross_validation"]

    assert fitted.temperatures["choice"] == pytest.approx(2.0, abs=0.2)
    assert cv["calibrated"]["brier"] < cv["raw"]["brier"]


@pytest.mark.parametrize("question_type", ["choice", "noul", "score"])
def test_already_calibrated_decisions_get_a_temperature_of_about_one(question_type):
    fitted = fit(_synthetic(3000, question_type))

    assert fitted.temperatures == {question_type: pytest.approx(1.0, abs=0.1)}


def test_each_question_type_gets_its_own_temperature():
    mixed = (
        _synthetic(2000, "choice", sharpened_by=2.0, seed=1)
        + _synthetic(2000, "noul", seed=2)
        + _synthetic(2000, "score", sharpened_by=0.5, seed=3)
    )
    fitted = fit(mixed)

    assert fitted.temperatures == {
        "choice": pytest.approx(2.0, abs=0.2),
        "noul": pytest.approx(1.0, abs=0.1),
        "score": pytest.approx(0.5, abs=0.05),
    }


def _most_likely(distribution) -> str:
    return max(sorted(distribution), key=distribution.__getitem__)


@pytest.mark.parametrize("temperature", [0.05, 0.5, 2.0, 20.0])
def test_calibration_never_changes_the_most_likely_label(temperature):
    decisions = _synthetic(500, sharpened_by=3.0) + _synthetic(500, "noul", sharpened_by=0.3)
    calibration = Calibration({"choice": temperature, "noul": temperature})

    for d in decisions:
        calibrated = calibration.apply(d.question_type, d.distribution)
        assert _most_likely(calibrated) == _most_likely(d.distribution)
        assert sum(calibrated.values()) == pytest.approx(1.0)


def test_cross_validated_accuracy_is_unchanged_by_calibration():
    cv = fit(_synthetic(1000, sharpened_by=2.0)).report["cross_validation"]

    assert cv["calibrated"]["accuracy"] == cv["raw"]["accuracy"]


def test_a_temperature_divides_log_probabilities_and_renormalises():
    calibration = Calibration({"choice": 2.0, "noul": 0.5})

    # T = 2: each p becomes sqrt(p), renormalised. T = 0.5: each p becomes p², renormalised.
    assert calibration.apply("choice", {"a": 0.64, "b": 0.36}) == pytest.approx(
        {"a": 0.8 / 1.4, "b": 0.6 / 1.4}
    )
    assert calibration.apply("noul", {"yes": 0.75, "no": 0.25}) == pytest.approx(
        {"yes": 0.9, "no": 0.1}
    )
    assert calibration.apply("score", {"0": 0.3, "1": 0.7}) == {"0": 0.3, "1": 0.7}


def test_raw_scores_are_computed_as_jevbench_computes_them():
    """Brier is the multi-class sum over the Label set (2·(p_yes − y)² for a noul); ECE is
    top-label over 10 equal-width bins; a Task without a valid answer is wrong but unscored."""
    decisions = [
        TaskDecision("choice", "a", {"a": 0.7, "b": 0.2, "c": 0.1}),
        TaskDecision("noul", "no", {"yes": 0.8, "no": 0.2}),
        TaskDecision("score", "1", None),
    ]
    raw = fit(decisions).report["cross_validation"]["raw"]

    # Brier: (0.09 + 0.04 + 0.01 + 0.64 + 0.64) / 2. ECE: ½·|1 − 0.7| + ½·|0 − 0.8|.
    assert raw == {
        "n": 3,
        "n_calibration": 2,
        "accuracy": pytest.approx(1 / 3),
        "brier": pytest.approx(0.71),
        "ece": pytest.approx(0.55),
    }


def _bench_run(
    runs: Path, decisions: list[TaskDecision], runtime: dict | None = None, name="run-public"
) -> Path:
    """A finished `local-jev bench` run directory holding `decisions`, as the runner writes it."""
    runtime = D1_RUNTIME if runtime is None else runtime
    run = runs / name
    run.mkdir(parents=True)
    with (run / "decisions.jsonl").open("w", encoding="utf-8") as out:
        for i, d in enumerate(decisions):
            labels = list(d.distribution) if d.distribution else [d.expected, "other"]
            predicted = _most_likely(d.distribution) if d.distribution else None
            line = {
                "task_id": f"synthetic-{i}",
                "task_set": "easy",
                "question_type": d.question_type,
                "labels": labels,
                "expected": d.expected,
                "distribution": d.distribution,
                "valid": d.distribution is not None,
                "predicted": predicted,
                "correct": predicted == d.expected,
                "latency_s": 0.1,
                "runtime": runtime,
            }
            out.write(json.dumps(line) + "\n")
    server = {k: runtime.get(k) for k in ("decider", "model", "revision", "device")}
    record = {"task_set": "public", "n_tasks": len(decisions), "server": server}
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run


def _calibrate(run: Path, calibrations: Path) -> None:
    main(["calibrate", str(run), "--calibrations-dir", str(calibrations)])


def test_calibrate_writes_the_deciders_calibration_for_its_model_and_revision(tmp_path):
    run = _bench_run(tmp_path / "runs", _synthetic(600, sharpened_by=2.0))
    _calibrate(run, tmp_path / "calibrations")
    written = json.loads((tmp_path / "calibrations" / "d1.json").read_text())

    assert written["decider"] == "d1"
    assert written["model"] == "LiquidAI/d1-2B"
    assert written["revision"] == "652f920e544ea37f81e4b0ce89b81d4ed0da90b1"
    assert written["temperatures"] == {"choice": pytest.approx(2.0, abs=0.3)}


def test_calibrate_writes_a_cross_validated_report_into_the_run(tmp_path):
    decisions = _synthetic(600, sharpened_by=2.0) + [TaskDecision("noul", "yes", None)]
    run = _bench_run(tmp_path / "runs", decisions)
    _calibrate(run, tmp_path / "calibrations")
    report = json.loads((run / "calibration-report.json").read_text())
    cv = report["cross_validation"]

    assert (report["decider"], report["model"], report["device"]) == (
        "d1",
        "LiquidAI/d1-2B",
        "mps",
    )
    assert report["temperatures"] == {"choice": pytest.approx(2.0, abs=0.3)}
    assert cv["folds"] == 5
    assert (cv["raw"]["n"], cv["raw"]["n_calibration"]) == (601, 600)
    assert cv["calibrated"]["accuracy"] == cv["raw"]["accuracy"]
    assert cv["calibrated"]["brier"] < cv["raw"]["brier"]
    assert cv["calibrated"]["ece"] < cv["raw"]["ece"]
    assert set(cv["by_question_type"]) == {"choice", "noul"}


def test_calibrate_refuses_a_run_served_with_calibration(tmp_path):
    served_calibrated = {**D1_RUNTIME, "calibration": {"temperatures": {"choice": 1.4}}}
    run = _bench_run(tmp_path / "runs", _synthetic(50), served_calibrated)

    with pytest.raises(SystemExit, match="served with calibration.*--no-calibration"):
        _calibrate(run, tmp_path / "calibrations")
    assert not (tmp_path / "calibrations" / "d1.json").exists()
    assert not (run / "calibration-report.json").exists()


def test_calibrate_takes_a_run_from_a_server_older_than_calibration_as_raw(tmp_path):
    """Such a server's runtime has no `calibration` field at all."""
    older = {k: v for k, v in D1_RUNTIME.items() if k != "calibration"}
    run = _bench_run(tmp_path / "runs", _synthetic(50), older)
    _calibrate(run, tmp_path / "calibrations")

    assert (tmp_path / "calibrations" / "d1.json").exists()


def test_calibrate_refuses_a_run_served_by_more_than_one_model(tmp_path):
    run = _bench_run(tmp_path / "runs", _synthetic(50))
    lines = (run / "decisions.jsonl").read_text().splitlines()
    other = json.loads(lines[-1])
    other["runtime"] = {**D1_RUNTIME, "revision": "0" * 40}
    (run / "decisions.jsonl").write_text("\n".join([*lines[:-1], json.dumps(other)]) + "\n")

    with pytest.raises(SystemExit, match="more than one Decider"):
        _calibrate(run, tmp_path / "calibrations")
    assert not (tmp_path / "calibrations" / "d1.json").exists()


def test_calibrate_refuses_a_directory_that_is_not_a_finished_run(tmp_path):
    with pytest.raises(SystemExit, match="not a finished `local-jev bench` run"):
        _calibrate(tmp_path, tmp_path / "calibrations")
