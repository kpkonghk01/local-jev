"""`local-jev calibrate` on runs that real jevbench made against a server (fake d1 model)."""

import json
import math
import random
import zlib
from pathlib import Path

import pytest
from harness.fake_d1 import FakeD1Model, d1_choice, d1_noul, d1_score
from harness.server import listening

from local_jev.cli import main
from local_jev.d1 import D1Decider

PROJECT_CACHE = Path(__file__).parents[1] / ".cache"


def _over_confident(state, question: dict) -> dict:
    """A d1 answer that differs from Task to Task (seeded by the State) and is over-confident."""
    rng = random.Random(zlib.crc32(json.dumps(state, sort_keys=True).encode()))
    match question["type"]:
        case "choice":
            labels = list(question["criteria"])
        case "noul":
            labels = ["yes", "no"]
        case "score":
            labels = [str(level) for level in range(len(question["criteria"]))]
    logits = [3 * rng.gauss(0, 1.5) for _ in labels]
    weights = [math.exp(z - max(logits)) for z in logits]
    p = {label: w / sum(weights) for label, w in zip(labels, weights, strict=True)}
    match question["type"]:
        case "choice":
            return d1_choice(p)
        case "noul":
            return d1_noul(p["yes"])
        case "score":
            return d1_score([p[label] for label in labels])


def _serving_fake_d1(calibrations: Path | None = None):
    model = FakeD1Model(_over_confident)
    return listening(lambda: D1Decider(model, device="cpu"), calibrations=calibrations)


def _bench(endpoint: str, runs_dir: Path, *args: str) -> Path:
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    main(
        ["bench", "--endpoint", endpoint, "--runs-dir", str(runs_dir)]
        + ["--cache-dir", str(PROJECT_CACHE), *args]
    )
    (run,) = set(runs_dir.iterdir()) - before
    return run


def _calibrate(run: Path, calibrations: Path) -> dict:
    main(["calibrate", str(run), "--calibrations-dir", str(calibrations)])
    return json.loads((run / "calibration-report.json").read_text())


@pytest.fixture(scope="module")
def raw_run(tmp_path_factory) -> Path:
    """All 231 public Tasks against a raw server."""
    with _serving_fake_d1() as endpoint:
        return _bench(endpoint, tmp_path_factory.mktemp("runs"), "--task-set", "public")


def test_raw_scores_are_jevbenchs_own_for_the_same_run(raw_run, tmp_path):
    raw = _calibrate(raw_run, tmp_path)["cross_validation"]["raw"]
    summary = json.loads((raw_run / "summary.json").read_text())

    assert raw["n"] == summary["n_scorable"] == 231
    assert raw["n_calibration"] == summary["calibration_n"]
    assert raw["accuracy"] == pytest.approx(summary["accuracy"], abs=1e-12)
    assert raw["brier"] == pytest.approx(summary["brier_mean"], abs=1e-9)
    assert raw["ece"] == pytest.approx(summary["ece"]["ece"], abs=1e-9)


def test_a_fitted_calibration_is_served_and_a_run_served_with_it_is_refused(raw_run, tmp_path):
    calibrations = tmp_path / "calibrations"
    fitted = _calibrate(raw_run, calibrations)["temperatures"]
    with _serving_fake_d1(calibrations) as endpoint:
        calibrated_run = _bench(endpoint, tmp_path / "runs", "--limit", "5")
    served = json.loads((calibrated_run / "run.json").read_text())["server"]["calibration"]

    assert fitted["choice"] > 1
    assert served == {"temperatures": fitted, "fitted_from": raw_run.name}
    with pytest.raises(SystemExit, match="served with calibration"):
        _calibrate(calibrated_run, calibrations)
