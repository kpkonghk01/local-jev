"""Calibration: one temperature per Question type, applied to a Decider's raw Distribution.

Temperature scaling divides a Distribution's log-probabilities by T and renormalises, so p_k
becomes p_k^(1/T) / Σ_j p_j^(1/T). T > 1 softens an over-confident Distribution, T < 1 sharpens
an under-confident one, T = 1 changes nothing. It never reorders the Labels, so the argmax (and
with it Accuracy) never changes; only calibration metrics move.

The fitter finds T for each Question type by minimising the negative log-likelihood of the
expected Labels over a benchmark run's raw Distributions, and reports k-fold cross-validated
Brier and ECE computed exactly as jevbench's `summarize` computes them.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_jev.decider import Label, PinnedModel, Runtime
from local_jev.distribution import argmax, normalised, softmax
from local_jev.files import read_json, read_jsonl, write_json

log = logging.getLogger(__name__)

QUESTION_TYPES = ("noul", "choice", "score")
CALIBRATIONS_DIR = Path(__file__).with_name("calibrations")
"""Where each Decider's committed calibration lives, as `<decider>.json`."""
REPORT_FILE = "calibration-report.json"
"""The fitter's cross-validated report, written into the run it was fitted on."""
FOLDS = 5
TEMPERATURE_RANGE = (0.05, 20.0)
"""The fitted T is searched for within these bounds."""
_EPSILON = 1e-12
"""A served probability of exactly 0 is read as this in the likelihood, so its log is finite."""


@dataclass(frozen=True)
class Calibration:
    """One temperature per Question type; a type without one is left as it is (T = 1).

    `fitted_from` names the bench run it was fitted on, when known.
    """

    temperatures: Mapping[str, float] = field(default_factory=dict)
    fitted_from: str | None = None

    def apply(self, question_type: str, distribution: Mapping[Label, float]) -> dict[Label, float]:
        """`distribution`, a Distribution for a Question of `question_type`, calibrated."""
        return _tempered(distribution, self.temperatures.get(question_type, 1.0))

    def describe(self) -> dict:
        """What the server says in every answer's `runtime` about the calibration it applies."""
        return {"temperatures": dict(self.temperatures), "fitted_from": self.fitted_from}


def load_calibration(directory: Path, runtime: Runtime) -> Calibration | None:
    """The calibration in `directory` fitted for the Decider `runtime` describes, if there is one.

    Only `<directory>/<decider>.json` is read, so Deciders never share a calibration. One fitted
    for another model or revision is not applied (None, with a warning). A file that is not a
    calibration raises ValueError, naming it.
    """
    path = directory / f"{runtime.decider}.json"
    if not path.is_file():
        log.info("no calibration for %s at %s: serving raw Distributions", runtime.decider, path)
        return None
    try:
        written = read_json(path)
        fitted_for = PinnedModel.of(written)
        temperatures = _temperatures(written["temperatures"])
        fitted_from = (written.get("fitted_from") or {}).get("run")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"{path} is not a calibration: {e}") from e
    if fitted_for != runtime.pinned_model:
        log.warning(
            "%s was fitted for %s, not %s: serving raw Distributions",
            path,
            fitted_for,
            runtime.pinned_model,
        )
        return None
    log.info("calibrating with %s: %s", path, temperatures)
    return Calibration(temperatures, fitted_from)


def _temperatures(written: object) -> dict[str, float]:
    if not isinstance(written, dict):
        raise ValueError(f'"temperatures" maps Question types to numbers, not {written!r}')
    for question_type, t in written.items():
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"{question_type!r} is not a Question type ({', '.join(QUESTION_TYPES)})"
            )
        if isinstance(t, bool) or not isinstance(t, int | float) or not 0 < t < math.inf:
            raise ValueError(f"the {question_type} temperature is {t!r}, not a positive number")
    return {question_type: float(t) for question_type, t in written.items()}


def _tempered(distribution: Mapping[Label, float], temperature: float) -> dict[Label, float]:
    if temperature == 1.0:
        return dict(distribution)
    logs = {label: math.log(p) if p > 0 else -math.inf for label, p in distribution.items()}
    return softmax({label: log / temperature for label, log in logs.items()})


@dataclass(frozen=True)
class TaskDecision:
    """One Task's raw Distribution beside its expected Label.

    `distribution` is None when the Task got no valid answer: it counts as wrong in Accuracy and
    is left out of Brier, ECE and the fit, as jevbench leaves it out.
    """

    question_type: str
    expected: Label
    distribution: Mapping[Label, float] | None


@dataclass(frozen=True)
class Fitted:
    """The temperatures fitted on every Decision, and the cross-validated report on them."""

    temperatures: dict[str, float]
    report: dict


class CalibrationRefused(Exception):
    """A run the fitter will not fit on, and why."""


def calibrate(run: Path, calibrations_dir: Path = CALIBRATIONS_DIR) -> dict:
    """Fits calibration on a finished `local-jev bench` run and returns the report.

    Writes the calibration of the Decider that served the run to `<calibrations_dir>/<decider>.json`
    (replacing any earlier one), and the report to `<run>/calibration-report.json`. Refuses a run
    served with calibration: temperatures are fitted on raw Distributions only.
    """
    lines = _decision_lines(run)
    served = _served_raw_by_one_decider(run, lines)
    decisions = [_task_decision(line) for line in lines if line.get("expected") is not None]
    if not any(d.distribution is not None for d in decisions):
        raise CalibrationRefused(f"{run} has no valid answer to fit on")
    fitted = fit(decisions)
    task_set = _run_record(run).get("task_set")

    calibrations_dir.mkdir(parents=True, exist_ok=True)
    calibration_file = calibrations_dir / f"{served['decider']}.json"
    calibration = {
        "decider": served["decider"],
        "model": served["model"],
        "revision": served["revision"],
        "temperatures": fitted.temperatures,
        "fitted_from": {
            "run": run.name,
            "task_set": task_set,
            "n_calibration": fitted.report["cross_validation"]["raw"]["n_calibration"],
        },
    }
    write_json(calibration_file, calibration)
    report = {
        "run": run.name,
        "task_set": task_set,
        **{key: served.get(key) for key in ("decider", "model", "revision", "device")},
        "temperatures": fitted.temperatures,
        "calibration_file": str(calibration_file),
        **fitted.report,
    }
    write_json(run / REPORT_FILE, report)
    return report


def _decision_lines(run: Path) -> list[dict]:
    path = run / "decisions.jsonl"
    if not path.is_file():
        raise CalibrationRefused(
            f"{run} has no decisions.jsonl: it is not a finished `local-jev bench` run"
        )
    return read_jsonl(path)


def _served_raw_by_one_decider(run: Path, lines: Sequence[dict]) -> dict:
    """The `runtime` of the server that answered every Decision in `run`, uncalibrated.

    A runtime without a `calibration` field comes from a server older than calibration: raw.
    """
    runtimes = [line["runtime"] for line in lines if line.get("runtime")]
    if not runtimes:
        raise CalibrationRefused(f"no answer in {run} says which Decider served it")
    for runtime in runtimes:
        if runtime.get("calibration"):
            raise CalibrationRefused(
                f"{run} was served with calibration {runtime['calibration']}, and temperatures "
                "are fitted on raw Distributions only: serve with --no-calibration, bench again "
                "and calibrate that run"
            )
    served_by = {PinnedModel.of(r) for r in runtimes}
    if len(served_by) > 1:
        names = ", ".join(sorted(map(str, served_by)))
        raise CalibrationRefused(f"{run} was served by more than one Decider: {names}")
    return runtimes[0]


def _task_decision(line: dict) -> TaskDecision:
    """A decisions.jsonl line as a TaskDecision, its Distribution rescaled to sum to 1 as
    jevbench rescales an answer it accepts."""
    distribution = line.get("distribution") if line.get("valid") else None
    if distribution is not None:
        distribution = normalised(distribution)
    return TaskDecision(line["question_type"], str(line["expected"]), distribution)


def _run_record(run: Path) -> dict:
    path = run / "run.json"
    return read_json(path) if path.is_file() else {}


def fit(decisions: Sequence[TaskDecision], folds: int = FOLDS, seed: int = 0) -> Fitted:
    """Fits one temperature per Question type on `decisions` and cross-validates it.

    Each Decision is held out once, in one of `folds` folds (stratified by Question type, shuffled
    with `seed`): it is calibrated with the temperatures fitted on the other folds. Brier and ECE
    are then computed over the held-out Distributions, raw and calibrated, pooled over every
    Question type as jevbench pools them, and per Question type.
    """
    decisions = list(decisions)
    fold_of = _folds(decisions, folds, seed)
    calibrated: list[Mapping[Label, float] | None] = [None] * len(decisions)
    fold_temperatures = []
    for fold in range(folds):
        training = [d for d, f in zip(decisions, fold_of, strict=True) if f != fold]
        calibration = Calibration(_fit_temperatures(training))
        fold_temperatures.append(dict(calibration.temperatures))
        for i, d in enumerate(decisions):
            if fold_of[i] == fold and d.distribution is not None:
                calibrated[i] = calibration.apply(d.question_type, d.distribution)
    raw = [d.distribution for d in decisions]

    by_type = {}
    for question_type in sorted({d.question_type for d in decisions}):
        of_type = [i for i, d in enumerate(decisions) if d.question_type == question_type]
        by_type[question_type] = {
            "raw": _metrics([decisions[i] for i in of_type], [raw[i] for i in of_type]),
            "calibrated": _metrics(
                [decisions[i] for i in of_type], [calibrated[i] for i in of_type]
            ),
        }
    report = {
        "cross_validation": {
            "folds": folds,
            "seed": seed,
            "fold_temperatures": fold_temperatures,
            "raw": _metrics(decisions, raw),
            "calibrated": _metrics(decisions, calibrated),
            "by_question_type": by_type,
        }
    }
    return Fitted(_fit_temperatures(decisions), report)


def _folds(decisions: Sequence[TaskDecision], folds: int, seed: int) -> list[int]:
    """The fold each Decision is held out in, dealt round-robin within each Question type."""
    rng = random.Random(seed)
    fold_of = [0] * len(decisions)
    for question_type in sorted({d.question_type for d in decisions}):
        of_type = [i for i, d in enumerate(decisions) if d.question_type == question_type]
        rng.shuffle(of_type)
        for n, i in enumerate(of_type):
            fold_of[i] = n % folds
    return fold_of


def _fit_temperatures(decisions: Sequence[TaskDecision]) -> dict[str, float]:
    """The NLL-minimising temperature for each Question type `decisions` has a valid answer for."""
    by_type: dict[str, list[tuple[list[float], int]]] = {}
    for d in decisions:
        if d.distribution is None:
            continue
        labels = list(d.distribution)
        logs = [math.log(max(d.distribution[label], _EPSILON)) for label in labels]
        by_type.setdefault(d.question_type, []).append((logs, labels.index(d.expected)))
    return {question_type: _temperature(rows) for question_type, rows in sorted(by_type.items())}


def _temperature(rows: Sequence[tuple[list[float], int]]) -> float:
    """The T that minimises the mean NLL of the expected Labels, within TEMPERATURE_RANGE.

    The NLL is convex in 1/T, so it has one minimum along log T: a golden-section search finds it.
    """

    def nll(log_t: float) -> float:
        inverse = math.exp(-log_t)
        total = 0.0
        for logs, expected in rows:
            scaled = [z * inverse for z in logs]
            top = max(scaled)
            total += top + math.log(sum(math.exp(s - top) for s in scaled)) - scaled[expected]
        return total / len(rows)

    lo, hi = (math.log(t) for t in TEMPERATURE_RANGE)
    golden = (math.sqrt(5) - 1) / 2
    a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fa, fb = nll(a), nll(b)
    while hi - lo > 1e-7:
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - golden * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + golden * (hi - lo)
            fb = nll(b)
    return round(math.exp((lo + hi) / 2), 4)


def _metrics(
    decisions: Sequence[TaskDecision], distributions: Sequence[Mapping[Label, float] | None]
) -> dict:
    """Accuracy, Brier and ECE as jevbench's `summarize` computes them.

    Accuracy counts a Task without a valid answer as wrong. Brier is the multi-class sum
    Σ_k (p_k − [k is expected])² over the Label set, averaged; ECE is top-label, over 10
    equal-width bins of confidence. Both leave out Tasks without a valid answer.
    """
    correct = 0
    briers = []
    pairs = []
    for d, p in zip(decisions, distributions, strict=True):
        if p is None:
            continue
        hit = argmax(p) == d.expected
        correct += hit
        briers.append(sum((q - (label == d.expected)) ** 2 for label, q in p.items()))
        pairs.append((max(p.values()), hit))
    return {
        "n": len(decisions),
        "n_calibration": len(briers),
        "accuracy": correct / len(decisions) if decisions else None,
        "brier": sum(briers) / len(briers) if briers else None,
        "ece": _ece(pairs) if pairs else None,
    }


def _ece(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> float:
    """Top-label expected calibration error over (confidence, correct) pairs, as jevbench's."""
    counts = [0] * bins
    confidence = [0.0] * bins
    hits = [0] * bins
    for conf, hit in pairs:
        conf = min(max(conf, 0.0), 1.0)
        b = min(int(conf * bins), bins - 1)
        counts[b] += 1
        confidence[b] += conf
        hits[b] += hit
    return sum(
        (n / len(pairs)) * abs(hits[b] / n - confidence[b] / n) for b, n in enumerate(counts) if n
    )
