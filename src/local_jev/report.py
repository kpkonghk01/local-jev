"""The comparison report: finished bench runs merged into one Markdown report.

Each row is one run (or the cross-validated calibration of one run) on one device: runs are
never pooled, so a latency figure never mixes hardware.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from local_jev.bench import JEVBENCH_COMMIT, RunFigures
from local_jev.calibration import REPORT_FILE
from local_jev.decider import PinnedModel
from local_jev.figures import number, percent
from local_jev.files import create_new, read_json, read_jsonl, utc_stamp


class ReportRefused(Exception):
    """Something given to the report that it will not report on, and why."""


PUBLIC_TASKS = 231


@dataclass(frozen=True)
class Row:
    """One line of the results table: one run, or the cross-validated calibration of one run."""

    pinned_model: PinnedModel
    """The Decider that answered, with its model and revision."""
    kind: Literal["raw", "cross-validated", "served calibrated"]
    calibration: str
    on_public_tasks: bool
    """Whether the run is of all 231 public Tasks in English, which the verdict is on."""
    language: str
    task_set: str
    tasks: int
    accuracy: float | None
    brier: float | None
    ece: float | None
    valid: float | None
    client_ms: tuple[float | None, float | None]
    server_ms: tuple[float | None, float | None]
    """p50 and p95: jevbench's wall clock around each request, and the server's `total_ms`.
    None when the run was answered on more than one device."""
    devices: tuple[str, ...]
    """Each device (cuda, mps, cpu) the run's answers say they were decided on, usually one."""
    run: str
    served_calibration: dict | None = None
    """The calibration the server applied, for a run served calibrated: its Brier and ECE are
    withheld (None), since they may be in-sample."""


@dataclass(frozen=True)
class Reference:
    """A system jevbench has published figures for, on one suite of Tasks; None: not published."""

    system: str
    key: str
    """Its key in jevbench's results files."""
    suite: str
    tasks: int
    accuracy: float
    brier: float | None
    ece: float | None
    valid: float | None
    client_p50_s: float
    client_p95_s: float
    hardware: str
    source: str


_PER_TASK = "results/v1.2/jevbench-v1.2-per-task.json"
_V1_RESULTS = "results/jevbench-v1-results.json"
_PUBLIC_231 = "the 231 public Tasks"
_V1_SUITE = "older v1 suite (242 decisions, 2026-09-19), not the public Tasks"
_JEVBENCH_AT = f"jevbench@{JEVBENCH_COMMIT[:7]}"
_JEV = "Jev 1.13.0 (TypeSafe AI)"
_JEV_HARDWARE = "TypeSafe's production API (hardware not published)"
_SYSTEM_ONE_OPEN = "system-one-open (Gemma 4 E2B with LoRA, on an L4)"
_SYSTEM_ONE_OPEN_HARDWARE = "one L4 GPU (the author's public demo endpoint on Modal)"

REFERENCES = (
    # On the public Tasks jevbench publishes each Task's outcome (c, w, f, n) and latency, not its
    # Distribution: Accuracy is the share of `c`, p50/p95 are over the 231 latencies.
    Reference(
        _JEV, "jev-1.13.0", _PUBLIC_231, 231, 200 / 231, None, None, None, 0.665, 0.8025,
        _JEV_HARDWARE,
        f"{_JEVBENCH_AT} `{_PER_TASK}`, `systems.jev-1.13.0.public_tasks`",
    ),
    Reference(
        _SYSTEM_ONE_OPEN, "system-one-open", _PUBLIC_231, 231, 169 / 231, None, None, None,
        0.649, 1.0475,
        _SYSTEM_ONE_OPEN_HARDWARE,
        f"{_JEVBENCH_AT} `{_PER_TASK}`, `systems.system-one-open.public_tasks`",
    ),
    # The spec's reference points: the v1 suite's `overall` figures, as in its RESULTS.md table.
    Reference(
        _JEV, "jev-1.13.0", _V1_SUITE, 242, 0.9628099173553719, 0.05558677685950413,
        0.027438016528926218, 1.0, 0.6524335257709026, 0.7221905551850795,
        _JEV_HARDWARE,
        f"{_JEVBENCH_AT} `{_V1_RESULTS}`, `overall` of jev-1.13.0 (`RESULTS.md` line 32)",
    ),
    Reference(
        _SYSTEM_ONE_OPEN, "system-one-open", _V1_SUITE, 242, 0.9008264462809917,
        0.13816078684373112, 0.06800454654184858, 1.0, 0.6517308317124844, 0.7724301926791667,
        _SYSTEM_ONE_OPEN_HARDWARE,
        f"{_JEVBENCH_AT} `{_V1_RESULTS}`, `overall` of system-one-open (`RESULTS.md` line 36)",
    ),
)  # fmt: skip
"""jevbench's published reference points, all at the pinned commit."""


@dataclass(frozen=True)
class Comparison:
    markdown: str
    overall: str
    """The overall verdict, as one line of plain text."""
    default_decider: str
    """The default Decider picked, and why, as one line of plain text."""


def compare(runs: Sequence[Path], deciders: Sequence[str] = ("d1", "lfm3")) -> Comparison:
    """The comparison report on `runs`, finished `local-jev bench` run directories.

    `deciders` names every Decider (as `serve --decider` does) the verdict waits for. Raises
    ReportRefused for a directory that is not a finished run, or a run that mixes languages.
    """
    rows = [row for run in runs for row in _rows(run)]
    notes = [*_RESULT_NOTES]
    for r in rows:
        if r.served_calibration is not None:
            notes.append(
                f"- `{r.run}` was served with calibration fitted on "
                f"{r.served_calibration.get('fitted_from') or 'an unnamed run'}: its Brier and ECE "
                "are withheld, since they are in-sample if those temperatures were fitted on "
                "these Tasks. Bench a raw server (`serve --no-calibration`) and calibrate that "
                "run for cross-validated figures."
            )
    verdict = _verdict(rows, deciders)
    overall = _overall(verdict)
    default = _default_decider_line([r for r in rows if r.on_public_tasks], deciders)
    commits = sorted({read_json(run / "run.json")["jevbench"]["commit"][:7] for run in runs})
    markdown = "\n".join(
        [
            "# local-jev comparison report",
            "",
            f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} from {len(runs)} "
            f"`local-jev bench` run{'s' if len(runs) != 1 else ''} "
            f"(jevbench {', '.join(commits)}).",
            "",
            "## Results",
            "",
            *_results_table(rows),
            "",
            *notes,
            "",
            "## jevbench reference points",
            "",
            *_references_table(REFERENCES),
            "",
            *_REFERENCE_NOTES,
            "",
            "## Feasibility verdict",
            "",
            _VERDICT_INTRO,
            "",
            *_table(
                ("Criterion", "Status", "Evidence"),
                [(c.criterion, f"**{c.status}**", "; ".join(c.evidence)) for c in verdict],
            ),
            "",
            overall.markdown,
            "",
            default.markdown,
            "",
        ]
    )
    return Comparison(markdown, overall.text, default.text)


def write_new_report(markdown: str, runs_dir: Path) -> Path:
    """Writes `markdown` to a new `report-<UTC time>.md` in `runs_dir`; never over another."""

    def write(path: Path) -> Path:
        with path.open("x", encoding="utf-8") as out:
            out.write(markdown)
        return path

    return create_new(write, runs_dir, f"report-{utc_stamp()}", ".md")


@dataclass(frozen=True)
class _Line:
    """A line of the report that starts in bold."""

    bold: str
    rest: str = ""

    @property
    def markdown(self) -> str:
        return f"**{self.bold}**{self.rest}"

    @property
    def text(self) -> str:
        return self.bold + self.rest


def _overall(verdict: Sequence[Checked]) -> _Line:
    """Feasible needs every criterion to pass: any fail fails it, else any pending holds it."""
    numbers = {
        status: [c.criterion.split(".")[0] for c in verdict if c.status == status]
        for status in ("fail", "pending")
    }
    status = "fail" if numbers["fail"] else "pending" if numbers["pending"] else "pass"
    why = "; ".join(f"{s}: {', '.join(n)}" for s, n in numbers.items() if n)
    return _Line(f"Overall: {status}", f" ({why})" if why else "")


_VERDICT_INTRO = (
    "Checked on the 231 public Tasks in English only (`--task-set public`, no `--limit`), "
    "zero-shot plus calibration, with cross-validated figures. The Traditional-Chinese rows are "
    "reported apart and do not affect the verdict. Feasible, worth further investment, needs "
    "every criterion to pass."
)


type Status = Literal["pass", "fail", "pending"]
"""A criterion's status; pending: not decided until more runs are in."""


@dataclass(frozen=True)
class Checked:
    criterion: str
    status: Status
    evidence: list[str]


def _verdict(rows: Sequence[Row], deciders: Sequence[str]) -> list[Checked]:
    public = [r for r in rows if r.on_public_tasks]
    return [
        _valid_answers(public, deciders),
        _accuracy_and_brier(public, deciders),
        _recorded_on_mps(public, deciders),
        _on_gpu(public),
    ]


GPU_P50_BAR_MS = 150.0


def _on_gpu(public: Sequence[Row]) -> Checked:
    """The default Decider's server-side p50 on CUDA, against the bar set for an L4-class GPU.

    Each CUDA run is checked on its own, and all must reach the bar. A run says only that it ran
    on `cuda`, not on which GPU.
    """
    criterion = f"3b. Server-side p50 ≤ {GPU_P50_BAR_MS:.0f} ms on an L4-class GPU"
    chosen = _default_decider(public)
    on_cuda = [
        r for r in _timed_on(public, "cuda") if chosen and r.pinned_model == chosen.pinned_model
    ]
    if not on_cuda:
        who = chosen.pinned_model if chosen else "the default Decider"
        return Checked(
            criterion, "pending", [f"no run of {who} on cuda on the {PUBLIC_TASKS} public Tasks"]
        )
    reached = all(r.server_ms[0] <= GPU_P50_BAR_MS for r in on_cuda)
    return Checked(
        criterion,
        "pass" if reached else "fail",
        [_server_p50(r) for r in on_cuda] + ["that the GPU is L4-class is not checked"],
    )


def _recorded_on_mps(public: Sequence[Row], deciders: Sequence[str]) -> Checked:
    """The spec sets no bar on the Mac: each Decider's server-side p50 on MPS is recorded."""
    on_mps = _timed_on(public, "mps")
    missing = [d for d in deciders if not any(r.pinned_model.decider == d for r in on_mps)]
    return Checked(
        "3a. Server-side p50 on the Mac (MPS) is recorded for every Decider",
        "pending" if missing else "pass",
        [_server_p50(r) for r in on_mps]
        + [f"no run of {d} on mps on the {PUBLIC_TASKS} public Tasks" for d in missing],
    )


def _timed_on(public: Sequence[Row], device: str) -> list[Row]:
    """One row per run answered on `device` alone with server timing."""
    return [
        r
        for r in public
        if r.kind != "cross-validated" and r.devices == (device,) and r.server_ms[0] is not None
    ]


def _server_p50(r: Row) -> str:
    return f"{r.pinned_model}: server p50 {r.server_ms[0]:.0f} ms ({r.run})"


ACCURACY_BAR, BRIER_BAR = 0.80, 0.20


@dataclass(frozen=True)
class Standing:
    """How one Decider (at one model and revision) did over its runs on the public Tasks."""

    pinned_model: PinnedModel
    rows: list[Row]

    @property
    def all_valid(self) -> bool:
        return all(r.valid is not None and r.valid >= 1 for r in self.rows)

    @property
    def accuracy(self) -> float:
        return max((r.accuracy for r in self.rows if r.accuracy is not None), default=0.0)

    @property
    def best(self) -> Row | None:
        """Its best raw or cross-validated figure: one reaching both bars, else the lowest Brier.

        A run served calibrated has no Brier here: it may be in-sample.
        """
        scored = [r for r in self.rows if r.brier is not None and r.accuracy is not None]
        return min(scored, key=lambda r: (not _meets_bar(r), r.brier), default=None)

    @property
    def status(self) -> Status:
        """Whether it reaches Accuracy and Brier bars.

        Calibration never changes the argmax, so below the Accuracy bar it fails whether calibrated
        or not; above it, with too high a Brier, it is pending until it is calibrated.
        """
        if self.best is not None and _meets_bar(self.best):
            return "pass"
        calibrated = any(r.kind == "cross-validated" for r in self.rows)
        return "pending" if self.accuracy >= ACCURACY_BAR and not calibrated else "fail"

    @property
    def figures(self) -> str:
        best = self.best
        figures = (
            f"Accuracy {percent(best.accuracy)}, Brier {number(best.brier)} "
            f"({best.calibration}, {best.run})"
            if best is not None
            else f"Accuracy {percent(self.accuracy)}, Brier withheld"
        )
        return f"not calibrated yet; {figures}" if self.status == "pending" else figures


def _meets_bar(r: Row) -> bool:
    return r.accuracy >= ACCURACY_BAR and r.brier <= BRIER_BAR


def _standings(public: Sequence[Row]) -> list[Standing]:
    grouped: dict[PinnedModel, list[Row]] = {}
    for r in public:
        grouped.setdefault(r.pinned_model, []).append(r)
    return [Standing(pinned_model, rows) for pinned_model, rows in grouped.items()]


def _accuracy_and_brier(public: Sequence[Row], deciders: Sequence[str]) -> Checked:
    """Passes when some Decider reaches both bars on a raw or cross-validated figure."""
    standings = _standings(public)
    statuses = {s.status for s in standings}
    missing = _missing(public, deciders)
    if "pass" in statuses:
        status = "pass"
    else:
        status = "pending" if missing or "pending" in statuses else "fail"
    return Checked(
        f"2. The best Decider reaches Accuracy ≥ {ACCURACY_BAR:.0%} with Brier ≤ {BRIER_BAR:.2f}",
        status,
        [f"{s.pinned_model}: {s.figures}" for s in standings] + [_no_run(d) for d in missing],
    )


def _default_decider(public: Sequence[Row]) -> Standing | None:
    """The Decider to serve by default: of those with only valid answers on the public Tasks,
    one reaching both bars of criterion 2, then the most accurate, then the lowest Brier."""
    return max(
        _standings(public),
        key=lambda s: (s.all_valid, s.status == "pass", s.accuracy, -_brier_of(s)),
        default=None,
    )


def _brier_of(s: Standing) -> float:
    return s.best.brier if s.best is not None else math.inf


def _default_decider_line(public: Sequence[Row], deciders: Sequence[str]) -> _Line:
    chosen = _default_decider(public)
    if chosen is None:
        return _Line("Default Decider: none yet", f": no run on the {PUBLIC_TASKS} public Tasks.")
    missing = _missing(public, deciders)
    return _Line(
        f"Default Decider: {chosen.pinned_model}",
        (", provisional" if missing else "")
        + f": {chosen.figures}. Ranked on only valid answers, then reaching both bars of "
        "criterion 2, then Accuracy, then the lower Brier"
        + "".join(f"; {_no_run(d)}" for d in missing)
        + ".",
    )


def _missing(public: Sequence[Row], deciders: Sequence[str]) -> list[str]:
    """The Deciders in `deciders` without a run on the public Tasks."""
    return [d for d in deciders if not any(r.pinned_model.decider == d for r in public)]


def _valid_answers(public: Sequence[Row], deciders: Sequence[str]) -> Checked:
    served = [r for r in public if r.kind != "cross-validated"]  # one per run
    missing = _missing(public, deciders)
    if any(r.valid is None or r.valid < 1 for r in served):
        status = "fail"
    else:
        status = "pending" if missing else "pass"
    return Checked(
        "1. Valid answers 100% for every Decider",
        status,
        [f"{r.pinned_model}: {percent(r.valid)} ({r.run})" for r in served]
        + [_no_run(d) for d in missing],
    )


def _no_run(decider: str) -> str:
    return f"no run of {decider} on the {PUBLIC_TASKS} public Tasks"


_REFERENCE_NOTES = [
    "- The spec's reference points (Jev 96.3% with Brier 0.056; system-one-open 90.1%) are "
    "jevbench's older v1 suite, not the 231 public Tasks the verdict is on. On the public Tasks "
    "jevbench publishes each Task's outcome and latency, not its Distribution, so Brier, ECE and "
    "Valid answers are not published there.",
    "- Reference latency is jevbench's maintainers' own client wall clock to each remote "
    "endpoint, raw (before their ×2 adjustment for self-hosted endpoints): it is not comparable "
    "with a local server's own timing.",
]


def _references_table(references: Sequence[Reference]) -> list[str]:
    def published(x: float | None, as_text) -> str:
        return "not published" if x is None else as_text(x)

    return _table(
        (
            "System",
            "Suite",
            "Tasks",
            "Accuracy",
            "Brier",
            "ECE",
            "Valid answers",
            "Client p50 / p95",
            "Hardware",
            "Source",
        ),
        [
            (
                r.system,
                r.suite,
                str(r.tasks),
                percent(r.accuracy),
                published(r.brier, number),
                published(r.ece, number),
                published(r.valid, percent),
                _latency((_ms(r.client_p50_s), _ms(r.client_p95_s))),
                r.hardware,
                r.source,
            )
            for r in references
        ],
    )


_RESULT_NOTES = [
    "- **raw** rows are jevbench's own figures for the run (its `summary.json`).",
    "- **calibrated, k-fold CV** rows re-score that raw run's own Distributions with the "
    "temperatures `local-jev calibrate` fitted, cross-validated: each Task is calibrated with "
    "temperatures fitted on the other folds (`calibration-report.json`), never in-sample. "
    "Calibration is applied after the fact, so Valid answers and latency are the raw run's.",
]


FINISHED_RUN_FILES = ("run.json", "summary.json", "decisions.jsonl")
"""What `local-jev bench` writes only once jevbench ran every Task and summarized them."""


def _rows(run: Path) -> list[Row]:
    for name in FINISHED_RUN_FILES:
        if not (run / name).is_file():
            raise ReportRefused(f"{run} is not a finished `local-jev bench` run: it has no {name}")
    record = read_json(run / "run.json")
    figures = RunFigures.of(record, read_json(run / "summary.json"))
    decisions = read_jsonl(run / "decisions.jsonl")
    server = record["server"]
    if server is None:  # every request failed, or the server predates `runtime`
        raise ReportRefused(f"{run}: no answer says which Decider served it")
    runtimes = [d["runtime"] for d in decisions if d.get("runtime")]
    devices = tuple(sorted({r["device"] for r in runtimes})) or (server["device"],)
    if len(devices) == 1:
        client_ms = (_ms(figures.client_p50_s), _ms(figures.client_p95_s))
        total_ms = [r["total_ms"] for r in runtimes]
        server_ms = (_percentile(total_ms, 0.5), _percentile(total_ms, 0.95))
    else:  # a latency figure never mixes hardware
        client_ms = server_ms = (None, None)
    language = _language(run, decisions)
    served = Row(
        pinned_model=PinnedModel.of(server),
        kind="served calibrated" if figures.calibrated else "raw",
        calibration="calibrated, served" if figures.calibrated else "raw",
        on_public_tasks=(
            language == ENGLISH
            and record["task_set"] == "public"
            and record.get("limit") is None
            and figures.tasks == PUBLIC_TASKS
        ),
        language=language,
        task_set=record["task_set"],
        tasks=figures.tasks,
        accuracy=figures.accuracy,
        brier=None if figures.calibrated else figures.brier,
        ece=None if figures.calibrated else figures.ece,
        valid=figures.valid,
        client_ms=client_ms,
        server_ms=server_ms,
        devices=devices,
        run=run.name,
        served_calibration=figures.served_calibration,
    )
    rows = [served]
    # `calibrate` writes a report into raw runs only.
    if not figures.calibrated and (run / REPORT_FILE).is_file():
        cv = read_json(run / REPORT_FILE)["cross_validation"]
        calibrated = cv["calibrated"]
        rows.append(
            replace(
                served,
                kind="cross-validated",
                calibration=f"calibrated, {cv['folds']}-fold CV",
                accuracy=calibrated["accuracy"],
                brier=calibrated["brier"],
                ece=calibrated["ece"],
            )
        )
    return rows


ENGLISH, TRADITIONAL_CHINESE = "English", "Traditional Chinese"
ZH_TW_PREFIX = "zh-tw-"
"""Every part of the Traditional-Chinese Task set is named with it, as `zh-tw-easy`."""


def _language(run: Path, decisions: Sequence[dict]) -> str:
    """The language of the States in `run`, from the part of the Task set each Decision is from."""
    chinese = {d["task_set"].startswith(ZH_TW_PREFIX) for d in decisions}
    if chinese == {True, False}:
        raise ReportRefused(
            f"{run} mixes English and Traditional-Chinese Tasks, which are reported apart: "
            "bench each language as a run of its own"
        )
    return TRADITIONAL_CHINESE if chinese == {True} else ENGLISH


RESULT_COLUMNS = (
    "Decider",
    "Calibration",
    "Language",
    "Task set",
    "Tasks",
    "Accuracy",
    "Brier",
    "ECE",
    "Valid answers",
    "Client p50 / p95",
    "Server p50 / p95",
    "Hardware",
    "Run",
)


def _results_table(rows: Sequence[Row]) -> list[str]:
    def calibration_metric(r: Row, x: float | None) -> str:
        return "withheld" if r.served_calibration is not None else number(x)

    def latency(r: Row, p50_p95: tuple[float | None, float | None]) -> str:
        return "mixed" if len(r.devices) > 1 else _latency(p50_p95)

    return _table(
        RESULT_COLUMNS,
        [
            (
                str(r.pinned_model),
                r.calibration,
                r.language,
                r.task_set,
                str(r.tasks),
                percent(r.accuracy),
                calibration_metric(r, r.brier),
                calibration_metric(r, r.ece),
                percent(r.valid),
                latency(r, r.client_ms),
                latency(r, r.server_ms),
                " + ".join(r.devices),
                r.run,
            )
            for r in rows
        ],
    )


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _percentile(values: Sequence[float], q: float) -> float | None:
    """jevbench's percentile (jevbench/metrics.py): linear interpolation between closest ranks."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else 1000 * seconds


def _latency(p50_p95: tuple[float | None, float | None]) -> str:
    p50, p95 = p50_p95
    if p50 is None or p95 is None:
        return "n/a"
    return f"{p50:.0f} / {p95:.0f} ms"
