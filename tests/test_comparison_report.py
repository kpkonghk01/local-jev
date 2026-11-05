"""`local-jev report` merges finished bench runs into one Markdown comparison report."""

import json
import statistics
from pathlib import Path

import pytest
from harness.fake_d1 import FakeD1Model, d1_choice, d1_noul, d1_score
from harness.markdown import table_with
from harness.runs import ZH_TW_PARTS, bench_run, calibration_report
from harness.runs import report as _report
from harness.server import listening

from local_jev.bench import fetch_jevbench
from local_jev.cli import main
from local_jev.d1 import D1Decider
from local_jev.report import REFERENCES

PROJECT_CACHE = Path(__file__).parents[1] / ".cache"
RESULTS_V1 = "results/jevbench-v1-results.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _results(report: str) -> list[dict[str, str]]:
    return table_with(report, "Server p50 / p95")


def test_a_raw_run_is_one_row_of_jevbenchs_figures_beside_the_servers_own_timing(tmp_path):
    run = bench_run(
        tmp_path / "runs",
        "20261105T085252Z-public",
        server_ms=[100.0, 200.0, 300.0, 400.0, 500.0],
        n_tasks=5,
        accuracy=0.76623,
        brier=0.28518,
        ece=0.07205,
        valid=1.0,
        client_p50_s=0.17503,
        client_p95_s=2.61972,
    )

    (row,) = _results(_report(tmp_path, run))

    assert row == {
        "Decider": "d1 LiquidAI/d1-2B@652f920",
        "Calibration": "raw",
        "Language": "English",
        "Task set": "public",
        "Tasks": "5",
        "Accuracy": "76.6%",
        "Brier": "0.285",
        "ECE": "0.072",
        "Valid answers": "100.0%",
        "Client p50 / p95": "175 / 2620 ms",
        "Server p50 / p95": "300 / 480 ms",
        "Hardware": "mps",
        "Run": "20261105T085252Z-public",
    }


def test_a_calibrated_row_has_the_fitters_cross_validated_figures_and_says_so(tmp_path):
    run = bench_run(tmp_path / "runs", "raw-public", accuracy=0.766, brier=0.285, ece=0.072)
    calibration_report(run, accuracy=0.766, brier=0.2751, ece=0.0604)

    report = _report(tmp_path, run)
    raw, calibrated = _results(report)

    assert (raw["Calibration"], raw["Brier"], raw["ECE"]) == ("raw", "0.285", "0.072")
    assert calibrated["Calibration"] == "calibrated, 5-fold CV"
    assert (calibrated["Accuracy"], calibrated["Brier"], calibrated["ECE"]) == (
        "76.6%",
        "0.275",
        "0.060",
    )
    # Re-scored after the fact from the raw run, so its answers and latency are that run's.
    for same in ("Valid answers", "Client p50 / p95", "Server p50 / p95", "Hardware", "Run"):
        assert calibrated[same] == raw[same]
    assert "cross-validated" in report
    assert "never in-sample" in report


def test_a_run_on_traditional_chinese_tasks_is_a_row_of_its_own(tmp_path):
    english = bench_run(tmp_path / "runs", "en", accuracy=0.766)
    chinese = bench_run(tmp_path / "runs", "zh", task_set="zh-tw", parts=ZH_TW_PARTS, accuracy=0.7)

    rows = _results(_report(tmp_path, english, chinese))

    assert [(r["Language"], r["Task set"], r["Accuracy"]) for r in rows] == [
        ("English", "public", "76.6%"),
        ("Traditional Chinese", "zh-tw", "70.0%"),
    ]


def test_a_run_mixing_english_and_traditional_chinese_tasks_is_refused(tmp_path):
    mixed = bench_run(tmp_path / "runs", "mixed", parts=("easy", "zh-tw-easy"))

    with pytest.raises(SystemExit, match="mixes English and Traditional-Chinese Tasks"):
        _report(tmp_path, mixed)


def test_a_run_served_with_calibration_shows_no_brier_or_ece_as_they_may_be_in_sample(tmp_path):
    served = {"temperatures": {"noul": 2.1}, "fitted_from": "20261105T085252Z-public"}
    run = bench_run(tmp_path / "runs", "served", calibration=served, accuracy=0.766, brier=0.2)

    report = _report(tmp_path, run)
    (row,) = _results(report)

    assert row["Calibration"] == "calibrated, served"
    assert (row["Accuracy"], row["Valid answers"], row["Hardware"]) == ("76.6%", "100.0%", "mps")
    assert (row["Brier"], row["ECE"]) == ("withheld", "withheld")
    assert "fitted on 20261105T085252Z-public" in report


def test_runs_of_one_decider_on_different_hardware_keep_their_own_latency(tmp_path):
    on_mps = bench_run(tmp_path / "runs", "mps", server_ms=100.0, client_p50_s=0.11)
    on_cpu = bench_run(tmp_path / "runs", "cpu", device="cpu", server_ms=900.0, client_p50_s=0.95)

    rows = _results(_report(tmp_path, on_mps, on_cpu))

    assert [(r["Hardware"], r["Server p50 / p95"], r["Client p50 / p95"]) for r in rows] == [
        ("mps", "100 / 100 ms", "110 / 400 ms"),
        ("cpu", "900 / 900 ms", "950 / 400 ms"),
    ]


def test_a_run_answered_on_more_than_one_device_has_no_latency_figure(tmp_path):
    devices = ["mps"] * 100 + ["cpu"] * 131
    run = bench_run(tmp_path / "runs", "restarted", device=devices)

    (row,) = _results(_report(tmp_path, run))

    assert row["Hardware"] == "cpu + mps"
    assert (row["Client p50 / p95"], row["Server p50 / p95"]) == ("mixed", "mixed")
    assert row["Accuracy"] == "80.0%"


def test_a_report_goes_into_the_runs_directory_unless_told_where_and_ends_with_its_verdict(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    run = bench_run(tmp_path / "runs", "d1-public", accuracy=0.766, brier=0.285)

    main(["report", str(run)])
    said = capsys.readouterr().out

    (written,) = (tmp_path / "runs").glob("report-*.md")
    assert f"[local-jev] report: {written.relative_to(tmp_path)}" in said
    assert "[local-jev] Overall: pending (pending: 1, 2, 3a, 3b)" in said  # lfm3 has no run
    assert "[local-jev] Default Decider: d1 LiquidAI/d1-2B@652f920, provisional" in said
    main(["report", str(run)])
    assert len(list((tmp_path / "runs").glob("report-*.md"))) == 2  # an earlier one is kept


def test_a_directory_that_is_not_a_finished_bench_run_is_refused(tmp_path):
    unfinished = tmp_path / "runs" / "stopped"
    unfinished.mkdir(parents=True)
    (unfinished / "results.jsonl").write_text("")

    with pytest.raises(SystemExit, match="stopped is not a finished `local-jev bench` run"):
        main(["report", str(unfinished), "--output", str(tmp_path / "report.md")])


def test_a_run_no_answer_of_which_says_what_served_it_is_refused(tmp_path):
    run = bench_run(tmp_path / "runs", "anonymous")
    record = _json(run / "run.json")
    (run / "run.json").write_text(json.dumps({**record, "server": None}))

    with pytest.raises(SystemExit, match="anonymous: no answer says which Decider served it"):
        _report(tmp_path, run)


def test_a_real_bench_run_and_its_calibration_are_reported_as_jevbench_and_the_fitter_scored_them(
    tmp_path,
):
    with listening(lambda: D1Decider(FakeD1Model(_plausible), device="cpu")) as endpoint:
        main(
            ["bench", "--endpoint", endpoint, "--runs-dir", str(tmp_path / "runs")]
            + ["--cache-dir", str(PROJECT_CACHE), "--limit", "20"]
        )
    (run,) = (tmp_path / "runs").iterdir()
    main(["calibrate", str(run), "--calibrations-dir", str(tmp_path / "calibrations")])
    summary = _json(run / "summary.json")
    cv = _json(run / "calibration-report.json")["cross_validation"]["calibrated"]
    total_ms = sorted(d["runtime"]["total_ms"] for d in map(json.loads, _lines(run)))

    raw, calibrated = _results(_report(tmp_path, run))

    assert (raw["Calibration"], raw["Task set"], raw["Tasks"], raw["Hardware"]) == (
        "raw",
        "public",
        "20",
        "cpu",
    )
    assert raw["Accuracy"] == f"{100 * summary['accuracy']:.1f}%"
    assert raw["Brier"] == f"{summary['brier_mean']:.3f}"
    assert raw["Valid answers"] == "100.0%"
    assert raw["Server p50 / p95"].startswith(f"{(total_ms[9] + total_ms[10]) / 2:.0f} / ")
    assert (calibrated["Brier"], calibrated["ECE"]) == (f"{cv['brier']:.3f}", f"{cv['ece']:.3f}")


def test_a_real_bench_run_of_the_traditional_chinese_set_is_reported_as_such(tmp_path):
    with listening(lambda: D1Decider(FakeD1Model(_plausible), device="cpu")) as endpoint:
        main(
            ["bench", "--endpoint", endpoint, "--runs-dir", str(tmp_path / "runs")]
            + ["--cache-dir", str(PROJECT_CACHE), "--task-set", "zh-tw", "--limit", "6"]
        )
    (run,) = (tmp_path / "runs").iterdir()

    (row,) = _results(_report(tmp_path, run))

    assert (row["Language"], row["Task set"], row["Tasks"]) == ("Traditional Chinese", "zh-tw", "6")


def _lines(run: Path) -> list[str]:
    return (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()


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


def test_jevbench_reference_rows_are_on_the_public_tasks_and_older_suite_rows_say_so(tmp_path):
    report = _report(tmp_path, bench_run(tmp_path / "runs", "run"))
    references = {(r["System"], r["Tasks"]): r for r in table_with(report, "Suite")}
    jev = references["Jev 1.13.0 (TypeSafe AI)", "231"]
    system_one_open = references["system-one-open (Gemma 4 E2B with LoRA, on an L4)", "231"]
    older_jev = references["Jev 1.13.0 (TypeSafe AI)", "242"]
    older_system_one_open = references["system-one-open (Gemma 4 E2B with LoRA, on an L4)", "242"]

    assert (jev["Accuracy"], jev["Client p50 / p95"]) == ("86.6%", "665 / 802 ms")
    assert (system_one_open["Accuracy"], system_one_open["Client p50 / p95"]) == (
        "73.2%",
        "649 / 1048 ms",
    )
    # jevbench publishes an outcome per public Task, not its Distribution.
    assert {jev[c] for c in ("Brier", "ECE", "Valid answers")} == {"not published"}
    assert jev["Suite"] == system_one_open["Suite"] == "the 231 public Tasks"
    assert (older_jev["Accuracy"], older_jev["Brier"], older_jev["ECE"]) == (
        "96.3%",
        "0.056",
        "0.027",
    )
    assert (older_system_one_open["Accuracy"], older_system_one_open["Brier"]) == (
        "90.1%",
        "0.138",
    )
    assert "older" in older_jev["Suite"] and "not the public Tasks" in older_jev["Suite"]
    assert all("jevbench@609a664" in r["Source"] for r in references.values())


def test_the_reference_figures_are_jevbenchs_own_at_the_pinned_commit():
    jevbench = fetch_jevbench(PROJECT_CACHE)
    public = [
        json.loads(line)["id"]
        for part in ("easy", "original", "hard")
        for line in (jevbench / "datasets" / "public" / f"{part}.jsonl").read_text().splitlines()
        if line
    ]
    per_task = _json(jevbench / "results" / "v1.2" / "jevbench-v1.2-per-task.json")["systems"]
    v1 = {s["key"]: s["overall"] for s in _json(jevbench / RESULTS_V1)["systems"]}

    for ref in REFERENCES:
        if ref.tasks == 231:
            outcomes = per_task[ref.key]["public_tasks"]
            latencies = [outcomes[task_id][1] for task_id in public]
            assert set(outcomes) == set(public)
            assert ref.accuracy == sum(outcomes[t][0] == "c" for t in public) / len(public)
            assert ref.client_p50_s == pytest.approx(statistics.median(latencies))
            p95 = statistics.quantiles(latencies, n=20, method="inclusive")[-1]
            assert ref.client_p95_s == pytest.approx(p95)
        else:
            overall = v1[ref.key]
            assert ref.tasks == overall["n_scorable"] == 242
            assert (ref.accuracy, ref.brier) == (overall["accuracy"], overall["brier_mean"])
            assert (ref.ece, ref.valid) == (overall["ece"]["ece"], overall["schema_validity"])
            latency = overall["latency_ok_s"]
            assert (ref.client_p50_s, ref.client_p95_s) == (latency["p50_s"], latency["p95_s"])
