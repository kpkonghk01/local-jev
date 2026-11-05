"""The comparison report checks the agreed feasibility verdict criteria (spec, "Feasibility verdict
criteria"): pass, fail or pending, from runs on the 231 public Tasks only."""

import pytest
from harness.markdown import table_with
from harness.runs import D1, LFM3, ZH_TW_PARTS, bench_run, calibration_report, report


def _verdict(markdown: str) -> dict[str, dict[str, str]]:
    """The verdict table's rows, keyed by criterion number ("1", "2", "3a", ...)."""
    return {row["Criterion"].split(".")[0]: row for row in table_with(markdown, "Criterion")}


def test_valid_answers_pass_when_every_decider_answered_all_public_tasks_validly(tmp_path):
    runs = [bench_run(tmp_path / "runs", "d1", D1), bench_run(tmp_path / "runs", "lfm3", LFM3)]

    valid = _verdict(report(tmp_path, *runs))["1"]

    assert valid["Status"] == "pass"
    assert "d1" in valid["Evidence"] and "lfm3" in valid["Evidence"]


def test_valid_answers_fail_when_any_decider_gave_an_invalid_answer(tmp_path):
    runs = [
        bench_run(tmp_path / "runs", "d1", D1),
        bench_run(tmp_path / "runs", "lfm3", LFM3, valid=230 / 231),
    ]

    valid = _verdict(report(tmp_path, *runs))["1"]

    assert valid["Status"] == "fail"
    assert "lfm3 LiquidAI/LFM3-1.7B@0dc4268: 99.6%" in valid["Evidence"]


def test_valid_answers_are_pending_until_every_decider_ran_on_all_public_tasks(tmp_path):
    runs = [
        bench_run(tmp_path / "runs", "d1", D1),
        # Neither a smoke run nor the Traditional-Chinese set counts toward the verdict.
        bench_run(tmp_path / "runs", "lfm3-smoke", LFM3, n_tasks=10, limit=10, valid=0.9),
        bench_run(tmp_path / "runs", "lfm3-zh", LFM3, task_set="zh-tw", parts=ZH_TW_PARTS),
    ]

    valid = _verdict(report(tmp_path, *runs))["1"]

    assert valid["Status"] == "pending"
    assert "no run of lfm3 on the 231 public Tasks" in valid["Evidence"]


def test_accuracy_and_brier_pass_on_the_cross_validated_calibrated_brier(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.85, brier=0.25)
    calibration_report(d1, accuracy=0.85, brier=0.18, ece=0.04)
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.7, brier=0.4)

    accuracy_and_brier = _verdict(report(tmp_path, d1, lfm3))["2"]

    assert accuracy_and_brier["Status"] == "pass"
    assert (
        "d1 LiquidAI/d1-2B@652f920: Accuracy 85.0%, Brier 0.180 (calibrated, 5-fold CV, d1)"
        in accuracy_and_brier["Evidence"]
    )


def test_accuracy_and_brier_fail_when_no_decider_reaches_both(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.85, brier=0.3)
    calibration_report(d1, accuracy=0.85, brier=0.24, ece=0.04)
    # Calibration never changes the argmax: below 80% Accuracy, calibrating cannot help.
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.79, brier=0.3)

    assert _verdict(report(tmp_path, d1, lfm3))["2"]["Status"] == "fail"


def test_accuracy_and_brier_are_pending_while_an_accurate_enough_decider_is_uncalibrated(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.85, brier=0.25)
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.7, brier=0.4)

    accuracy_and_brier = _verdict(report(tmp_path, d1, lfm3))["2"]

    assert accuracy_and_brier["Status"] == "pending"
    assert "d1 LiquidAI/d1-2B@652f920: not calibrated yet" in accuracy_and_brier["Evidence"]


def test_accuracy_and_brier_are_pending_while_a_decider_has_no_run(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.7, brier=0.3)

    accuracy_and_brier = _verdict(report(tmp_path, d1))["2"]

    assert accuracy_and_brier["Status"] == "pending"
    assert "no run of lfm3 on the 231 public Tasks" in accuracy_and_brier["Evidence"]


def test_a_calibrated_servers_in_sample_brier_never_passes_accuracy_and_brier(tmp_path):
    served = {"temperatures": {"noul": 2.0}, "fitted_from": "d1-raw"}
    d1 = bench_run(tmp_path / "runs", "d1", D1, calibration=served, accuracy=0.85, brier=0.1)
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.7, brier=0.4)

    assert _verdict(report(tmp_path, d1, lfm3))["2"]["Status"] == "pending"


def test_mps_latency_passes_once_every_deciders_server_p50_on_mps_is_recorded(tmp_path):
    runs = [
        bench_run(tmp_path / "runs", "d1-mps", D1, server_ms=173.0),
        bench_run(tmp_path / "runs", "d1-cpu", D1, device="cpu", server_ms=900.0),
        bench_run(tmp_path / "runs", "lfm3-mps", LFM3, server_ms=140.0),
    ]

    on_mps = _verdict(report(tmp_path, *runs))["3a"]

    assert on_mps["Status"] == "pass"
    assert on_mps["Evidence"] == (
        "d1 LiquidAI/d1-2B@652f920: server p50 173 ms (d1-mps); "
        "lfm3 LiquidAI/LFM3-1.7B@0dc4268: server p50 140 ms (lfm3-mps)"
    )


def test_mps_latency_is_pending_while_a_decider_has_no_run_on_mps(tmp_path):
    runs = [
        bench_run(tmp_path / "runs", "d1-mps", D1, server_ms=173.0),
        bench_run(tmp_path / "runs", "lfm3-cpu", LFM3, device="cpu", server_ms=1400.0),
    ]

    on_mps = _verdict(report(tmp_path, *runs))["3a"]

    assert on_mps["Status"] == "pending"
    assert "no run of lfm3 on mps on the 231 public Tasks" in on_mps["Evidence"]
    assert "1400" not in on_mps["Evidence"]


def test_gpu_latency_is_pending_until_the_default_decider_ran_on_cuda(tmp_path):
    runs = [
        bench_run(tmp_path / "runs", "d1-mps", D1, accuracy=0.766, server_ms=100.0),
        bench_run(tmp_path / "runs", "lfm3-mps", LFM3, accuracy=0.82, server_ms=140.0),
        # Not the default Decider, which is lfm3 (more accurate).
        bench_run(tmp_path / "runs", "d1-cuda", D1, accuracy=0.766, device="cuda", server_ms=40.0),
    ]

    on_gpu = _verdict(report(tmp_path, *runs))["3b"]

    assert on_gpu["Status"] == "pending"
    assert "no run of lfm3 LiquidAI/LFM3-1.7B@0dc4268 on cuda" in on_gpu["Evidence"]


@pytest.mark.parametrize("p50_ms, status", [(150.0, "pass"), (151.0, "fail")])
def test_gpu_latency_is_the_default_deciders_server_p50_on_cuda(tmp_path, p50_ms, status):
    served = {"temperatures": {"noul": 2.0}, "fitted_from": "d1-mps"}
    runs = [
        bench_run(tmp_path / "runs", "d1-mps", D1, server_ms=100.0),
        bench_run(
            tmp_path / "runs", "d1-cuda", D1, device="cuda", server_ms=p50_ms, calibration=served
        ),
    ]

    on_gpu = _verdict(report(tmp_path, *runs))["3b"]

    assert on_gpu["Status"] == status
    assert f"server p50 {p50_ms:.0f} ms (d1-cuda)" in on_gpu["Evidence"]
    assert "not checked" in on_gpu["Evidence"]  # that the GPU is L4-class


def _overall(markdown: str) -> str:
    (line,) = [line for line in markdown.splitlines() if line.startswith("**Overall")]
    return line.replace("**", "")


def test_feasibility_is_pending_while_only_the_gpu_check_is_left(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.85, brier=0.15)
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.75, brier=0.3)

    assert _overall(report(tmp_path, d1, lfm3)) == "Overall: pending (pending: 3b)"


def test_feasibility_fails_when_any_criterion_fails(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.766, brier=0.285, valid=230 / 231)
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.75, brier=0.3)

    assert _overall(report(tmp_path, d1, lfm3)) == "Overall: fail (fail: 1, 2; pending: 3b)"


def _default_decider(markdown: str) -> str:
    (line,) = [line for line in markdown.splitlines() if line.startswith("**Default Decider")]
    return line.replace("**", "")


def test_the_default_decider_is_the_most_accurate_on_the_public_tasks(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.766, brier=0.285)
    calibration_report(d1, accuracy=0.766, brier=0.275, ece=0.06)
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.82, brier=0.3)
    calibration_report(lfm3, accuracy=0.82, brier=0.25, ece=0.05)
    smoke = bench_run(tmp_path / "runs", "d1-smoke", D1, n_tasks=10, limit=10, accuracy=1.0)

    assert _default_decider(report(tmp_path, d1, lfm3, smoke)).startswith(
        "Default Decider: lfm3 LiquidAI/LFM3-1.7B@0dc4268"
    )


def test_a_decider_with_an_invalid_answer_is_not_the_default(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1, accuracy=0.8)
    lfm3 = bench_run(tmp_path / "runs", "lfm3", LFM3, accuracy=0.9, valid=230 / 231)

    assert _default_decider(report(tmp_path, d1, lfm3)).startswith(
        "Default Decider: d1 LiquidAI/d1-2B@652f920"
    )


def test_the_default_decider_is_provisional_while_a_decider_has_no_run(tmp_path):
    d1 = bench_run(tmp_path / "runs", "d1", D1)

    default = _default_decider(report(tmp_path, d1))

    assert default.startswith("Default Decider: d1 LiquidAI/d1-2B@652f920, provisional")
    assert "no run of lfm3 on the 231 public Tasks" in default


def test_there_is_no_default_decider_without_a_run_on_the_public_tasks(tmp_path):
    smoke = bench_run(tmp_path / "runs", "d1-smoke", D1, n_tasks=10, limit=10)

    assert _default_decider(report(tmp_path, smoke)).startswith("Default Decider: none yet")
