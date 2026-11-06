"""The `local-jev` command line."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from local_jev.bench import TASK_SETS, BenchFailed, bench, headline
from local_jev.calibration import CALIBRATIONS_DIR, REPORT_FILE, CalibrationRefused, calibrate
from local_jev.decider import Decider, PinnedModel
from local_jev.device import DEVICES, DTYPES
from local_jev.figures import number, percent
from local_jev.report import ReportRefused, compare, write_new_report

LoadDecider = Callable[[argparse.Namespace], Decider]
"""Loads a Decider as the `serve` arguments say (device, dtype, model and so on)."""


def _load_d1(args: argparse.Namespace) -> Decider:
    from local_jev.d1 import D1_MODEL_ID, D1_REVISION, load_d1

    return load_d1(
        device=args.device,
        dtype=args.dtype,
        model_id=args.model or D1_MODEL_ID,
        revision=args.revision or D1_REVISION,
    )


def _load_lfm3(args: argparse.Namespace) -> Decider:
    from local_jev.option_id import (
        LFM3_CHAT_TEMPLATE_OPTIONS,
        LFM3_MODEL_ID,
        LFM3_REVISION,
        load_lfm3,
    )

    return load_lfm3(
        device=args.device,
        dtype=args.dtype,
        model_id=args.model or LFM3_MODEL_ID,
        revision=args.revision or LFM3_REVISION,
        chat_template_options={**LFM3_CHAT_TEMPLATE_OPTIONS, **(args.chat_template_options or {})},
    )


DECIDERS: dict[str, LoadDecider] = {"d1": _load_d1, "lfm3": _load_lfm3}
DEFAULT_DECIDER = "d1"
"""Served when `serve` is given no `--decider`: on the 231 public Tasks d1 was the more accurate
Decider (76.6% against 62.3%; docs/feasibility-report.md)."""
CHAT_MODEL_DECIDERS = ("lfm3",)
"""Deciders over a Chat model, which take chat-template options."""

TOKEN_ENV = "LOCAL_JEV_TOKEN"
"""When set and non-empty, `serve` requires this bearer token on every request but /health,
and `bench` has jevbench send it."""
MODEL_ENV = "LOCAL_JEV_MODEL"
REVISION_ENV = "LOCAL_JEV_REVISION"
CHAT_TEMPLATE_OPTIONS_ENV = "LOCAL_JEV_CHAT_TEMPLATE_OPTIONS"


def _json_object(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not JSON: {e}") from e
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError('a JSON object, as {"enable_thinking": false}')
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="local-jev", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve one Decider over /v1/systemone")
    serve.add_argument(
        "--decider", choices=sorted(DECIDERS), default=DEFAULT_DECIDER, help="default: %(default)s"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", choices=DEVICES, default="auto", help="default: cuda, mps, cpu")
    serve.add_argument("--dtype", choices=DTYPES, default="auto", help="default: bf16, fp32 on cpu")
    serve.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV) or None,
        help=f"Hugging Face model id; default: the Decider's own (env {MODEL_ENV})",
    )
    serve.add_argument(
        "--revision",
        default=os.environ.get(REVISION_ENV) or None,
        help=f"commit to pin the model to; required with --model (env {REVISION_ENV})",
    )
    serve.add_argument(
        "--chat-template-options",
        type=_json_object,  # argparse applies it to a string default too
        default=os.environ.get(CHAT_TEMPLATE_OPTIONS_ENV) or None,
        metavar="JSON",
        help="lfm3 only: options for the chat template, merged over "
        f'{{"enable_thinking": false}} (env {CHAT_TEMPLATE_OPTIONS_ENV})',
    )
    serve.add_argument(
        "--calibrations-dir",
        type=Path,
        default=CALIBRATIONS_DIR,
        help="where the Decider's calibration <decider>.json is read from; "
        "default: the committed calibrations",
    )
    serve.add_argument(
        "--no-calibration",
        action="store_true",
        help="serve raw Distributions, as a bench run for `calibrate` needs",
    )

    bench_cmd = commands.add_parser("bench", help="run jevbench against a running server")
    bench_cmd.add_argument(
        "--endpoint", default="http://127.0.0.1:8000", help="the server's base URL"
    )
    bench_cmd.add_argument("--task-set", choices=sorted(TASK_SETS), default="public")
    bench_cmd.add_argument("--limit", type=int, help="run only N of the Task set's Tasks")
    bench_cmd.add_argument("--runs-dir", type=Path, default=Path("runs"))
    bench_cmd.add_argument("--cache-dir", type=Path, default=Path(".cache"))

    calibrate_cmd = commands.add_parser(
        "calibrate", help="fit the temperatures of the Decider that served a bench run"
    )
    calibrate_cmd.add_argument("run", type=Path, help="a finished `local-jev bench` run directory")
    calibrate_cmd.add_argument(
        "--calibrations-dir",
        type=Path,
        default=CALIBRATIONS_DIR,
        help="where to write <decider>.json; default: the committed calibrations",
    )

    report = commands.add_parser(
        "report", help="merge finished bench runs into one Markdown comparison report"
    )
    report.add_argument(
        "runs", type=Path, nargs="+", help="finished `local-jev bench` run directories"
    )
    report.add_argument(
        "--output",
        type=Path,
        help="the Markdown file to write; default: a new report-<UTC time>.md in --runs-dir",
    )
    report.add_argument("--runs-dir", type=Path, default=Path("runs"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.model and not args.revision:
            serve.error("--model needs --revision: pinning the revision pins the code that runs")
        if args.chat_template_options and args.decider not in CHAT_MODEL_DECIDERS:
            serve.error(f"--chat-template-options is for a Chat model, not {args.decider}")
        _serve(args)
    elif args.command == "bench":
        _bench(args)
    elif args.command == "calibrate":
        _calibrate(args)
    elif args.command == "report":
        _report(args)


def _bench(args: argparse.Namespace) -> None:
    try:
        run = bench(
            args.endpoint,
            args.task_set,
            limit=args.limit,
            runs_dir=args.runs_dir,
            cache_dir=args.cache_dir,
            token=os.environ.get(TOKEN_ENV) or None,
        )
    except BenchFailed as e:
        raise SystemExit(f"local-jev bench: {e}") from e
    print(f"[local-jev] run: {run}")
    print(f"[local-jev] {headline(run)}")


def _calibrate(args: argparse.Namespace) -> None:
    try:
        report = calibrate(args.run, args.calibrations_dir)
    except CalibrationRefused as e:
        raise SystemExit(f"local-jev calibrate: {e}") from e
    cv = report["cross_validation"]
    raw, calibrated = cv["raw"], cv["calibrated"]
    print(f"[local-jev] {PinnedModel.of(report)}, run {report['run']}, {raw['n']} Tasks")
    temperatures = ", ".join(f"{t} {v:g}" for t, v in report["temperatures"].items())
    print(f"[local-jev] temperatures: {temperatures}")
    print(f"[local-jev] {cv['folds']}-fold cross-validated, raw -> calibrated:")
    rows = [("all", raw, calibrated)] + [
        (t, scores["raw"], scores["calibrated"]) for t, scores in cv["by_question_type"].items()
    ]
    for name, before, after in rows:
        print(
            f"[local-jev]   {name:<6} {before['n']:>4} Tasks: "
            f"Accuracy {percent(before['accuracy'])} -> {percent(after['accuracy'])}, "
            f"Brier {number(before['brier'])} -> {number(after['brier'])}, "
            f"ECE {number(before['ece'])} -> {number(after['ece'])}"
        )
    print(f"[local-jev] calibration: {report['calibration_file']}")
    print(f"[local-jev] report: {args.run / REPORT_FILE}")


def _report(args: argparse.Namespace) -> None:
    try:
        comparison = compare(args.runs, deciders=sorted(DECIDERS))
    except ReportRefused as e:
        raise SystemExit(f"local-jev report: {e}") from e
    if args.output is None:
        written = write_new_report(comparison.markdown, args.runs_dir)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(comparison.markdown, encoding="utf-8")
        written = args.output
    print(f"[local-jev] report: {written}")
    print(f"[local-jev] {comparison.overall}")
    print(f"[local-jev] {comparison.default_decider}")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from local_jev.wire import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load = DECIDERS[args.decider]
    token = os.environ.get(TOKEN_ENV) or None
    calibrations = None if args.no_calibration else args.calibrations_dir
    app = create_app(lambda: load(args), token=token, calibrations_dir=calibrations)
    uvicorn.run(app, host=args.host, port=args.port)
