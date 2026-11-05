"""The `local-jev` command line."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from local_jev.bench import TASK_SETS, BenchFailed, bench, headline
from local_jev.decider import Decider
from local_jev.device import DEVICES, DTYPES

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
CHAT_MODEL_DECIDERS = ("lfm3",)
"""Deciders over a Chat model, which take chat-template options."""

TOKEN_ENV = "LOCAL_JEV_TOKEN"
"""When set and non-empty, `serve` requires this bearer token on every request but /health."""
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
    serve.add_argument("--decider", choices=sorted(DECIDERS), required=True)
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

    bench = commands.add_parser("bench", help="run jevbench against a running server")
    bench.add_argument("--endpoint", default="http://127.0.0.1:8000", help="the server's base URL")
    bench.add_argument("--task-set", choices=sorted(TASK_SETS), default="public")
    bench.add_argument("--limit", type=int, help="run only N of the Task set's Tasks")
    bench.add_argument("--model", default="local-jev", help="the `model` jevbench sends")
    bench.add_argument("--runs-dir", type=Path, default=Path("runs"))
    bench.add_argument("--cache-dir", type=Path, default=Path(".cache"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.model and not args.revision:
            serve.error("--model needs --revision: pinning the revision pins the code that runs")
        if args.chat_template_options and args.decider not in CHAT_MODEL_DECIDERS:
            serve.error(f"--chat-template-options is for a Chat model, not {args.decider}")
        _serve(args)
    elif args.command == "bench":
        _bench(args)


def _bench(args: argparse.Namespace) -> None:
    try:
        run = bench(
            args.endpoint,
            args.task_set,
            limit=args.limit,
            model=args.model,
            runs_dir=args.runs_dir,
            cache_dir=args.cache_dir,
            token=os.environ.get(TOKEN_ENV) or None,
        )
    except BenchFailed as e:
        raise SystemExit(f"local-jev bench: {e}") from e
    print(f"[local-jev] run: {run}")
    print(f"[local-jev] {headline(run)}")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from local_jev.wire import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load = DECIDERS[args.decider]
    token = os.environ.get(TOKEN_ENV) or None
    app = create_app(lambda: load(args), token=token)
    uvicorn.run(app, host=args.host, port=args.port)
