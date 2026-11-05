"""The `local-jev` command line."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from local_jev.bench import TASK_SETS, BenchFailed, bench, headline
from local_jev.decider import Decider
from local_jev.device import DEVICES, DTYPES

LoadDecider = Callable[[str, str], Decider]
"""Loads a Decider onto (device, dtype); each may be "auto"."""


def _load_d1(device: str, dtype: str) -> Decider:
    from local_jev.d1 import load_d1

    return load_d1(device=device, dtype=dtype)


DECIDERS: dict[str, LoadDecider] = {"d1": _load_d1}

TOKEN_ENV = "LOCAL_JEV_TOKEN"
"""When set and non-empty, `serve` requires this bearer token on every request but /health."""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="local-jev", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve one Decider over /v1/systemone")
    serve.add_argument("--decider", choices=sorted(DECIDERS), required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", choices=DEVICES, default="auto", help="default: cuda, mps, cpu")
    serve.add_argument("--dtype", choices=DTYPES, default="auto", help="default: bf16, fp32 on cpu")

    bench = commands.add_parser("bench", help="run jevbench against a running server")
    bench.add_argument("--endpoint", default="http://127.0.0.1:8000", help="the server's base URL")
    bench.add_argument("--task-set", choices=sorted(TASK_SETS), default="public")
    bench.add_argument("--limit", type=int, help="run only N of the Task set's Tasks")
    bench.add_argument("--model", default="local-jev", help="the `model` jevbench sends")
    bench.add_argument("--runs-dir", type=Path, default=Path("runs"))
    bench.add_argument("--cache-dir", type=Path, default=Path(".cache"))

    args = parser.parse_args(argv)
    if args.command == "serve":
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
    app = create_app(lambda: load(args.device, args.dtype), token=token)
    uvicorn.run(app, host=args.host, port=args.port)
