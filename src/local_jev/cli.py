"""The `local-jev` command line."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence

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

    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from local_jev.wire import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load = DECIDERS[args.decider]
    token = os.environ.get(TOKEN_ENV) or None
    app = create_app(lambda: load(args.device, args.dtype), token=token)
    uvicorn.run(app, host=args.host, port=args.port)
