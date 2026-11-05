"""Drive the Wire API in-process, with a Decider built behind the Decider seam."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import uvicorn
from fastapi.testclient import TestClient

from harness import fake_d1
from local_jev.control_tokens import ControlTokens
from local_jev.d1 import D1Decider
from local_jev.decider import Decider
from local_jev.option_id import OptionIDDecider
from local_jev.wire import create_app


@contextmanager
def starting(
    load_decider: Callable[[], Decider],
    token: str | None = None,
    calibrations: Path | None = None,
) -> Iterator[TestClient]:
    """A server that has just started; its Decider may still be loading.

    With `calibrations`, a directory, it calibrates with the Decider's calibration there, if any.
    """
    app = create_app(load_decider, token=token, calibrations_dir=calibrations)
    with TestClient(app) as client:
        yield client


@contextmanager
def serving(
    load_decider: Callable[[], Decider],
    ready_within: float = 5.0,
    token: str | None = None,
    calibrations: Path | None = None,
) -> Iterator[TestClient]:
    """A running server whose Decider comes from `load_decider`, once its health reports ready.

    With a `token`, the server requires it as a bearer token.
    """
    with starting(load_decider, token=token, calibrations=calibrations) as client:
        wait_until_ready(client, ready_within)
        yield client


def serving_d1(
    model, device: str = "cpu", token: str | None = None, calibrations: Path | None = None
) -> Iterator[TestClient]:
    """A running server with the d1 Decider over `model`, a fake d1, on `device`."""
    control_tokens = ControlTokens(fake_d1.CONTROL_TOKENS)
    return serving(
        lambda: D1Decider(model, device=device, control_tokens=control_tokens),
        token=token,
        calibrations=calibrations,
    )


def serving_option_id(
    backend, device: str = "cpu", calibrations: Path | None = None
) -> Iterator[TestClient]:
    """A running server with the LFM3 Option-ID Decider over `backend` (a fake or the real one)."""
    return serving(lambda: OptionIDDecider(backend, device=device), calibrations=calibrations)


@contextmanager
def listening(
    load_decider: Callable[[], Decider],
    token: str | None = None,
    calibrations: Path | None = None,
) -> Iterator[str]:
    """A server listening on a free localhost port, as jevbench reaches it; yields its base URL.

    The server starts at once; its Decider may still be loading.
    """
    app = create_app(load_decider, token=token, calibrations_dir=calibrations)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the server did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def wait_until_ready(client: TestClient, within: float) -> None:
    deadline = time.monotonic() + within
    while (health := client.get("/health")).status_code != 200:
        if time.monotonic() > deadline:
            raise TimeoutError(f"server not ready after {within} s: {health.json()}")
        time.sleep(0.01)
