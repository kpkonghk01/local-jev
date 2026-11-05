"""The Wire API: TypeSafe's `/v1/systemone` format, as jevbench's `typesafe` adapter speaks it."""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, JsonValue

from local_jev.decider import ChoiceQuestion, Decider, Label, Question, Usage
from local_jev.distribution import NotADistribution, over_labels

log = logging.getLogger(__name__)


class ChoiceQuestionIn(BaseModel):
    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str | None]

    def to_question(self) -> ChoiceQuestion:
        return ChoiceQuestion(instructions=self.instructions, criteria=self.criteria)


class SystemOneRequest(BaseModel):
    state: JsonValue
    model: str | None = None
    questions: dict[str, ChoiceQuestionIn]


class _DeciderLoader:
    """Loads the Decider in the background, so the server can report health while it loads."""

    def __init__(self, load_decider: Callable[[], Decider]):
        self._load_decider = load_decider
        self.decider: Decider | None = None
        self.error: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._load, name="decider-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            self.decider = self._load_decider()
        except Exception as e:
            log.exception("the Decider failed to load")
            self.error = f"{type(e).__name__}: {e}"


def create_app(load_decider: Callable[[], Decider], token: str | None = None) -> FastAPI:
    """The server, answering with the Decider that `load_decider` returns once it has loaded.

    With a `token`, every request but `/health` must carry it as `Authorization: Bearer <token>`.
    Without one, any Authorization header (jevbench always sends one) or none is accepted.
    """
    loader = _DeciderLoader(load_decider)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader.start()
        yield

    app = FastAPI(title="local-jev", lifespan=lifespan)

    if token:

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            if request.url.path == "/health" or _bears(request, token):
                return await call_next(request)
            return JSONResponse(
                {"detail": "a missing or wrong bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health")
    def health():
        if loader.error is not None:
            return JSONResponse({"status": "failed", "error": loader.error}, status_code=503)
        if loader.decider is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ready"}

    # The model has one set of weights and caches: requests take turns, each one whole.
    one_request_at_a_time = threading.Lock()

    @app.post("/v1/systemone")
    def systemone(request: SystemOneRequest):
        started = time.perf_counter()
        decider = loader.decider
        if decider is None:
            raise HTTPException(503, "the Decider is not ready; see /health")
        answers = {}
        usage = Usage()
        deciding = 0.0
        with one_request_at_a_time:
            for name, wire_question in request.questions.items():
                question = wire_question.to_question()
                asked = time.perf_counter()
                decision = decider.decide(request.state, question)
                deciding += time.perf_counter() - asked
                usage += decision.usage
                try:
                    distribution = over_labels(decision.distribution, question.labels)
                except NotADistribution as e:
                    log.error("the Decider's answer to %r is not a Distribution: %s", name, e)
                    raise HTTPException(
                        500, f"the Decider's answer to {name!r} is unusable: {e}"
                    ) from e
                answers[name] = _answer(question, distribution)
        # Counted from the parsed request, so it includes any wait for an earlier request.
        timing = {"total_ms": _ms(time.perf_counter() - started), "decider_ms": _ms(deciding)}
        return {
            "answers": answers,
            "usage": asdict(usage),
            "model": request.model,
            "timing": timing,
            # jevbench copies `runtime` into every results record, so its records carry these too.
            "runtime": {**asdict(decider.runtime), **timing},
        }

    return app


def _bears(request: Request, token: str) -> bool:
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(
        credentials.strip().encode(), token.encode()
    )


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def _answer(question: Question, distribution: Mapping[Label, float]) -> dict:
    return {
        "type": question.type,
        "choice": max(distribution, key=distribution.__getitem__),
        "probabilities": dict(distribution),
    }
