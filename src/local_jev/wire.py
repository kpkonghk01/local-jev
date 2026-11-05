"""The Wire API: TypeSafe's `/v1/systemone` format, as jevbench's `typesafe` adapter speaks it."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, JsonValue

from local_jev.decider import ChoiceQuestion, Decider, Label, Question
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


def create_app(load_decider: Callable[[], Decider]) -> FastAPI:
    """The server, answering with the Decider that `load_decider` returns once it has loaded."""
    loader = _DeciderLoader(load_decider)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader.start()
        yield

    app = FastAPI(title="local-jev", lifespan=lifespan)

    @app.get("/health")
    def health():
        if loader.error is not None:
            return JSONResponse({"status": "failed", "error": loader.error}, status_code=503)
        if loader.decider is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ready"}

    @app.post("/v1/systemone")
    def systemone(request: SystemOneRequest):
        decider = loader.decider
        if decider is None:
            raise HTTPException(503, "the Decider is not ready; see /health")
        answers = {}
        for name, wire_question in request.questions.items():
            question = wire_question.to_question()
            decision = decider.decide(request.state, question)
            try:
                distribution = over_labels(decision.distribution, question.labels)
            except NotADistribution as e:
                log.error("the Decider's answer to %r is not a Distribution: %s", name, e)
                raise HTTPException(
                    500, f"the Decider's answer to {name!r} is unusable: {e}"
                ) from e
            answers[name] = _answer(question, distribution)
        return {"answers": answers, "model": request.model}

    return app


def _answer(question: Question, distribution: Mapping[Label, float]) -> dict:
    return {
        "type": question.type,
        "choice": max(distribution, key=distribution.__getitem__),
        "probabilities": dict(distribution),
    }
