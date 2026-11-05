"""The Wire API: TypeSafe's `/v1/systemone` format, as jevbench's `typesafe` adapter speaks it."""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, JsonValue, PlainValidator, StrictBool

from local_jev.calibration import Calibration, load_calibration
from local_jev.decider import (
    ChoiceQuestion,
    Decider,
    Label,
    NoulQuestion,
    Question,
    ScoreQuestion,
    Usage,
)
from local_jev.distribution import NotADistribution, argmax, over_labels

log = logging.getLogger(__name__)


_SUPPORTED = "this server answers noul, choice and score Questions"
_UNSUPPORTED_TYPES = ("text", "number", "date", "box", "boxes")
"""TypeSafe's other Question types, which d1 knows but local-jev does not answer."""
MAX_CHOICE_OPTIONS = 26
"""The Option-ID Decider names `choice` options A to Z; every Decider keeps to the same limit."""


def _question(wire: JsonValue) -> Question:
    """A Question as the wire carries it, as a Question; a ValueError says what is wrong with it."""
    if not isinstance(wire, dict):
        raise ValueError(
            'a Question is an object, as {"type": "noul", "instructions": "Is it urgent?"}'
        )
    kind = wire.get("type")
    if kind in _UNSUPPORTED_TYPES:
        raise ValueError(f"Question type {kind!r} is not supported; {_SUPPORTED}")
    if kind not in _QUESTION_TYPES:
        if "type" not in wire:
            raise ValueError(f'the Question has no "type"; {_SUPPORTED}')
        raise ValueError(f"{kind!r} is not a Question type; {_SUPPORTED}")
    instructions = wire.get("instructions")
    if not (isinstance(instructions, str) and instructions.strip()):
        raise ValueError('"instructions" is the Question as text, as "Is it urgent?"')
    return _QUESTION_TYPES[kind](wire, instructions)


def _noul(wire: dict, instructions: str) -> NoulQuestion:
    criteria = wire.get("criteria")
    if criteria is None:
        return NoulQuestion(instructions)
    says_yes_and_no = (
        isinstance(criteria, dict)
        and criteria.keys() == {"true", "false"}
        and all(isinstance(d, str) for d in criteria.values())
    )
    if not says_yes_and_no:
        raise ValueError(
            'a noul\'s "criteria", when given, says what yes and no mean, as '
            '{"true": "The order has shipped", "false": "It has not"}'
        )
    return NoulQuestion(instructions, {"yes": criteria["true"], "no": criteria["false"]})


def _choice(wire: dict, instructions: str) -> ChoiceQuestion:
    if wire.get("multiple"):
        raise ValueError(f'multi-select choice ("multiple": true) is not supported; {_SUPPORTED}')
    criteria = wire.get("criteria")
    descriptions = criteria.values() if isinstance(criteria, dict) else ()
    # Every Label's Criterion is shown to the model, so each must say something.
    if not (descriptions and all(isinstance(d, str) and d.strip() for d in descriptions)):
        hint = "; a list of levels is a score" if isinstance(criteria, list) else ""
        raise ValueError(
            'a choice\'s "criteria" names its options and says what each means, as '
            f'{{"billing": "Charges and refunds", "shipping": "Delivery"}}{hint}'
        )
    if len(criteria) > MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"a choice has at most {MAX_CHOICE_OPTIONS} options (one letter each); "
            f"this one has {len(criteria)}"
        )
    return ChoiceQuestion(instructions, criteria)


def _score(wire: dict, instructions: str) -> ScoreQuestion:
    levels = wire.get("criteria")
    problem = None
    if not isinstance(levels, list):
        problem = ""
    elif not 2 <= len(levels) <= 10:
        problem = f"; this one lists {len(levels)}"
    elif not all(isinstance(level, str) for level in levels):
        problem = "; each level is a text"
    if problem is not None:
        raise ValueError(
            'a score\'s "criteria" lists 2 to 10 levels, lowest first, as '
            f'["Can wait", "Today", "Now"]{problem}'
        )
    return ScoreQuestion(instructions, tuple(levels))


_QUESTION_TYPES: dict[str, Callable[[dict, str], Question]] = {
    "noul": _noul,
    "choice": _choice,
    "score": _score,
}


WireQuestion = Annotated[Question, PlainValidator(_question, json_schema_input_type=JsonValue)]


def _has_state(state: JsonValue) -> JsonValue:
    if state is None or isinstance(state, str) and not state.strip():
        raise ValueError("is empty; it is the text or JSON the Questions are about")
    return state


def _has_questions(questions: dict[str, Question]) -> dict[str, Question]:
    if not questions:
        raise ValueError('is empty; name at least one Question, as in {"decision": {...}}')
    return questions


class SystemOneRequest(BaseModel):
    state: Annotated[JsonValue, AfterValidator(_has_state)]
    model: str | None = None
    explain: StrictBool = False
    """Our extension to the wire format: ask for a Reason in every answer."""
    questions: Annotated[dict[str, WireQuestion], AfterValidator(_has_questions)]


class _DeciderLoader:
    """Loads the Decider, then its calibration, in the background, so the server can report health
    while it loads."""

    def __init__(self, load_decider: Callable[[], Decider], calibrations_dir: Path | None):
        self._load_decider = load_decider
        self._calibrations_dir = calibrations_dir
        self.decider: Decider | None = None
        self.calibration: Calibration | None = None
        self.error: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._load, name="decider-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            decider = self._load_decider()
            if self._calibrations_dir is not None:
                self.calibration = load_calibration(self._calibrations_dir, decider.runtime)
            self.decider = decider  # last: once it is set, requests are answered
        except Exception as e:
            log.exception("the Decider or its calibration failed to load")
            self.error = f"{type(e).__name__}: {e}"


def create_app(
    load_decider: Callable[[], Decider],
    token: str | None = None,
    calibrations_dir: Path | None = None,
) -> FastAPI:
    """The server, answering with the Decider that `load_decider` returns once it has loaded.

    With a `token`, every request but `/health` must carry it as `Authorization: Bearer <token>`.
    Without one, any Authorization header (jevbench always sends one) or none is accepted.

    With a `calibrations_dir`, the Decider's calibration there (if one was fitted for its model and
    revision) is applied to every Distribution. Without one, Distributions are served raw.
    """
    loader = _DeciderLoader(load_decider, calibrations_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader.start()
        yield

    app = FastAPI(title="local-jev", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, _refuse_malformed)

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
        decider, calibration = loader.decider, loader.calibration
        if decider is None:
            raise HTTPException(503, "the Decider is not ready; see /health")
        answers = {}
        usage = Usage()
        deciding = 0.0
        with one_request_at_a_time:
            for name, question in request.questions.items():
                asked = time.perf_counter()
                decision = decider.decide(request.state, question, explain=request.explain)
                deciding += time.perf_counter() - asked
                usage += decision.usage
                try:
                    distribution = over_labels(decision.distribution, question.labels)
                except NotADistribution as e:
                    log.error("the Decider's answer to %r is not a Distribution: %s", name, e)
                    raise HTTPException(
                        500, f"the Decider's answer to {name!r} is unusable: {e}"
                    ) from e
                if calibration is not None:
                    distribution = calibration.apply(question.type, distribution)
                answers[name] = _answer(question, distribution)
                if decision.reason is not None:
                    answers[name]["reason"] = decision.reason
        # Counted from the parsed request, so it includes any wait for an earlier request.
        timing = {"total_ms": _ms(time.perf_counter() - started), "decider_ms": _ms(deciding)}
        return {
            "answers": answers,
            "usage": asdict(usage),
            "model": request.model,
            "timing": timing,
            # jevbench copies `runtime` into every results record, so its records carry these too.
            # `calibration` is null when the Distributions are served raw.
            "runtime": {
                **asdict(decider.runtime),
                "calibration": None if calibration is None else calibration.describe(),
                **timing,
            },
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
    """The wire answer to `question` whose Distribution is `distribution`."""
    most_likely = argmax(distribution)
    match question:
        case NoulQuestion():
            return {"type": "noul", "noul": distribution["yes"]}
        case ChoiceQuestion():
            return {"type": "choice", "choice": most_likely, "probabilities": dict(distribution)}
        case ScoreQuestion():
            return {"type": "score", "score": int(most_likely), "probabilities": dict(distribution)}


async def _refuse_malformed(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with one plain message saying where the request is wrong and how.

    jevbench counts a 422 as a wrong answer without counting it towards the consecutive failures
    that stop its run.
    """
    return JSONResponse({"detail": "; ".join(map(_say, exc.errors()))}, status_code=422)


def _say(error: Mapping) -> str:
    if error["type"] == "json_invalid":
        return f"the request body is not valid JSON ({error['ctx']['error']} at {error['loc'][-1]})"
    where = ".".join(str(part) for part in error["loc"][1:])  # without the leading "body"
    if error["type"] == "value_error":
        what = str(error["ctx"]["error"])
    elif error["type"] == "missing":
        what = "is missing"
    else:
        what = error["msg"]
    return f"{where}: {what}" if where else what
