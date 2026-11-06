"""A fake LM backend: the capabilities of a real one, over a next-token distribution a test gives.

The next token is distributed with the probabilities the test gives for some tokens of the fake's
small vocabulary, and the rest of the mass spread evenly over the other tokens. (Fixed
probabilities are fixed logits up to a constant.) Given a function instead, the fake asks it for
those probabilities with the text the prompt shows it, so its answer can depend on what it read.
Like the real backend, the fake refuses a candidate that is not one token at the scored position,
such as `" 0"`.

Asked to generate, the fake writes the text the test gives it (or, given a function, what that
function makes of the text it was shown), one token per word, and like the real backend stops at
the bound it is given.

The fake counts the forward passes its model runs in `forward_passes`: one to score a prompt, one
per token written.
"""

from __future__ import annotations

import math
import string
from collections.abc import Callable, Mapping, Sequence

from local_jev.lm_backend import Generated, Messages, NextToken, NotOneToken

VOCABULARY = (
    *string.ascii_uppercase,
    *string.digits,
    *("yes", "Yes", "no", "No"),
    *("The", "I", "\n"),  # what a Chat model may also want to say first
)

type Probabilities = Mapping[str, float]


class FakeLMBackend:
    def __init__(
        self,
        probabilities: Probabilities | Callable[[str], Probabilities],
        prompt_tokens: int = 100,
        writes: str | Callable[[str], str] = "",
    ):
        """`probabilities` for some tokens in `VOCABULARY`, summing to at most 1, or what makes them
        from the text the model was shown; `writes` is the text the model writes whenever it
        generates, or what makes it from the text it was shown."""
        if not callable(probabilities):
            _logprobs(probabilities)  # refuses tokens the fake does not know at once
        self._probabilities = probabilities
        self._prompt_tokens = prompt_tokens
        self._writes = writes
        self.forward_passes = 0

    def next_token(self, messages: Messages, candidates: Sequence[str]) -> NextToken:
        for candidate in candidates:
            if candidate not in VOCABULARY:
                raise NotOneToken(candidate)
        self.forward_passes += 1
        logprobs = _logprobs(_for_prompt(self._probabilities, messages))
        return NextToken(
            logprobs={candidate: logprobs[candidate] for candidate in candidates},
            prompt_tokens=self._prompt_tokens,
        )

    def generate(self, messages: Messages, max_new_tokens: int) -> Generated:
        writes = _for_prompt(self._writes, messages)
        written = (writes.split(" ") if writes else [])[:max_new_tokens]
        self.forward_passes += len(written)
        return Generated(
            text=" ".join(written), prompt_tokens=self._prompt_tokens, output_tokens=len(written)
        )


def _for_prompt[T](given: T | Callable[[str], T], messages: Messages) -> T:
    """`given`, or what it makes of all the text the prompt shows the model."""
    if not callable(given):
        return given
    return given("\n".join(message["content"] for message in messages))


def _logprobs(probabilities: Probabilities) -> dict[str, float]:
    unknown = set(probabilities) - set(VOCABULARY)
    if unknown:
        raise ValueError(f"not in the fake vocabulary: {sorted(unknown)}")
    rest = [token for token in VOCABULARY if token not in probabilities]
    leftover = max(0.0, 1.0 - sum(probabilities.values())) / len(rest)
    everything = {token: probabilities.get(token, leftover) for token in VOCABULARY}
    return {t: math.log(p) if p > 0 else -math.inf for t, p in everything.items()}
