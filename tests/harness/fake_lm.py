"""A fake LM backend: the capabilities of a real one, over a fixed next-token distribution.

Whatever the prompt, the next token is distributed the same way: with the probabilities the test
gives for some tokens of the fake's small vocabulary, and the rest of the mass spread evenly over
the other tokens. (Fixed probabilities are fixed logits up to a constant.) Like the real backend,
the fake refuses a candidate that is not one token at the scored position, such as `" 0"`.

The fake keeps the messages of every prompt it was asked to score in `asked`.
"""

from __future__ import annotations

import math
import string
from collections.abc import Mapping, Sequence

from local_jev.lm_backend import Messages, NextToken, NotOneToken

VOCABULARY = (
    *string.ascii_uppercase,
    *string.digits,
    *("yes", "Yes", "no", "No"),
    *("The", "I", "\n"),  # what a Chat model may also want to say first
)


class FakeLMBackend:
    def __init__(self, probabilities: Mapping[str, float], prompt_tokens: int = 100):
        """`probabilities` for some tokens in `VOCABULARY`, summing to at most 1."""
        unknown = set(probabilities) - set(VOCABULARY)
        if unknown:
            raise ValueError(f"not in the fake vocabulary: {sorted(unknown)}")
        rest = [token for token in VOCABULARY if token not in probabilities]
        leftover = max(0.0, 1.0 - sum(probabilities.values())) / len(rest)
        everything = {token: probabilities.get(token, leftover) for token in VOCABULARY}
        self._logprobs = {t: math.log(p) if p > 0 else -math.inf for t, p in everything.items()}
        self._prompt_tokens = prompt_tokens
        self.asked: list[Messages] = []

    def next_token(self, messages: Messages, candidates: Sequence[str]) -> NextToken:
        for candidate in candidates:
            if candidate not in self._logprobs:
                raise NotOneToken(candidate)
        self.asked.append([dict(message) for message in messages])
        return NextToken(
            logprobs={candidate: self._logprobs[candidate] for candidate in candidates},
            prompt_tokens=self._prompt_tokens,
        )
