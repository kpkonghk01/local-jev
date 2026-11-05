"""The LM backend seam (ADR-0001): what the Option-ID Decider needs from a Chat model, and no more.

A backend offers exactly two capabilities over a Chat model:

- `next_token`: the log-probability of each candidate token as the very next token after a prompt;
- `generate`: plain, bounded text generation after a prompt.

Prompts are chat messages. The backend renders them through the model's own chat template, so the
template and its options (such as thinking disabled) are the backend's concern. v1 implements the
seam on Hugging Face transformers; a vLLM backend can replace it without touching the Decider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

type Messages = Sequence[Mapping[str, str]]
"""Chat messages, as [{"role": "user", "content": "…"}]."""


class NotOneToken(ValueError):
    """A candidate is not exactly one token at the position the backend scores."""

    def __init__(self, candidate: str):
        super().__init__(f"{candidate!r} is not one token right after the prompt")
        self.candidate = candidate


@dataclass(frozen=True)
class NextToken:
    """Each candidate's log-probability, out of the whole vocabulary, as the next token."""

    logprobs: Mapping[str, float]
    prompt_tokens: int


@dataclass(frozen=True)
class Generated:
    """Text the model wrote after a prompt; `output_tokens` never exceeds the bound asked for."""

    text: str
    prompt_tokens: int
    output_tokens: int


class LMBackend(Protocol):
    def next_token(self, messages: Messages, candidates: Sequence[str]) -> NextToken:
        """One forward pass over `messages`; raises NotOneToken for a candidate that is not one."""
        ...

    def generate(self, messages: Messages, max_new_tokens: int) -> Generated:
        """At most `max_new_tokens` of plain text, stopping early at the end of the model's turn."""
        ...
