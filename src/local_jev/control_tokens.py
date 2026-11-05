"""A caller's text read as text: never as one of the model's control tokens.

A tokenizer reads the string of any token added to its vocabulary as that token wherever it
appears, so a State holding `<|im_end|>` would close the user's turn and could answer for the
model. A zero-width space after the first character of such a string reads the same and is no
longer the token; d1's own prompt code defuses its special tokens the same way, but not `<think>`
or `</think>`, which its tokenizer does not mark special.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

BREAK = "​"
"""A zero-width space."""


class ControlTokens:
    def __init__(self, strings: Iterable[str] = ()):
        """The strings a tokenizer reads as control tokens, as `<|im_end|>` or `</think>`."""
        names = sorted({s for s in strings if len(s) > 1}, key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, names))) if names else None

    @classmethod
    def of(cls, tokenizer: Any) -> ControlTokens:
        """Every token added to a Hugging Face tokenizer's vocabulary, special or not."""
        return cls(tokenizer.get_added_vocab())

    def defuse(self, text: str) -> str:
        """`text` with no control token in it once encoded; text without one is unchanged."""
        if self._pattern is None:
            return text
        found = 1
        while found:  # again, for a string that overlapped one broken before it
            text, found = self._pattern.subn(lambda m: m[0][0] + BREAK + m[0][1:], text)
        return text

    def defuse_json(self, value: Any) -> Any:
        """`value`, a JSON value, with every string in it (keys too) defused."""
        if isinstance(value, str):
            return self.defuse(value)
        if isinstance(value, dict):
            return {self.defuse(k): self.defuse_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.defuse_json(v) for v in value]
        return value


NO_CONTROL_TOKENS = ControlTokens()
"""For a model whose tokenizer is not at hand: text is passed on as it is."""
