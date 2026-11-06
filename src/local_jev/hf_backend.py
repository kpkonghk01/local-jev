"""The LM backend (ADR-0001) on Hugging Face transformers, on CUDA, MPS or CPU."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from local_jev.control_tokens import ControlTokens
from local_jev.device import pick_device, pick_dtype
from local_jev.lm_backend import Generated, Messages, NextToken, NotOneToken

log = logging.getLogger(__name__)

_PROBE = [{"role": "user", "content": "?"}]


class TransformersBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        chat_template_options: Mapping[str, Any] | None = None,
    ):
        """`model` (a causal LM on `device`) and its `tokenizer`, prompted through the tokenizer's
        chat template with `chat_template_options` (such as `enable_thinking=False`)."""
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._template_options = dict(chat_template_options or {})
        # Message content is the caller's text: a control token spelled in it stays text.
        self._control_tokens = ControlTokens.of(tokenizer)
        # Every prompt ends with the same generation prompt (LFM3: `…assistant\n<think></think>`),
        # so a candidate that is one token after this probe is one token after any prompt.
        self._answer_start = self._render(_PROBE)
        self._token_ids: dict[str, int] = {}
        # The template's own control tokens (LFM3: BOS, <|im_start|>, <|im_end|>, <think>,
        # </think>): the model writing any of them has left its answer, so generation stops there.
        added = set(tokenizer.get_added_vocab().values())
        self._stops = sorted({i for i in self._encode(self._answer_start) if i in added})

    def next_token(self, messages: Messages, candidates: Sequence[str]) -> NextToken:
        import torch

        ids = [self._token_id(candidate) for candidate in candidates]
        prompt = self._prompt_ids(messages)
        with torch.inference_mode():  # also what lets LFM3's Triton conv kernel run on CUDA
            logits = self._model(input_ids=prompt, use_cache=False, logits_to_keep=1).logits
        logprobs = torch.log_softmax(logits[0, -1].float(), dim=-1)[ids].tolist()
        return NextToken(
            logprobs=dict(zip(candidates, logprobs, strict=True)), prompt_tokens=prompt.shape[1]
        )

    def generate(self, messages: Messages, max_new_tokens: int) -> Generated:
        """Sampled with the model's own generation defaults (LFM3: temperature 0.1, repetition
        penalty 1.1); generation stops at the bound or at any control token of the chat template,
        such as the end of the model's turn, which is not part of the text.

        LFM3 marks only some of those tokens special (not `</think>`), so the text is cut at the
        stop rather than left to `skip_special_tokens`.
        """
        import torch

        prompt = self._prompt_ids(messages)
        with torch.inference_mode():
            out = self._model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=max_new_tokens,
                eos_token_id=self._stops,
            )
        new = out[0, prompt.shape[1] :].tolist()
        text = new[:-1] if new and new[-1] in self._stops else new
        return Generated(
            text=self._tokenizer.decode(text, skip_special_tokens=True).strip(),
            prompt_tokens=prompt.shape[1],
            output_tokens=len(new),
        )

    def _render(self, messages: Messages) -> str:
        return self._tokenizer.apply_chat_template(
            [
                {**message, "content": self._control_tokens.defuse(message["content"])}
                for message in messages
            ],
            add_generation_prompt=True,
            tokenize=False,
            **self._template_options,
        )

    def _encode(self, text: str) -> list[int]:
        # The chat template writes the BOS and control tokens itself.
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def _prompt_ids(self, messages: Messages):
        import torch

        return torch.tensor([self._encode(self._render(messages))], device=self._device)

    def _token_id(self, candidate: str) -> int:
        """`candidate`'s id as the one token right after the prompt; NotOneToken if it is not."""
        if candidate not in self._token_ids:
            before = self._encode(self._answer_start)
            after = self._encode(self._answer_start + candidate)
            if len(after) != len(before) + 1 or after[: len(before)] != before:
                raise NotOneToken(candidate)
            self._token_ids[candidate] = after[-1]
        return self._token_ids[candidate]


def load_transformers_backend(
    model_id: str,
    revision: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    chat_template_options: Mapping[str, Any] | None = None,
) -> TransformersBackend:
    """`model_id` at `revision` with its remote code trusted, downloaded on first use."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    dtype = pick_dtype(device, dtype)
    log.info("loading %s@%s on %s (%s)", model_id, revision, device, dtype)
    started = time.perf_counter()
    # The tokenizer needs trust_remote_code too: without it transformers asks on the terminal.
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, dtype=getattr(torch, dtype)
    )
    model = model.to(device).eval()
    log.info("%s loaded in %.1f s", model_id, time.perf_counter() - started)
    return TransformersBackend(
        model, tokenizer, device=device, chat_template_options=chat_template_options
    )
