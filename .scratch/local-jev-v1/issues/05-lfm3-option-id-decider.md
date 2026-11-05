# 05: LFM3 Option-ID Decider

**What to build:** The developer can start the server with `LiquidAI/LFM3-1.7B` instead of d1 and get the same kinds of answers. The Chat model answers through Option-ID scoring: one forward pass with thinking disabled, normalised over the Label set (ADR-0002). It sits on the LM backend interface (ADR-0001), so a vLLM backend can replace transformers later.

**Blocked by:** 02 (d1 answers `noul` and `score`; bad requests get a clear 4xx)

**Status:** ready-for-agent

- [ ] The LM backend interface offers exactly two capabilities, implemented on Hugging Face transformers:
  - next-token probabilities over a given set of candidate tokens after a prompt;
  - plain, bounded text generation.
- [ ] `local-jev serve --decider lfm3` loads `LiquidAI/LFM3-1.7B` at the pinned revision with remote code trusted. It renders prompts through the model's chat template with thinking disabled. Device and dtype rules are the same as for d1.
- [ ] Option-ID scoring:
  - `choice` options are presented as letters (up to 26 options).
  - `noul` is scored on yes/no.
  - `score` is scored on the level numbers.
  - Each Decision costs one forward pass, is normalised over the Label set and is mapped back to Labels.
  - More than 26 `choice` options returns a 4xx.
- [ ] Model id, revision and chat-template options are configuration, so another ≤ 3B Chat model can be swapped in without code changes.
- [ ] HTTP-contract tests with a fake LM backend (fixed next-token logits) cover all three Question types:
  - the argmax and the letter-to-Label mapping;
  - normalisation, with the Distribution summing to 1;
  - the error on more than 26 options.
- [ ] `real_model` tests with the real LFM3-1.7B:
  - every Option ID is a single token at the exact position scored;
  - one public Task of each type is answered validly.
- [ ] The README says how to choose the Decider.
