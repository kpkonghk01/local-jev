# 06: Reason via `explain`

**What to build:** When exploring their own data, the developer can ask why a Decision was made. With `explain: true`, the LFM3 Decider adds a short Reason written in the State's language. The Reason is produced only after the Distribution is fixed, so asking for one never changes the Decision. The d1 Decider cannot generate text: it ignores the flag gracefully.

**Blocked by:** 05 (LFM3 Option-ID Decider)

**Status:** ready-for-agent

- [ ] With `explain: true`, the LFM3 Decider returns a `reason` in each answer.
  - The Reason is length-capped and written in the State's language (Traditional Chinese or English).
  - The Distribution is identical with and without `explain`.
- [ ] With `explain: true`, the d1 Decider returns its normal answer without a `reason` and without an error.
- [ ] `explain` defaults to false, and the bench runner never sets it.
- [ ] HTTP-contract tests:
  - fake LM backend: the Reason is present, the Distribution is unchanged and the length cap is honoured;
  - fake d1: the response is 200 with no Reason.
- [ ] A `real_model` test checks that a Traditional-Chinese State gets a Chinese Reason and an English State gets an English one. A loose check such as the presence of CJK characters is enough.
- [ ] The README documents `explain` and states that d1 does not support it.
