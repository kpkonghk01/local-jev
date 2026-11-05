# 01: d1 Decider answers a `choice` Question over `/v1/systemone`

**What to build:** This is the thinnest path from end to end. The developer starts the server with the d1 Decider on their Mac. They POST a jevbench-style request that carries one `choice` Question. They get back a valid answer: the chosen Label and a Distribution over exactly that Question's Label set.

This ticket also lays down the foundations that every later ticket builds on:
- the uv project
- the CLI entry point
- the Decider seam (ADR-0003)
- the in-process HTTP-contract test harness

See the feature spec in this folder for the wire contract and pinned revisions.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The uv project is set up as follows:
  - Project `local-jev`, package `local_jev`, Python 3.12, with the lockfile committed.
  - Runtime dependencies are as the spec lists. Torch comes from the default index for now; ticket 10 adds the CUDA index.
  - Dev dependencies are pytest, ruff and httpx.
  - Ruff and pytest are configured.
  - A `real_model` pytest marker is registered and excluded by default.
- [x] `local-jev serve --decider d1` (with host and port options) loads `LiquidAI/d1-2B` at the pinned revision with remote code trusted.
  - It picks CUDA, then MPS, then CPU automatically, with an override.
  - It uses bf16 on CUDA/MPS and fp32 on CPU.
- [x] The health endpoint reports not-ready until the model has loaded, and ready afterwards.
- [x] A `/v1/systemone` request with one `choice` Question returns an answer under the same name with:
  - `type: "choice"`;
  - a `choice` that is one of the Labels;
  - `probabilities` keyed by exactly the Labels and summing to 1.
  The request's `model` is echoed back.
- [x] The wire API depends only on the Decider interface (State + Question → Decision); the d1 Decider is one implementation of it.
- [x] Fast HTTP-contract tests drive the endpoint in-process with a fake d1 model that returns canned `system_one` answers. They assert jevbench's acceptance rules for `choice`.
- [x] A `real_model` test loads the real d1-2B and answers one public jevbench `choice` Task validly.
- [x] The README (in Traditional Chinese) has a section on running natively on the Mac with `uv run`, including a note that the first start downloads the model.

## Comments

- 2026-11-05 (implement-spec): Merged in c1d999f. torch pinned to 2.11.* (the PyTorch cu128 index stops there). The real-model test answers easy-intent-00 and the JSON-State hard-opus-b-ambiguous-10 on MPS. Integration branch `local-jev-v1`; code-review fixes merged in ce0131c and 2338b2f.
