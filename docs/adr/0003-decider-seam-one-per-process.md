# Two Deciders behind one Decider seam, one Decider per process

`LiquidAI/d1-2B` is a Decision model whose `system_one` call already returns Jev-style typed answers with probabilities in a single pass, so it does not fit the backend seam of ADR-0001 (next-token probabilities plus text generation). We put a Decider seam above that one (State and Question in, Decision out) with two implementations: Option-ID scoring over a chat model (LFM3-1.7B by default, ADR-0002) and d1's native head. Each server process runs exactly one Decider chosen at startup, and compose runs them as separate services, because holding both in one CPU container (about 15 GB in fp32) would overrun the dev machine's container memory. Only Deciders that can generate text offer a Reason, so d1 answers without one.

## Considered Options

- **One process, Decider picked per request via the `model` field**: rejected for v1 because of the memory ceiling above, and because it makes latency depend on which other model happens to be resident.
- **d1 paired with a chat model just to write the Reason**: rejected; it loads a second model and adds generation latency for a field jevbench does not score.
