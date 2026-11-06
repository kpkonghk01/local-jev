# 03: Wire API completeness: several Questions, usage, timing, optional bearer token

**What to build:** Make the server honour the rest of the TypeSafe wire format and report what the benchmark report needs. The changes:
- One request can carry several named Questions about the same State.
- Responses carry token usage and the server's own timing.
- The server can optionally require a bearer token.

**Blocked by:** 01 (d1 Decider answers a `choice` Question over `/v1/systemone`)

**Status:** resolved

- [x] A request with several named Questions returns one answer per Question under the same names, in a single response.
- [x] The response includes `usage` with input and output token counts. Output is 0 when nothing is generated.
- [x] The response includes the server's own timing for the request: total time and time spent in the Decider. This lets the report separate server time from client wall-clock.
- [x] Bearer token, set through the environment:
  - When a token is configured, a missing or wrong token gets 401.
  - When none is configured, any token or none is accepted. jevbench always sends a dummy one.
- [x] Requests are processed one at a time; concurrent requests do not interleave model access.
- [x] HTTP-contract tests cover each of the above with the fake d1 model.

## Comments

- 2026-11-05 (implement-spec): Merged in 23ce0a8. One Decider call per Question (an answer never depends on the other Questions in the request). Timing is also in a top-level `runtime` object with decider, model, revision and device, which jevbench copies into every results record. Integration branch `local-jev-v1`; code-review fixes merged in ce0131c and 2338b2f.
