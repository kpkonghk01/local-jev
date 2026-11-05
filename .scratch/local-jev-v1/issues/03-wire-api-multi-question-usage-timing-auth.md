# 03: Wire API completeness: several Questions, usage, timing, optional bearer token

**What to build:** Make the server honour the rest of the TypeSafe wire format and report what the benchmark report needs. The changes:
- One request can carry several named Questions about the same State.
- Responses carry token usage and the server's own timing.
- The server can optionally require a bearer token.

**Blocked by:** 01 (d1 Decider answers a `choice` Question over `/v1/systemone`)

**Status:** ready-for-agent

- [ ] A request with several named Questions returns one answer per Question under the same names, in a single response.
- [ ] The response includes `usage` with input and output token counts. Output is 0 when nothing is generated.
- [ ] The response includes the server's own timing for the request: total time and time spent in the Decider. This lets the report separate server time from client wall-clock.
- [ ] Bearer token, set through the environment:
  - When a token is configured, a missing or wrong token gets 401.
  - When none is configured, any token or none is accepted. jevbench always sends a dummy one.
- [ ] Requests are processed one at a time; concurrent requests do not interleave model access.
- [ ] HTTP-contract tests cover each of the above with the fake d1 model.
