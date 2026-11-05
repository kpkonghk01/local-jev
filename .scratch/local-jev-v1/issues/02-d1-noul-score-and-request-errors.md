# 02: d1 answers `noul` and `score`; bad requests get a clear 4xx

**What to build:** Extend the d1 path to the other two Question types that jevbench uses, so that every public Task can be answered. Make misuse fail loudly instead of returning nonsense.

**Blocked by:** 01 (d1 Decider answers a `choice` Question over `/v1/systemone`)

**Status:** ready-for-agent

- [ ] A `noul` Question returns `type: "noul"` and `noul` set to P(yes), as a non-boolean number in [0, 1].
  - Criteria keyed `"true"`/`"false"` reach the model meaningfully.
- [ ] A `score` Question with 2 to 10 levels (Criteria listed lowest first) returns:
  - `type: "score"`;
  - `score` set to the argmax level;
  - `probabilities` keyed by level index strings (`"0"` to `"N-1"`) and summing to 1.
- [ ] Unsupported Question types return a 4xx whose message names the type. These are `text`, `number`, `date`, `box`, `boxes` and multi-select `choice`.
- [ ] Malformed requests return a 4xx with a clear message. Examples:
  - a missing State or missing Questions;
  - a `choice` Question without Criteria;
  - a `score` Question with fewer than 2 or more than 10 levels.
- [ ] HTTP-contract tests with the fake d1 model cover both new types and every error case, against jevbench's acceptance rules.
- [ ] A `real_model` test answers one public `noul` Task and one `score` Task validly.
  - If d1's real answer fields or `score` key format differ from its model card, the mapping handles what the model actually returns.
  - Record what was observed as a short note in the test.
