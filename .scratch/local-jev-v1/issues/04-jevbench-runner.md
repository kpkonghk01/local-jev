# 04: jevbench runner: benchmark any running server with one command

**What to build:** The developer points one command at a running server and gets a reproducible jevbench run. It fetches jevbench at the pinned commit, runs the built-in `typesafe` adapter on the chosen Task sets, summarises the run, and keeps the per-Task raw Distributions for calibration later. Once this lands, the developer has d1's first numbers on all 231 public Tasks.

**Blocked by:** 02 (d1 answers `noul` and `score`; bad requests get a clear 4xx)

**Status:** ready-for-agent

- [ ] `local-jev bench` fetches jevbench at the pinned commit into a git-ignored cache. It reuses the cache when the commit matches.
- [ ] It selects Task sets: easy, original, hard, all public Tasks, or a `--limit N` subset for smoke runs.
- [ ] Every run goes to a new timestamped directory outside jevbench's checkout. The directory is git-ignored and never overwritten. It holds:
  - jevbench's results, raw evidence, ledger and summary;
  - a record of the Decider, model id and revision, and hardware (CUDA, MPS or CPU) taken from the server.
- [ ] The run keeps, for each Task, the raw Distribution and the expected Label in a form the calibration fitter (ticket 07) can read.
- [ ] jevbench's quirks are handled:
  - a dummy API key environment variable when the server has no token;
  - the cost-reservation flags it needs for an unknown endpoint;
  - results written to new files only.
- [ ] A `real_model` end-to-end smoke test runs jevbench's own `typesafe` adapter on 10 Tasks against a real d1 server. All 10 answers must be valid.
- [ ] The README (in Traditional Chinese) has a benchmarking section.
