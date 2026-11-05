# 12: Full benchmark and feasibility report

**What to build:** The developer gets the answer this project exists for: a feasibility report in Traditional Chinese with measured numbers and a verdict against the agreed criteria. The same work picks the default Decider.

**Blocked by:** 05 (LFM3 Option-ID Decider), 07 (calibration), 08 (Traditional-Chinese Task set), 09 (comparison report)

**Status:** ready-for-agent

- [ ] Benchmark both Deciders natively on the Mac (MPS):
  - on all 231 public Tasks and on the Traditional-Chinese set;
  - fit calibration for each Decider and commit the fitted calibrations;
  - produce the comparison report.
  Raw run outputs stay uncommitted.
- [ ] Write the feasibility report in Traditional Chinese, with technical terms kept in English. It covers:
  - what jevbench is and how to run it;
  - the Jev wire format;
  - the original proposal claim by claim (the table in the spec);
  - findings on models and backends (vLLM, xgrammar, d1, the licence limit);
  - the measured results;
  - the verdict per criterion, with GPU latency marked pending;
  - recommended next steps.
- [ ] Make the winning Decider the default and say so in the README.
- [ ] Note in the report anything observed that contradicts an ADR.
