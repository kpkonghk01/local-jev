# 12: Full benchmark and feasibility report

**What to build:** The developer gets the answer this project exists for: a feasibility report in Traditional Chinese with measured numbers and a verdict against the agreed criteria. The same work picks the default Decider.

**Blocked by:** 05 (LFM3 Option-ID Decider), 07 (calibration), 08 (Traditional-Chinese Task set), 09 (comparison report)

**Status:** resolved

- [x] Benchmark both Deciders natively on the Mac (MPS):
  - on all 231 public Tasks and on the Traditional-Chinese set;
  - fit calibration for each Decider and commit the fitted calibrations;
  - produce the comparison report.
  Raw run outputs stay uncommitted.
- [x] Write the feasibility report in Traditional Chinese, with technical terms kept in English. It covers:
  - what jevbench is and how to run it;
  - the Jev wire format;
  - the original proposal claim by claim (the table in the spec);
  - findings on models and backends (vLLM, xgrammar, d1, the licence limit);
  - the measured results;
  - the verdict per criterion, with GPU latency marked pending;
  - recommended next steps.
- [x] Make the winning Decider the default and say so in the README.
- [x] Note in the report anything observed that contradicts an ADR.

## Comments

- 2026-11-05 (implement-spec): Merged in 1408717. Verdict: fail on criterion 2 (best Decider d1: Accuracy 76.6%, CV Brier 0.275); valid answers 100%; GPU latency pending. Default Decider is d1. Raw runs stay uncommitted in runs/. The report notes that Option-ID scoring did not make LFM3-1.7B usable (62.3%, first-option lean), so ADR-0002 is worth reopening if the Chat-model path continues. Integration branch `local-jev-v1`; code-review fixes merged in ce0131c and 2338b2f.
