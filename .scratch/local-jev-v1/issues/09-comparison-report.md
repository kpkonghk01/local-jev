# 09: Comparison report across Deciders, calibration and language

**What to build:** The developer turns a set of benchmark runs into one readable report that picks the default Decider and checks the agreed verdict criteria automatically.

**Blocked by:** 03 (wire API completeness: server timing), 04 (jevbench runner), 07 (calibration)

**Status:** resolved

- [x] A command merges the selected runs into one Markdown report.
  - **Rows:** Decider × raw/calibrated × English/Traditional Chinese.
  - **Columns:** Accuracy, Brier, ECE, Valid answers, client p50/p95, server-side timing p50/p95, and the hardware used.
- [x] Calibrated rows use the cross-validated figures from the calibration fitter, never in-sample numbers, and say so.
- [x] jevbench reference rows (Jev 1.13.0; system-one-open, Gemma 4 E2B with LoRA) are shown for context.
- [x] Each agreed verdict criterion is evaluated as pass, fail or pending:
  - Valid answers at 100%;
  - Accuracy ≥ 80% with Brier ≤ 0.20;
  - latency, where the GPU check stays pending without a GPU run.
- [x] Runs from different hardware are never merged into one latency figure.
- [x] Tests generate a report from fixture runs and check the figures and the verdict logic.

## Comments

- 2026-11-05 (implement-spec): Merged in 8c7b566. Reference rows use jevbench's own figures on the same 231 public Tasks (Jev 86.6%, system-one-open 73.2%); the spec's 96.3% / 90.1% rows are kept and labelled as the older 242-decision suite. Integration branch `local-jev-v1`; code-review fixes merged in ce0131c and 2338b2f.
