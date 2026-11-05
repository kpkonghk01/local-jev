# 09: Comparison report across Deciders, calibration and language

**What to build:** The developer turns a set of benchmark runs into one readable report that picks the default Decider and checks the agreed verdict criteria automatically.

**Blocked by:** 03 (wire API completeness: server timing), 04 (jevbench runner), 07 (calibration)

**Status:** ready-for-agent

- [ ] A command merges the selected runs into one Markdown report.
  - **Rows:** Decider × raw/calibrated × English/Traditional Chinese.
  - **Columns:** Accuracy, Brier, ECE, Valid answers, client p50/p95, server-side timing p50/p95, and the hardware used.
- [ ] Calibrated rows use the cross-validated figures from the calibration fitter, never in-sample numbers, and say so.
- [ ] jevbench reference rows (Jev 1.13.0; system-one-open, Gemma 4 E2B with LoRA) are shown for context.
- [ ] Each agreed verdict criterion is evaluated as pass, fail or pending:
  - Valid answers at 100%;
  - Accuracy ≥ 80% with Brier ≤ 0.20;
  - latency, where the GPU check stays pending without a GPU run.
- [ ] Runs from different hardware are never merged into one latency figure.
- [ ] Tests generate a report from fixture runs and check the figures and the verdict logic.
