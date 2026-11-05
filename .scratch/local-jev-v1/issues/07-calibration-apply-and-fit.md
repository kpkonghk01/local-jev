# 07: Calibration: apply per-Question-type temperatures and fit them from a run

**What to build:** The developer fits calibration for a Decider from one benchmark run, gets honest cross-validated numbers, and the server applies the result from then on. There is one temperature per Question type, stored per Decider. With no calibration fitted, Distributions pass through unchanged. Temperature scaling never changes the argmax, so Accuracy is unaffected.

**Blocked by:** 02 (d1 answers `noul` and `score`; bad requests get a clear 4xx), 04 (jevbench runner)

**Status:** ready-for-agent

- [ ] Calibration in the server:
  - Every Decider's raw Distribution goes through per-Question-type temperature scaling: log-probabilities are scaled by 1/T and renormalised.
  - Calibration is loaded at server start for the active Decider only, and Deciders never share it.
  - With no calibration, Distributions are unchanged.
- [ ] `local-jev calibrate` reads a bench run's raw per-Task Distributions and expected Labels, then:
  - fits T for each Question type by minimising negative log-likelihood;
  - reports 5-fold cross-validated Brier and ECE before and after, with Accuracy shown unchanged;
  - writes the calibration for that Decider.
- [ ] Tests at the calibration-fitter seam, using synthetic data:
  - over-confident data yields T > 1 and a lower Brier;
  - already-calibrated data yields T ≈ 1;
  - the argmax never changes;
  - the cross-validation report is produced.
- [ ] An HTTP-contract test shows that a loaded calibration changes the served Distribution as expected.
- [ ] The README has a calibration section.
