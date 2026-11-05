# local-jev comparison report

Generated 2026-11-05T10:15:07+00:00 from 4 `local-jev bench` runs (jevbench 609a664).

## Results

| Decider | Calibration | Language | Task set | Tasks | Accuracy | Brier | ECE | Valid answers | Client p50 / p95 | Server p50 / p95 | Hardware | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1 LiquidAI/d1-2B@652f920 | raw | English | public | 231 | 76.6% | 0.285 | 0.072 | 100.0% | 129 / 2181 ms | 128 / 2180 ms | mps | 20261105T100421Z-public |
| d1 LiquidAI/d1-2B@652f920 | calibrated, 5-fold CV | English | public | 231 | 76.6% | 0.275 | 0.060 | 100.0% | 129 / 2181 ms | 128 / 2180 ms | mps | 20261105T100421Z-public |
| lfm3 LiquidAI/LFM3-1.7B@0dc4268 | raw | English | public | 231 | 62.3% | 0.626 | 0.272 | 100.0% | 163 / 2342 ms | 162 / 2340 ms | mps | 20261105T100910Z-public |
| lfm3 LiquidAI/LFM3-1.7B@0dc4268 | calibrated, 5-fold CV | English | public | 231 | 62.3% | 0.495 | 0.038 | 100.0% | 163 / 2342 ms | 162 / 2340 ms | mps | 20261105T100910Z-public |
| d1 LiquidAI/d1-2B@652f920 | raw | Traditional Chinese | zh-tw | 231 | 78.8% | 0.291 | 0.069 | 100.0% | 153 / 3397 ms | 151 / 3395 ms | mps | 20261105T100622Z-zh-tw |
| d1 LiquidAI/d1-2B@652f920 | calibrated, 5-fold CV | Traditional Chinese | zh-tw | 231 | 78.8% | 0.287 | 0.057 | 100.0% | 153 / 3397 ms | 151 / 3395 ms | mps | 20261105T100622Z-zh-tw |
| lfm3 LiquidAI/LFM3-1.7B@0dc4268 | raw | Traditional Chinese | zh-tw | 231 | 58.0% | 0.693 | 0.302 | 100.0% | 167 / 3574 ms | 165 / 3572 ms | mps | 20261105T101118Z-zh-tw |
| lfm3 LiquidAI/LFM3-1.7B@0dc4268 | calibrated, 5-fold CV | Traditional Chinese | zh-tw | 231 | 58.0% | 0.522 | 0.050 | 100.0% | 167 / 3574 ms | 165 / 3572 ms | mps | 20261105T101118Z-zh-tw |

- **raw** rows are jevbench's own figures for the run (its `summary.json`).
- **calibrated, k-fold CV** rows re-score that raw run's own Distributions with the temperatures `local-jev calibrate` fitted, cross-validated: each Task is calibrated with temperatures fitted on the other folds (`calibration-report.json`), never in-sample. Calibration is applied after the fact, so Valid answers and latency are the raw run's.

## jevbench reference points

| System | Suite | Tasks | Accuracy | Brier | ECE | Valid answers | Client p50 / p95 | Hardware | Source |
|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe AI) | the 231 public Tasks | 231 | 86.6% | not published | not published | not published | 665 / 802 ms | TypeSafe's production API (hardware not published) | jevbench@609a664 `results/v1.2/jevbench-v1.2-per-task.json`, `systems.jev-1.13.0.public_tasks` |
| system-one-open (Gemma 4 E2B with LoRA, on an L4) | the 231 public Tasks | 231 | 73.2% | not published | not published | not published | 649 / 1048 ms | one L4 GPU (the author's public demo endpoint on Modal) | jevbench@609a664 `results/v1.2/jevbench-v1.2-per-task.json`, `systems.system-one-open.public_tasks` |
| Jev 1.13.0 (TypeSafe AI) | older v1 suite (242 decisions, 2026-09-19), not the public Tasks | 242 | 96.3% | 0.056 | 0.027 | 100.0% | 652 / 722 ms | TypeSafe's production API (hardware not published) | jevbench@609a664 `results/jevbench-v1-results.json`, `overall` of jev-1.13.0 (`RESULTS.md` line 32) |
| system-one-open (Gemma 4 E2B with LoRA, on an L4) | older v1 suite (242 decisions, 2026-09-19), not the public Tasks | 242 | 90.1% | 0.138 | 0.068 | 100.0% | 652 / 772 ms | one L4 GPU (the author's public demo endpoint on Modal) | jevbench@609a664 `results/jevbench-v1-results.json`, `overall` of system-one-open (`RESULTS.md` line 36) |

- The spec's reference points (Jev 96.3% with Brier 0.056; system-one-open 90.1%) are jevbench's older v1 suite, not the 231 public Tasks the verdict is on. On the public Tasks jevbench publishes each Task's outcome and latency, not its Distribution, so Brier, ECE and Valid answers are not published there.
- Reference latency is jevbench's maintainers' own client wall clock to each remote endpoint, raw (before their ×2 adjustment for self-hosted endpoints): it is not comparable with a local server's own timing.

## Feasibility verdict

Checked on the 231 public Tasks in English only (`--task-set public`, no `--limit`), zero-shot plus calibration, with cross-validated figures. The Traditional-Chinese rows are reported apart and do not affect the verdict. Feasible, worth further investment, needs every criterion to pass.

| Criterion | Status | Evidence |
|---|---|---|
| 1. Valid answers 100% for every Decider | **pass** | d1 LiquidAI/d1-2B@652f920: 100.0% (20261105T100421Z-public); lfm3 LiquidAI/LFM3-1.7B@0dc4268: 100.0% (20261105T100910Z-public) |
| 2. The best Decider reaches Accuracy ≥ 80% with Brier ≤ 0.20 | **fail** | d1 LiquidAI/d1-2B@652f920: Accuracy 76.6%, Brier 0.275 (calibrated, 5-fold CV, 20261105T100421Z-public); lfm3 LiquidAI/LFM3-1.7B@0dc4268: Accuracy 62.3%, Brier 0.495 (calibrated, 5-fold CV, 20261105T100910Z-public) |
| 3a. Server-side p50 on the Mac (MPS) is recorded for every Decider | **pass** | d1 LiquidAI/d1-2B@652f920: server p50 128 ms (20261105T100421Z-public); lfm3 LiquidAI/LFM3-1.7B@0dc4268: server p50 162 ms (20261105T100910Z-public) |
| 3b. Server-side p50 ≤ 150 ms on an L4-class GPU | **pending** | no run of d1 LiquidAI/d1-2B@652f920 on cuda on the 231 public Tasks |

**Overall: fail** (fail: 2; pending: 3b)

**Default Decider: d1 LiquidAI/d1-2B@652f920**: Accuracy 76.6%, Brier 0.275 (calibrated, 5-fold CV, 20261105T100421Z-public). Ranked on only valid answers, then reaching both bars of criterion 2, then Accuracy, then the lower Brier.
