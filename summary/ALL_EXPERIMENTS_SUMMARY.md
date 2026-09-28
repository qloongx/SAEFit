# SAEFit Baseline Reproduction - Consolidated Experiment Results

Generated 2026-09-26 by `/root/autodl-tmp/generate_summary.py`.

- Results directory: `/root/autodl-tmp/auto-baseline-v6-output/work/results`
- Repository: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main`
- Datasets: `/root/autodl-tmp/auto-baseline-v6-output/work/results/../data`
- Environment: PyTorch Lightning 2.5.0 / torch 2.8.0+cu128, RTX 4090 (49 GB)
- Stage 1: Timing-Focused (tau=5.0, tauc=0.05, lam=0.01, lr=3e-4, 100 epochs)
- Stage 2: Amplitude-Focused (tau=3.0, tauc=0.05, lam=0.05, lr=2e-4, 100 epochs)

## Overview

| Exp | Name | Samples/stage | Conditions | Status | Result files |
|---|---|---|---|---|---|
| 0 | Calibration and Baseline | 112,000 | 1 | superseded by 13 | `calibration_stage{1,2}.json` |
| 2 | Amplitude Failure Stratified | 560,000 | 5 strata | valid | `amplitude_failure_stage{1,2}.json` |
| 3 | Complete Metrics | 112,000 | 1 | valid | `complete_metrics_stage{1,2}.json` |
| 6 | Distribution Shift Robustness | 560,000 | 5 | valid | `rmse_stage{1,2}_corrected.json` |
| 8 | Gain Drift Robustness | 784,000 | 7 | valid | `gain_drift_stage{1,2}.json` |
| 9 | Pulse Shape Variation (Parametric) | 1,792,000 | 16 | valid | `pulse_shape_parametric_stage{1,2}.json` |
| 11 | OER Threshold Optimization (original) | 784,000 | 7 | superseded by 14 | `oer_threshold_stage{1,2}.json` |
| 12 | Multi-Seed Independent Repeated Training | 224,000/seed | seeds x 2 stages | NEW | `multiseed_summary.json` |
| 13 | Unified Temperature Scaling | 224,000 | 10 | NEW | `temp_scaling_stage{1,2}.json` |
| 14 | OER Threshold Sensitivity (unified OER) | 224,000 | 12 tau x 4 variants | NEW | `oer_sensitivity_stage{1,2}.json` |

## Unified evaluation protocol (experiments 12-14)

Shared implementation: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/lab/utils/unified_eval.py`

- Task: balanced binary classification, single pulse -> label 0, double pulse -> label 1, as in control/saefit.py::test_step
- Test samples per condition: 224,000
- `OER_cond(tau)` = P(yhat != y | conf >= tau); NaN when no sample clears tau
- `OER_abs(tau)` = P(yhat != y AND conf >= tau); monotone non-increasing in tau
- ECE bins: 15
- Determinism: seed_eval_dataset() pins np.random and torch.manual_seed before every eval dataloader; verified bit-identical across runs

Experiments 0 and 11 predate this protocol. They derived labels from
`(dt > 0)`, which is constant 1 for every sample `PulsePrepare`
produces, so their accuracy/ECE/OER numbers were measured on a
degenerate single-class task. They are retained for traceability but
superseded by experiments 13 and 14.

## Experiment 0: Calibration and Baseline

- Priority: Foundation
- Reviewer concerns: Establishes reference baseline
- Samples per stage: 112,000
- Conditions: 1
- Status: **superseded_by_experiment_13**
- Caveat: Labels derived from (dt > 0), which is constant 1; metrics were computed on a degenerate single-class task. Use experiment_13 instead.

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/calibration_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/calibration_stage2.json`

### stage1

| Metric | Value |
|---|---|
| aggregate_uncalibrated.accuracy | 0.999973 |
| aggregate_uncalibrated.ece | 2.676e-05 |
| aggregate_uncalibrated.nll | 0.018397 |
| aggregate_uncalibrated.oer_0.85 | 2.679e-05 |
| aggregate_calibrated.accuracy | 0.999973 |
| aggregate_calibrated.ece | 2.676e-05 |
| aggregate_calibrated.nll | 0.012265 |
| aggregate_calibrated.oer_0.85 | 2.679e-05 |
| temperature | 1.5 |

### stage2

| Metric | Value |
|---|---|
| aggregate_uncalibrated.accuracy | 0.999991 |
| aggregate_uncalibrated.ece | 8.941e-06 |
| aggregate_uncalibrated.nll | 0.001021 |
| aggregate_uncalibrated.oer_0.85 | 8.929e-06 |
| aggregate_calibrated.accuracy | 0.999991 |
| aggregate_calibrated.ece | 8.941e-06 |
| aggregate_calibrated.nll | 6.807e-04 |
| aggregate_calibrated.oer_0.85 | 8.929e-06 |
| temperature | 1.5 |

## Experiment 2: Amplitude Failure Stratified Analysis

- Priority: Medium
- Reviewer concerns: Reviewer 1 amplitude reconstruction; Reviewer 2 #4-5
- Samples per stage: 560,000
- Conditions: 5 stratification dimensions
- Status: **valid**

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/amplitude_failure_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/amplitude_failure_stage2.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment2_amplitude_failure_analysis.md`

### stage1

| Metric | Value |
|---|---|
| a1_rmse | 1.551354 |
| a2_rmse | 0.380758 |
| n_samples | 112000 |

### stage2

| Metric | Value |
|---|---|
| a1_rmse | 1.551296 |
| a2_rmse | 0.380755 |
| n_samples | 112000 |

## Experiment 3: Complete Metrics Evaluation

- Priority: High
- Reviewer concerns: Reviewer 2 #3 complete metric set
- Samples per stage: 112,000
- Conditions: 1
- Status: **valid**

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/complete_metrics_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/complete_metrics_stage2.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment3_complete_metrics_analysis.md`

### stage1

| Metric | Value |
|---|---|
| accuracy | 0.999955 |
| timing_rmse.t1 | 0.011636 |
| timing_rmse.t2 | 1.396515 |
| timing_rmse.combined | 0.987519 |
| amplitude_rmse.a1 | 1.551098 |
| amplitude_rmse.a2 | 0.380672 |
| amplitude_rmse.combined | 1.12934 |
| joint_error | 1.500201 |
| asnr.timing | 0.032536 |
| asnr.amplitude | 0.001849 |
| asnr.combined | 0.017192 |
| calibration.ece | 5.018e-08 |
| calibration.nll | 5.032e-08 |

### stage2

| Metric | Value |
|---|---|
| accuracy | 0.999973 |
| timing_rmse.t1 | 0.01144 |
| timing_rmse.t2 | 1.396814 |
| timing_rmse.combined | 0.987729 |
| amplitude_rmse.a1 | 1.551796 |
| amplitude_rmse.a2 | 0.380671 |
| amplitude_rmse.combined | 1.129819 |
| joint_error | 1.5007 |
| asnr.timing | 0.02771 |
| asnr.amplitude | 0.001849 |
| asnr.combined | 0.01478 |
| calibration.ece | 0 |
| calibration.nll | -0 |

## Experiment 6: Distribution Shift Robustness

- Priority: High
- Reviewer concerns: Reviewer 2 #2 distribution shift robustness
- Samples per stage: 560,000
- Conditions: 5
- Status: **valid**

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/rmse_stage1_corrected.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/rmse_stage2_corrected.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment6_distribution_shift_analysis.md`

### stage1

| Metric | Value |
|---|---|
| timing_rmse_pixels | 83.730675 |
| timing_rmse_periods | 1.395511 |
| amplitude_rmse_p1_raw | 38803.269531 |
| amplitude_rmse_p1_normalized | 1.552131 |
| amplitude_rmse_p2_raw | 9516.717773 |
| amplitude_rmse_p2_normalized | 0.380669 |
| n_samples | 112000 |

### stage2

| Metric | Value |
|---|---|
| timing_rmse_pixels | 83.716125 |
| timing_rmse_periods | 1.395269 |
| amplitude_rmse_p1_raw | 38802.71875 |
| amplitude_rmse_p1_normalized | 1.552109 |
| amplitude_rmse_p2_raw | 9516.739258 |
| amplitude_rmse_p2_normalized | 0.38067 |
| n_samples | 112000 |

## Experiment 8: Gain Drift Robustness

- Priority: Medium
- Reviewer concerns: Reviewer 2 #6 gain variation robustness
- Samples per stage: 784,000
- Conditions: 7
- Status: **valid**

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/gain_drift_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/gain_drift_stage2.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment8_gain_drift_analysis.md`

### stage1

| Metric | Value |
|---|---|
| baseline.drift_factor | 1 |
| baseline.n_samples | 112000 |
| baseline.classification.accuracy | 0.999973 |
| baseline.classification.n_errors | 3 |
| baseline.timing.t1_rmse | 0.012963 |
| baseline.timing.t2_rmse | 1.390607 |
| baseline.timing.t1_mae | 0.012963 |
| baseline.timing.t2_mae | 1.151068 |
| baseline.amplitude.a1_rmse | 1.550861 |
| baseline.amplitude.a2_rmse | 0.380753 |
| baseline.amplitude.a1_mae | 0.988362 |
| baseline.amplitude.a2_mae | 0.369165 |
| baseline.amplitude.a1_rel_error | 0.996051 |
| baseline.amplitude.a2_rel_error | 0.998403 |
| baseline.joint.joint_error | 1.15509 |
| drift_factors_tested | [0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2] |
| n_conditions | 7 |

### stage2

| Metric | Value |
|---|---|
| baseline.drift_factor | 1 |
| baseline.n_samples | 112000 |
| baseline.classification.accuracy | 0.999991 |
| baseline.classification.n_errors | 1 |
| baseline.timing.t1_rmse | 0.012963 |
| baseline.timing.t2_rmse | 1.39105 |
| baseline.timing.t1_mae | 0.012963 |
| baseline.timing.t2_mae | 1.151298 |
| baseline.amplitude.a1_rmse | 1.55255 |
| baseline.amplitude.a2_rmse | 0.380755 |
| baseline.amplitude.a1_mae | 0.989259 |
| baseline.amplitude.a2_mae | 0.369166 |
| baseline.amplitude.a1_rel_error | 0.996035 |
| baseline.amplitude.a2_rel_error | 0.998401 |
| baseline.joint.joint_error | 1.155954 |
| drift_factors_tested | [0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2] |
| n_conditions | 7 |

## Experiment 9: Pulse Shape Variation Robustness (Parametric)

- Priority: Medium
- Reviewer concerns: Reviewer 1 & 2 pulse shape variation robustness
- Samples per stage: 1,792,000
- Conditions: 16
- Status: **valid**

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/pulse_shape_parametric_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/pulse_shape_parametric_stage2.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment9_pulse_shape_parametric_analysis.md`
- `implementation`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/parametric_pulse_generator.py`

### stage1

| Metric | Value |
|---|---|
| baseline_accuracy | 1 |
| baseline_a1_rmse | 2.047e+05 |
| baseline_a2_rmse | 2.051e+05 |
| baseline_t2_rmse | 94.363785 |
| n_conditions | 16 |

### stage2

| Metric | Value |
|---|---|
| baseline_accuracy | 1 |
| baseline_a1_rmse | 2.020e+05 |
| baseline_a2_rmse | 2.018e+05 |
| baseline_t2_rmse | 94.854362 |
| n_conditions | 16 |

## Experiment 11: OER Threshold Optimization (original)

- Priority: High
- Reviewer concerns: Reviewer 2 #1 OER threshold selection
- Samples per stage: 784,000
- Conditions: 7
- Status: **superseded_by_experiment_14**
- Caveat: Consumed prediction dumps built from the constant (dt > 0) label and returned 0.0 for empty high-confidence sets. Use experiment_14 instead.

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/oer_threshold_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/oer_threshold_stage2.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment11_oer_threshold_analysis.md`

### stage1

| Metric | Value |
|---|---|
| n_samples | 112000 |
| thresholds_evaluated | <10 entries> |
| lowest_oer | 2.679e-05 |
| lowest_oer_threshold | 0.5 |
| oer_at_0.85 | 2.679e-05 |

### stage2

| Metric | Value |
|---|---|
| n_samples | 112000 |
| thresholds_evaluated | <10 entries> |
| lowest_oer | 8.929e-06 |
| lowest_oer_threshold | 0.5 |
| oer_at_0.85 | 8.929e-06 |

## Experiment 12: Multi-Seed Independent Repeated Training

- Priority: High
- Reviewer concerns: single-seed results; run-to-run variability unknown
- Status: **valid**

Traceable paths:

- `summary`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/results/multiseed_summary.json`
- `per_seed`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/results/multiseed_per_seed.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment12_multiseed_analysis.md`
- `train_script`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/run_multiseed.sh`
- `aggregate_script`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/aggregate_multiseed.py`

### Across-seed statistics (clean test data, uncalibrated)

| Stage | n seeds | seeds | Accuracy (mean +- std) | 95% CI | ECE | OER@0.85 | T |
|---|---|---|---|---|---|---|---|
| stage1 | 5 | 0, 1, 2, 3, 4 | 0.988531 +- 0.00125 | [0.98698, 0.990083] | 0.002641 +- 0.001163 | 0.006817 +- 7.433e-04 | 0.959606 |
| stage2 | 5 | 0, 1, 2, 3, 4 | 0.992096 +- 9.064e-04 | [0.990971, 0.993222] | 0.001713 +- 8.800e-04 | 0.004508 +- 2.965e-04 | 0.970162 |

### Regression metrics across seeds

| Stage | dt RMSE (samples) | a1 RMSE (norm) | a2 RMSE (norm) |
|---|---|---|---|
| stage1 | 2.477562 +- 0.17731 | 0.414656 +- 0.020113 | 0.423923 +- 0.052706 |
| stage2 | 2.184947 +- 0.219635 | 0.399223 +- 0.006532 | 0.39872 +- 0.005991 |

### Paired stage2 - stage1 deltas (same seed, clean data)

| Metric | Mean delta | 95% CI | n | Significant at 95% |
|---|---|---|---|---|
| accuracy | +0.00356519 | [+0.00235468, +0.0047757] | 5 | yes |
| ece | -0.000928169 | [-0.00271669, +0.000860354] | 5 | no |
| nll | -0.0122894 | [-0.0175421, -0.00703672] | 5 | yes |
| aurc | -0.000413237 | [-0.000710013, -0.000116461] | 5 | yes |
| e_aurc | -0.000378225 | [-0.000663574, -9.28764e-05] | 5 | yes |
| oer_conditional@0.85 | -0.00230883 | [-0.00303129, -0.00158637] | 5 | yes |
| oer_absolute@0.85 | -0.0022375 | [-0.00293617, -0.00153883] | 5 | yes |
| accuracy_single | +0.000528574 | [-0.00189817, +0.00295532] | 5 | no |
| accuracy_double | +0.00660176 | [+0.00448822, +0.00871531] | 5 | yes |
| dt_rmse_samples | -0.292615 | [-0.477311, -0.107919] | 5 | yes |
| a1_rmse_norm | -0.0154328 | [-0.0335896, +0.00272393] | 5 | no |
| a2_rmse_norm | -0.0252026 | [-0.0926984, +0.0422932] | 5 | no |

## Experiment 13: Unified Temperature Scaling Calibration

- Priority: High
- Reviewer concerns: unified post-hoc calibration protocol
- Status: **valid** (supersedes experiment_0)

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/results/temp_scaling_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/results/temp_scaling_stage2.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment13_temperature_scaling_analysis.md`
- `script`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/evaluate_temperature_scaling.py`
- `shared_core`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/lab/utils/unified_eval.py`

Temperature is fitted on the validation split by minimizing NLL over a log-space grid plus golden-section refinement. Two fit sources are reported: `clean` (the headline protocol) and `shifted` (fitted under the `combined` perturbation, an upper bound on what a single temperature can recover).

### stage1

T (clean fit) = 1.028994, T (shifted fit) = 35.355308

| Condition | Accuracy | ECE uncal | ECE clean-fit | ECE shift-fit |
|---|---|---|---|---|
| clean | 0.988473 | 0.001702 | 0.001409 | 0.405203 |
| graded_sigma0.002 | 0.921098 | 0.041222 | 0.040135 | 0.338816 |
| graded_sigma0.005 | 0.835424 | 0.125245 | 0.124134 | 0.267815 |
| graded_sigma0.01 | 0.775156 | 0.187311 | 0.186193 | 0.251779 |
| graded_sigma0.02 | 0.688786 | 0.270357 | 0.269095 | 0.190125 |
| noise_only | 0.702388 | 0.261744 | 0.26068 | 0.116022 |
| baseline_only | 0.576263 | 0.396838 | 0.395901 | 0.090124 |
| combined | 0.559045 | 0.415523 | 0.414717 | 0.085637 |
| mild_combined | 0.643321 | 0.317509 | 0.316301 | 0.140794 |
| strong_combined | 0.530933 | 0.452272 | 0.45173 | 0.128386 |

Aggregate over the five named paper shift conditions:

| Variant | Accuracy | ECE | NLL | E-AURC |
|---|---|---|---|---|
| uncalibrated | 0.60239 | 0.368777 | 4.538884 | 0.176709 |
| calibrated_clean_fit | 0.60239 | 0.367866 | 4.413017 | 0.171955 |
| calibrated_shifted_fit | 0.60239 | 0.112193 | 0.648369 | 0.182104 |

### stage2

T (clean fit) = 0.972805, T (shifted fit) = 31.051455

| Condition | Accuracy | ECE uncal | ECE clean-fit | ECE shift-fit |
|---|---|---|---|---|
| clean | 0.991232 | 0.002258 | 0.001851 | 0.385629 |
| graded_sigma0.002 | 0.910924 | 0.058772 | 0.059635 | 0.304426 |
| graded_sigma0.005 | 0.821022 | 0.152013 | 0.152816 | 0.225964 |
| graded_sigma0.01 | 0.76067 | 0.217383 | 0.21805 | 0.172354 |
| graded_sigma0.02 | 0.694228 | 0.280834 | 0.281561 | 0.121874 |
| noise_only | 0.710594 | 0.259625 | 0.260442 | 0.110801 |
| baseline_only | 0.579384 | 0.400734 | 0.401307 | 0.129687 |
| combined | 0.559554 | 0.417935 | 0.418567 | 0.100667 |
| mild_combined | 0.65567 | 0.316608 | 0.317379 | 0.120346 |
| strong_combined | 0.530897 | 0.452936 | 0.45339 | 0.109397 |

Aggregate over the five named paper shift conditions:

| Variant | Accuracy | ECE | NLL | E-AURC |
|---|---|---|---|---|
| uncalibrated | 0.60722 | 0.369567 | 4.542492 | 0.178948 |
| calibrated_clean_fit | 0.60722 | 0.370217 | 4.667938 | 0.174223 |
| calibrated_shifted_fit | 0.60722 | 0.11418 | 0.63624 | 0.160801 |

## Experiment 14: OER Threshold Sensitivity (Unified OER Definition)

- Priority: High
- Reviewer concerns: OER threshold sensitivity; unify OER definition
- Status: **valid** (supersedes experiment_11)

Traceable paths:

- `stage1`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/results/oer_sensitivity_stage1.json`
- `stage2`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/results/oer_sensitivity_stage2.json`
- `analysis`: `/root/autodl-tmp/auto-baseline-v6-output/work/results/experiment14_oer_sensitivity_analysis.md`
- `script`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/evaluate_oer_sensitivity.py`
- `shared_core`: `/root/autodl-tmp/auto-baseline-v6-output/work/repo/SAEFit-main/lab/utils/unified_eval.py`

Unified OER definition:

- `conditional`: P(yhat != y | conf >= tau); nan if no sample clears tau
- `absolute`: P(yhat != y AND conf >= tau); monotone non-increasing in tau
- `legacy`: strict conf > tau, conditional, 0.0 when empty (deprecated)

### stage1 (T = 1.028994, 1000 bootstrap resamples)

Threshold sweep, clean test data, uncalibrated:

| tau | OER_cond | 95% CI | OER_abs | Coverage | N selected | N errors |
|---|---|---|---|---|---|---|
| 0.5000 | 0.011527 | [0.011093, 0.01196] | 0.011527 | 1 | 224,000 | 2,582 |
| 0.6000 | 0.010231 | [0.00982, 0.010643] | 0.010196 | 0.996638 | 223,247 | 2,284 |
| 0.7000 | 0.008957 | [0.008579, 0.009339] | 0.008888 | 0.992313 | 222,278 | 1,991 |
| 0.7500 | 0.008395 | [0.008024, 0.008759] | 0.008308 | 0.989665 | 221,685 | 1,861 |
| 0.8000 | 0.00776 | [0.007401, 0.008116] | 0.007652 | 0.986013 | 220,867 | 1,714 |
| 0.8500 | 0.006965 | [0.006626, 0.007296] | 0.00683 | 0.980607 | 219,656 | 1,530 |
| 0.9000 | 0.006022 | [0.005712, 0.006333] | 0.005853 | 0.971817 | 217,687 | 1,311 |
| 0.9500 | 0.004677 | [0.004394, 0.004962] | 0.004455 | 0.952665 | 213,397 | 998 |
| 0.9900 | 0.001989 | [0.001788, 0.002189] | 0.001683 | 0.846201 | 189,549 | 377 |
| 0.9950 | 0.001198 | [0.001036, 0.001358] | 8.973e-04 | 0.749317 | 167,847 | 201 |
| 0.9990 | 1.585e-04 | [8.408e-05, 2.330e-04] | 7.589e-05 | 0.478768 | 107,244 | 17 |
| 0.9999 | 5.379e-05 | [1.076e-05, 1.076e-04] | 2.232e-05 | 0.415 | 92,960 | 5 |

Summary across variants:

| Variant | Accuracy | OER@0.85 | OER_cond min | max | Spread | AURC | E-AURC |
|---|---|---|---|---|---|---|---|
| clean/uncalibrated | 0.988473 | 0.006965 | 5.379e-05 | 0.011527 | 214.3x | 9.616e-04 | 8.948e-04 |
| clean/calibrated | 0.988473 | 0.006885 | 5.447e-05 | 0.011527 | 211.6x | 9.616e-04 | 8.949e-04 |
| combined/uncalibrated | 0.559045 | 0.433476 | 0.309985 | 0.440955 | 1.4x | 0.327036 | 0.211179 |
| combined/calibrated | 0.559045 | 0.433118 | 0.306682 | 0.440955 | 1.4x | 0.319154 | 0.203297 |


### stage2 (T = 0.972805, 1000 bootstrap resamples)

Threshold sweep, clean test data, uncalibrated:

| tau | OER_cond | 95% CI | OER_abs | Coverage | N selected | N errors |
|---|---|---|---|---|---|---|
| 0.5000 | 0.008768 | [0.008406, 0.009156] | 0.008768 | 1 | 224,000 | 1,964 |
| 0.6000 | 0.007416 | [0.007072, 0.007753] | 0.007393 | 0.996844 | 223,293 | 1,656 |
| 0.7000 | 0.00634 | [0.006016, 0.006652] | 0.006295 | 0.992857 | 222,400 | 1,410 |
| 0.7500 | 0.005839 | [0.005522, 0.00614] | 0.005781 | 0.990174 | 221,799 | 1,295 |
| 0.8000 | 0.005432 | [0.005116, 0.00572] | 0.005362 | 0.987 | 221,088 | 1,201 |
| 0.8500 | 0.004933 | [0.004632, 0.005209] | 0.004844 | 0.98192 | 219,950 | 1,085 |
| 0.9000 | 0.004344 | [0.004055, 0.004594] | 0.004228 | 0.973134 | 217,982 | 947 |
| 0.9500 | 0.003489 | [0.003221, 0.003719] | 0.003321 | 0.951942 | 213,235 | 744 |
| 0.9900 | 0.002093 | [0.001867, 0.002285] | 0.001768 | 0.844772 | 189,229 | 396 |
| 0.9950 | 0.00167 | [0.001469, 0.001848] | 0.001304 | 0.780607 | 174,856 | 292 |
| 0.9990 | 5.382e-04 | [4.211e-04, 6.610e-04] | 3.304e-04 | 0.61383 | 137,498 | 74 |
| 0.9999 | 6.971e-05 | [1.993e-05, 1.290e-04] | 3.125e-05 | 0.448259 | 100,410 | 7 |

Summary across variants:

| Variant | Accuracy | OER@0.85 | OER_cond min | max | Spread | AURC | E-AURC |
|---|---|---|---|---|---|---|---|
| clean/uncalibrated | 0.991232 | 0.004933 | 6.971e-05 | 0.008768 | 125.8x | 8.597e-04 | 8.211e-04 |
| clean/calibrated | 0.991232 | 0.005006 | 7.872e-05 | 0.008768 | 111.4x | 8.593e-04 | 8.207e-04 |
| combined/uncalibrated | 0.559554 | 0.435561 | 0.365651 | 0.440446 | 1.2x | 0.33665 | 0.221088 |
| combined/calibrated | 0.559554 | 0.435698 | 0.369661 | 0.440446 | 1.2x | 0.314544 | 0.198982 |

## Reviewer coverage map

| Reviewer concern | Experiment(s) |
|---|---|
| Reviewer 2 #1 OER threshold selection | 14 (11 superseded) |
| Reviewer 2 #2 distribution shift robustness | 6, 13 |
| Reviewer 2 #3 complete metric set | 3 |
| Reviewer 2 #4-5 amplitude reconstruction | 2 |
| Reviewer 2 #6 gain variation robustness | 8 |
| Reviewer 1 & 2 pulse shape variation | 9 |
| Unified post-hoc calibration | 13 (0 superseded) |
| Single-seed results / run-to-run variability | 12 |

## How to cite a number from this file

Every experiment section lists the absolute path of the JSON it was
parsed from. To trace a value: open that JSON and follow the dotted
key shown in the table. To regenerate everything from raw results:

```bash
python3 /root/autodl-tmp/generate_summary.py
```

To re-run an individual experiment, see the `script` path in its
section; experiments 12-14 all share `lab/utils/unified_eval.py` as their evaluation core.
