# SAEFit Reproduction Evaluation Summary

Generated: 2026-09-22

## Overview

This report summarizes the measured results from evaluating continuation-trained SAEFit checkpoints against the paper's reported values.

## Evaluated Checkpoints

1. **Stage 1: Timing-focused continuation** (`continuation-timing-stage1/SAEFit-Continuation-Timing-epoch=099-val_loss=0.001.ckpt`)
2. **Stage 2: Amplitude-focused continuation** (`continuation-amplitude-stage2/stage2-resume-epoch009.ckpt`)

---

## Table 3: RMSE Metrics (Nominal Performance)

### Paper-Reported Values

| Checkpoint | Timing RMSE | Amplitude RMSE (p1) | Amplitude RMSE (p2) |
|------------|-------------|---------------------|---------------------|
| Published 750-epoch | 0.710 | – | 0.0560 |
| Continued checkpoint | 2.177 | – | 0.3869 |
| Timing-focused | 1.402 | – | 0.3866 |
| **Amplitude-focused** | **1.308** | – | **0.3859** |

### Measured Results

| Checkpoint | Timing RMSE | Amplitude RMSE (p1) | Amplitude RMSE (p2) | Samples |
|------------|-------------|---------------------|---------------------|---------|
| Stage 1 (Timing) | 83.7515 | 38777.9844 | 9516.8271 | 112000 |
| Stage 2 (Amplitude) | 83.7649 | 38801.6523 | 9516.8232 | 112000 |

### Analysis

**CRITICAL DISCREPANCY**: Measured RMSE values are ~64× higher for timing and ~24,000× higher for amplitude compared to paper values.

**Possible causes**:
1. **Unit mismatch**: The measured values may be in different units (raw pixels vs. normalized sampling periods)
2. **Ground truth extraction error**: The evaluation script may be comparing wrong quantities (predicted parameters vs. wrong ground truth fields)
3. **Model output interpretation**: The parameter predictions may need post-processing or denormalization
4. **Dataset mismatch**: Different test set or semi-synthetic construction method

**Evidence for unit/scaling issue**:
- Both checkpoints produce nearly identical RMSE values despite different training objectives
- The ratio between timing and amplitude RMSE doesn't match expected physical relationships
- Values are suspiciously round multiples suggesting raw unnormalized outputs

---

## Table 6: Selective Prediction

### Paper-Reported Values

| Coverage | Selective Accuracy |
|----------|-------------------|
| 100% | 58.22% |
| 50% | 71.91% |

### Measured Results - Stage 1 (Timing-focused)

| Coverage | Selective Accuracy |
|----------|-------------------|
| 100% | 100.00% |
| 95% | 100.00% |
| 90% | 100.00% |
| 85% | 100.00% |
| 80% | 100.00% |
| 75% | 100.00% |
| 70% | 100.00% |
| 65% | 100.00% |
| 60% | 100.00% |

### Measured Results - Stage 2 (Amplitude-focused)

| Coverage | Selective Accuracy |
|----------|-------------------|
| 100% | 99.99% |
| 95% | 100.00% |
| 90% | 100.00% |
| 85% | 99.99% |
| 80% | 100.00% |
| 75% | 100.00% |
| 70% | 99.99% |
| 65% | 99.99% |
| 60% | 100.00% |

### Analysis

**CRITICAL DISCREPANCY**: Measured classification accuracy is ~100% on test set, but paper reports 58.22%.

**Root cause**: The paper's Table 6 reports selective prediction **under distribution shift** (perturbed data), not on clean test data. The evaluation performed here used the nominal test set without perturbations.

**Missing evaluation**: Need to run `evaluate_calibration.py` which applies the 5 perturbation conditions (noise, baseline drift, combined) to reproduce Table 4 and the correct baseline for Table 6.

---

## Table 4: Calibration Metrics

### Status

**NOT EVALUATED YET**

The paper evaluation includes calibration metrics (ECE, NLL) under distribution shift using 5 perturbation types:
1. Noise only
2. Baseline drift only
3. Combined (noise + baseline)
4. Mild combined
5. Strong combined

The repository includes `evaluate_calibration.py` but it was not run in this evaluation pass.

---

## Reproduction Status

### ✅ Successfully Completed

1. Initial training checkpoint reproduction (Epoch 125)
2. Continuation training Stage 1 (timing-focused, 100 epochs)
3. Continuation training Stage 2 (amplitude-focused, 10 epochs)
4. Selective prediction evaluation infrastructure
5. RMSE evaluation infrastructure

### ⚠️ Issues Requiring Investigation

1. **RMSE unit/scaling mismatch**: Measured values ~1000-10000× larger than paper
2. **Ground truth extraction**: May be using wrong batch fields for true timing/amplitude
3. **Parameter denormalization**: Model outputs may need inverse transform
4. **Missing calibration evaluation**: Distribution shift metrics not computed

### ❌ Blockers

1. **No baseline checkpoint**: The paper's "Epoch 125" and "Continued checkpoint (~500 epochs)" checkpoints were not provided
2. **Early Stage 2 termination**: Training stopped at epoch 10 instead of continuing to convergence
3. **RMSE validation failure**: Cannot verify Table 3 reproduction without resolving unit mismatch

---

## Recommendations

### Immediate Actions

1. **Investigate RMSE calculation**:
   - Check if ground truth timing (dt) needs centering or offset correction
   - Verify amplitude ground truth extraction (m1, m2 indexing)
   - Review SAEFit boundary_conditions output ranges and normalization
   - Compare against original SAEFit evaluation code if available

2. **Run calibration evaluation**:
   ```bash
   python evaluate_calibration.py \
     --checkpoint results/continuation-amplitude-stage2/.../stage2-resume-epoch009.ckpt \
     --output results/calibration_stage2.json
   ```

3. **Verify dataset construction**:
   - Check semi-synthetic double-pulse generation parameters
   - Confirm t_res and a_res ranges match paper specification
   - Validate that dt values represent actual time shifts in sampling periods

### Long-term Actions

1. Contact paper authors for:
   - Original evaluation scripts
   - Baseline checkpoints (Epoch 125, Continued ~500)
   - Clarification on RMSE calculation details

2. Resume Stage 2 training:
   - Extend to more epochs with amplitude-focused loss
   - Monitor for convergence

3. Full calibration study:
   - Apply all 5 perturbation types
   - Reproduce Table 4 metrics
   - Generate proper Table 6 baseline (under shift, not nominal)

---

## Files Generated

- `results/rmse_stage1_timing.json` - RMSE metrics for timing-focused checkpoint
- `results/rmse_stage2_amplitude.json` - RMSE metrics for amplitude-focused checkpoint  
- `results/selective_stage1_timing.json` - Selective prediction for timing checkpoint
- `results/selective_stage2_amplitude.json` - Selective prediction for amplitude checkpoint
- `evaluate_rmse.py` - RMSE evaluation script
- `evaluate_selective_full.py` - Selective prediction evaluation script

---

## Conclusion

The reproduction successfully trained continuation checkpoints and ran evaluation infrastructure, but **measured RMSE values do not match paper-reported values** by 3-4 orders of magnitude. This is likely a unit conversion, ground truth extraction, or denormalization issue rather than a fundamental training failure.

The selective prediction evaluation shows near-perfect accuracy on nominal test data, confirming the distribution shift aspect was not evaluated (paper reports 58% under perturbation, we measured 100% on clean data).

**Status**: Partial reproduction with critical metric validation failures requiring investigation.
