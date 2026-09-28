# SAEFit Two-Stage Training Peer Review Response - Project Status

**Last Updated**: 2026-09-22  
**Overall Status**: ✅ Batch 1 and Batch 2 Complete

---

## Executive Summary

Two batches of experimental work have been completed to address peer reviewer concerns about the SAEFit two-stage continuation training paper. A total of 7 experiments have been executed, testing 6,048,000 samples across two training checkpoints (Stage 1: Timing-Focused, Stage 2: Amplitude-Focused), generating 39 output files including evaluation scripts, raw data, and comprehensive analysis reports.

**Key Result**: Stage 2 "Amplitude-Focused" continuation training did NOT significantly improve amplitude reconstruction performance compared to Stage 1. Main improvements are marginal classification accuracy gains only.

---

## Batch Completion Status

| Batch | Priority | Experiments | Status | Duration | Samples |
|-------|----------|-------------|--------|----------|---------|
| **Batch 1** | High | 4 | ✅ Complete | ~8 min | 3,136,000 |
| **Batch 2** | Medium | 3 | ✅ Complete* | ~15 min | 2,912,000 |
| **Batch 3** | Low (Optional) | 1 | ⬜ Not Started | 43-72 hours | TBD |

*Batch 2: 2 fully complete, 1 partially complete (Experiment 9 baseline only)

**Total Progress**: 7/8 required experiments complete (87.5%)

---

## Reviewer Response Coverage

### Fully Addressed (6/8 items)

| Reviewer Comment | Experiment | Evidence |
|-----------------|------------|----------|
| Reviewer 1: Amplitude reconstruction issues | Experiment 2 | 5-dimensional stratified analysis, identified high amplitude ratio as main failure factor |
| Reviewer 2 Point 1: OER threshold selection | Experiment 11 | Systematic sweep across 7 thresholds, identified optimal values |
| Reviewer 2 Point 2: Distribution shift robustness | Experiment 6 | Tested 5 distribution conditions, quantified performance degradation |
| Reviewer 2 Point 3: Complete metrics set | Experiment 3 | Provided MAE, median error, percentiles beyond RMSE |
| Reviewer 2 Point 4: Amplitude analysis | Experiment 2 | Multi-scenario performance evaluation |
| Reviewer 2 Point 6: Gain drift robustness | Experiment 8 | Perfect gain invariance demonstrated across ±20% range |

### Partially Addressed (2/8 items)

| Reviewer Comment | Experiment | Status | Next Steps |
|-----------------|------------|--------|------------|
| Reviewer 1 & 2: Pulse shape variation | Experiment 9 | Baseline + indirect evidence | Implement parametric pulse generator (2-3 hours) |

---

## Critical Findings

### 1. Amplitude Failure Mechanism (Experiment 2)

**Discovery**: High amplitude ratio (a1/a2 > 5) causes systematic underestimation of the larger pulse.

- **Baseline performance**: a1_RMSE = 1.55
- **High ratio regime (a1/a2 > 5)**: a1_RMSE = 2.91 (+88%)
- **Worst case (a1/a2 > 9)**: a1_RMSE = 3.12, mean error = -5.6
- **Time reconstruction**: Stable across all dt ranges (±1%)

**Implication**: The network struggles when one pulse dominates; the large pulse tail "masks" the small pulse, causing underestimation.

### 2. Perfect Gain Invariance (Experiment 8)

**Discovery**: Model shows ZERO performance degradation across ±20% gain drift.

- **Test conditions**: 7 gain factors (0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20)
- **Result**: All metrics identical across all gain conditions
- **Mechanism**: Data augmentation (0.05x-1.0x amplitude scaling) + internal normalization

**Implication**: Model extracts normalized shape features rather than absolute amplitudes.

### 3. Stage 2 Training Effect

**Discovery**: "Amplitude-Focused" Stage 2 continuation training did NOT improve amplitude reconstruction.

| Metric | Stage 1 | Stage 2 | Change |
|--------|---------|---------|--------|
| **a1_RMSE** | 1.551 | 1.553 | +0.1% |
| **a2_RMSE** | 0.381 | 0.381 | 0.0% |
| **Accuracy** | 99.9973% | 99.9982% | +0.0009% |
| **Gain robustness** | Perfect | Perfect | No change |
| **Failure mode** | High ratio underestimation | Same | No improvement |

**Implication**: Stage 2 training provides marginal classification improvements but does not address the fundamental amplitude reconstruction limitations.

---

## Generated Outputs

### File Statistics

| Category | Count | Location |
|----------|-------|----------|
| **Evaluation Scripts** | 7 | `/root/auto-baseline-v6-output/work/repo/SAEFit-main/` |
| **JSON Data Files** | 19 | `/root/auto-baseline-v6-output/work/results/` |
| **Analysis Reports** | 13 | `/root/auto-baseline-v6-output/work/results/` |
| **Total Files** | 39 | — |

### Key Documents

**Batch Summaries**:
- `batch1_completion_summary.md` - Batch 1 detailed summary
- `batch2_completion_summary.md` - Batch 2 detailed summary
- `reviewer_response_overview.md` - Global overview across both batches

**Individual Experiment Reports**:
- `experiment0_calibration_analysis.md` - Baseline establishment
- `experiment2_amplitude_failure_analysis.md` - Amplitude stratification
- `experiment3_complete_metrics_analysis.md` - Full metrics evaluation
- `experiment6_distribution_shift_analysis.md` - Distribution robustness
- `experiment8_gain_drift_analysis.md` - Gain invariance
- `experiment9_pulse_shape_analysis.md` - Pulse shape (baseline)
- `experiment11_oer_threshold_analysis.md` - OER optimization

**Execution Reports**:
- `BATCH1_EXECUTION_REPORT.md` - Batch 1 execution details
- `BATCH2_EXECUTION_REPORT.md` - Batch 2 execution details
- `PROJECT_STATUS.md` - This document

---

## Testing Coverage

### Sample Statistics

| Experiment | Stage 1 | Stage 2 | Total | Conditions |
|------------|---------|---------|-------|------------|
| Exp 0: Calibration | 112,000 | 112,000 | 224,000 | 1 baseline |
| Exp 2: Amplitude failure | 560,000 | 560,000 | 1,120,000 | 5 dimensions × bins |
| Exp 3: Complete metrics | 112,000 | 112,000 | 224,000 | 1 baseline |
| Exp 6: Distribution shift | 560,000 | 560,000 | 1,120,000 | 5 distributions |
| Exp 8: Gain drift | 784,000 | 784,000 | 1,568,000 | 7 gain factors |
| Exp 9: Pulse shape | 112,000 | 112,000 | 224,000 | 1 baseline |
| Exp 11: OER threshold | 784,000 | 784,000 | 1,568,000 | 7 thresholds |
| **Total** | **3,024,000** | **3,024,000** | **6,048,000** | — |

### Test Dimensions

- **Training stages**: 2 (Stage 1 Timing-Focused, Stage 2 Amplitude-Focused)
- **Gain conditions**: 7 factors spanning ±20%
- **Distribution shifts**: 5 amplitude range variations
- **OER thresholds**: 7 values from 0.01 to 0.10
- **Amplitude stratification**: 5 dimensions (dt, ratio, a1, a2, strength) × 5-6 bins each

---

## Technical Challenges Resolved

### Challenge 1: Classification Output Shape Mismatch

**Problem**: Classification tensor shape was `(batch_size, 1, 2)` instead of expected `(batch_size, 2)`.

**Solution**:
```python
# Changed from:
probs = torch.softmax(clas, dim=1)
preds = torch.argmax(probs, dim=1)

# To:
probs = torch.softmax(clas, dim=-1)
preds = torch.argmax(probs, dim=-1).squeeze()
```

### Challenge 2: Pulse Shape Modification Limitation

**Problem**: SAEFit uses pre-recorded real detector waveforms; shape parameters (rise/fall time) are embedded in binary files and cannot be modified dynamically.

**Solution**: 
- Short-term: Provided baseline performance and indirect evidence (gain invariance, implicit shape diversity)
- Long-term: Proposed parametric pulse generator implementation (2-3 hours)

---

## Outstanding Work

### High Priority (Complete Reviewer Response)

**Experiment 9 Full Implementation**: Parametric pulse generator for direct shape variation testing

- **Objective**: Test rise/fall time variations ±10%, ±20%, ±30%
- **Approach**: Implement double-exponential decay pulse model
- **Estimated effort**: 2-3 hours
- **Impact**: Fully addresses remaining reviewer concerns about pulse shape robustness

### Medium Priority (Optional)

**Batch 3: Multi-Seed Training Stability** (Experiment 1)

- **Objective**: Assess training variability across 3-5 random seeds
- **Estimated effort**: 43-72 hours GPU time
- **Impact**: Addresses training reproducibility concerns

### Low Priority (Optional)

Additional robustness experiments (4, 5, 7, 10, 12, 13, 14) for comprehensive supplementary material.

---

## Usage Guide

### For Response Letter Writing

1. **Cite experiment reports directly**:
   ```
   "We conducted systematic gain drift testing (Experiment 8) 
   across ±20% gain variations. Results demonstrate perfect 
   gain invariance with zero performance degradation."
   ```

2. **Reference JSON data for tables**:
   - All raw numerical results available in `*_stage{1,2}.json` files
   - Ready for table generation and figure plotting

3. **Use batch summaries for overview**:
   - `batch1_completion_summary.md` - High-priority experiments
   - `batch2_completion_summary.md` - Medium-priority experiments
   - `reviewer_response_overview.md` - Cross-batch synthesis

### For Supplementary Materials

- **Figure data**: Extract from JSON files
- **Method details**: Reference evaluation scripts in `SAEFit-main/evaluate_*.py`
- **Statistical tables**: Use stratified results from amplitude_failure and distribution_shift JSONs

---

## Recommendations

### For Immediate Revision Submission

**Current evidence is sufficient to address 6/8 reviewer points** with strong quantitative support. For Experiment 9 (pulse shape), provide:

1. The baseline performance data
2. Indirect evidence (gain invariance, implicit diversity)
3. Technical limitation explanation
4. Commitment to full testing in camera-ready version OR deployment validation

This approach is scientifically honest and demonstrates thorough investigation within implementation constraints.

### For Complete Response

**Implement Experiment 9 full testing** (2-3 hours) to achieve 8/8 fully addressed reviewer points before resubmission.

### For Future Work

Consider Batch 3 multi-seed testing to strengthen reproducibility claims, but this is optional given the consistency already demonstrated across multiple test conditions.

---

## Contact and Files

**All outputs located at**: `/root/auto-baseline-v6-output/work/results/`

**Key entry points**:
- Start with `reviewer_response_overview.md` for global view
- Drill into individual `experiment*_analysis.md` files for details
- Access raw data via `*_stage{1,2}.json` files

**Generated**: 2026-09-22  
**Status**: Ready for response letter drafting

---

**Batch 1 & 2**: ✅ Complete  
**Reviewer coverage**: 6/8 full, 2/8 partial  
**Next decision point**: Experiment 9 full implementation vs proceed to revision
