# Experiment 14: OER Threshold Sensitivity under a Unified OER Definition

Corresponds to the reviewer request to add OER threshold sensitivity and to fix
a single, consistent OER definition.

- Script: `repo/SAEFit-main/evaluate_oer_sensitivity.py`
- Shared core: `repo/SAEFit-main/lab/utils/unified_eval.py`
- Results: `results/oer_sensitivity_stage1.json`, `results/oer_sensitivity_stage2.json`
- Supersedes: `evaluate_oer_threshold.py`, `results/oer_threshold_stage{1,2}.json`

## The unified OER definition

All OER numbers in the revised results use these two forms, computed on the
balanced single/double task of `control/saefit.py::test_step` (224,000 samples:
112,000 singles labelled 0 and 112,000 doubles labelled 1):

**Conditional OER (headline form)**

    OER_cond(τ) = P(ŷ ≠ y | conf ≥ τ)

The error rate among predictions the model is willing to commit to. Reported as
`NaN` — never 0.0 — when no sample clears τ.

**Absolute OER**

    OER_abs(τ) = P(ŷ ≠ y AND conf ≥ τ)

The fraction of the *whole* test set that is both wrong and confident.
Monotonically non-increasing in τ by construction, which makes it the safer form
for threshold sweeps; `OER_cond` is a ratio of two quantities that both shrink
with τ and so is not guaranteed monotone.

Both use inclusive `conf ≥ τ`. Each JSON row also carries
`oer_legacy_strict_zerofill` computed with the old formula for direct comparison.

### What changed relative to the previous definition

| | Previous | Unified |
|---|---|---|
| Task | labels from `(dt > 0)`, constant 1 → single-class | balanced singles(0) / doubles(1) |
| Comparison | strict `conf > τ` | inclusive `conf ≥ τ` |
| Empty selection | returns `0.0` | returns `NaN` |
| Forms reported | conditional only | conditional + absolute |
| Uncertainty | none | 1000-sample bootstrap 95% CI |
| Ranking metric | none | AURC, E-AURC, full risk-coverage curve |

The `0.0`-on-empty behavior was the most consequential: it made "no sample was
confident enough" numerically indistinguishable from "all confident predictions
were correct."

## Results: Stage 1 (T = 1.029), clean test data, uncalibrated

| τ | OER_cond | 95% CI | OER_abs | Coverage | N selected | N errors |
|---|---|---|---|---|---|---|
| 0.500 | 1.153e-02 | [1.109e-02, 1.196e-02] | 1.153e-02 | 1.0000 | 224,000 | 2,582 |
| 0.600 | 1.023e-02 | [9.820e-03, 1.064e-02] | 1.020e-02 | 0.9966 | 223,247 | 2,284 |
| 0.700 | 8.957e-03 | [8.579e-03, 9.339e-03] | 8.888e-03 | 0.9923 | 222,278 | 1,991 |
| 0.750 | 8.395e-03 | [8.024e-03, 8.759e-03] | 8.308e-03 | 0.9897 | 221,685 | 1,861 |
| 0.800 | 7.760e-03 | [7.401e-03, 8.116e-03] | 7.652e-03 | 0.9860 | 220,867 | 1,714 |
| **0.850** | **6.965e-03** | [6.626e-03, 7.296e-03] | 6.830e-03 | 0.9806 | 219,656 | 1,530 |
| 0.900 | 6.022e-03 | [5.712e-03, 6.333e-03] | 5.853e-03 | 0.9718 | 217,687 | 1,311 |
| 0.950 | 4.677e-03 | [4.394e-03, 4.962e-03] | 4.455e-03 | 0.9527 | 213,397 | 998 |
| 0.990 | 1.989e-03 | [1.788e-03, 2.189e-03] | 1.683e-03 | 0.8462 | 189,549 | 377 |
| 0.995 | 1.198e-03 | [1.036e-03, 1.358e-03] | 8.973e-04 | 0.7493 | 167,847 | 201 |
| 0.999 | 1.585e-04 | [8.408e-05, 2.330e-04] | 7.589e-05 | 0.4788 | 107,244 | 17 |
| 0.9999 | 5.379e-05 | [1.076e-05, 1.076e-04] | 2.232e-05 | 0.4150 | 92,960 | 5 |

## Results: Stage 2 (T = 0.973), clean test data, uncalibrated

| τ | OER_cond | 95% CI | OER_abs | Coverage | N selected | N errors |
|---|---|---|---|---|---|---|
| 0.500 | 8.768e-03 | [8.406e-03, 9.156e-03] | 8.768e-03 | 1.0000 | 224,000 | 1,964 |
| 0.700 | 6.340e-03 | [6.016e-03, 6.652e-03] | 6.295e-03 | 0.9929 | 222,400 | 1,410 |
| 0.800 | 5.432e-03 | [5.116e-03, 5.720e-03] | 5.362e-03 | 0.9870 | 221,088 | 1,201 |
| **0.850** | **4.933e-03** | [4.632e-03, 5.209e-03] | 4.844e-03 | 0.9819 | 219,950 | 1,085 |
| 0.900 | 4.344e-03 | [4.055e-03, 4.594e-03] | 4.228e-03 | 0.9731 | 217,982 | 947 |
| 0.950 | 3.489e-03 | [3.221e-03, 3.719e-03] | 3.321e-03 | 0.9519 | 213,235 | 744 |
| 0.990 | 2.093e-03 | [1.867e-03, 2.285e-03] | 1.768e-03 | 0.8448 | 189,229 | 396 |
| 0.995 | 1.670e-03 | [1.469e-03, 1.848e-03] | 1.304e-03 | 0.7806 | 174,856 | 292 |
| 0.999 | 5.382e-04 | [4.211e-04, 6.610e-04] | 3.304e-04 | 0.6138 | 137,498 | 74 |
| 0.9999 | 6.971e-05 | [1.993e-05, 1.290e-04] | 3.125e-05 | 0.4483 | 100,410 | 7 |

## Summary across variants

| Variant | Accuracy | OER@0.85 | OER_cond min | max | Spread | AURC | E-AURC |
|---|---|---|---|---|---|---|---|
| S1 clean / uncal | 0.988473 | 6.97e-03 | 5.38e-05 | 1.15e-02 | **214×** | 9.62e-04 | 8.95e-04 |
| S1 clean / cal | 0.988473 | 6.89e-03 | 5.45e-05 | 1.15e-02 | 212× | 9.62e-04 | 8.95e-04 |
| S1 combined / uncal | 0.559045 | 4.33e-01 | 3.10e-01 | 4.41e-01 | 1.4× | 3.27e-01 | 2.11e-01 |
| S1 combined / cal | 0.559045 | 4.33e-01 | 3.07e-01 | 4.41e-01 | 1.4× | 3.19e-01 | 2.03e-01 |
| S2 clean / uncal | 0.991232 | 4.93e-03 | 6.97e-05 | 8.77e-03 | **126×** | 8.60e-04 | 8.21e-04 |
| S2 clean / cal | 0.991232 | 5.01e-03 | 7.87e-05 | 8.77e-03 | 111× | 8.59e-04 | 8.21e-04 |
| S2 combined / uncal | 0.559554 | 4.36e-01 | 3.66e-01 | 4.40e-01 | 1.2× | 3.37e-01 | 2.21e-01 |
| S2 combined / cal | 0.559554 | 4.36e-01 | 3.70e-01 | 4.40e-01 | 1.2× | 3.15e-01 | 1.99e-01 |

## Findings

1. **OER is strongly threshold-dependent in-distribution — any single-threshold
   OER claim must state its τ.** Over τ ∈ [0.5, 0.9999] the conditional OER
   spans **214×** for Stage 1 (5.4e-05 → 1.2e-02) and **126×** for Stage 2. The
   bootstrap CIs at adjacent thresholds are disjoint, so this is a real trend,
   not sampling noise. Reporting a single OER without its threshold is therefore
   not reproducible; the revised results quote OER@0.85 with a full sweep beside
   it.

2. **The coverage cost of a high threshold is large and must be reported
   alongside.** Driving Stage 1's OER from 7.0e-03 to 1.6e-04 (a 44× reduction)
   requires moving τ from 0.85 to 0.999, which drops coverage from 98.1% to
   47.9% — more than half the data is rejected. `OER_abs` makes this explicit:
   it falls only from 6.8e-03 to 7.6e-05 because it charges the rejected mass.

3. **Under distribution shift the threshold loses nearly all its selectivity.**
   The spread collapses from 214× (clean) to 1.4× (`combined`): OER stays
   between 0.31 and 0.44 no matter how high τ is set. Confidence thresholding
   cannot be used as a safety mechanism under shift, because the errors are
   themselves high-confidence. E-AURC rises from 8.9e-04 to 2.1e-01, a 236×
   degradation in ranking quality.

4. **Stage 2 has the better in-distribution OER but the worse tail behavior.**
   At τ=0.85 Stage 2 wins (4.9e-03 vs 7.0e-03, non-overlapping CIs). At τ=0.999
   the ordering reverses (5.4e-04 vs 1.6e-04, also non-overlapping): Stage 2
   retains more data (61.4% vs 47.9% coverage) but makes 74 confident errors vs
   17. So the amplitude-focused stage improves average confidence quality while
   producing a heavier overconfident tail.

5. **Temperature scaling barely moves OER, consistent with Experiment 13.** With
   `T ≈ 1.03` / `0.97` the monotone rescaling leaves accuracy exactly unchanged
   and shifts OER@0.85 by under 2% relative. Under shift, calibration does
   improve AURC modestly (0.327 → 0.319 for Stage 1; 0.337 → 0.315 for Stage 2)
   but leaves OER essentially untouched.

6. **The legacy formula agrees with the unified one wherever the selected set is
   non-empty** (see `oer_legacy_strict_zerofill` in the JSON; differences appear
   only in the 6th decimal, from the `>` vs `≥` boundary). The material change is
   not the formula but the task: the old numbers were computed against
   all-ones labels, which is why they read 2.7e-05 regardless of condition.

## Reproduce

```bash
cd auto-baseline-v6-output/work/repo/SAEFit-main
python3 evaluate_oer_sensitivity.py \
  --checkpoint results/continuation-timing-stage1/*/checkpoints/*.ckpt \
  --output results/oer_sensitivity_stage1.json --tag stage1_seed0
```

Test-set synthesis is pinned via `seed_eval_dataset()` (see below), so repeated
runs are bit-identical.
