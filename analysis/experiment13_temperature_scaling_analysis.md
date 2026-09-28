# Experiment 13: Unified Temperature Scaling Calibration

Corresponds to the reviewer request for a unified post-hoc calibration experiment.

- Script: `repo/SAEFit-main/evaluate_temperature_scaling.py`
- Shared core: `repo/SAEFit-main/lab/utils/unified_eval.py`
- Results: `results/temp_scaling_stage1.json`, `results/temp_scaling_stage2.json`
- Predictions: `results/temp_scaling_stage{1,2}_{clean,combined}_predictions.pt`

## Protocol

One protocol applied identically to every checkpoint, so calibrated numbers are
comparable across training stages and across seeds:

1. Collect classifier logits on the **balanced single/double task** — the task
   the model is actually trained and tested on in `control/saefit.py::test_step`:
   the single pulse `x1/m1` carries label 0, the double pulse `xd/m` carries
   label 1. 112,000 test samples per branch → 224,000 evaluated per condition.
2. Fit one scalar temperature `T` on the **validation** split by minimizing NLL.
   Two fit sources are reported: `clean` (unperturbed validation, the headline
   protocol) and `shifted` (validation under the `combined` perturbation, an
   oracle-ish upper bound on what temperature scaling can recover).
3. Apply `T` to the test split under every condition and report accuracy, ECE
   (15 bins), NLL, OER, AURC and E-AURC before and after.

Fitting uses a log-space grid search over `T ∈ [0.02, 500]` followed by
golden-section refinement, and warns if the optimum lands on a search boundary.

## Two corrections to the previous calibration code

The earlier `evaluate_calibration.py` produced accuracy 0.99997 and ECE 2.7e-5
under every perturbation. Both numbers were artifacts:

| Issue | Previous behavior | Consequence |
|---|---|---|
| **Degenerate labels** | Label derived as `(dt > 0).long()`, and `dt` is `t2_shift ≥ 2` for every sample of `PulsePrepare` | Label was constant 1. Accuracy/ECE/OER were measured on a single-class problem, which any constant predictor solves. |
| **Un-normalized input** | Perturbation applied to the raw ADC waveform, which the model never sees; the network consumes `x/m` | With raw amplitudes of order 1e4, a nominal `noise_std=0.05` was ~1e-6 in normalized units — effectively no perturbation at all. |
| **Temperature never fitted** | LBFGS from `T=1.5` init on an already-tiny NLL | Returned `T = 1.5` unchanged in every reported run; no calibration was actually applied. |

Under the corrected protocol the perturbations are genuinely damaging and
temperature scaling has something to work on.

## Results

Accuracy and calibration vs. perturbation severity. `→ clean-fit / shift-fit`
denotes ECE and NLL after applying the clean-fitted and shifted-fitted `T`.

### Stage 1 (Timing-Focused), T_clean = 1.029, T_shifted = 35.36

| Condition | Accuracy | ECE (uncal → clean-fit / shift-fit) | NLL (uncal → clean-fit / shift-fit) |
|---|---|---|---|
| clean | 0.9885 | 0.0017 → 0.0014 / 0.4052 | 0.042 → 0.042 / 0.546 |
| graded σ=0.002 | 0.9211 | 0.0412 → 0.0401 / 0.3388 | 0.206 → 0.203 / 0.552 |
| graded σ=0.005 | 0.8354 | 0.1252 → 0.1241 / 0.2678 | 0.556 → 0.543 / 0.557 |
| graded σ=0.01 | 0.7752 | 0.1873 → 0.1862 / 0.2518 | 0.981 → 0.956 / 0.568 |
| graded σ=0.02 | 0.6888 | 0.2704 → 0.2691 / 0.1901 | 1.673 → 1.629 / 0.596 |
| mild_combined | 0.6433 | 0.3175 → 0.3163 / 0.1408 | 2.245 → 2.184 / 0.612 |
| noise_only | 0.7024 | 0.2617 → 0.2607 / 0.1160 | 2.401 → 2.336 / 0.605 |
| baseline_only | 0.5763 | 0.3968 → 0.3959 / 0.0901 | 4.377 → 4.256 / 0.648 |
| combined | 0.5590 | 0.4155 → 0.4147 / 0.0856 | 5.102 → 4.960 / 0.662 |
| strong_combined | 0.5309 | 0.4523 → 0.4517 / 0.1284 | 8.570 → 8.329 / 0.713 |

Aggregate over the five named paper conditions: ECE 0.3688 → 0.3679 (clean fit)
→ 0.1122 (shifted fit); NLL 4.539 → 4.413 → 0.648.

### Stage 2 (Amplitude-Focused), T_clean = 0.973, T_shifted = 31.05

| Condition | Accuracy | ECE (uncal → clean-fit / shift-fit) | NLL (uncal → clean-fit / shift-fit) |
|---|---|---|---|
| clean | 0.9912 | 0.0023 → 0.0019 / 0.3856 | 0.035 → 0.035 / 0.511 |
| graded σ=0.002 | 0.9109 | 0.0588 → 0.0596 / 0.3044 | 0.291 → 0.298 / 0.517 |
| graded σ=0.005 | 0.8210 | 0.1520 → 0.1528 / 0.2260 | 0.849 → 0.870 / 0.527 |
| graded σ=0.01 | 0.7607 | 0.2174 → 0.2181 / 0.1724 | 1.524 → 1.565 / 0.545 |
| graded σ=0.02 | 0.6942 | 0.2808 → 0.2816 / 0.1219 | 2.391 → 2.457 / 0.582 |
| mild_combined | 0.6557 | 0.3166 → 0.3174 / 0.1203 | 2.896 → 2.975 / 0.601 |
| noise_only | 0.7106 | 0.2596 → 0.2604 / 0.1108 | 3.023 → 3.106 / 0.595 |
| baseline_only | 0.5794 | 0.4007 → 0.4013 / 0.1297 | 4.215 → 4.331 / 0.634 |
| combined | 0.5596 | 0.4179 → 0.4186 / 0.1007 | 5.004 → 5.142 / 0.653 |
| strong_combined | 0.5309 | 0.4529 → 0.4534 / 0.1094 | 7.574 → 7.785 / 0.698 |

Aggregate over the five named conditions: ECE 0.3696 → 0.3702 (clean fit)
→ 0.1142 (shifted fit); NLL 4.542 → 4.668 → 0.636.

## Findings

1. **The model is already well calibrated in-distribution, so clean-fitted
   temperature scaling has almost no room to act.** `T ≈ 1.03` (Stage 1) and
   `≈ 0.97` (Stage 2) are both near the identity, and clean ECE is already
   1.7e-3 / 2.3e-3. Clean ECE improves only marginally (0.0017 → 0.0014).

2. **Clean-fitted temperature scaling does not fix miscalibration under shift.**
   Aggregate ECE moves from 0.3688 to 0.3679 for Stage 1 — a 0.2% relative
   reduction — and for Stage 2 it *worsens* slightly (0.3696 → 0.3702, since
   `T < 1` sharpens already-overconfident predictions). This is the central
   negative result: a single temperature fitted in-distribution cannot transfer
   to shifted data.

3. **The miscalibration is nevertheless largely recoverable if shifted data are
   available at fit time.** The shifted fit drives aggregate ECE down by 3.3×
   (0.369 → 0.112) and NLL by 7× (4.54 → 0.65), at the cost of badly
   *under*confident clean predictions (clean ECE rises to 0.41). The required
   temperature is large (`T ≈ 31–35`), i.e. the logits under shift are
   overconfident by more than an order of magnitude.

4. **Temperature scaling cannot improve ranking, as expected.** A single
   positive scalar is monotone in the logits, so accuracy is identical before
   and after in every row, and E-AURC changes only through floating-point
   reordering of near-ties.

5. **Both stages degrade at essentially the same rate under shift** (e.g.
   `combined`: 0.5590 vs 0.5596 accuracy), so the amplitude-focused second
   stage brings no robustness benefit on the classification task. Stage 2 is
   slightly better in-distribution (0.9912 vs 0.9885).

## Caveat

The named perturbation configs from the paper (`noise_std=0.05`,
`baseline_range=±0.1`) are severe once correctly applied to the normalized
waveform: they push accuracy to 0.53–0.71, near the 0.5 chance level of the
balanced task. The `graded_sigma*` ladder (σ = 0.002…0.02) was added to expose
the degradation trend in the regime where the model is still usable. Whether the
paper's nominal σ values were intended as fractions of the normalized amplitude
is a question for the authors; the graded ladder makes the conclusion independent
of that choice.
