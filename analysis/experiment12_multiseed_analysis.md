# Experiment 12: Multi-Seed Independent Repeated Training

Corresponds to the reviewer request for independently repeated training runs. Every previously reported number came from a single seed, so it was impossible to tell whether the Stage 1 -> Stage 2 deltas exceeded run-to-run noise.

- Training driver: `repo/SAEFit-main/run_multiseed.sh`
- Aggregation: `repo/SAEFit-main/aggregate_multiseed.py`
- Document generator: `repo/SAEFit-main/write_multiseed_doc.py`
- Shared evaluation core: `repo/SAEFit-main/lab/utils/unified_eval.py`
- Summary: `results/multiseed_summary.json`
- Per-seed cache: `results/multiseed_per_seed.json`

## Protocol

- Seeds trained: 0, 1, 2, 3, 4 (5 with both stages complete). Seed 0 reuses the original `continuation-*` runs.
- Each seed retrains **both stages from scratch**: Stage 1 Timing-Focused then Stage 2 Amplitude-Focused resuming from that seed's own Stage 1 checkpoint. Only `manual_seed` differs between seeds; all hyperparameters are identical.
- `pl.seed_everything(manual_seed, workers=True)` varies weight init and data order. The **dataset split seed is held at 0** for every run, so all seeds are evaluated on the same test set and the comparison isolates training stochasticity.
- Evaluation uses the unified protocol of Experiments 13/14: balanced single(0)/double(1) task, 224,000 test samples, ECE with 15 bins, temperature fitted per seed on the clean validation split, test-set synthesis pinned by `seed_eval_dataset()`.
- Across-seed uncertainty is a t-based 95% CI of the mean (small n), not a normal approximation.

## Per-seed results (clean test data, uncalibrated)

| Seed | Stage | Accuracy | ECE | NLL | OER_cond@0.85 | T | Checkpoint |
|---|---|---|---|---|---|---|---|
| 0 | stage1 | 0.988473 | 0.001702 | 0.042063 | 0.006965 | 1.0290 | `results/continuation-timing-stage1/09-22__01-33-39---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Timing-epoch=099-val_loss=0.001.ckpt` |
| 0 | stage2 | 0.991232 | 0.002258 | 0.035313 | 0.004933 | 0.9728 | `results/continuation-amplitude-stage2/09-22__17-14-55---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Amplitude-epoch=099-val_loss=0.001.ckpt` |
| 1 | stage1 | 0.989143 | 0.002775 | 0.040818 | 0.006644 | 0.9233 | `results/multiseed/seed1/stage1/09-25__23-14-19---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Timing-epoch=099-val_loss=0.001.ckpt` |
| 1 | stage2 | 0.993036 | 0.001603 | 0.025317 | 0.004367 | 1.0577 | `results/multiseed/seed1/stage2/09-25__23-49-02---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Amplitude-epoch=099-val_loss=0.001.ckpt` |
| 2 | stage1 | 0.988915 | 0.001380 | 0.039743 | 0.007162 | 1.0445 | `results/multiseed/seed2/stage1/09-26__00-21-46---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Timing-epoch=099-val_loss=0.001.ckpt` |
| 2 | stage2 | 0.991594 | 0.000739 | 0.029474 | 0.004490 | 0.9755 | `results/multiseed/seed2/stage2/09-26__00-55-18---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Amplitude-epoch=099-val_loss=0.001.ckpt` |
| 3 | stage1 | 0.989688 | 0.004301 | 0.038062 | 0.005660 | 0.8577 | `results/multiseed/seed3/stage1/09-26__01-28-04---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Timing-epoch=099-val_loss=0.020.ckpt` |
| 3 | stage2 | 0.993121 | 0.001061 | 0.026521 | 0.004135 | 0.9610 | `results/multiseed/seed3/stage2/09-26__02-01-41---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Amplitude-epoch=099-val_loss=0.001.ckpt` |
| 4 | stage1 | 0.986437 | 0.003048 | 0.049641 | 0.007654 | 0.9435 | `results/multiseed/seed4/stage1/09-26__02-34-17---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Timing-epoch=099-val_loss=0.001.ckpt` |
| 4 | stage2 | 0.991500 | 0.002903 | 0.032256 | 0.004617 | 0.8837 | `results/multiseed/seed4/stage2/09-26__03-08-01---dataset_Pulse_SAEFit/checkpoints/SAEFit-Continuation-Amplitude-epoch=099-val_loss=0.001.ckpt` |

## Across-seed statistics

### stage1 (n = 5 seeds: 0, 1, 2, 3, 4)

| Metric | Clean mean +- std | Clean 95% CI | Clean min-max | Shifted (combined) mean +- std |
|---|---|---|---|---|
| Accuracy | 0.988531 +- 0.001250 | [0.986980, 0.990083] | 0.986437-0.989688 | 0.612706 +- 0.035645 |
| ECE | 0.002641 +- 0.001163 | [0.001197, 0.004085] | 0.001380-0.004301 | 0.363166 +- 0.031623 |
| NLL | 0.042066 +- 0.004483 | [0.036501, 0.047630] | 0.038062-0.049641 | 3.810316 +- 1.025110 |
| OER_cond@0.85 | 0.006817 +- 0.000743 | [0.005894, 0.007740] | 0.005660-0.007654 | 0.378239 +- 0.035488 |
| OER_abs@0.85 | 0.006685 +- 0.000715 | [0.005797, 0.007573] | 0.005554-0.007446 | 0.355766 +- 0.030231 |
| E-AURC | 0.000942 +- 0.000190 | [0.000706, 0.001178] | 0.000750-0.001244 | 0.159503 +- 0.044581 |
| Temperature T | 0.9596 +- 0.0774 | [0.8635, 1.0557] | 0.8577-1.0445 | n/a |

| Regression metric | Mean +- std | 95% CI | min-max |
|---|---|---|---|
| dt RMSE (samples) | 2.477562 +- 0.177310 | [2.257438, 2.697687] | 2.289947-2.706163 |
| a1 RMSE (normalized) | 0.414656 +- 0.020113 | [0.389687, 0.439625] | 0.391618-0.446614 |
| a2 RMSE (normalized) | 0.423923 +- 0.052706 | [0.358490, 0.489356] | 0.396512-0.518004 |

### stage2 (n = 5 seeds: 0, 1, 2, 3, 4)

| Metric | Clean mean +- std | Clean 95% CI | Clean min-max | Shifted (combined) mean +- std |
|---|---|---|---|---|
| Accuracy | 0.992096 +- 0.000906 | [0.990971, 0.993222] | 0.991232-0.993121 | 0.599204 +- 0.029648 |
| ECE | 0.001713 +- 0.000880 | [0.000621, 0.002805] | 0.000739-0.002903 | 0.380987 +- 0.025266 |
| NLL | 0.029776 +- 0.004104 | [0.024681, 0.034872] | 0.025317-0.035313 | 4.927956 +- 1.077631 |
| OER_cond@0.85 | 0.004508 +- 0.000296 | [0.004140, 0.004876] | 0.004135-0.004933 | 0.395148 +- 0.029634 |
| OER_abs@0.85 | 0.004447 +- 0.000277 | [0.004103, 0.004791] | 0.004089-0.004844 | 0.375592 +- 0.023959 |
| E-AURC | 0.000564 +- 0.000165 | [0.000359, 0.000769] | 0.000397-0.000821 | 0.180091 +- 0.041874 |
| Temperature T | 0.9702 +- 0.0618 | [0.8935, 1.0468] | 0.8837-1.0577 | n/a |

| Regression metric | Mean +- std | 95% CI | min-max |
|---|---|---|---|
| dt RMSE (samples) | 2.184947 +- 0.219635 | [1.912278, 2.457616] | 1.912795-2.414805 |
| a1 RMSE (normalized) | 0.399223 +- 0.006532 | [0.391113, 0.407332] | 0.388778-0.406384 |
| a2 RMSE (normalized) | 0.398720 +- 0.005991 | [0.391282, 0.406158] | 0.391301-0.405765 |

## Paired Stage 2 - Stage 1 comparison

Each row is the within-seed difference, so the training-seed effect cancels. A delta is called significant only when the t-based 95% CI of the mean difference excludes zero.

| Metric | Mean delta | 95% CI | n seeds | Significant at 95% |
|---|---|---|---|---|
| Accuracy (clean) | +3.565e-03 | [+2.355e-03, +4.776e-03] | 5 | **yes** |
| ECE (clean) | -9.282e-04 | [-2.717e-03, +8.604e-04] | 5 | no |
| NLL (clean) | -1.229e-02 | [-1.754e-02, -7.037e-03] | 5 | **yes** |
| OER_cond@0.85 (clean) | -2.309e-03 | [-3.031e-03, -1.586e-03] | 5 | **yes** |
| OER_abs@0.85 (clean) | -2.237e-03 | [-2.936e-03, -1.539e-03] | 5 | **yes** |
| E-AURC (clean) | -3.782e-04 | [-6.636e-04, -9.288e-05] | 5 | **yes** |
| dt RMSE (samples) | -2.926e-01 | [-4.773e-01, -1.079e-01] | 5 | **yes** |
| a1 RMSE (normalized) | -1.543e-02 | [-3.359e-02, +2.724e-03] | 5 | no |
| a2 RMSE (normalized) | -2.520e-02 | [-9.270e-02, +4.229e-02] | 5 | no |
| Accuracy (shifted) | -1.350e-02 | [-2.373e-02, -3.280e-03] | 5 | **yes** |
| ECE (shifted) | +1.782e-02 | [+6.532e-03, +2.911e-02] | 5 | **yes** |
| NLL (shifted) | +1.118e+00 | [+5.251e-02, +2.183e+00] | 5 | **yes** |
| OER_cond@0.85 (shifted) | +1.691e-02 | [+6.284e-03, +2.753e-02] | 5 | **yes** |
| OER_abs@0.85 (shifted) | +1.983e-02 | [+7.703e-03, +3.195e-02] | 5 | **yes** |
| E-AURC (shifted) | +2.059e-02 | [+6.370e-03, +3.481e-02] | 5 | **yes** |

## Findings

1. **Run-to-run spread in clean accuracy is 0.001250 (Stage 1) and 0.000906 (Stage 2)** over 5 independent training seeds. Every seed uses an identical dataset split (split seed pinned to 0) and an identical evaluation protocol, so this is training stochasticity alone: weight init, data order, and cuDNN non-determinism.

2. **The Stage 1 -> Stage 2 clean accuracy gain (+0.003565) exceeds the seed noise at 95% confidence** (paired CI [+0.002355, +0.004776] excludes zero, n=5). The mean difference is 2.9x the single-stage seed standard deviation. Because the comparison is paired within seed, it is more sensitive than comparing the two unpaired across-seed means, whose CIs overlap heavily.

3. **Which Stage 2 improvements survive the multi-seed test.** Significant on clean data: Accuracy, NLL, OER_cond@0.85, OER_abs@0.85, E-AURC. Not significant: ECE. Any claim in the single-seed results that rests on a non-significant metric should be stated as a point estimate without a directional conclusion.

4. **Seed variability is far larger under distribution shift than in-distribution.** Stage 1 accuracy under `combined` is 0.612706 +- 0.035645 against 0.988531 +- 0.001250 on clean data -- the standard deviation is 29x larger (0.035645 vs 0.001250), spanning 0.559045-0.641991. Every seed still collapses toward the 0.5 chance level, so the *failure* under shift is a property of the architecture and training recipe rather than of a particular run, but the *magnitude* of the collapse is seed-dependent. Single-seed shift numbers should therefore be quoted with this spread, and small between-condition differences under shift (see Experiments 6, 8, 9) cannot be resolved from one seed.

5. **The fitted temperature itself varies across seeds** (stage1 0.8577-1.0445, stage2 0.8837-1.0577), straddling T=1. A per-seed fit is therefore required; reusing one seed's temperature for another run would introduce a calibration error of the same order as the calibration improvement being reported (see Experiment 13).

## Reproduce

```bash
cd auto-baseline-v6-output/work/repo/SAEFit-main
bash run_multiseed.sh 1 2 3 4
python3 aggregate_multiseed.py --output results/multiseed_summary.json
python3 write_multiseed_doc.py
```

Per-seed evaluations are cached in `results/multiseed_per_seed.json` keyed by checkpoint path, so `aggregate_multiseed.py` can be re-run incrementally as seeds finish.
