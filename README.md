# SAEFit Reproduction Package

Complete code and dataset for the SAEFit baseline reproduction, covering the
original 7 experiments plus the 3 experiments added in response to the first
round of Sensors review (multi-seed repeated training, unified temperature
scaling, OER threshold sensitivity).

Packaged 2026-09-28.

> **This is derivative work.** The model, losses, dataloader, training loop and
> dataset are unmodified from the published SAEFit
> ([10.1016/j.nima.2025.170971](https://doi.org/10.1016/j.nima.2025.170971),
> Apache 2.0). Our contribution is the two-stage continuation training schedule
> and the entire evaluation/robustness methodology. Read **`PROVENANCE.md`** for
> the file-by-file breakdown before reusing or citing anything here.

## Layout

```
SAEFit-reproduction/
  code/                   the full SAEFit-main tree: source, configs, dataset
    data/                 real_single_traces.bin, real_double_traces.bin (83 MB)
    lab/                  models, datasets, losses, metrics, utils
      utils/unified_eval.py    shared evaluation core (see below)
    control/saefit.py     LightningModule; test_step defines the real task
    exps/*.ini            training configs, incl. the 8 generated multiseed ones
    results/              result JSONs + training logs (checkpoints excluded)
    evaluate_*.py         one script per experiment
    run_multiseed.sh      experiment 12 driver
  analysis/               per-experiment analysis write-ups + raw result JSONs
  summary/                ALL_EXPERIMENTS_SUMMARY.{json,md} + regenerator
  paper/                  peer-review PDF
```

## What is NOT in this archive

Model checkpoints (`*.ckpt`) are excluded: 10 files, 16 GB, which is 130× the
size of everything else combined. Every number in `analysis/` and `summary/` is
reproducible from the code and dataset here by rerunning the commands below.
Also excluded: TensorBoard event files (~300 MB), per-epoch pulse dumps
(15,020 `.txt`), and progress-bar spam in the training logs (downsampled).

## Environment

torch 2.8.0+cu128, pytorch-lightning 2.5.0, Python 3.11, one RTX 4090.
Stage runtime is ~32 s/epoch × 100 epochs ≈ 55 min; a full 4-seed multi-seed
run is ~8 h.

## Reproduce

Training, two stages:

```bash
cd code
python main.py --config exps/continuation_timing.ini      # stage 1, timing-focused
python main.py --config exps/continuation_amplitude.ini   # stage 2, amplitude-focused
```

Experiment 13 — unified temperature scaling:

```bash
python evaluate_temperature_scaling.py \
  --checkpoint results/continuation-timing-stage1/*/checkpoints/*.ckpt \
  --output results/temp_scaling_stage1.json --tag stage1_seed0 --save-predictions
```

Experiment 14 — OER threshold sensitivity:

```bash
python evaluate_oer_sensitivity.py \
  --checkpoint results/continuation-timing-stage1/*/checkpoints/*.ckpt \
  --output results/oer_sensitivity_stage1.json --tag stage1_seed0
```

Experiment 12 — multi-seed repeated training, then aggregate:

```bash
bash run_multiseed.sh 1 2 3 4
python aggregate_multiseed.py --output results/multiseed_summary.json
python write_multiseed_doc.py
```

Regenerate the consolidated summary:

```bash
python summary/generate_summary.py
```

## Two corrections that invalidate the earlier numbers

Experiments 0 and 11 are retained in the summary but marked **superseded**, by
13 and 14 respectively. Two bugs in the original evaluation code:

1. **Degenerate labels.** The label was `(dt > 0).long()`, and `PulsePrepare`
   emits `t2_shift >= 2` for every sample, so the label was constant 1.
   Accuracy, ECE and OER were measured on a single-class problem that a constant
   predictor solves — hence the reported 0.99997 accuracy and 2.7e-5 ECE under
   every perturbation. The real balanced single/double task gives 0.988 (stage 1)
   and 0.991 (stage 2).
2. **Un-normalized perturbations.** Noise was added to the raw ADC waveform
   (magnitude ~1e4), but the network consumes `x/m`. A nominal `noise_std=0.05`
   was therefore ~1e-6 in the units the model sees — effectively no perturbation.

A third, smaller issue: temperature was never actually fitted. LBFGS from a
`T=1.5` init on an already-tiny NLL returned 1.5 unchanged in every run.
`evaluate_temperature_scaling.py` uses a log-space grid plus golden-section
refinement over `T ∈ [0.02, 500]` and warns on a boundary hit.

## Determinism

`PulsePrepare` synthesizes the eval set with bare `np.random.*`, so repeated
evaluation of one checkpoint drew a different test set each time — observed
spread 0.98783 vs 0.988339, comparable to the stage2−stage1 effect being
measured. `lab/utils/unified_eval.seed_eval_dataset()` pins both the numpy and
torch RNGs and is called before every eval dataloader construction; perturbation
draws are seeded separately inside `collect_balanced_predictions`. Results are
now bit-identical across runs.

## Headline results

Balanced single/double task, 224,000 test samples per condition.

| | Stage 1 | Stage 2 |
|---|---|---|
| Clean accuracy | 0.988473 | 0.991232 |
| T (clean fit) | 1.029 | 0.973 |
| Clean ECE | 0.0017 | 0.0023 |
| OER_cond@0.85 | 6.97e-03 | 4.93e-03 |
| `combined` accuracy | 0.5590 | 0.5596 |

Multi-seed, 5 seeds × 2 stages (clean accuracy 0.988531 ± 0.001250 stage 1,
0.992096 ± 0.000906 stage 2), paired stage2−stage1 on clean data: accuracy
+3.57e-03, NLL −1.23e-02, OER_cond@0.85 −2.31e-03, E-AURC −3.78e-04 and dt RMSE
−0.293 samples are all significant at 95%; **ECE (−9.28e-04) and both amplitude
RMSEs are not** — the amplitude-focused stage's amplitude gains sit inside seed
noise. The accuracy gain is carried entirely by the double class (+6.60e-03,
significant) while singles do not move (+5.29e-04, not significant). Under
distribution shift every metric moves significantly in the *worse* direction
(shifted accuracy −1.35e-02, ECE +1.78e-02).

Seed variability under shift is ~28× the clean spread (±0.0356 vs ±0.00125), so
the single-seed shift numbers in experiments 6, 8 and 9 cannot resolve small
between-condition differences. The fitted temperature also varies across seeds
(0.858–1.044 stage 1, 0.884–1.058 stage 2), straddling T=1, so it must be fitted
per run.

## Where to start reading

`summary/ALL_EXPERIMENTS_SUMMARY.md` — all 10 experiments with traceable paths
to every source file. Then `analysis/experiment1{2,3,4}_*.md` for the three new
experiments.
