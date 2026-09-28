# Provenance: what is upstream SAEFit and what is ours

Verified 2026-09-28 by byte-comparing every packaged file against the pristine
upstream archive `SAEFit-main.zip`. "Upstream" is the published work:

> *A Machine Learning Framework for Accurate and Robust Analysis of Radiation
> Detector Pulses*, Nucl. Instrum. Methods A (2025).
> https://doi.org/10.1016/j.nima.2025.170971
> Licensed Apache 2.0 (see `code/LICENSE`), which permits derivative work with
> attribution and a statement of changes — this file is that statement.

## Summary

| | Files | Lines | Provenance |
|---|---|---|---|
| Model, losses, datasets, training loop, dataset files | 26 | 2,306 | **upstream, byte-identical** |
| Evaluation / experiment layer | 20 | 3,813 | **ours** |
| Training entry point `main.py` | 1 | 2 changed lines | upstream + our 2 edits |
| Two-stage training configs | 2 | — | ours (hyperparameters only) |

Nothing upstream was deleted. Every upstream file is either byte-identical or,
in the single case of `main.py`, modified in the two ways listed below.

## Byte-identical to upstream — NOT our work

These carry the published authors' contribution and must not be presented as
ours:

- `code/lab/models/net.py` — the entire model architecture
- `code/control/saefit.py` — the LightningModule: training/validation/test steps,
  including the `test_step` that defines the classification task
- `code/lab/losses/{classification,regression}.py` — all loss functions
- `code/lab/datasets/{pulse_saefit,pulse_actual}.py` — the dataloader and the
  double-pulse synthesis procedure
- `code/lab/{checkpoints.py,schedulers/schedulers.py,utils/misc.py,metrics/__init__.py}`
- `code/config.py`, `code/exps/saefit.ini`, `code/requirements.txt`
- `code/data/real_single_traces.bin`, `code/data/real_double_traces.bin` —
  **the dataset is the upstream authors' measured data**, not ours
- `code/README.md`, `code/LICENSE`, `code/docs/*.png` — retained for attribution

## Ours

### Evaluation layer (3,813 lines, none of it upstream)

- `lab/utils/unified_eval.py` — shared evaluation core; the single definition of
  the balanced single/double task, both OER forms, risk-coverage/AURC, bootstrap
  CIs, and `seed_eval_dataset()` determinism pinning
- `lab/utils/calibration.py`, `lab/utils/perturbations.py`
- `evaluate_temperature_scaling.py` (exp 13), `evaluate_oer_sensitivity.py` (exp 14)
- `evaluate_{calibration,complete_metrics,amplitude_failure,gain_drift,pulse_shape,selective,oer_threshold}.py`
  (exps 0, 2, 3, 6, 8, 9, 11)
- `run_multiseed.sh`, `aggregate_multiseed.py`, `write_multiseed_doc.py`,
  `finalize_multiseed.sh` (exp 12)
- `run_reproduction.sh`, `check_status.sh`
- Everything in `analysis/` and `summary/`

### Two-stage continuation training (configs only)

`exps/continuation_timing.ini` and `exps/continuation_amplitude.ini`, plus the 8
generated `multiseed_seed*_stage*.ini`. These change **hyperparameters only** —
the training code executing them is upstream's:

| | upstream `saefit.ini` | Stage 1 (timing) | Stage 2 (amplitude) |
|---|---|---|---|
| `tau` | 0.75 | 5.0 | 3.0 |
| `tauc` | 0.5 | 0.05 | 0.05 |
| `lam` | 0.01 | 0.01 | 0.05 |
| `learning_rate` | 1e-3 | 3e-4 | 2e-4 |
| `nepochs` | 750 | 100 | 100 |
| `scheduler step_size` | 50 | 25 | 25 |
| `monitor` | `loss` | `val_loss` | `val_loss` |

### The two `main.py` edits

Neither changes scientific behavior:

1. `torch.load(..., weights_only=False)` — required by torch ≥ 2.6, which
   flipped the default. Without it, resuming any checkpoint raises.
2. `ModelCheckpoint(save_last=False, every_n_epochs=100)` — saves one checkpoint
   per stage instead of one per validation. Purely to fit 50 GB of disk; each
   checkpoint is 1.6 GB.

## Honest characterization of the contribution

The model, the losses, the dataset and the training procedure are unmodified
upstream. **Our contribution as packaged is (a) the two-stage continuation
training schedule, expressed as hyperparameters, and (b) the entire evaluation
and robustness-analysis methodology.** No architectural or algorithmic change to
the model was made.

The substantive findings are therefore methodological, and several correct the
upstream evaluation rather than improve the model:

- The original evaluation labelled samples `(dt > 0)`, which is constant 1
  because `PulsePrepare` always emits `t2_shift >= 2`. Accuracy/ECE/OER were
  measured on a single-class problem — the source of the reported 0.99997
  accuracy and 2.7e-5 ECE under every perturbation.
- Perturbations were applied to the raw ADC waveform (~1e4) while the network
  consumes `x/m`, making a nominal `noise_std=0.05` about 1e-6 in effective
  units — i.e. no perturbation.
- Temperature was never actually fitted (LBFGS returned its 1.5 init).
- Evaluation-set synthesis was unseeded, giving a run-to-run accuracy spread
  comparable to the stage2−stage1 effect being measured.

Under the corrected protocol, clean accuracy is 0.988 (Stage 1) / 0.991
(Stage 2), and the multi-seed results show the amplitude-focused stage's
amplitude-RMSE and ECE gains are **not** significant against seed noise.

## Known loose end

`code/lab/metrics/amplitude_rmse.py` and `code/lab/metrics/timing_rmse.py` are
ours (50 lines each, `torchmetrics.Metric` subclasses) but are **never imported
anywhere** — `lab/metrics/__init__.py` is upstream's unmodified
`from torchmetrics import *`, and the RMSE numbers actually reported are computed
inline with numpy in `evaluate_complete_metrics.py`. They are dead code; either
wire them in or delete them before submission.
