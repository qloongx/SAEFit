#!/usr/bin/env python
"""Experiment 13: unified temperature-scaling calibration.

One protocol, applied identically to every checkpoint (both training stages and
every multi-seed run) so that calibrated numbers are comparable across runs:

  1. Collect logits on the balanced single/double task defined in
     ``control/saefit.py::test_step`` (see ``lab/utils/unified_eval``).
  2. Fit a single scalar temperature T on the *validation* split by minimizing
     NLL.  Two fitting sources are reported: ``clean`` (unperturbed val) and
     ``shifted`` (val under the ``combined`` perturbation).  The clean fit is
     the headline protocol; the shifted fit quantifies how much of the residual
     miscalibration is recoverable if shifted data were available at fit time.
  3. Apply T to the test split under every perturbation condition and report
     accuracy, ECE, NLL, OER (both forms) and AURC/E-AURC before and after.

The fit uses a deterministic log-space grid plus golden-section refinement
rather than LBFGS.  With NLL already at ~1e-2 the LBFGS gradients are ~1e-5, so
the previous implementation returned its 1.5 initialization essentially
unchanged and no calibration was actually performed.

Usage:
    python evaluate_temperature_scaling.py \
        --checkpoint path/to.ckpt --output results/temp_scaling_stage1.json
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import lab.datasets as datasets
from control.saefit import SAEFit as SAEFitModule
from lab.utils.unified_eval import (PERTURBATION_CONFIGS,
                                    collect_balanced_predictions,
                                    seed_eval_dataset, summarize_predictions)


def fit_temperature(logits, labels, lo=0.02, hi=500.0, n_grid=400, n_refine=80):
    """Minimize NLL over T by log-space grid search + golden-section refinement.

    Returns (temperature, nll_at_T, nll_at_T=1).
    """
    logits, labels = logits.double(), labels

    def nll(t):
        return float(F.cross_entropy(logits / t, labels).item())

    grid = [math.exp(x) for x in torch.linspace(math.log(lo), math.log(hi), n_grid).tolist()]
    best = min(grid, key=nll)

    i = grid.index(best)
    a = grid[max(0, i - 1)]
    b = grid[min(len(grid) - 1, i + 1)]
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(n_refine):
        c = b - (b - a) * inv_phi
        d = a + (b - a) * inv_phi
        if nll(c) < nll(d):
            b = d
        else:
            a = c
    t = 0.5 * (a + b)
    if t <= lo * 1.01 or t >= hi * 0.99:
        print(f'  WARNING: fitted T={t:.4f} is at the search boundary '
              f'[{lo}, {hi}]; widen the range')
    return float(t), nll(t), nll(1.0)


def apply_temperature(data, temperature):
    """Rescale logits and recompute confidences; predictions are unchanged."""
    logits = data['logits'] / temperature
    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    return {**data, 'logits': logits,
            'confidences': confidences, 'predictions': predictions}


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    opts = ckpt['hyper_parameters']['opts']
    model = SAEFitModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    datamodule = getattr(datasets, opts.dataset)(opts)
    return model, datamodule, opts


def main():
    p = argparse.ArgumentParser(description='Unified temperature scaling')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tag', default=None, help='label for this run in the JSON')
    p.add_argument('--threshold', type=float, default=0.85)
    p.add_argument('--save-predictions', action='store_true',
                   help='also dump clean/combined test logits as .pt')
    args = p.parse_args()

    print('=' * 72)
    print('Experiment 13: unified temperature-scaling calibration')
    print('=' * 72)
    print(f'checkpoint: {args.checkpoint}')

    model, datamodule, _opts = load_model(args.checkpoint)
    seed_eval_dataset()
    val_loader = datamodule.val_dataloader()
    seed_eval_dataset()
    test_loader = datamodule.test_dataloader()

    # --- fit temperature on validation, two sources -----------------------
    fits = {}
    for source, pert in (('clean', None), ('shifted', 'combined')):
        v = collect_balanced_predictions(model, val_loader, args.device, pert)
        t, nll_t, nll_1 = fit_temperature(v['logits'], v['labels'])
        fits[source] = {
            'temperature': t,
            'val_nll_at_T': nll_t,
            'val_nll_at_1': nll_1,
            'val_nll_reduction': nll_1 - nll_t,
            'val_n_samples': int(len(v['labels'])),
            'fit_perturbation': pert,
        }
        print(f'  fit on {source:8s} val -> T = {t:.4f} '
              f'(NLL {nll_1:.6f} -> {nll_t:.6f})')

    T = fits['clean']['temperature']

    # --- evaluate on test under every condition ---------------------------
    # The named configs from the paper turn out to be severe once applied to
    # the normalized waveform, so a graded ladder is added to expose the trend
    # between clean and 'combined' rather than a single cliff.
    conditions = {'clean': None}
    for s in (0.002, 0.005, 0.01, 0.02):
        conditions[f'graded_sigma{s}'] = {
            'noise_std': s, 'baseline_range': (-s * 2, s * 2)}
    conditions.update({k: k for k in PERTURBATION_CONFIGS})

    per_condition = {}
    for name, pert in conditions.items():
        d = collect_balanced_predictions(model, test_loader, args.device, pert)
        uncal = summarize_predictions(d, threshold=args.threshold)
        cal = summarize_predictions(apply_temperature(d, T),
                                    threshold=args.threshold)
        cal_shift = summarize_predictions(
            apply_temperature(d, fits['shifted']['temperature']),
            threshold=args.threshold)
        per_condition[name] = {
            'uncalibrated': uncal,
            'calibrated_clean_fit': cal,
            'calibrated_shifted_fit': cal_shift,
        }
        print(f'  {name:16s} acc {uncal["accuracy"]:.6f}  '
              f'ECE {uncal["ece"]:.6f} -> {cal["ece"]:.6f}  '
              f'NLL {uncal["nll"]:.6f} -> {cal["nll"]:.6f}')

        if args.save_predictions and name in ('clean', 'combined'):
            out = Path(args.output)
            torch.save({'confidences': d['confidences'], 'logits': d['logits'],
                        'predictions': d['predictions'], 'labels': d['labels'],
                        'group': d['group'], 'temperature': T},
                       out.parent / f'{out.stem}_{name}_predictions.pt')

    # --- aggregate across the five named paper shift conditions -----------
    shift_keys = list(PERTURBATION_CONFIGS)
    aggregate = {}
    for variant in ('uncalibrated', 'calibrated_clean_fit', 'calibrated_shifted_fit'):
        agg = {}
        for metric in ('accuracy', 'ece', 'nll', 'aurc', 'e_aurc',
                       f'oer_conditional@{args.threshold}',
                       f'oer_absolute@{args.threshold}'):
            vals = [per_condition[k][variant][metric] for k in shift_keys]
            agg[metric] = sum(vals) / len(vals)
        aggregate[variant] = agg

    result = {
        'metadata': {
            'experiment': 'unified_temperature_scaling',
            'checkpoint': str(args.checkpoint),
            'tag': args.tag,
            'oer_threshold': args.threshold,
            'task_definition': 'balanced single(label 0)/double(label 1) as in '
                               'control/saefit.py::test_step',
            'fit_method': 'log-space grid + golden-section on val NLL',
        },
        'temperature_fits': fits,
        'per_condition': per_condition,
        'aggregate_over_shift_conditions': aggregate,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print('=' * 72)
    for variant, agg in aggregate.items():
        print(f'{variant:26s} ECE {agg["ece"]:.6f}  NLL {agg["nll"]:.6f}  '
              f'E-AURC {agg["e_aurc"]:.6f}')
    print(f'saved -> {out_path}')


if __name__ == '__main__':
    main()
