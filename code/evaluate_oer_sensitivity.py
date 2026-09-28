#!/usr/bin/env python
"""Experiment 14: OER threshold sensitivity under a unified OER definition.

Supersedes ``evaluate_oer_threshold.py``, which (a) consumed prediction dumps
built from the degenerate ``(dt > 0)`` label -- constant 1, so every sample was
"double" and OER was measured on a single-class problem -- and (b) used a strict
``conf > tau`` comparison while returning 0.0 for the undefined empty-selection
case, which makes a high-threshold OER of 0 indistinguishable from "no sample
qualified".

Unified definition (``lab/utils/unified_eval.compute_oer``), on the balanced
single/double task of ``control/saefit.py::test_step``:

    OER_cond(tau) = P(yhat != y | conf >= tau)     (headline; nan if none clear tau)
    OER_abs(tau)  = P(yhat != y AND conf >= tau)   (monotone in tau)

Reported for uncalibrated and temperature-scaled confidences, clean and shifted
test data, with bootstrap CIs, the full risk-coverage curve and AURC/E-AURC.

Usage:
    python evaluate_oer_sensitivity.py --checkpoint path/to.ckpt \
        --output results/oer_sensitivity_stage1.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import lab.datasets as datasets
from control.saefit import SAEFit as SAEFitModule
from evaluate_temperature_scaling import apply_temperature, fit_temperature
from lab.utils.unified_eval import (collect_balanced_predictions, compute_aurc,
                                    risk_coverage_curve, seed_eval_dataset,
                                    summarize_predictions, threshold_sweep)

DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                      0.99, 0.995, 0.999, 0.9999]


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    opts = ckpt['hyper_parameters']['opts']
    model = SAEFitModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    return model, getattr(datasets, opts.dataset)(opts)


def legacy_oer(confidences, predictions, labels, threshold):
    """The pre-unification formula, kept to document the numerical difference.

    Strict ``>`` comparison, conditional form, and 0.0 when nothing clears the
    threshold.
    """
    mask = confidences > threshold
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((predictions[mask] != labels[mask]).float().mean().item())


def main():
    p = argparse.ArgumentParser(description='OER threshold sensitivity')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tag', default=None)
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=DEFAULT_THRESHOLDS)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rc-points', type=int, default=101)
    args = p.parse_args()

    print('=' * 72)
    print('Experiment 14: OER threshold sensitivity (unified definition)')
    print('=' * 72)

    model, datamodule = load_model(args.checkpoint)
    seed_eval_dataset()
    val_loader = datamodule.val_dataloader()
    seed_eval_dataset()
    test_loader = datamodule.test_dataloader()

    val = collect_balanced_predictions(model, val_loader, args.device, None)
    T, _, _ = fit_temperature(val['logits'], val['labels'])
    print(f'temperature fitted on clean val: T = {T:.4f}')

    variants = {}
    for cond_name, pert in (('clean', None), ('combined', 'combined')):
        d = collect_balanced_predictions(model, test_loader, args.device, pert)
        for cal_name, dd in (('uncalibrated', d),
                             ('calibrated', apply_temperature(d, T))):
            key = f'{cond_name}/{cal_name}'
            sweep = threshold_sweep(dd['confidences'], dd['predictions'],
                                    dd['labels'], args.thresholds,
                                    inclusive=True, n_boot=args.n_boot,
                                    seed=args.seed)
            for row in sweep:
                row['oer_legacy_strict_zerofill'] = legacy_oer(
                    dd['confidences'], dd['predictions'], dd['labels'],
                    row['threshold'])

            rc = risk_coverage_curve(dd['confidences'], dd['predictions'],
                                     dd['labels'], n_points=args.rc_points)
            finite = [r['oer_conditional'] for r in sweep
                      if not np.isnan(r['oer_conditional'])]
            variants[key] = {
                'summary': summarize_predictions(dd, threshold=0.85),
                'oer_by_threshold': sweep,
                'risk_coverage_curve': rc,
                'aurc': compute_aurc(dd['confidences'], dd['predictions'],
                                     dd['labels']),
                'oer_conditional_range': {
                    'min': min(finite), 'max': max(finite),
                    'mean': float(np.mean(finite)),
                    'spread_factor': (max(finite) / min(finite)
                                      if min(finite) > 0 else None),
                    'n_defined_thresholds': len(finite),
                },
            }
            print(f'\n{key}')
            print(f"{'tau':>8} {'OER_cond':>11} {'OER_abs':>11} "
                  f"{'cover':>9} {'n_sel':>9} {'n_err':>7} {'legacy':>11}")
            for row in sweep:
                print(f"{row['threshold']:>8.4f} {row['oer_conditional']:>11.6f} "
                      f"{row['oer_absolute']:>11.6f} {row['coverage']:>9.4f} "
                      f"{row['n_selected']:>9d} {row['n_errors_selected']:>7d} "
                      f"{row['oer_legacy_strict_zerofill']:>11.6f}")

    result = {
        'metadata': {
            'experiment': 'oer_threshold_sensitivity_unified',
            'checkpoint': str(args.checkpoint),
            'tag': args.tag,
            'temperature': T,
            'thresholds': args.thresholds,
            'n_bootstrap': args.n_boot,
            'seed': args.seed,
            'oer_definitions': {
                'conditional': 'P(yhat != y | conf >= tau); nan if no sample clears tau',
                'absolute': 'P(yhat != y AND conf >= tau); monotone non-increasing in tau',
                'legacy': 'strict conf > tau, conditional, 0.0 when empty (deprecated)',
            },
            'task_definition': 'balanced single(0)/double(1) as in '
                               'control/saefit.py::test_step',
        },
        'variants': variants,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()
