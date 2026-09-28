#!/usr/bin/env python
"""Experiment 12: aggregate the multi-seed independent training runs.

Discovers every trained seed (seed 0 = the original ``continuation-*`` runs,
seeds 1..N = ``results/multiseed/seed*/stage{1,2}``), evaluates each checkpoint
with the *same* unified protocol used by experiments 13 and 14, and reports
mean +/- std with a t-based 95% CI across seeds.

Why this matters for the review: every number reported so far comes from a
single seed, so it is impossible to tell whether the Stage 1 -> Stage 2 deltas
(and the Stage 2 amplitude improvement in particular) exceed run-to-run noise.
This script produces the paired per-seed comparison that settles that.

Usage:
    python aggregate_multiseed.py --output results/multiseed_summary.json
"""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

import lab.datasets as datasets
from control.saefit import SAEFit as SAEFitModule
from evaluate_temperature_scaling import apply_temperature, fit_temperature
from lab.utils.unified_eval import (collect_balanced_predictions,
                                    seed_eval_dataset, summarize_predictions,
                                    threshold_sweep)

SEED0 = {
    'stage1': 'results/continuation-timing-stage1/*/checkpoints/*.ckpt',
    'stage2': 'results/continuation-amplitude-stage2/09-22__17-14-55*/checkpoints/*.ckpt',
}
TIMING_FACTOR = 60.0
AMPLITUDE_FACTOR = 25000.0


def discover_checkpoints():
    """Map seed -> {stage: checkpoint path} for every completed run."""
    found = {}
    for stage, pat in SEED0.items():
        hits = sorted(glob.glob(pat))
        if hits:
            found.setdefault(0, {})[stage] = hits[-1]

    for path in sorted(glob.glob('results/multiseed/seed*/stage*/*/checkpoints/*.ckpt')):
        m = re.search(r'multiseed/seed(\d+)/(stage[12])/', path)
        if m:
            found.setdefault(int(m.group(1)), {})[m.group(2)] = path
    return found


@torch.no_grad()
def regression_metrics(model, dataloader, device):
    """t/a RMSE on the double-pulse branch, in physical units.

    Uses the same normalization and scale factors as the existing experiments:
    predictions are multiplied by TIMING_FACTOR / AMPLITUDE_FACTOR before being
    compared with the dataset ground truth.
    """
    net = model.model.to(device).eval()
    a_err = []
    dt_all, t2mt1 = [], []

    for batch in dataloader:
        x1, x2, xd, m1, m2, m, dt = batch
        xd_n = xd.to(device) / m.to(device)[:, None]
        _out, _clas, t1, _a1s, a1, _k1, _r1, t2, a2, _k2, _r2 = net(xd_n, control=False)
        n = xd_n.shape[0]

        t1 = t1.expand(n, -1).mean(dim=1)
        t2 = t2.expand(n, -1).mean(dim=1)
        a1 = a1.expand(n, -1).mean(dim=1)
        a2 = a2.expand(n, -1).mean(dim=1)

        # dt is the ground-truth separation in samples; the model predicts
        # normalized positions, so compare (t2 - t1) * n_t_samples with dt.
        t2mt1.append(((t2 - t1) * 250.0).cpu())
        dt_all.append(dt.cpu())

        gt_a1 = (m1 / m).to(device)
        gt_a2 = (m2 / m).to(device)
        a_err.append(torch.stack(((a1 - gt_a1).cpu(), (a2 - gt_a2).cpu())))

    dt_all = torch.cat(dt_all).double()
    t2mt1 = torch.cat(t2mt1).double()
    a_err = torch.cat(a_err, dim=1).double()

    d = t2mt1 - dt_all
    return {
        'dt_rmse_samples': float(torch.sqrt((d ** 2).mean()).item()),
        'dt_mae_samples': float(d.abs().mean().item()),
        'a1_rmse_norm': float(torch.sqrt((a_err[0] ** 2).mean()).item()),
        'a2_rmse_norm': float(torch.sqrt((a_err[1] ** 2).mean()).item()),
        'n_samples': int(len(dt_all)),
    }


def evaluate_seed(ckpt, device, thresholds):
    model = SAEFitModule.load_from_checkpoint(ckpt, map_location='cpu')
    model.eval()
    opts = torch.load(ckpt, map_location='cpu',
                      weights_only=False)['hyper_parameters']['opts']
    dm = getattr(datasets, opts.dataset)(opts)

    seed_eval_dataset()
    val = collect_balanced_predictions(model, dm.val_dataloader(), device, None)
    T, _, _ = fit_temperature(val['logits'], val['labels'])

    seed_eval_dataset()
    test_loader = dm.test_dataloader()
    clean = collect_balanced_predictions(model, test_loader, device, None)
    shifted = collect_balanced_predictions(model, test_loader, device, 'combined')

    out = {
        'checkpoint': ckpt,
        'temperature': T,
        'clean': summarize_predictions(clean, threshold=0.85),
        'clean_calibrated': summarize_predictions(apply_temperature(clean, T),
                                                  threshold=0.85),
        'shifted_combined': summarize_predictions(shifted, threshold=0.85),
        'oer_by_threshold_clean': threshold_sweep(
            clean['confidences'], clean['predictions'], clean['labels'],
            thresholds, n_boot=0),
        'regression': regression_metrics(model, test_loader, device),
    }
    del model
    torch.cuda.empty_cache()
    return out


def stats(values):
    """mean, std (ddof=1), and t-based 95% CI of the mean."""
    a = np.asarray([v for v in values if v is not None and not
                    (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(a)
    if n == 0:
        return None
    mean = float(a.mean())
    if n == 1:
        return {'mean': mean, 'std': 0.0, 'n': 1,
                'ci95_lower': mean, 'ci95_upper': mean,
                'min': mean, 'max': mean}
    sd = float(a.std(ddof=1))
    # two-sided 95% t quantiles for small n
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return {'mean': mean, 'std': sd, 'n': n,
            'ci95_lower': mean - half, 'ci95_upper': mean + half,
            'min': float(a.min()), 'max': float(a.max())}


def paired_delta(per_seed, stage_a, stage_b, block, metric):
    """Paired stage2 - stage1 difference over seeds that trained both stages."""
    diffs, seeds = [], []
    for seed, stages in sorted(per_seed.items()):
        if stage_a in stages and stage_b in stages:
            va = stages[stage_a][block].get(metric)
            vb = stages[stage_b][block].get(metric)
            if va is None or vb is None:
                continue
            diffs.append(vb - va)
            seeds.append(seed)
    if not diffs:
        return None
    s = stats(diffs)
    s['seeds'] = seeds
    s['significant_at_95'] = bool(
        s['ci95_lower'] > 0 or s['ci95_upper'] < 0) if s['n'] > 1 else False
    return s


def main():
    p = argparse.ArgumentParser(description='Aggregate multi-seed runs')
    p.add_argument('--output', default='results/multiseed_summary.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=[0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 0.999])
    p.add_argument('--per-seed-cache', default='results/multiseed_per_seed.json')
    args = p.parse_args()

    found = discover_checkpoints()
    print('=' * 72)
    print('Experiment 12: multi-seed independent repeated training')
    print('=' * 72)
    for seed in sorted(found):
        print(f'  seed {seed}: {sorted(found[seed])}')

    cache = {}
    if os.path.exists(args.per_seed_cache):
        cache = json.load(open(args.per_seed_cache))

    per_seed = {}
    for seed in sorted(found):
        per_seed[seed] = {}
        for stage in sorted(found[seed]):
            ckpt = found[seed][stage]
            key = f'{seed}/{stage}'
            if key in cache and cache[key].get('checkpoint') == ckpt:
                per_seed[seed][stage] = cache[key]
                print(f'  [cached] seed {seed} {stage}')
                continue
            print(f'\n--- evaluating seed {seed} {stage}')
            r = evaluate_seed(ckpt, args.device, args.thresholds)
            per_seed[seed][stage] = r
            cache[key] = r
            with open(args.per_seed_cache, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"    clean acc {r['clean']['accuracy']:.6f}  "
                  f"ECE {r['clean']['ece']:.6f}  "
                  f"OER@.85 {r['clean']['oer_conditional@0.85']:.6f}  "
                  f"T {r['temperature']:.4f}")

    # --- across-seed statistics per stage ---------------------------------
    blocks = ('clean', 'clean_calibrated', 'shifted_combined')
    metrics = ('accuracy', 'ece', 'nll', 'aurc', 'e_aurc',
               'oer_conditional@0.85', 'oer_absolute@0.85',
               'accuracy_single', 'accuracy_double')
    across = {}
    for stage in ('stage1', 'stage2'):
        seeds = [s for s in per_seed if stage in per_seed[s]]
        if not seeds:
            continue
        entry = {'seeds': sorted(seeds), 'n_seeds': len(seeds)}
        for block in blocks:
            entry[block] = {
                m: stats([per_seed[s][stage][block].get(m) for s in seeds])
                for m in metrics}
        entry['regression'] = {
            m: stats([per_seed[s][stage]['regression'].get(m) for s in seeds])
            for m in ('dt_rmse_samples', 'dt_mae_samples',
                      'a1_rmse_norm', 'a2_rmse_norm')}
        entry['temperature'] = stats([per_seed[s][stage]['temperature']
                                      for s in seeds])
        across[stage] = entry

    # --- paired stage2 - stage1 deltas ------------------------------------
    deltas = {}
    for block in blocks:
        deltas[block] = {m: paired_delta(per_seed, 'stage1', 'stage2', block, m)
                         for m in metrics}
    deltas['regression'] = {
        m: paired_delta(per_seed, 'stage1', 'stage2', 'regression', m)
        for m in ('dt_rmse_samples', 'a1_rmse_norm', 'a2_rmse_norm')}

    result = {
        'metadata': {
            'experiment': 'multiseed_independent_training',
            'seeds_found': {str(k): sorted(v) for k, v in found.items()},
            'protocol': 'unified balanced single/double task; temperature fitted '
                        'per seed on clean val (see lab/utils/unified_eval)',
            'note': 'seed 0 reuses the original continuation-* runs; '
                    'dataset split seed held at 0 for all runs so test sets match',
        },
        'per_seed': {str(k): v for k, v in per_seed.items()},
        'across_seeds': across,
        'paired_stage2_minus_stage1': deltas,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print('\n' + '=' * 72)
    for stage, e in across.items():
        c = e['clean']
        print(f"{stage}  n={e['n_seeds']}  "
              f"acc {c['accuracy']['mean']:.6f}+-{c['accuracy']['std']:.6f}  "
              f"ECE {c['ece']['mean']:.6f}+-{c['ece']['std']:.6f}  "
              f"OER@.85 {c['oer_conditional@0.85']['mean']:.6f}"
              f"+-{c['oer_conditional@0.85']['std']:.6f}")
    d = deltas['clean']['accuracy']
    if d:
        print(f"\npaired stage2-stage1 clean accuracy: {d['mean']:+.6f} "
              f"(95% CI [{d['ci95_lower']:+.6f}, {d['ci95_upper']:+.6f}], "
              f"n={d['n']}, significant={d['significant_at_95']})")
    print(f'saved -> {args.output}')


if __name__ == '__main__':
    main()
