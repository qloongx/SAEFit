#!/usr/bin/env python
"""
实验8: 增益漂移鲁棒性测试

测试不同增益漂移条件下的性能，评估模型对增益变化的鲁棒性。

对应审稿意见：评论员2第6条关于增益变化的鲁棒性

Usage:
    python evaluate_gain_drift.py --checkpoint checkpoints/stage2.ckpt --output gain_drift_stage2.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pytorch_lightning as pl


# Constants
TIMING_FACTOR = 60.0  # pixels per sampling period
AMPLITUDE_FACTOR = 25000.0  # ADC fullscale


def apply_gain_drift(amplitudes, drift_factor):
    """
    Apply gain drift to amplitudes

    Args:
        amplitudes: tensor of shape (N,)
        drift_factor: multiplicative factor (e.g., 0.9 for -10%, 1.1 for +10%)

    Returns:
        Modified amplitudes
    """
    return amplitudes * drift_factor


def evaluate_with_gain_drift(model, test_loader, drift_factor, device):
    """
    Evaluate model with gain drift applied to ground truth amplitudes

    Args:
        model: trained model
        test_loader: test data loader
        drift_factor: gain drift factor
        device: torch device

    Returns:
        dict with metrics
    """
    all_dt = []
    all_a1_true = []
    all_a2_true = []
    all_a1_pred = []
    all_a2_pred = []
    all_t1_pred = []
    all_t2_pred = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Drift={drift_factor:.2f}"):
            x1, x2, x, m1, m2, m, dt = batch
            x1, x2, x, m1, m2, m, dt = x1.to(device), x2.to(device), x.to(device), \
                                       m1.to(device), m2.to(device), m.to(device), dt.to(device)

            # Apply gain drift to input signal (simulating gain change in detector)
            x_drifted = x * drift_factor
            x1_drifted = x1 * drift_factor
            x2_drifted = x2 * drift_factor

            # Model forward
            outputs = model.model(x_drifted.unsqueeze(1))
            out, clas, t1, a1s, a1, k1, r1, t2, a2, k2, r2 = outputs

            # Get predictions
            batch_size = x.shape[0]
            probs = torch.softmax(clas, dim=-1)
            preds = torch.argmax(probs, dim=-1).squeeze()
            labels = torch.ones(batch_size, dtype=torch.long, device=device)  # All are double-pulse

            # Store results
            all_dt.append(dt.cpu())
            all_a1_true.append(m1.cpu())
            all_a2_true.append(m2.cpu())
            all_a1_pred.append(a1.expand(batch_size, -1).mean(dim=1).cpu())
            all_a2_pred.append(a2.expand(batch_size, -1).mean(dim=1).cpu())
            all_t1_pred.append(t1.expand(batch_size, -1).mean(dim=1).cpu())
            all_t2_pred.append(t2.expand(batch_size, -1).mean(dim=1).cpu())
            all_predictions.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    dt_intervals = torch.cat(all_dt)
    a1_true = torch.cat(all_a1_true)
    a2_true = torch.cat(all_a2_true)
    a1_pred = torch.cat(all_a1_pred)
    a2_pred = torch.cat(all_a2_pred)
    t1_pred = torch.cat(all_t1_pred)
    t2_pred = torch.cat(all_t2_pred)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    # Convert to numpy
    dt_intervals = dt_intervals.numpy()
    a1_true = a1_true.numpy()
    a2_true = a2_true.numpy()
    a1_pred = a1_pred.numpy()
    a2_pred = a2_pred.numpy()
    t1_pred = t1_pred.numpy()
    t2_pred = t2_pred.numpy()
    predictions = predictions.numpy()
    labels = labels.numpy()

    # Convert to physical units
    a1_true_phys = a1_true / AMPLITUDE_FACTOR
    a2_true_phys = a2_true / AMPLITUDE_FACTOR
    a1_pred_phys = a1_pred / AMPLITUDE_FACTOR
    a2_pred_phys = a2_pred / AMPLITUDE_FACTOR

    t1_pred_phys = t1_pred / TIMING_FACTOR
    t2_pred_phys = t2_pred / TIMING_FACTOR
    dt_true_phys = dt_intervals / TIMING_FACTOR

    # Compute errors
    a1_errors = a1_pred_phys - a1_true_phys
    a2_errors = a2_pred_phys - a2_true_phys
    t1_errors = t1_pred_phys  # t1 true is 0 after centering
    t2_errors = t2_pred_phys - dt_true_phys

    # Compute metrics
    metrics = {
        'drift_factor': float(drift_factor),
        'n_samples': len(dt_intervals),
        'classification': {
            'accuracy': float((predictions == labels).mean()),
            'n_errors': int((predictions != labels).sum())
        },
        'timing': {
            't1_rmse': float(np.sqrt(np.mean(t1_errors**2))),
            't2_rmse': float(np.sqrt(np.mean(t2_errors**2))),
            't1_mae': float(np.mean(np.abs(t1_errors))),
            't2_mae': float(np.mean(np.abs(t2_errors)))
        },
        'amplitude': {
            'a1_rmse': float(np.sqrt(np.mean(a1_errors**2))),
            'a2_rmse': float(np.sqrt(np.mean(a2_errors**2))),
            'a1_mae': float(np.mean(np.abs(a1_errors))),
            'a2_mae': float(np.mean(np.abs(a2_errors))),
            'a1_rel_error': float(np.mean(np.abs(a1_errors / a1_true_phys))),
            'a2_rel_error': float(np.mean(np.abs(a2_errors / a2_true_phys)))
        },
        'joint': {
            'joint_error': float(np.mean(np.sqrt(a1_errors**2 + a2_errors**2)))
        }
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Gain drift robustness test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--drift_factors', type=float, nargs='+',
                        default=[0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20],
                        help='Gain drift factors to test')

    args = parser.parse_args()

    print("="*70)
    print("实验8: 增益漂移鲁棒性测试")
    print("="*70)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    from control.saefit import SAEFit as SAEFitModule
    import lab.datasets as datasets

    model = SAEFitModule.load_from_checkpoint(args.checkpoint, map_location='cpu')
    model.eval()

    # Load checkpoint to get opts
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', )
    opts = hparams.get('opts', None)

    # Create datamodule
    if opts is not None:
        datamodule = getattr(datasets, opts.dataset)(opts)
    else:
        raise ValueError("Cannot find opts in checkpoint")

    # Get test loader
    test_loader = datamodule.test_dataloader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Test different gain drift factors
    print(f"\nTesting {len(args.drift_factors)} gain drift conditions...")
    print(f"{'Drift':<10} {'Accuracy':<12} {'a1_RMSE':<12} {'a2_RMSE':<12} {'t2_RMSE':<12}")
    print("-"*70)

    results = []
    for drift_factor in args.drift_factors:
        metrics = evaluate_with_gain_drift(model, test_loader, drift_factor, device)
        results.append(metrics)

        print(f"{drift_factor:<10.2f} "
              f"{metrics['classification']['accuracy']:<12.6f} "
              f"{metrics['amplitude']['a1_rmse']:<12.6f} "
              f"{metrics['amplitude']['a2_rmse']:<12.6f} "
              f"{metrics['timing']['t2_rmse']:<12.6f}")

    # Compute degradation relative to baseline (drift=1.0)
    baseline = next((r for r in results if abs(r['drift_factor'] - 1.0) < 0.01), None)
    if baseline:
        print(f"\n{'='*70}")
        print("Degradation Relative to Baseline (drift=1.0)")
        print(f"{'='*70}")
        print(f"{'Drift':<10} {'Δ Acc%':<12} {'Δ a1_RMSE%':<15} {'Δ a2_RMSE%':<15} {'Δ t2_RMSE%':<15}")
        print("-"*70)

        for r in results:
            if abs(r['drift_factor'] - 1.0) > 0.01:
                acc_change = (r['classification']['accuracy'] - baseline['classification']['accuracy']) * 100
                a1_change = (r['amplitude']['a1_rmse'] - baseline['amplitude']['a1_rmse']) / baseline['amplitude']['a1_rmse'] * 100
                a2_change = (r['amplitude']['a2_rmse'] - baseline['amplitude']['a2_rmse']) / baseline['amplitude']['a2_rmse'] * 100
                t2_change = (r['timing']['t2_rmse'] - baseline['timing']['t2_rmse']) / baseline['timing']['t2_rmse'] * 100

                print(f"{r['drift_factor']:<10.2f} "
                      f"{acc_change:<12.4f} "
                      f"{a1_change:<15.2f} "
                      f"{a2_change:<15.2f} "
                      f"{t2_change:<15.2f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'checkpoint': args.checkpoint,
            'drift_factors_tested': args.drift_factors,
            'timing_factor': TIMING_FACTOR,
            'amplitude_factor': AMPLITUDE_FACTOR
        },
        'results_by_drift': results,
        'baseline': baseline
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()