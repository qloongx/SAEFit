#!/usr/bin/env python
"""
实验2: 振幅失败分层分析

按照脉冲间隔、重叠程度、振幅比、绝对振幅、信号强度和噪声水平对测试样本进行分组，
分别报告第一分量和第二分量的振幅RMSE、相对误差及联合分解误差，
并识别典型成功与失败样本。

对应审稿意见：评论员1关于振幅重建问题和失败条件；评论员2第4条和第5条

Usage:
    python evaluate_amplitude_failure.py --checkpoint checkpoints/stage2.ckpt --output amplitude_failure_stage2.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytorch_lightning as pl


# Constants
TIMING_FACTOR = 60.0  # pixels per sampling period
AMPLITUDE_FACTOR = 25000.0  # ADC fullscale


def compute_stratified_errors(dt_intervals, a_ratios, a1_values, a2_values, snr_values,
                              a1_errors, a2_errors, joint_errors):
    """
    按多个维度分层计算误差统计

    Returns:
        dict with stratified results
    """
    results = {}

    # 1. 按脉冲间隔分层 (时间差)
    dt_bins = [(0, 10), (10, 20), (20, 40), (40, 60), (60, np.inf)]
    results['by_pulse_interval'] = []
    for dt_min, dt_max in dt_bins:
        mask = (dt_intervals >= dt_min) & (dt_intervals < dt_max)
        if mask.sum() > 0:
            results['by_pulse_interval'].append({
                'range': f'{dt_min}-{dt_max if dt_max != np.inf else "inf"}',
                'n_samples': int(mask.sum()),
                'a1_rmse': float(np.sqrt(np.mean(a1_errors[mask]**2))),
                'a2_rmse': float(np.sqrt(np.mean(a2_errors[mask]**2))),
                'joint_error': float(np.mean(joint_errors[mask])),
                'a1_rel_error': float(np.mean(np.abs(a1_errors[mask] / a1_values[mask])))
            })

    # 2. 按振幅比分层 (a1/a2)
    ratio_bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, np.inf)]
    results['by_amplitude_ratio'] = []
    for r_min, r_max in ratio_bins:
        mask = (a_ratios >= r_min) & (a_ratios < r_max)
        if mask.sum() > 0:
            results['by_amplitude_ratio'].append({
                'range': f'{r_min}-{r_max if r_max != np.inf else "inf"}',
                'n_samples': int(mask.sum()),
                'a1_rmse': float(np.sqrt(np.mean(a1_errors[mask]**2))),
                'a2_rmse': float(np.sqrt(np.mean(a2_errors[mask]**2))),
                'joint_error': float(np.mean(joint_errors[mask]))
            })

    # 3. 按第一分量绝对振幅分层
    a1_percentiles = np.percentile(a1_values, [0, 25, 50, 75, 100])
    results['by_a1_magnitude'] = []
    for i in range(len(a1_percentiles) - 1):
        a_min, a_max = a1_percentiles[i], a1_percentiles[i+1]
        if i == len(a1_percentiles) - 2:
            mask = (a1_values >= a_min) & (a1_values <= a_max)
        else:
            mask = (a1_values >= a_min) & (a1_values < a_max)
        if mask.sum() > 0:
            results['by_a1_magnitude'].append({
                'range': f'{a_min:.1f}-{a_max:.1f}',
                'quartile': f'Q{i+1}',
                'n_samples': int(mask.sum()),
                'a1_rmse': float(np.sqrt(np.mean(a1_errors[mask]**2))),
                'a2_rmse': float(np.sqrt(np.mean(a2_errors[mask]**2))),
                'joint_error': float(np.mean(joint_errors[mask]))
            })

    # 4. 按第二分量绝对振幅分层
    a2_percentiles = np.percentile(a2_values, [0, 25, 50, 75, 100])
    results['by_a2_magnitude'] = []
    for i in range(len(a2_percentiles) - 1):
        a_min, a_max = a2_percentiles[i], a2_percentiles[i+1]
        if i == len(a2_percentiles) - 2:
            mask = (a2_values >= a_min) & (a2_values <= a_max)
        else:
            mask = (a2_values >= a_min) & (a2_values < a_max)
        if mask.sum() > 0:
            results['by_a2_magnitude'].append({
                'range': f'{a_min:.1f}-{a_max:.1f}',
                'quartile': f'Q{i+1}',
                'n_samples': int(mask.sum()),
                'a1_rmse': float(np.sqrt(np.mean(a1_errors[mask]**2))),
                'a2_rmse': float(np.sqrt(np.mean(a2_errors[mask]**2))),
                'joint_error': float(np.mean(joint_errors[mask]))
            })

    # 5. 按信噪比分层 (使用总振幅作为信号强度代理)
    snr_percentiles = np.percentile(snr_values, [0, 25, 50, 75, 100])
    results['by_signal_strength'] = []
    for i in range(len(snr_percentiles) - 1):
        s_min, s_max = snr_percentiles[i], snr_percentiles[i+1]
        if i == len(snr_percentiles) - 2:
            mask = (snr_values >= s_min) & (snr_values <= s_max)
        else:
            mask = (snr_values >= s_min) & (snr_values < s_max)
        if mask.sum() > 0:
            results['by_signal_strength'].append({
                'range': f'{s_min:.1f}-{s_max:.1f}',
                'quartile': f'Q{i+1}',
                'n_samples': int(mask.sum()),
                'a1_rmse': float(np.sqrt(np.mean(a1_errors[mask]**2))),
                'a2_rmse': float(np.sqrt(np.mean(a2_errors[mask]**2))),
                'joint_error': float(np.mean(joint_errors[mask]))
            })

    return results


def identify_failure_cases(a1_errors, a2_errors, joint_errors,
                           dt_intervals, a_ratios, a1_values, a2_values,
                           top_k=10):
    """
    识别最严重的失败案例

    Returns:
        dict with worst cases
    """
    # 找到振幅误差最大的案例
    a1_worst_idx = np.argsort(np.abs(a1_errors))[-top_k:][::-1]
    a2_worst_idx = np.argsort(np.abs(a2_errors))[-top_k:][::-1]
    joint_worst_idx = np.argsort(joint_errors)[-top_k:][::-1]

    failure_cases = {
        'worst_a1_errors': [],
        'worst_a2_errors': [],
        'worst_joint_errors': []
    }

    for idx in a1_worst_idx:
        failure_cases['worst_a1_errors'].append({
            'index': int(idx),
            'a1_error': float(a1_errors[idx]),
            'a1_true': float(a1_values[idx]),
            'a2_true': float(a2_values[idx]),
            'dt': float(dt_intervals[idx]),
            'amplitude_ratio': float(a_ratios[idx])
        })

    for idx in a2_worst_idx:
        failure_cases['worst_a2_errors'].append({
            'index': int(idx),
            'a2_error': float(a2_errors[idx]),
            'a1_true': float(a1_values[idx]),
            'a2_true': float(a2_values[idx]),
            'dt': float(dt_intervals[idx]),
            'amplitude_ratio': float(a_ratios[idx])
        })

    for idx in joint_worst_idx:
        failure_cases['worst_joint_errors'].append({
            'index': int(idx),
            'joint_error': float(joint_errors[idx]),
            'a1_error': float(a1_errors[idx]),
            'a2_error': float(a2_errors[idx]),
            'a1_true': float(a1_values[idx]),
            'a2_true': float(a2_values[idx]),
            'dt': float(dt_intervals[idx]),
            'amplitude_ratio': float(a_ratios[idx])
        })

    return failure_cases


def main():
    parser = argparse.ArgumentParser(description='Amplitude failure stratification analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of worst cases to identify')

    args = parser.parse_args()

    print("="*70)
    print("实验2: 振幅失败分层分析")
    print("="*70)

    # Load model using the same approach as evaluate_complete_metrics.py
    print(f"\nLoading model from {args.checkpoint}")
    from control.saefit import SAEFit as SAEFitModule
    import lab.datasets as datasets

    model = SAEFitModule.load_from_checkpoint(args.checkpoint, map_location='cpu')
    model.eval()

    # Load checkpoint to get opts
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', {})
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

    # Collect predictions and ground truth
    print("\nEvaluating on test set...")
    all_dt = []
    all_a1_true = []
    all_a2_true = []
    all_a1_pred = []
    all_a2_pred = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            x1, x2, x, m1, m2, m, dt = batch
            x1, x2, x, m1, m2, m, dt = x1.to(device), x2.to(device), x.to(device), \
                                       m1.to(device), m2.to(device), m.to(device), dt.to(device)

            # Model forward (use model.model for the underlying network)
            outputs = model.model(x.unsqueeze(1))
            out, clas, t1, a1s, a1, k1, r1, t2, a2, k2, r2 = outputs

            # Store results - note a1/a2 need to be expanded to match batch size
            batch_size = x.shape[0]
            all_dt.append(dt.cpu())
            all_a1_true.append(m1.cpu())
            all_a2_true.append(m2.cpu())
            all_a1_pred.append(a1.expand(batch_size, -1).mean(dim=1).cpu())
            all_a2_pred.append(a2.expand(batch_size, -1).mean(dim=1).cpu())

    # Concatenate all batches
    dt_intervals = torch.cat(all_dt).numpy()
    a1_true = torch.cat(all_a1_true).numpy()
    a2_true = torch.cat(all_a2_true).numpy()
    a1_pred = torch.cat(all_a1_pred).numpy()
    a2_pred = torch.cat(all_a2_pred).numpy()

    # Convert to physical units
    a1_true_phys = a1_true / AMPLITUDE_FACTOR
    a2_true_phys = a2_true / AMPLITUDE_FACTOR
    a1_pred_phys = a1_pred / AMPLITUDE_FACTOR
    a2_pred_phys = a2_pred / AMPLITUDE_FACTOR

    # Compute errors
    a1_errors = a1_pred_phys - a1_true_phys
    a2_errors = a2_pred_phys - a2_true_phys
    joint_errors = np.sqrt(a1_errors**2 + a2_errors**2)

    # Compute derived quantities
    a_ratios = a1_true_phys / (a2_true_phys + 1e-8)
    snr_values = a1_true_phys + a2_true_phys  # Total signal as proxy

    print(f"\nTotal samples: {len(dt_intervals)}")
    print(f"a1 RMSE (overall): {np.sqrt(np.mean(a1_errors**2)):.6f}")
    print(f"a2 RMSE (overall): {np.sqrt(np.mean(a2_errors**2)):.6f}")
    print(f"Joint error (overall): {np.mean(joint_errors):.6f}")

    # Compute stratified errors
    print(f"\n{'='*70}")
    print("Stratified Error Analysis")
    print(f"{'='*70}")

    stratified_results = compute_stratified_errors(
        dt_intervals, a_ratios, a1_true_phys, a2_true_phys, snr_values,
        a1_errors, a2_errors, joint_errors
    )

    # Print stratified results
    print("\n1. By Pulse Interval (dt):")
    print(f"{'Range':<15} {'N':<10} {'a1_RMSE':<12} {'a2_RMSE':<12} {'Joint':<12}")
    print("-"*70)
    for result in stratified_results['by_pulse_interval']:
        print(f"{result['range']:<15} {result['n_samples']:<10} "
              f"{result['a1_rmse']:<12.6f} {result['a2_rmse']:<12.6f} "
              f"{result['joint_error']:<12.6f}")

    print("\n2. By Amplitude Ratio (a1/a2):")
    print(f"{'Range':<15} {'N':<10} {'a1_RMSE':<12} {'a2_RMSE':<12} {'Joint':<12}")
    print("-"*70)
    for result in stratified_results['by_amplitude_ratio']:
        print(f"{result['range']:<15} {result['n_samples']:<10} "
              f"{result['a1_rmse']:<12.6f} {result['a2_rmse']:<12.6f} "
              f"{result['joint_error']:<12.6f}")

    print("\n3. By a1 Magnitude (Quartiles):")
    print(f"{'Quartile':<12} {'Range':<20} {'N':<10} {'a1_RMSE':<12}")
    print("-"*70)
    for result in stratified_results['by_a1_magnitude']:
        print(f"{result['quartile']:<12} {result['range']:<20} {result['n_samples']:<10} "
              f"{result['a1_rmse']:<12.6f}")

    # Identify failure cases
    print(f"\n{'='*70}")
    print(f"Worst {args.top_k} Failure Cases")
    print(f"{'='*70}")

    failure_cases = identify_failure_cases(
        a1_errors, a2_errors, joint_errors,
        dt_intervals, a_ratios, a1_true_phys, a2_true_phys,
        top_k=args.top_k
    )

    print("\nWorst a1 errors:")
    print(f"{'Index':<10} {'Error':<12} {'a1_true':<12} {'a2_true':<12} {'dt':<10}")
    print("-"*70)
    for case in failure_cases['worst_a1_errors'][:5]:
        print(f"{case['index']:<10} {case['a1_error']:<12.6f} {case['a1_true']:<12.6f} "
              f"{case['a2_true']:<12.6f} {case['dt']:<10.2f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'checkpoint': args.checkpoint,
            'n_samples': len(dt_intervals),
            'timing_factor': TIMING_FACTOR,
            'amplitude_factor': AMPLITUDE_FACTOR
        },
        'overall_metrics': {
            'a1_rmse': float(np.sqrt(np.mean(a1_errors**2))),
            'a2_rmse': float(np.sqrt(np.mean(a2_errors**2))),
            'joint_error_mean': float(np.mean(joint_errors)),
            'a1_rel_error': float(np.mean(np.abs(a1_errors / a1_true_phys))),
            'a2_rel_error': float(np.mean(np.abs(a2_errors / a2_true_phys)))
        },
        'stratified_analysis': stratified_results,
        'failure_cases': failure_cases
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
