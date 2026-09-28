#!/usr/bin/env python
"""
实验3: 计算完整指标集，补全Table 3中的缺失值

评估所有检查点的完整指标：
- Accuracy
- Timing RMSE (两个分量)
- Amplitude RMSE (两个分量)
- Joint Error
- ASNR
- ECE, NLL (校准指标)
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path


def load_checkpoint_and_evaluate(checkpoint_path, output_path):
    """加载检查点并计算完整指标"""
    import pytorch_lightning as pl
    from control.saefit import SAEFit as SAEFitModule
    import lab.datasets as datasets

    print(f"Loading checkpoint: {checkpoint_path}")

    # 加载模型
    model = SAEFitModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()

    # 加载数据
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', {})
    opts = hparams.get('opts', None)

    if opts is not None:
        datamodule = getattr(datasets, opts.dataset)(opts)
    else:
        from argparse import Namespace
        default_opts = Namespace(
            dataset='Pulse_SAEFit',
            batch_size=10000,
            num_workers=4,
            data_root='../data',
        )
        datamodule = datasets.Pulse_SAEFit(default_opts)

    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    # 收集所有预测
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_gt_timing = []
    all_gt_amplitude = []
    all_timing_preds = []
    all_amplitude_preds = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch: x1, x2, x, m1, m2, m, dt
            if isinstance(batch, (list, tuple)) and len(batch) >= 7:
                x = batch[2].to(device)  # Combined pulse
                m1_true = batch[3]  # Amplitude 1
                m2_true = batch[4]  # Amplitude 2
                dt_true = batch[6]  # Timing offset

                # Only evaluate on double pulses
                double_mask = dt_true > 0
                if double_mask.sum() == 0:
                    continue

                # 前向传播 - SAEFit returns: out, clas, t1, a1s, a1, k1, r1, t2, a2, k2, r2
                outputs = model.model(x)
                clas = outputs[1]  # Classification logits
                t1_pred = outputs[2][double_mask]  # First pulse timing
                a1_pred = outputs[4][double_mask]  # First pulse amplitude
                t2_pred = outputs[7][double_mask]  # Second pulse timing
                a2_pred = outputs[8][double_mask]  # Second pulse amplitude

                # 收集预测 - classification probabilities
                probs = torch.softmax(clas, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.append(probs[double_mask].cpu())
                all_labels.append(preds[double_mask].cpu())

                # 收集时序和振幅真值和预测
                # Timing: [t1=0 (fixed), t2=dt_true]
                dt_masked = dt_true[double_mask].to(device)
                gt_timing = torch.stack([torch.zeros_like(dt_masked), dt_masked], dim=1)
                pred_timing = torch.stack([t1_pred.squeeze(), t2_pred.squeeze()], dim=1)

                # Amplitude: [a1, a2]
                gt_amplitude = torch.stack([m1_true[double_mask].to(device), m2_true[double_mask].to(device)], dim=1)
                pred_amplitude = torch.stack([a1_pred.squeeze(), a2_pred.squeeze()], dim=1)

                all_gt_timing.append(gt_timing.cpu())
                all_gt_amplitude.append(gt_amplitude.cpu())

                all_timing_preds.append(pred_timing.cpu())
                all_amplitude_preds.append(pred_amplitude.cpu())

    # 合并所有批次
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_gt_timing = torch.cat(all_gt_timing, dim=0).numpy()
    all_gt_amplitude = torch.cat(all_gt_amplitude, dim=0).numpy()
    all_timing_preds = torch.cat(all_timing_preds, dim=0).numpy()
    all_amplitude_preds = torch.cat(all_amplitude_preds, dim=0).numpy()

    print(f"Total double-pulse test samples: {len(all_labels)}")

    # 计算指标
    results = {}

    # 1. Accuracy
    accuracy = np.mean(all_labels == 1)  # 假设类别1是双脉冲
    results['accuracy'] = float(accuracy)
    print(f"Accuracy: {accuracy:.6f}")

    # 2. Timing RMSE
    # 单位转换因子
    TIMING_FACTOR = 60.0  # pixels → sampling periods
    AMPLITUDE_FACTOR = 25000.0  # ADC counts → normalized

    # 计算RMSE（原始单位）
    timing_rmse_t1_raw = np.sqrt(np.mean((all_timing_preds[:, 0] - all_gt_timing[:, 0])**2))
    timing_rmse_t2_raw = np.sqrt(np.mean((all_timing_preds[:, 1] - all_gt_timing[:, 1])**2))
    amplitude_rmse_a1_raw = np.sqrt(np.mean((all_amplitude_preds[:, 0] - all_gt_amplitude[:, 0])**2))
    amplitude_rmse_a2_raw = np.sqrt(np.mean((all_amplitude_preds[:, 1] - all_gt_amplitude[:, 1])**2))

    # 转换为论文单位
    timing_rmse_t1 = timing_rmse_t1_raw / TIMING_FACTOR
    timing_rmse_t2 = timing_rmse_t2_raw / TIMING_FACTOR
    amplitude_rmse_a1 = amplitude_rmse_a1_raw / AMPLITUDE_FACTOR
    amplitude_rmse_a2 = amplitude_rmse_a2_raw / AMPLITUDE_FACTOR

    # 综合RMSE（论文中报告的）
    timing_rmse_combined = np.sqrt(np.mean(((all_timing_preds - all_gt_timing) / TIMING_FACTOR)**2))
    amplitude_rmse_combined = np.sqrt(np.mean(((all_amplitude_preds - all_gt_amplitude) / AMPLITUDE_FACTOR)**2))

    results['timing_rmse'] = {
        't1': float(timing_rmse_t1),
        't2': float(timing_rmse_t2),
        'combined': float(timing_rmse_combined)
    }
    results['amplitude_rmse'] = {
        'a1': float(amplitude_rmse_a1),
        'a2': float(amplitude_rmse_a2),
        'combined': float(amplitude_rmse_combined)
    }

    print(f"Timing RMSE - t1: {timing_rmse_t1:.4f}, t2: {timing_rmse_t2:.4f}, combined: {timing_rmse_combined:.4f}")
    print(f"Amplitude RMSE - a1: {amplitude_rmse_a1:.4f}, a2: {amplitude_rmse_a2:.4f}, combined: {amplitude_rmse_combined:.4f}")

    # 3. Joint Error (timing + amplitude的联合误差)
    joint_error = np.sqrt(
        np.mean(((all_timing_preds - all_gt_timing) / TIMING_FACTOR)**2) +
        np.mean(((all_amplitude_preds - all_gt_amplitude) / AMPLITUDE_FACTOR)**2)
    )
    results['joint_error'] = float(joint_error)
    print(f"Joint Error: {joint_error:.4f}")

    # 4. ASNR (Average Signal-to-Noise Ratio)
    # ASNR = mean(signal_power / noise_power)
    signal_power_timing = np.mean(all_gt_timing**2)
    noise_power_timing = np.mean((all_timing_preds - all_gt_timing)**2)
    asnr_timing = 10 * np.log10(signal_power_timing / noise_power_timing) if noise_power_timing > 0 else np.inf

    signal_power_amplitude = np.mean(all_gt_amplitude**2)
    noise_power_amplitude = np.mean((all_amplitude_preds - all_gt_amplitude)**2)
    asnr_amplitude = 10 * np.log10(signal_power_amplitude / noise_power_amplitude) if noise_power_amplitude > 0 else np.inf

    asnr_combined = (asnr_timing + asnr_amplitude) / 2

    results['asnr'] = {
        'timing': float(asnr_timing),
        'amplitude': float(asnr_amplitude),
        'combined': float(asnr_combined)
    }
    print(f"ASNR - timing: {asnr_timing:.2f} dB, amplitude: {asnr_amplitude:.2f} dB, combined: {asnr_combined:.2f} dB")

    # 5. 校准指标 (ECE, NLL)
    # ECE: Expected Calibration Error
    def compute_ece(probs, labels, n_bins=15):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(accuracies[mask])
                bin_conf = np.mean(confidences[mask])
                bin_weight = np.sum(mask) / len(confidences)
                ece += bin_weight * np.abs(bin_acc - bin_conf)

        return ece

    # NLL: Negative Log-Likelihood
    def compute_nll(probs, labels):
        epsilon = 1e-12
        probs = np.clip(probs, epsilon, 1 - epsilon)
        nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        return nll

    # 假设类别1是正类（双脉冲）
    ece = compute_ece(all_preds, all_labels)
    nll = compute_nll(all_preds, all_labels)

    results['calibration'] = {
        'ece': float(ece),
        'nll': float(nll)
    }
    print(f"ECE: {ece:.6f}, NLL: {nll:.6f}")

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate complete metrics for Table 3')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')

    args = parser.parse_args()

    results = load_checkpoint_and_evaluate(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
