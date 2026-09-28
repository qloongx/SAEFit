#!/usr/bin/env python
"""
实验11: OER阈值敏感性分析

计算多个置信度阈值下的OER（过度自信错误率）、拒绝率和错误率，
绘制风险-覆盖率曲线，验证结论不依赖单一阈值。

对应审稿意见：评论员2第12条

Usage:
    python evaluate_oer_threshold.py --predictions calibration_stage1_predictions.pt --output oer_threshold_stage1.json
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_oer_at_threshold(confidences, predictions, labels, threshold):
    """
    计算指定阈值下的OER（过度自信错误率）

    Args:
        confidences: 预测置信度 (N,)
        predictions: 预测标签 (N,)
        labels: 真实标签 (N,)
        threshold: 置信度阈值

    Returns:
        dict with oer, rejection_rate, error_rate, n_high_conf, n_errors
    """
    high_conf_mask = confidences > threshold
    n_high_conf = high_conf_mask.sum().item()
    n_total = len(confidences)

    if n_high_conf == 0:
        return {
            'threshold': threshold,
            'oer': 0.0,
            'rejection_rate': 1.0,
            'error_rate': 0.0,
            'n_high_conf': 0,
            'n_errors': 0
        }

    # OER: 高置信度样本中的错误率
    errors = (predictions[high_conf_mask] != labels[high_conf_mask]).float()
    oer = errors.mean().item()

    # 拒绝率: 低于阈值的样本比例
    rejection_rate = 1.0 - (n_high_conf / n_total)

    # 整体错误率
    total_errors = (predictions != labels).float().sum().item()
    error_rate = total_errors / n_total

    # 高置信度样本中的错误数量
    n_errors = int(errors.sum().item())

    return {
        'threshold': threshold,
        'oer': oer,
        'rejection_rate': rejection_rate,
        'error_rate': error_rate,
        'n_high_conf': n_high_conf,
        'n_errors': n_errors
    }


def compute_risk_coverage_curve(confidences, predictions, labels, n_points=20):
    """
    计算风险-覆盖率曲线

    在不同覆盖率水平下，按置信度排序保留top-k样本，计算选择性风险

    Args:
        confidences: 预测置信度 (N,)
        predictions: 预测标签 (N,)
        labels: 真实标签 (N,)
        n_points: 曲线点数

    Returns:
        List of dicts with coverage, risk, accuracy
    """
    # 按置信度降序排序
    sorted_indices = torch.argsort(confidences, descending=True)

    coverage_levels = np.linspace(1.0, 0.0, n_points + 1)[:-1]  # 不包括0

    results = []
    for coverage in coverage_levels:
        n_retain = max(1, int(len(confidences) * coverage))

        # 保留top-n样本
        retain_indices = sorted_indices[:n_retain]

        # 计算选择性准确率和风险
        correct = (predictions[retain_indices] == labels[retain_indices]).float()
        selective_acc = correct.mean().item()
        selective_risk = 1.0 - selective_acc

        # 最低置信度（阈值）
        min_conf = confidences[retain_indices].min().item()

        results.append({
            'coverage': coverage,
            'risk': selective_risk,
            'accuracy': selective_acc,
            'threshold': min_conf,
            'n_samples': n_retain
        })

    return results


def plot_risk_coverage_curve(results, output_path):
    """绘制风险-覆盖率曲线"""
    coverages = [r['coverage'] for r in results]
    risks = [r['risk'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(coverages, risks, 'b-', linewidth=2, label='Selective Risk')
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Selective Risk (1 - Accuracy)', fontsize=12)
    plt.title('Risk-Coverage Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # 标注关键点
    for coverage in [1.0, 0.9, 0.8, 0.7, 0.5]:
        result = next((r for r in results if abs(r['coverage'] - coverage) < 0.05), None)
        if result:
            plt.plot(result['coverage'], result['risk'], 'ro', markersize=6)
            plt.text(result['coverage'], result['risk'] + 0.0001,
                    f"{result['coverage']:.0%}", fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Risk-coverage curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='OER threshold sensitivity analysis')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to saved predictions (.pt file)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999],
                        help='Confidence thresholds to evaluate')
    parser.add_argument('--plot', action='store_true',
                        help='Generate risk-coverage curve plot')

    args = parser.parse_args()

    print("="*70)
    print("实验11: OER阈值敏感性分析")
    print("="*70)

    # 加载预测数据
    print(f"\nLoading predictions from {args.predictions}")
    pred_data = torch.load(args.predictions, weights_only=True)
    confidences = pred_data['confidences']
    predictions = pred_data['predictions']
    labels = pred_data['labels']

    print(f"Total samples: {len(confidences)}")
    print(f"Overall accuracy: {(predictions == labels).float().mean().item():.6f}")

    # 1. 计算多个阈值下的OER
    print(f"\n{'='*70}")
    print("OER at Different Confidence Thresholds")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'OER':<12} {'Rejection%':<14} {'N_High_Conf':<14} {'N_Errors':<12}")
    print("-"*70)

    oer_results = []
    for threshold in sorted(args.thresholds):
        result = compute_oer_at_threshold(confidences, predictions, labels, threshold)
        oer_results.append(result)

        print(f"{result['threshold']:<12.3f} "
              f"{result['oer']:<12.6f} "
              f"{result['rejection_rate']*100:<14.2f} "
              f"{result['n_high_conf']:<14d} "
              f"{result['n_errors']:<12d}")

    # 2. 计算风险-覆盖率曲线
    print(f"\n{'='*70}")
    print("Risk-Coverage Curve (20 points)")
    print(f"{'='*70}")

    risk_coverage = compute_risk_coverage_curve(confidences, predictions, labels, n_points=20)

    print(f"{'Coverage':<12} {'Risk':<14} {'Accuracy':<14} {'Threshold':<14} {'N_Samples':<12}")
    print("-"*70)
    for r in risk_coverage[::2]:  # 每隔一个显示
        print(f"{r['coverage']:<12.2%} "
              f"{r['risk']:<14.6f} "
              f"{r['accuracy']:<14.6f} "
              f"{r['threshold']:<14.6f} "
              f"{r['n_samples']:<12d}")

    # 3. 关键发现总结
    print(f"\n{'='*70}")
    print("关键发现")
    print(f"{'='*70}")

    # OER@0.85 (论文关注的阈值)
    oer_85 = next((r for r in oer_results if abs(r['threshold'] - 0.85) < 0.001), None)
    if oer_85:
        print(f"\n1. OER@0.85 (论文报告值):")
        print(f"   OER = {oer_85['oer']:.6f}")
        print(f"   拒绝率 = {oer_85['rejection_rate']*100:.2f}%")
        print(f"   高置信度样本数 = {oer_85['n_high_conf']}")

    # OER范围
    oer_values = [r['oer'] for r in oer_results if r['n_high_conf'] > 0]
    print(f"\n2. OER范围 (阈值0.5-0.999):")
    print(f"   最小 OER = {min(oer_values):.6f}")
    print(f"   最大 OER = {max(oer_values):.6f}")
    print(f"   平均 OER = {np.mean(oer_values):.6f}")

    # 风险-覆盖率趋势
    risk_at_50 = next((r for r in risk_coverage if abs(r['coverage'] - 0.5) < 0.03), None)
    risk_at_90 = next((r for r in risk_coverage if abs(r['coverage'] - 0.9) < 0.03), None)
    if risk_at_50 and risk_at_90:
        print(f"\n3. 选择性风险趋势:")
        print(f"   90%覆盖率: Risk = {risk_at_90['risk']:.6f}, Acc = {risk_at_90['accuracy']:.6f}")
        print(f"   50%覆盖率: Risk = {risk_at_50['risk']:.6f}, Acc = {risk_at_50['accuracy']:.6f}")
        print(f"   改善幅度: {(risk_at_90['risk'] - risk_at_50['risk'])/risk_at_90['risk']*100:.1f}%")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'n_samples': len(confidences),
            'overall_accuracy': (predictions == labels).float().mean().item(),
            'thresholds_evaluated': args.thresholds
        },
        'oer_by_threshold': oer_results,
        'risk_coverage_curve': risk_coverage,
        'summary': {
            'oer_at_0.85': oer_85,
            'oer_range': {
                'min': min(oer_values),
                'max': max(oer_values),
                'mean': float(np.mean(oer_values))
            },
            'selective_prediction': {
                'coverage_90': risk_at_90,
                'coverage_50': risk_at_50
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n结果已保存到: {output_path}")

    # 绘制曲线
    if args.plot:
        plot_path = output_path.with_suffix('.png')
        plot_risk_coverage_curve(risk_coverage, plot_path)


if __name__ == '__main__':
    main()
