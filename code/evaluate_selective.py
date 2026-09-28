#!/usr/bin/env python
"""
Evaluate selective prediction (risk-coverage analysis)

This script computes selective accuracy at different coverage levels
by ranking predictions by confidence and retaining top-k%.

Usage:
    python evaluate_selective.py --calibration_results calibration_results.json --output selective_results.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path


def compute_selective_accuracy(confidences, predictions, labels, coverage_levels):
    """
    Compute selective accuracy at different coverage levels

    Args:
        confidences: Prediction confidences (N,)
        predictions: Predicted labels (N,)
        labels: True labels (N,)
        coverage_levels: List of coverage fractions (e.g., [1.0, 0.9, 0.8, ...])

    Returns:
        List of dicts with coverage, selective_accuracy, selective_risk
    """
    # Sort by confidence (descending)
    sorted_indices = torch.argsort(confidences, descending=True)

    results = []
    for coverage in coverage_levels:
        n_retain = int(len(confidences) * coverage)

        if n_retain == 0:
            continue

        # Retain top n_retain samples
        retain_indices = sorted_indices[:n_retain]

        # Compute selective accuracy
        correct = (predictions[retain_indices] == labels[retain_indices]).float()
        selective_acc = correct.mean().item()
        selective_risk = 1.0 - selective_acc

        results.append({
            'coverage': coverage,
            'n_samples': n_retain,
            'selective_accuracy': selective_acc,
            'selective_risk': selective_risk
        })

    return results


def load_predictions_from_calibration(calibration_file):
    """
    Load prediction data from calibration results

    Note: This is a placeholder. In practice, you'll need to re-run
    evaluation and save predictions, or modify evaluate_calibration.py
    to save predictions.
    """
    with open(calibration_file, 'r') as f:
        cal_results = json.load(f)

    # This is a placeholder - actual implementation needs predictions
    print("WARNING: This script requires prediction data from calibration evaluation")
    print("Please modify evaluate_calibration.py to save predictions")

    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate selective prediction')
    parser.add_argument('--calibration_results', type=str, required=True,
                        help='Path to calibration_results.json')
    parser.add_argument('--predictions', type=str,
                        help='Path to saved predictions (confidences, predictions, labels)')
    parser.add_argument('--output', type=str, default='selective_results.json',
                        help='Output JSON file')
    parser.add_argument('--coverage_levels', type=float, nargs='+',
                        default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                        help='Coverage levels to evaluate')

    args = parser.parse_args()

    print("Selective Prediction Evaluation")
    print("="*60)

    # Load predictions
    if args.predictions:
        print(f"Loading predictions from {args.predictions}")
        pred_data = torch.load(args.predictions)
        confidences = pred_data['confidences']
        predictions = pred_data['predictions']
        labels = pred_data['labels']
    else:
        print("ERROR: --predictions argument is required")
        print("\nTo generate predictions, modify evaluate_calibration.py to save:")
        print("  torch.save({'confidences': confidences, 'predictions': predictions, 'labels': labels}, 'predictions.pt')")
        return

    # Compute selective accuracy
    print(f"\nComputing selective accuracy at {len(args.coverage_levels)} coverage levels...")
    results = compute_selective_accuracy(
        confidences,
        predictions,
        labels,
        args.coverage_levels
    )

    # Display results
    print("\n" + "="*60)
    print("SELECTIVE PREDICTION RESULTS (Table 6 reproduction)")
    print("="*60)
    print(f"{'Coverage':<12} {'N Samples':<12} {'Selective Acc':<16} {'Selective Risk':<16}")
    print("-"*60)

    for r in results:
        print(f"{r['coverage']:<12.1%} {r['n_samples']:<12d} {r['selective_accuracy']:<16.4f} {r['selective_risk']:<16.4f}")

    # Highlight 50% coverage (paper's focus)
    coverage_50 = next((r for r in results if abs(r['coverage'] - 0.5) < 0.01), None)
    if coverage_50:
        print("\n" + "="*60)
        print("KEY RESULT (50% Coverage):")
        print(f"  Selective Accuracy: {coverage_50['selective_accuracy']:.4f} ({coverage_50['selective_accuracy']*100:.2f}%)")
        print(f"  Selective Risk: {coverage_50['selective_risk']:.4f} ({coverage_50['selective_risk']*100:.2f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'coverage_levels': args.coverage_levels,
        'results': results,
        'summary': {
            'coverage_50': coverage_50
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
