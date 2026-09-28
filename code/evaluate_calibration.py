#!/usr/bin/env python
"""
Evaluate model calibration under distribution shift

This script evaluates a trained SAEFit checkpoint on perturbed evaluation data
and computes calibration metrics (ECE, NLL) before and after temperature scaling.

Usage:
    python evaluate_calibration.py --checkpoint path/to/checkpoint.ckpt --output results.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

# Import SAEFit components
from lab.utils.perturbations import PERTURBATION_CONFIGS, apply_perturbation
from lab.utils.calibration import compute_ece, compute_nll, compute_overconfident_error_rate, TemperatureScaling


def load_model_and_data(checkpoint_path, batch_size=10000):
    """Load model checkpoint and prepare dataloaders"""
    import pytorch_lightning as pl
    from control.saefit import SAEFit as SAEFitModule
    import lab.datasets as datasets

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract hyperparameters to recreate datamodule
    hparams = checkpoint.get('hyper_parameters', {})
    opts = hparams.get('opts', None)

    # Load model
    model = SAEFitModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()

    # Recreate datamodule using the same pattern as main.py
    if opts is not None:
        datamodule = getattr(datasets, opts.dataset)(opts)
    else:
        # Fallback: load Pulse_SAEFit with default args
        from argparse import Namespace
        default_opts = Namespace(
            dataset='Pulse_SAEFit',
            batch_size=batch_size,
            num_workers=4,
            data_root='../data',
            manual_seed=42
        )
        datamodule = datasets.Pulse_SAEFit(default_opts)

    return model, datamodule


def evaluate_on_perturbed_data(model, dataloader, perturbation_type='combined', device='cuda'):
    """
    Evaluate model on perturbed data and collect predictions

    Returns:
        dict with logits, predictions, confidences, labels
    """
    model = model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Get waveforms and labels
            # PulsePrepare returns: x1, x2, x, m1, m2, m, dt
            # x is the combined pulse, dt is just t2_shift (1D per sample)
            if isinstance(batch, dict):
                x = batch['x'].to(device)
                labels = batch.get('label', torch.zeros(x.shape[0], dtype=torch.long))
            elif isinstance(batch, (list, tuple)) and len(batch) >= 7:
                # Unpack: x1, x2, x, m1, m2, m, dt
                x = batch[2].to(device)  # Combined pulse
                dt = batch[6]  # t2_shift values (1D, batch_size)
                # Label is 1 if t2_shift > 0 (double pulse), 0 if t2_shift <= 0 (single pulse)
                labels = (dt > 0).long()
            else:
                x = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else batch[0]
                labels = batch[1].long() if len(batch) > 1 else torch.zeros(x.shape[0], dtype=torch.long)

            # Apply perturbation
            x_perturbed = apply_perturbation(x, perturbation_type)

            # Forward pass - use model.model since SAEFit is a LightningModule wrapper
            outputs = model.model(x_perturbed)

            # Extract classifier logits (adjust based on actual model output structure)
            if isinstance(outputs, dict):
                logits = outputs.get('classifier', outputs.get('clas', None))
            elif isinstance(outputs, (list, tuple)):
                # SAEFit returns: (out, clas, t1, a1s, a1, k1, r1, t2, a2, k2, r2)
                logits = outputs[1]  # clas is second element
            else:
                logits = outputs

            if logits is None:
                raise ValueError("Could not extract classifier logits from model outputs")

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute predictions and confidences
    probs = torch.softmax(all_logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)

    return {
        'logits': all_logits,
        'predictions': predictions,
        'confidences': confidences,
        'labels': all_labels
    }


def compute_calibration_metrics(logits, predictions, confidences, labels):
    """Compute all calibration metrics"""
    accuracy = (predictions == labels).float().mean().item()
    ece = compute_ece(confidences, predictions, labels)
    nll = compute_nll(logits, labels)
    oer = compute_overconfident_error_rate(confidences, predictions, labels, threshold=0.85)

    return {
        'accuracy': accuracy,
        'ece': ece,
        'nll': nll,
        'oer_0.85': oer
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate calibration under distribution shift')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='calibration_results.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for evaluation')

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model, datamodule = load_model_and_data(args.checkpoint, args.batch_size)

    # Get test dataloader
    test_loader = datamodule.test_dataloader()

    print(f"\nEvaluating on {len(PERTURBATION_CONFIGS)} perturbation conditions...")

    all_results = {}
    aggregate_results = {'uncalibrated': [], 'calibrated': []}

    # First, collect validation data for temperature scaling
    print("Collecting validation data for temperature scaling...")
    val_loader = datamodule.val_dataloader()
    val_data = evaluate_on_perturbed_data(model, val_loader, 'combined', args.device)

    # Fit temperature on validation set
    temp_scaler = TemperatureScaling()
    temperature = temp_scaler.fit(val_data['logits'], val_data['labels'])
    print(f"Fitted temperature: {temperature:.4f}")

    # Evaluate on each perturbation condition
    for pert_name, pert_config in PERTURBATION_CONFIGS.items():
        print(f"\nEvaluating: {pert_name}")

        # Collect predictions
        results = evaluate_on_perturbed_data(model, test_loader, pert_name, args.device)

        # Uncalibrated metrics
        uncal_metrics = compute_calibration_metrics(
            results['logits'],
            results['predictions'],
            results['confidences'],
            results['labels']
        )

        # Apply temperature scaling
        logits_calibrated = temp_scaler.transform(results['logits'])
        probs_calibrated = torch.softmax(logits_calibrated, dim=-1)
        confidences_calibrated, _ = probs_calibrated.max(dim=-1)

        # Calibrated metrics
        cal_metrics = compute_calibration_metrics(
            logits_calibrated,
            results['predictions'],  # predictions unchanged
            confidences_calibrated,
            results['labels']
        )

        # Store results
        all_results[pert_name] = {
            'uncalibrated': uncal_metrics,
            'calibrated': cal_metrics,
            'temperature': temperature
        }

        # Aggregate
        aggregate_results['uncalibrated'].append(uncal_metrics)
        aggregate_results['calibrated'].append(cal_metrics)

        print(f"  Uncalibrated - Acc: {uncal_metrics['accuracy']:.4f}, ECE: {uncal_metrics['ece']:.4f}, NLL: {uncal_metrics['nll']:.4f}")
        print(f"  Calibrated   - Acc: {cal_metrics['accuracy']:.4f}, ECE: {cal_metrics['ece']:.4f}, NLL: {cal_metrics['nll']:.4f}")

    # Compute aggregate metrics (average across perturbations)
    aggregate_uncal = {
        'accuracy': np.mean([m['accuracy'] for m in aggregate_results['uncalibrated']]),
        'ece': np.mean([m['ece'] for m in aggregate_results['uncalibrated']]),
        'nll': np.mean([m['nll'] for m in aggregate_results['uncalibrated']]),
        'oer_0.85': np.mean([m['oer_0.85'] for m in aggregate_results['uncalibrated']])
    }

    aggregate_cal = {
        'accuracy': np.mean([m['accuracy'] for m in aggregate_results['calibrated']]),
        'ece': np.mean([m['ece'] for m in aggregate_results['calibrated']]),
        'nll': np.mean([m['nll'] for m in aggregate_results['calibrated']]),
        'oer_0.85': np.mean([m['oer_0.85'] for m in aggregate_results['calibrated']])
    }

    all_results['aggregate'] = {
        'uncalibrated': aggregate_uncal,
        'calibrated': aggregate_cal,
        'temperature': temperature
    }

    print("\n" + "="*60)
    print("AGGREGATE RESULTS (Table 4 reproduction)")
    print("="*60)
    print(f"Uncalibrated - Acc: {aggregate_uncal['accuracy']:.4f}, ECE: {aggregate_uncal['ece']:.4f}, NLL: {aggregate_uncal['nll']:.4f}")
    print(f"Calibrated   - Acc: {aggregate_cal['accuracy']:.4f}, ECE: {aggregate_cal['ece']:.4f}, NLL: {aggregate_cal['nll']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Save predictions for selective prediction evaluation
    predictions_path = output_path.parent / (output_path.stem + '_predictions.pt')

    # Use the 'combined' perturbation as representative
    combined_data = evaluate_on_perturbed_data(model, test_loader, 'combined', args.device)

    torch.save({
        'confidences': combined_data['confidences'].cpu(),
        'predictions': combined_data['predictions'].cpu(),
        'labels': combined_data['labels'].cpu()
    }, predictions_path)

    print(f"Predictions saved to {predictions_path}")


if __name__ == '__main__':
    main()
