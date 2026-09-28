"""Perturbation functions for distribution shift experiments"""
import torch


def add_noise_perturbation(waveforms, noise_std=0.05):
    """
    Add Gaussian noise perturbation to waveforms

    Args:
        waveforms: Input waveforms (batch_size, length)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Perturbed waveforms
    """
    noise = torch.randn_like(waveforms) * noise_std
    return waveforms + noise


def add_baseline_perturbation(waveforms, baseline_range=(-0.1, 0.1)):
    """
    Add baseline shift perturbation to waveforms

    Args:
        waveforms: Input waveforms (batch_size, length)
        baseline_range: Tuple of (min, max) baseline shift values

    Returns:
        Perturbed waveforms
    """
    batch_size = waveforms.shape[0]
    device = waveforms.device

    # Random baseline shift per sample
    baseline_shifts = torch.rand(batch_size, 1, device=device) * (baseline_range[1] - baseline_range[0]) + baseline_range[0]

    return waveforms + baseline_shifts


def add_combined_perturbation(waveforms, noise_std=0.05, baseline_range=(-0.1, 0.1)):
    """
    Add combined noise + baseline perturbation

    Args:
        waveforms: Input waveforms (batch_size, length)
        noise_std: Standard deviation of Gaussian noise
        baseline_range: Tuple of (min, max) baseline shift values

    Returns:
        Perturbed waveforms
    """
    perturbed = add_noise_perturbation(waveforms, noise_std)
    perturbed = add_baseline_perturbation(perturbed, baseline_range)
    return perturbed


# Perturbation configurations from paper
PERTURBATION_CONFIGS = {
    'noise_only': {
        'noise_std': 0.05,
        'baseline_range': (0.0, 0.0)
    },
    'baseline_only': {
        'noise_std': 0.0,
        'baseline_range': (-0.1, 0.1)
    },
    'combined': {
        'noise_std': 0.05,
        'baseline_range': (-0.1, 0.1)
    },
    'mild_combined': {
        'noise_std': 0.03,
        'baseline_range': (-0.05, 0.05)
    },
    'strong_combined': {
        'noise_std': 0.08,
        'baseline_range': (-0.15, 0.15)
    }
}


def apply_perturbation(waveforms, perturbation_type='combined'):
    """
    Apply named perturbation configuration

    Args:
        waveforms: Input waveforms (batch_size, length)
        perturbation_type: One of PERTURBATION_CONFIGS keys

    Returns:
        Perturbed waveforms
    """
    if perturbation_type not in PERTURBATION_CONFIGS:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}. "
                         f"Must be one of {list(PERTURBATION_CONFIGS.keys())}")

    config = PERTURBATION_CONFIGS[perturbation_type]

    return add_combined_perturbation(
        waveforms,
        noise_std=config['noise_std'],
        baseline_range=config['baseline_range']
    )
