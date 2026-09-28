"""Calibration metrics and temperature scaling"""
import torch
import torch.nn.functional as F


def compute_ece(confidences, predictions, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE)

    Args:
        confidences: Predicted confidences (batch_size,)
        predictions: Predicted class labels (batch_size,)
        labels: True class labels (batch_size,)
        n_bins: Number of bins for binning predictions

    Returns:
        ECE value (scalar)
    """
    device = confidences.device
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Identify samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            # Compute accuracy in this bin
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
            # Compute average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            # Add weighted absolute difference
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def compute_nll(logits, labels):
    """
    Compute Negative Log-Likelihood (NLL)

    Args:
        logits: Model logits (batch_size, num_classes)
        labels: True class labels (batch_size,)

    Returns:
        NLL value (scalar)
    """
    nll = F.cross_entropy(logits, labels, reduction='mean')
    return nll.item()


def compute_overconfident_error_rate(confidences, predictions, labels, threshold=0.85):
    """
    Compute fraction of predictions with confidence > threshold that are incorrect

    Args:
        confidences: Predicted confidences (batch_size,)
        predictions: Predicted class labels (batch_size,)
        labels: True class labels (batch_size,)
        threshold: Confidence threshold

    Returns:
        OER value (scalar)
    """
    high_conf_mask = confidences > threshold

    if high_conf_mask.sum() == 0:
        return 0.0

    errors = (predictions[high_conf_mask] != labels[high_conf_mask]).float()
    oer = errors.mean().item()

    return oer


class TemperatureScaling:
    """
    Post-hoc temperature scaling for probability calibration

    Rescales logits by dividing by a learned temperature parameter:
        p_calibrated = softmax(logits / T)

    Temperature T is fitted on a validation set to minimize NLL.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Fit temperature parameter on validation set using LBFGS

        Args:
            logits: Validation logits (batch_size, num_classes)
            labels: Validation labels (batch_size,)
            lr: Learning rate for LBFGS
            max_iter: Maximum LBFGS iterations

        Returns:
            Fitted temperature value
        """
        # Initialize temperature parameter
        temperature = torch.nn.Parameter(torch.ones(1, device=logits.device) * 1.5)

        # Detach inputs (no gradients needed for data)
        logits = logits.detach()
        labels = labels.detach()

        # Optimizer
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss

        # Optimize
        optimizer.step(eval_loss)

        # Store fitted temperature
        self.temperature = temperature.item()

        return self.temperature

    def transform(self, logits):
        """
        Apply temperature scaling to logits

        Args:
            logits: Model logits (batch_size, num_classes)

        Returns:
            Scaled logits (batch_size, num_classes)
        """
        return logits / self.temperature

    def __repr__(self):
        return f"TemperatureScaling(temperature={self.temperature:.4f})"
