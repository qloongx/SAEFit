"""Amplitude RMSE metric for pulse parameter estimation"""
import torch
from torchmetrics import Metric


class AmplitudeRMSE(Metric):
    """Root Mean Squared Error for pulse amplitude estimation

    Computes RMSE between predicted and true amplitude values,
    only for valid samples (e.g., double pulses where second component exists).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, a_pred: torch.Tensor, a_true: torch.Tensor, mask: torch.Tensor = None):
        """
        Update metric state with batch predictions

        Args:
            a_pred: Predicted amplitude values (batch_size,) or (batch_size, 1)
            a_true: Ground truth amplitude values (batch_size,) or (batch_size, 1)
            mask: Optional binary mask for valid samples (batch_size,)
                  If None, all samples are considered valid
        """
        # Flatten to 1D if needed
        if a_pred.dim() > 1:
            a_pred = a_pred.squeeze()
        if a_true.dim() > 1:
            a_true = a_true.squeeze()

        # Default mask (all valid)
        if mask is None:
            mask = torch.ones_like(a_pred, dtype=torch.bool)
        else:
            mask = mask.bool()

        # Compute squared error for valid samples only
        squared_error = ((a_pred - a_true) ** 2) * mask.float()

        self.sum_squared_error += squared_error.sum()
        self.total += mask.sum()

    def compute(self):
        """Compute final RMSE"""
        if self.total == 0:
            return torch.tensor(0.0, device=self.sum_squared_error.device)
        return torch.sqrt(self.sum_squared_error / self.total)
