"""Timing RMSE metric for pulse parameter estimation"""
import torch
from torchmetrics import Metric


class TimingRMSE(Metric):
    """Root Mean Squared Error for pulse timing estimation

    Computes RMSE between predicted and true timing values,
    only for valid samples (e.g., double pulses where timing is defined).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, t_pred: torch.Tensor, t_true: torch.Tensor, mask: torch.Tensor = None):
        """
        Update metric state with batch predictions

        Args:
            t_pred: Predicted timing values (batch_size,) or (batch_size, 1)
            t_true: Ground truth timing values (batch_size,) or (batch_size, 1)
            mask: Optional binary mask for valid samples (batch_size,)
                  If None, all samples are considered valid
        """
        # Flatten to 1D if needed
        if t_pred.dim() > 1:
            t_pred = t_pred.squeeze()
        if t_true.dim() > 1:
            t_true = t_true.squeeze()

        # Default mask (all valid)
        if mask is None:
            mask = torch.ones_like(t_pred, dtype=torch.bool)
        else:
            mask = mask.bool()

        # Compute squared error for valid samples only
        squared_error = ((t_pred - t_true) ** 2) * mask.float()

        self.sum_squared_error += squared_error.sum()
        self.total += mask.sum()

    def compute(self):
        """Compute final RMSE"""
        if self.total == 0:
            return torch.tensor(0.0, device=self.sum_squared_error.device)
        return torch.sqrt(self.sum_squared_error / self.total)
