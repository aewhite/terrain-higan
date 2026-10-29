import torch
import torch.nn.functional as F


def residual_aggregate_loss(
    generated_normalized_residuals: torch.Tensor,
    real_normalized_residuals: torch.Tensor,
) -> torch.Tensor:
    """Match normalized residual mean and standard deviation across the full batch."""
    return F.mse_loss(
        generated_normalized_residuals.mean(),
        real_normalized_residuals.mean(),
    ) + F.mse_loss(
        generated_normalized_residuals.std(correction=0),
        real_normalized_residuals.std(correction=0),
    )
