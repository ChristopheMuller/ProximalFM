import torch
import torch.nn.functional as F


def compute_po_metrics(
    pred_y0: torch.Tensor,
    pred_y1: torch.Tensor,
    true_y0: torch.Tensor,
    true_y1: torch.Tensor,
    y0_scale: torch.Tensor,
    y1_scale: torch.Tensor,
) -> dict:
    """
    Computes the Normalized Root Mean Square Error (NRMSE) for Y(0) and Y(1).
    Normalizes by the standard deviation of the context outcomes.
    """
    # 1. Compute raw MSE per dataset in the batch (dim=1 is the sequence dimension)
    mse_y0 = F.mse_loss(pred_y0, true_y0, reduction="none").mean(dim=1)
    mse_y1 = F.mse_loss(pred_y1, true_y1, reduction="none").mean(dim=1)

    # 2. Convert to RMSE
    rmse_y0 = torch.sqrt(mse_y0)
    rmse_y1 = torch.sqrt(mse_y1)

    # 3. Normalize using the context standard deviation
    # y_scales have shape (B, 1), so we squeeze to (B,) to match rmse
    nrmse_y0 = rmse_y0 / y0_scale.squeeze(-1)
    nrmse_y1 = rmse_y1 / y1_scale.squeeze(-1)

    # Return the mean NRMSE across the batch, mapped to the specific WandB folder
    return {
        "po/nrmse_y0": nrmse_y0.mean().item(),
        "po/nrmse_y1": nrmse_y1.mean().item(),
    }
