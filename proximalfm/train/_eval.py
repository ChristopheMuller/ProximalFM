import torch
import torch.nn.functional as F


class CausalEvaluator:
    """
    Handles the computation and optional caching of causal evaluation metrics.
    Separates Potential Outcomes (PO) and Conditional Average Treatment Effects (CATE).
    """

    def __init__(self):
        # If you decide to cache static validation stats later,
        # you can store them here (e.g., self.val_stats_cache = {}).
        pass

    def compute_po_metrics(
        self,
        pred_y0: torch.Tensor,
        pred_y1: torch.Tensor,
        true_y0: torch.Tensor,
        true_y1: torch.Tensor,
        y_obs_std: torch.Tensor,
    ) -> dict:
        """Computes NRMSE for Y(0) and Y(1)."""
        mse_y0 = F.mse_loss(pred_y0, true_y0, reduction="none").mean(dim=1)
        mse_y1 = F.mse_loss(pred_y1, true_y1, reduction="none").mean(dim=1)

        nrmse_y0 = torch.sqrt(mse_y0) / y_obs_std.squeeze(-1)
        nrmse_y1 = torch.sqrt(mse_y1) / y_obs_std.squeeze(-1)

        return {
            "po/nrmse_y0": nrmse_y0.mean().item(),
            "po/nrmse_y1": nrmse_y1.mean().item(),
        }

    def compute_cate_metrics(
        self, pred_cate: torch.Tensor, true_cate: torch.Tensor, y_obs_std: torch.Tensor
    ) -> dict:
        """Computes RMSE and NRMSE for the CATE."""
        mse_cate = F.mse_loss(pred_cate, true_cate, reduction="none").mean(dim=1)
        rmse_cate = torch.sqrt(mse_cate)

        nrmse_cate = rmse_cate / y_obs_std.squeeze(-1)

        return {
            "cate/nrmse": nrmse_cate.mean().item(),
        }

    def evaluate_all(
        self,
        pred_y0: torch.Tensor,
        pred_y1: torch.Tensor,
        true_y0: torch.Tensor,
        true_y1: torch.Tensor,
        pred_cate: torch.Tensor,
        true_cate: torch.Tensor,
        y0_scale: torch.Tensor,
        y1_scale: torch.Tensor,
        y_obs_std: torch.Tensor,
    ) -> dict:
        """Aggregates all evaluation metrics into a single dictionary."""
        results = {}
        results.update(
            self.compute_po_metrics(pred_y0, pred_y1, true_y0, true_y1, y_obs_std)
        )
        results.update(self.compute_cate_metrics(pred_cate, true_cate, y_obs_std))
        return results
