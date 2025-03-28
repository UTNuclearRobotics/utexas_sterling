from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    def __init__(
        self,
        inv_coeff: float = 25.0,
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        # z1 = x
        # z2 = y
        # repr_loss = F.mse_loss(z1, z2)

        # # Standard deviation loss
        # std_z1 = torch.sqrt(torch.var(z1, dim=0) + 1e-4)
        # std_z2 = torch.sqrt(torch.var(z2, dim=0) + 1e-4)
        # std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # # Covariance loss
        # z1 = z1 - z1.mean(dim=0)
        # z2 = z2 - z2.mean(dim=0)
        # cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        # cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        # cov_loss = (self.off_diagonal(cov_z1).pow(2).sum() + self.off_diagonal(cov_z2).pow(2).sum()) / z1.shape[1]

        # # Total loss
        # loss = self.inv_coeff * repr_loss + self.var_coeff * std_loss + self.cov_coeff * cov_loss
        # return loss

        """Computes the VICReg loss.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].
            y: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The VICReg loss.
                Dictionary where values are of shape of [1,].
        """
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        
        metrics = dict()
        if torch.isnan(x).any() or torch.isnan(y).any() or torch.isinf(x).any() or torch.isinf(y).any():
            print(f"NaN or inf detected in inputs: x={x}, y={y}")

        inv_loss = self.inv_coeff * self.representation_loss(x, y)
        metrics["inv-loss"] = inv_loss if not torch.isnan(inv_loss) else torch.tensor(0.0, device=x.device)

        var_loss_x = self.variance_loss(x, self.gamma)
        var_loss_y = self.variance_loss(y, self.gamma)
        var_loss = self.var_coeff * (var_loss_x + var_loss_y) / 2
        metrics["var-loss"] = var_loss if not torch.isnan(var_loss) else torch.tensor(0.0, device=x.device)

        cov_loss_x = self.covariance_loss(x)
        cov_loss_y = self.covariance_loss(y)
        cov_loss = self.cov_coeff * (cov_loss_x + cov_loss_y) / 2
        metrics["cov-loss"] = cov_loss if not torch.isnan(cov_loss) else torch.tensor(0.0, device=x.device)

        metrics["loss"] = sum(metrics.values())
        if torch.isnan(metrics["loss"]):
            print(f"NaN in total loss. Components: inv={metrics['inv-loss'].item()}, var={metrics['var-loss'].item()}, cov={metrics['cov-loss'].item()}")
            metrics["loss"] = torch.tensor(0.0, device=x.device)

        return metrics["loss"]

    @staticmethod
    def representation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the representation loss.
        Force the representations of the same object to be similar.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].
            y: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The representation loss.
                Shape of [1,].
        """
        return F.mse_loss(x, y)

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes the variance loss.
        Push the representations across the batch
        to be different between each other.
        Avoid the model to collapse to a single point.

        The gamma parameter is used as a threshold so that
        the model is no longer penalized if its std is above
        that threshold.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The variance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        std = torch.sqrt(var.clamp(min=1e-4))  # Larger epsilon
        var_loss = F.relu(gamma - std).mean()
        if torch.isnan(var_loss) or torch.isinf(var_loss):
            print(f"Variance loss unstable: var={var}, std={std}")
        return var_loss

    @staticmethod
    def off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Returns the off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def covariance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the covariance loss.
        Decorrelates the embeddings' dimensions, which pushes
        the model to capture more information per dimension.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The covariance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1 + 1e-6)  # Small epsilon for stability
        cov = cov.clamp(min=-1e6, max=1e6)  # Prevent extreme values
        cov_loss = self.off_diagonal(cov).pow(2).sum() / x.shape[1]
        if torch.isnan(cov_loss) or torch.isinf(cov_loss):
            print(f"Covariance loss unstable: cov={cov}")
        return cov_loss
