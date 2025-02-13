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
        metrics = dict()
        metrics["inv-loss"] = self.inv_coeff * self.representation_loss(x, y)
        metrics["var-loss"] = (
            self.var_coeff
            * (self.variance_loss(x, self.gamma) + self.variance_loss(y, self.gamma))
            / 2
        )
        metrics["cov-loss"] = (
            self.cov_coeff * (self.covariance_loss(x) + self.covariance_loss(y)) / 2
        )
        metrics["loss"] = sum(metrics.values())
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
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(gamma - std).mean()
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
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = self.off_diagonal(cov).pow(2).sum() / x.shape[1]
        return cov_loss
