import torch.nn as nn
from visual_encoder_model import VisualEncoderModel
from vicreg import VICRegLoss
import torch.nn.functional as F


class SterlingRepresentation(nn.Module):
    def __init__(self, device):
        super(SterlingRepresentation, self).__init__()  # Call the parent class's __init__ method
        self.device = device
        self.latent_size = 64
        self.visual_encoder = VisualEncoderModel(self.latent_size)
        self.projector = nn.Sequential(
            nn.Linear(self.visual_encoder.rep_size, self.latent_size),
            nn.PReLU(),
            nn.Linear(self.latent_size, self.latent_size),
        )

        self.vicreg_loss = VICRegLoss()

        self.l1_coeff = 0.5

    def forward(self, patch1, patch2):
        """
        Args:
            patch1 (torch.Tensor): First patch image of shape (3, 64, 64)
            patch2 (torch.Tensor): Second patch image of shape (3, 64, 64)
        """
        # Shape should be [batch size, 2, 3, 64, 64]
        # patch1 = x[:, 0:1, :, :, :]
        # patch2 = x[:, 1:2, :, :, :]
        # print("In Shape:    ", patch1.shape)

        # Encode visual patches
        patch1 = patch1.to(self.device)
        patch2 = patch2.to(self.device)
        v_encoded_1 = self.visual_encoder(patch1)
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2)
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        # Project encoded representations to latent space
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)

        return zv1, zv2, v_encoded_1, v_encoded_2

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on the batch of data.
        Args:
            batch (tuple): A tuple containing the input data.
            batch_idx (int): The index of the current batch.
        Returns:
            torch.Tensor: The computed loss value.
        """
        patch1, patch2 = batch
        zv1, zv2, _, _ = self.forward(patch1, patch2)
        return self.vicreg_loss(zv1, zv2)