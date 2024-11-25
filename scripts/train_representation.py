import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from utils import load_dataset, load_model
from vicreg import VICRegLoss
from visual_encoder_model import VisualEncoderModel


class SterlingRepresentation(nn.Module):
    def __init__(self, device):
        super(SterlingRepresentation, self).__init__()
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

if __name__ == "__main__":
    """
    Train the model using the given dataset.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Sterling Representation Model")
    parser.add_argument("--batch_size", "-b", type=int, default=8192, help="Batch size for training")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs for training")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    data_pkl = load_dataset()
    dataset = TerrainDataset(patches=data_pkl["patches"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = SterlingRepresentation(device).to(device)
    model_path = load_model(model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
