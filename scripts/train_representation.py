import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from utils import load_bag_pkl, load_bag_pt_model
from vicreg import VICRegLoss
from visual_encoder_model import VisualEncoderModel
from torchvision import transforms


class SterlingRepresentation(nn.Module):
    def __init__(self, device):
        super(SterlingRepresentation, self).__init__()
        self.device = device
        self.latent_size = 128
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
            patch1 (torch.Tensor): First patch image of shape (3, 128, 128)
            patch2 (torch.Tensor): Second patch image of shape (3, 128, 128)
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
    
    def encode_single_patch(self, patch):
        """
        Encode a single patch and return its representation vector.
        Args:
            patch (torch.Tensor): Single patch image of shape (1, 3, H, W).
        Returns:
            torch.Tensor: Encoded and normalized representation vector.
        """
        # Ensure the input is on the correct device
        patch = patch.to(self.device)

        # Encode the patch
        v_encoded = self.visual_encoder(patch)
        v_encoded = F.normalize(v_encoded, dim=-1)  # Normalize the representation vector
        return v_encoded

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

        # Compute VICReg loss
        vicreg_loss = self.vicreg_loss(zv1, zv2)

        return vicreg_loss

if __name__ == "__main__":
    """
    Train the model using the given dataset.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Sterling Representation Model")
    parser.add_argument("-bag", "-b", type=str, required=True, help="Bag directory with VICReg dataset pickle file inside.")
    parser.add_argument("-batch_size", "-batch", type=int, default=256, help="Batch size for training")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs for training")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    data_pkl = load_bag_pkl(args.bag, "vicreg")
    # Define the augmentation pipeline
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Flips tensor horizontally
        transforms.RandomVerticalFlip(),    # Flips tensor vertically
        transforms.RandomRotation(15),      # Rotates the tensor by ±15 degrees
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes RGB channels
    ])
    dataset = TerrainDataset(patches=data_pkl, transform=augment_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = SterlingRepresentation(device).to(device)
    save_path = load_bag_pt_model(args.bag, "terrain_rep", model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

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

    torch.save(model.state_dict(), save_path)
