import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from vicreg import VICRegLoss

matplotlib.use("TkAgg")


class TerrainDataset(Dataset):
    def __init__(self, patches, dtype=torch.float32):
        # Convert patches to tensor
        patches_array = np.array(patches)
        self.patches = torch.tensor(patches_array, dtype=dtype)  # torch.Size([N_SAMPLES, N_PATCHES, 64, 64, 3])

        # Convert to RGB, 64x64
        self.patches = self.patches.permute(0, 1, 4, 2, 3)  # torch.Size([N_SAMPLES, N_PATCHES, 3, 64, 64])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sample = self.patches[idx]
        num_patches = sample.shape[0]

        # Pick a patch from first half and second half
        patch1_idx = torch.randint(0, num_patches // 2, (1,)).item()
        patch2_idx = torch.randint(num_patches // 2, num_patches, (1,)).item()
        patch1 = sample[patch1_idx]  # torch.Size([3, 64, 64])
        patch2 = sample[patch2_idx]  # torch.Size([3, 64, 64])

        # Combine the two patches
        # combined_patches = torch.stack((patch1, patch2))  # torch.Size([2, 3, 64, 64])
        # return combined_patches

        return patch1, patch2


class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderModel, self).__init__()
        self.rep_size = 64
        self.latent_size = latent_size

        self.model = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(3, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.PReLU(),  # torch.Size([batch_size, 8, 31, 31])
            nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),  # torch.Size([batch_size, 16, 15, 15])
            nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # torch.Size([batch_size, 32, 7, 7])
            nn.Conv2d(32, self.rep_size, kernel_size=3, stride=2),
            nn.PReLU(),  # torch.Size([batch_size, rep_size, 3, 3])
            nn.AvgPool2d(kernel_size=3),  # torch.Size([batch_size, rep_size, 1, 1])
            nn.Flatten(),
            nn.Linear(self.rep_size, latent_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class SterlingRepresentation(nn.Module):
    def __init__(self):
        super(SterlingRepresentation, self).__init__()  # Call the parent class's __init__ method
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

        # Encode visual patches
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
        metrics = self.vicreg_loss(zv1, zv2)
        return metrics["loss"]


if __name__ == "__main__":
    """
    N_SAMPLES = Number of total patch samples taken
    N_PATCHES = 10 consecutive patch frames
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "../datasets/")
    dataset_pkl = "nrg_ahg_courtyard.pkl"
    dataset_file = dataset_dir + dataset_pkl

    with open(dataset_file, "rb") as file:
        data_pkl = pickle.load(file)

    # Contains N_SAMPLES of N_PATCHES each
    patches = data_pkl["patches"]
    
    # Create dataset and dataloader
    dataset = TerrainDataset(patches)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = SterlingRepresentation()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

"""
# for index, patch in enumerate(patches):
#     plt.imshow(patch[0])
#     plt.axis('off')  # Turn off the axes for a cleaner image
#     print(index)
#     plt.pause(0.001)   # Pause for 0.2 seconds

# print(data_pkl.keys())
"""
