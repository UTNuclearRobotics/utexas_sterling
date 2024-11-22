from terrain_dataset import TerrainDataset

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
from util import load_dataset

matplotlib.use("TkAgg")


def train_model():
    # Training loop
    num_epochs = 50
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
    torch.save(model.state_dict(), model_file)

if __name__ == "__main__":
    """
    N_SAMPLES = Number of total patch samples taken
    N_PATCHES = 10 consecutive patch frames
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../models/")
    model_filename = "vis_rep.pt"
    model_file = model_dir + model_filename

    data_pkl = load_dataset()

    # Contains N_SAMPLES of N_PATCHES each
    patches = data_pkl["patches"]
    
    # Create dataset and dataloader
    dataset = TerrainDataset(patches)
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)

    # Initialize model
    model = SterlingRepresentation(device).to(device)    

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    train_model()

    # dataloader = DataLoader(dataset, batch_size=8192, shuffle=False)
    # batch = next(iter(dataloader))
    # patch1, _ = batch

    # # patch1 = torch.cat([data[0] for data in dataloader])
    # # patch2 = torch.cat([data[1] for data in dataloader])
    # print("long_tensor.shape:  ", patch1.shape)
    # # patch1 = patch1[0]
    # patch1 = patch1.to(device)
    # representation_vectors = model.visual_encoder(patch1)

    

"""
# for index, patch in enumerate(patches):
#     plt.imshow(patch[0])
#     plt.axis('off')  # Turn off the axes for a cleaner image
#     print(index)
#     plt.pause(0.001)   # Pause for 0.2 seconds

# print(data_pkl.keys())
"""
