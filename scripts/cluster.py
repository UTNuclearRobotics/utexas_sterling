import os
import torch
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from train_representation import SterlingRepresentation

from utils import load_dataset, load_model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    data_pkl = load_dataset()
    dataset = TerrainDataset(patches=data_pkl["patches"])
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)

    # Initialize model
    model = SterlingRepresentation(device).to(device)
    model_path = load_model(model)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    batch = next(iter(dataloader))
    patch1, _ = batch

    # patch1 = torch.cat([data[0] for data in dataloader])
    # patch2 = torch.cat([data[1] for data in dataloader])
    print("long_tensor.shape:  ", patch1.shape)
    # patch1 = patch1[0]
    patch1 = patch1.to(device)
    representation_vectors = model.visual_encoder(patch1)
