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

    patch1 = patch1.to(device)
    representation_vectors = model.visual_encoder(patch1)

    print("long_tensor.shape:  ", patch1.shape)
    print("representation_vectors.shape:  ", representation_vectors.shape)

    # K Means
    k = 5
    iterations = 2
    centroids = representation_vectors[torch.randperm(representation_vectors.size(0))[:k]]

    for _ in range(iterations):
        distances = torch.cdist(representation_vectors, centroids)
        clusters = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([representation_vectors[clusters == i].mean(dim=0) for i in range(k)])

        if torch.allclose(centroids, new_centroids):
            break

    print("I made (k) clusters: ", k)
    print("Number of items in each cluster.")
    for i in range(0,k):
        print(" [", i, "]: ", representation_vectors[clusters == i].shape)
    
    #Find the 5 farthest apart vectors for each cluster
    for i in range(0,k):
        cluster = representation_vectors[clusters == i]
        row_norms = torch.norm(representation_vectors, dim=1, keepdim=True)
        normalized_tensor = representation_vectors / row_norms


    '''
    TO DO:
    Find the 5 farthest apart vectors for each cluster
    Index them from the bigger representation vectors
    Get the indices where they occur in patch1
    Render the patches from patch1 corresponding to each cluster to get a representative sample
        of each cluster    
    '''