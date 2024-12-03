import os
import torch
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from train_representation import SterlingRepresentation
from PIL import Image

from utils import load_dataset, load_model

script_dir = os.path.dirname(os.path.abspath(__file__))


def image_grid(images, save_path):
    """
    Create and save a 5x5 image grid.
    Args:
        images (list): A list of images to be arranged in a grid.
        save_path (str): The path where the grid image will be saved.
    """
    # Validation
    if not save_path.lower().endswith('.png'):
        raise ValueError("The save_path must end with '.png'")
    if len(images) < 25:
        raise ValueError("The images list must contain at least 25 images.")
    
    # Initialize grid
    grid_size = 5
    image_size = (64, 64)
    new_im = Image.new("RGB", (image_size[0] * grid_size, image_size[1] * grid_size))

    for idx in range(25):
        vp = images[idx]

        # Calculate grid position
        row = idx // grid_size
        col = idx % grid_size

        # Format and paste individual patches to grid
        im = Image.fromarray(vp)
        im = im.convert("RGB")
        im.thumbnail(image_size)
        new_im.paste(im, (col * image_size[0], row * image_size[1]))

    # Save grid image
    new_im.save(save_path)


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

    print("I made (K) clusters: ", k)
    print("Number of items in each cluster.")
    for i in range(0, k):
        print(" [", i, "]: ", representation_vectors[clusters == i].shape)

    # Find the K farthest apart vectors for each cluster
    for i in range(0, k):
        cluster = representation_vectors[clusters == i]
        row_norms = torch.norm(representation_vectors, dim=1, keepdim=True)
        normalized_tensor = representation_vectors / row_norms
        # normalized_tensor = cluster / row_norms

        # images = []
        # image_grid(images, os.path.join(script_dir, "clusters", f"cluster{i}.png"))
    
    """
    TO DO:
    Find the 5 farthest apart vectors for each cluster
    Index them from the bigger representation vectors
    Get the indices where they occur in patch1
    Render the patches from patch1 corresponding to each cluster to get a representative sample
        of each cluster    
    """