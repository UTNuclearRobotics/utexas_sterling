import os
import torch
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from train_representation import SterlingRepresentation
from PIL import Image

from utils import load_dataset, load_model
import random

script_dir = os.path.dirname(os.path.abspath(__file__))


def render_patch(patch):
    """
    Render a single patch image.
    Args:
        patch (torch.Tensor): A single patch image tensor.
    Returns:
        A numpy array representing the patch image.
    """
    patch = patch.cpu().numpy()
    patch = patch.transpose(1, 2, 0)
    patch = (patch * 255).astype("uint8")
    return patch

def render_clusters(clusters, patch):
    """
    Render the patches for each cluster.
    Args:
        clusters (list): A 2d list of indicies to vectors in patches.
        patch (torch.Tensor): The tensor containing the patches.
    Returns:
        A 2D list where each row contains the rendered patches for a cluster.
    """
    rendered_clusters = []
    for row in enumerate(clusters):
        rendered_patches = []
        for col in row:
            single_patch = patch[col]
            rendered_patch = render_patch(single_patch)
            rendered_patches.append(rendered_patch)
        rendered_clusters.append(rendered_patches)
    return rendered_clusters

def image_grid(images, save_path):
    """
    Create and save an image grid row by row.
    Args:
        images (list of list): A 2D list of images to be arranged in a grid.
        save_path (str): The path where the grid image will be saved.
    """
    # Validation
    if not all(isinstance(row, list) for row in images):
        raise ValueError("The images must be in a 2D list where each row is a cluster.")

    max_row_len = max(len(row) for row in images)
    grid_height = len(images)
    image_size = (64, 64)
    new_im = Image.new("RGB", (image_size[0] * max_row_len, image_size[1] * grid_height))

    for row_idx, row in enumerate(images):
        for col_idx, vp in enumerate(row):
            # Format and paste individual patches to grid
            im = Image.fromarray(vp)
            im = im.convert("RGB")
            im.thumbnail(image_size)
            new_im.paste(im, (col_idx * image_size[0], row_idx * image_size[1]))
    
    if save_path is None:
        import matplotlib.pyplot as plt
        plt.imshow(new_im)
        plt.axis('off')
        plt.show()
    else :
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
    
    # Pick 25 random indices in patch1
    images = []
    random_indices = random.sample(range(patch1.size(0)), 25)
    for i in range(0, 25):
        patch = patch1[random_indices[i]]
        patch = render_patch(patch)
        images.append(patch)
    clusters = []
    clusters.append(images)
    image_grid(clusters, os.path.join(script_dir, "random_patches.png"))

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