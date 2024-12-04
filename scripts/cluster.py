import os
import torch
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from train_representation import SterlingRepresentation
from PIL import Image

from utils import load_dataset, load_model

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

def render_clusters(indicies, patches):
    """
    Render the patches for each cluster.
    Args:
        indicies (list): A 2d list of indicies to vectors in patches.
        patches (torch.Tensor): The tensor containing the patches.
    Returns:
        A 2D list where each row contains the rendered patches for a cluster.
    """
    rendered_clusters = []
    for cluster in indicies:
        rendered_patches = []
        for index in cluster:
            single_patch = patches[index]
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
    cluster_rep_vectors = []
    for i in range(0, k):
        cluster = representation_vectors[clusters == i]
        row_norms = torch.norm(cluster, dim=1, keepdim=True)
        normalized_tensor = cluster / row_norms
        clusterT = cluster.transpose(0,1)
        clusterSim = torch.matmul(cluster, clusterT)

        cluster_indices = []
        while len(cluster_indices) < 5:
            min_value = clusterSim.min()
            min_idx = (clusterSim == min_value).nonzero(as_tuple=False)
            cluster_indices.append(min_idx[0,0].item())
            cluster_indices.append(min_idx[0,1].item())
            # clusterSim[min_row, min_col] = 1
            # cluster_indices = list(set(cluster_indices))

        print("cluster.shape:   ", cluster.shape)
        print("cluster_indices:   ", cluster_indices)
        cluster_subtensor = cluster[cluster_indices]
        cluster_rep_vectors.append(cluster_subtensor)

    all_cluster_image_indices = []
    for index, cluster in enumerate(cluster_rep_vectors):
        # print("CLUSTER: ", index)
        cluster_image_indices = []
        for row in cluster:
            match = torch.all(representation_vectors == row, dim=1)
            index = torch.nonzero(match, as_tuple=True)[0]
            cluster_image_indices.append(index.item())
        all_cluster_image_indices.append(cluster_image_indices)
    
    for index, images in enumerate(all_cluster_image_indices):
        print("CLUSTER: ", index)
        print(images)
    
    # Ensure save path exists
    save_dir = os.path.join(script_dir, "clusters")
    os.makedirs(save_dir, exist_ok=True)
    
    rendered_clusters = render_clusters(all_cluster_image_indices, patch1)
    image_grid(rendered_clusters, os.path.join(save_dir, "rendered_clusters.png"))
    
    """
    TO DO:
    Find the 5 farthest apart vectors for each cluster
    Index them from the bigger representation vectors
    Get the indices where they occur in patch1
    Render the patches from patch1 corresponding to each cluster to get a representative sample
        of each cluster  

    []

    [ [] [] [] [] [] [] ]

    [
        [ [] [] [] [] ]

        [ [] [] [] [] [] [] ]

        [ [] [] [] [] [] [] ]
    ]
  
    """