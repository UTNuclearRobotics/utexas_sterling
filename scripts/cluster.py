import os
import pickle

import torch
from PIL import Image
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from train_representation import SterlingRepresentation
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))


class PatchRenderer:
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
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
                rendered_patch = PatchRenderer.render_patch(single_patch)
                rendered_patches.append(rendered_patch)
            rendered_clusters.append(rendered_patches)
        return rendered_clusters

    @staticmethod
    def image_grid(images):
        """
        Create and save a 5x5 image grid.
        Args:
            images (list): A list of images to be arranged in a grid.
            save_path (str): The path where the grid image will be saved.
        """
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

        return new_im

    @staticmethod
    def image_streaks(images):
        """
        Create and save an image grid row by row.
        Args:
            images (list of list): A 2D list of images to be arranged in a grid.
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

        return new_im


class Cluster:
    def __init__(self, data_pkl_path, model_path, batch_size=8192):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        if not os.path.exists(data_pkl_path):
            raise FileNotFoundError(f"Data pickle file not found at: {data_pkl_path}")
        with open(data_pkl_path, "rb") as file:
            data_pkl = pickle.load(file)

        # Load model weights
        self.model = SterlingRepresentation(device).to(device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

        # Create dataset and dataloader
        dataset = TerrainDataset(patches=data_pkl["patches"])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        batch = next(iter(dataloader))
        patch1, _ = batch

        self.patches = patch1.to(device)

    def generate_clusters(self, k_values, iterations=100):
        representation_vectors = self.model.visual_encoder(self.patches)
        row_sums = representation_vectors.sum(dim=1, keepdim=True)
        representation_vectors = representation_vectors / row_sums

        silhouette_scores = []
        all_k_cluster_indices = {}

        for k in k_values:
            # Initialize centroids
            centroids = representation_vectors[torch.randperm(representation_vectors.size(0))[:k]]

            for _ in range(iterations):
                distances = torch.cdist(representation_vectors, centroids)
                min_values, min_indices = torch.min(distances, dim=1)
                new_centroids = torch.stack(
                    [representation_vectors[min_indices == i].mean(dim=0) for i in range(k)]
                )

                if torch.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            print(f"Clusters formed for k={k}:")
            for cluster_idx in range(k):
                cluster_size = representation_vectors[min_indices == cluster_idx].shape[0]
                print(f"  Cluster {cluster_idx}: {cluster_size} points")

            # Calculate Silhouette Score
            silhouette_values = []
            for i, vector in enumerate(representation_vectors):
                # Intra-cluster distance (a)
                cluster = representation_vectors[min_indices == min_indices[i]]
                a = torch.mean(torch.norm(cluster - vector, dim=1))

                # Nearest-cluster distance (b)
                nearest_distances = []
                for j in range(k):
                    if j != min_indices[i]:
                        other_cluster = representation_vectors[min_indices == j]
                        if other_cluster.size(0) > 0:
                            nearest_distances.append(torch.mean(torch.norm(other_cluster - vector, dim=1)))
                b = min(nearest_distances) if nearest_distances else 0

                # Silhouette score for the point
                s = (b - a) / max(a, b) if max(a, b) > 0 else 0
                silhouette_values.append(s)

            avg_silhouette_score = torch.mean(torch.tensor(silhouette_values)).item()
            silhouette_scores.append((k, avg_silhouette_score))

            print(f"Silhouette Score for k={k}: {avg_silhouette_score}")

            # Store cluster indices
            all_cluster_image_indices = []
            for cluster_idx in range(k):
                cluster_vectors = representation_vectors[min_indices == cluster_idx]
                cluster_image_indices = []

                # Find indices of representative vectors in the cluster
                for row in cluster_vectors:
                    match = torch.all(representation_vectors == row, dim=1)
                    indices = torch.nonzero(match, as_tuple=True)[0]
                    if indices.numel() == 1:
                        cluster_image_indices.append(indices.item())
                    else:
                        cluster_image_indices.append(indices[0].item())

                all_cluster_image_indices.append(cluster_image_indices)

            all_k_cluster_indices[k] = all_cluster_image_indices

        print("Silhouette Scores for all k-values:")
        for k, score in silhouette_scores:
            print(f"  k={k}: {score}")

        best_k = max(silhouette_scores, key=lambda x: x[1])[0]
        print(f"Best k according to silhouette score: {best_k}")

        return all_k_cluster_indices[best_k], silhouette_scores



if __name__ == "__main__":
    # Save directory
    save_dir = os.path.join(script_dir, "clusters")
    os.makedirs(save_dir, exist_ok=True)

    # Generate clusters
    cluster = Cluster(
        data_pkl_path=os.path.join(script_dir, "../datasets/nrg_ahg_courtyard.pkl"),
        model_path=os.path.join(script_dir, "../models/vis_rep.pt"),
    )

    k_values = range(2, 10)
    k_best_cluster_image_indices, silhouette_scores = cluster.generate_clusters(k_values)
    

    # Render clusters
    rendered_clusters = PatchRenderer.render_clusters(k_best_cluster_image_indices, cluster.patches)
    
    for i, cluster in enumerate(rendered_clusters):
        grid_image = PatchRenderer.image_grid(cluster)
        grid_image.save(os.path.join(save_dir, f"cluster_{i}.png"))
    
    # image = PatchRenderer.image_streaks(rendered_clusters)
    # image.save(os.path.join(save_dir, "cluster_grid.png"))
