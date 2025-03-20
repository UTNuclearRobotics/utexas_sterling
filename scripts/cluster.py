import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
from kneed import KneeLocator
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from train_representation import SterlingRepresentation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, normalize, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import joblib
import cv2
import math

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
        patch = patch.permute(1, 2, 0).cpu().numpy().astype("uint8")

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
    def image_grid(cluster_images):
        """
        Creates a dynamically sized image grid containing all images in a cluster.

        Args:
            cluster_images (list): A list of images (NumPy arrays) belonging to the cluster.

        Returns:
            A NumPy array representing the image grid.
        """
        if not cluster_images:
            raise ValueError("No images provided for the cluster grid.")

        # Image parameters
        image_size = (64, 64)  # Resize all images to a uniform size
        resized_images = [cv2.resize(img, image_size) for img in cluster_images]

        # Compute optimal grid dimensions (square-like shape)
        num_images = len(resized_images)
        grid_cols = math.ceil(math.sqrt(num_images))  # Approximate square grid
        grid_rows = math.ceil(num_images / grid_cols)  # Compute needed rows

        # Create the grid row by row
        rows = []
        for i in range(grid_rows):
            row_images = resized_images[i * grid_cols : (i + 1) * grid_cols]
            
            # Fill missing slots with black images if needed
            while len(row_images) < grid_cols:
                row_images.append(np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))
            
            rows.append(np.hstack(row_images))

        # Stack rows vertically to form the final grid
        grid = np.vstack(rows)

        return grid


class Cluster:
    def __init__(self, data_pkl_path, synced_pkl_path, model_path, batch_size=256):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        if not os.path.exists(data_pkl_path):
            raise FileNotFoundError(f"Data pickle file not found at: {data_pkl_path}")
        with open(data_pkl_path, "rb") as file:
            data_pkl = pickle.load(file)

        if not os.path.exists(synced_pkl_path):
            raise FileNotFoundError(f"Synced pickle file not found at: {synced_pkl_path}")
        with open(synced_pkl_path, "rb") as file:
            synced_pkl = pickle.load(file)

        # Load model weights
        self.model = SterlingRepresentation("cpu").to("cpu")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

        # Create dataset and dataloader
        self.dataset = TerrainDataset(patches=data_pkl, synced_data=synced_pkl, incl_orientation=False)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        # Store patches and inertial data for clustering
        all_patches_list = []
        all_inertial_list = []
        for batch in self.dataloader:
            # Each batch is a tuple (patch1, patch2, inertial)
            patch1, _, inertial = batch
            all_patches_list.append(patch1.cpu())  # Shape: [batch_size, channels, height, width]
            all_inertial_list.append(inertial.cpu())  # Shape: [batch_size, 1, num_features]

        # Concatenate all batches into single tensors
        self.patches = torch.cat(all_patches_list, dim=0)  # Shape: [num_samples, channels, height, width]
        self.inertial = torch.cat(all_inertial_list, dim=0)  # Shape: [num_samples, 1, num_features]

    def generate_clusters(
        self,
        k,
        iterations,
        save_model_path="scripts/clusters/kmeans_model.pkl",
        save_scaler_path="scripts/clusters_sim/scaler.pkl",
    ):
        """
        Generate clusters using K-means algorithm on combined visual and inertial embeddings.

        Args:
            k (int): Number of clusters to generate.
            iterations (int): Number of iterations for K-means.
            save_model_path (str): Path to save the K-means model.
            save_scaler_path (str): Path to save the scaler (if used).

        Returns:
            list: List of lists containing indices of samples in each cluster.
        """
        # Move model to CPU and compute combined representations
        self.model.cpu()
        self.model.eval()
        combined_embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.patches), self.dataloader.batch_size):
                # Process in batches to avoid memory issues
                patch_batch = self.patches[i:i + self.dataloader.batch_size]
                inertial_batch = self.inertial[i:i + self.dataloader.batch_size]
                embeddings = self.model.get_terrain_embedding(patch_batch, inertial_batch)
                combined_embeddings.append(embeddings.cpu())

        # Concatenate all embeddings
        representation_vectors = torch.cat(combined_embeddings, dim=0)  # Shape: [num_samples, 2 * latent_size]
        representation_vectors_np = representation_vectors.numpy()
        representation_vectors_np = normalize(representation_vectors_np, axis=1, norm='l2')

        # Apply K-means clustering with sklearn
        kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=iterations, n_init=10, random_state=42)
        kmeans.fit(representation_vectors_np)
        cluster_labels = kmeans.labels_

        # Save the K-means model
        joblib.dump(kmeans, save_model_path)

        print("I made (K) clusters: ", k)
        print("Number of items in each cluster:")
        for i in range(k):
            print(f" [Cluster {i}]: {(cluster_labels == i).sum()} items")

        # Organize image indices into clusters
        all_cluster_image_indices = [[] for _ in range(k)]
        for idx, cluster in enumerate(cluster_labels):
            all_cluster_image_indices[cluster].append(idx)  # Assign image index to corresponding cluster

        # Plot clusters, passing the directory of save_model_path
        save_plot_dir = os.path.dirname(save_model_path)
        self.plot_clusters(representation_vectors, torch.tensor(cluster_labels), k, save_plot_path=save_plot_dir)

        return all_cluster_image_indices

    def plot_clusters(self, representation_vectors, min_indices, k, save_plot_path=None):
        """
        Visualizes the k-means clusters after performing dimensionality reduction
        using PCA.
        """
        # Step 1: Normalize the representation vectors
        representation_vectors_np = representation_vectors.detach().cpu().numpy()
        #scaler = MinMaxScaler()
        #representation_vectors_np = scaler.fit_transform(representation_vectors_np)
        representation_vectors_np = normalize(representation_vectors_np, norm='l2', axis=1)

        # Step 2: Apply PCA for dimensionality reduction (First to 20D, then to 2D)
        pca_high = PCA(n_components=20, random_state=42)
        intermediate_vectors = pca_high.fit_transform(representation_vectors_np)

        pca_final = PCA(n_components=2, whiten=True, random_state=42)
        reduced_vectors = pca_final.fit_transform(intermediate_vectors)

        # Step 3: Compute centroids in PCA-reduced space
        reduced_centroids = np.array([
            reduced_vectors[min_indices == i].mean(axis=0) for i in range(k)
        ])

        # Step 4: Plot clusters
        plt.figure(figsize=(8, 6))
        for cluster_idx in range(k):
            cluster_points = reduced_vectors[min_indices == cluster_idx]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}", alpha=0.6)

        # Step 5: Plot centroids correctly
        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c="black", marker="x", label="Centroids")
        plt.title(f"K-means Clusters with k={k}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)

        # Save the plot to the specified directory
        if save_plot_path:
            os.makedirs(save_plot_path, exist_ok=True)
            plot_file_path = os.path.join(save_plot_path, f"clusters_k{k}.png")
            plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
            print(f"Saved cluster plot to: {plot_file_path}")
        
        plt.close()


if __name__ == "__main__":
    # Save directory
    save_dir = os.path.join(script_dir, "clusters")
    os.makedirs(save_dir, exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Sterling Representation Model")
    parser.add_argument(
        "-bag", "-b", type=str, required=True, help="Bag directory with VICReg dataset pickle file inside."
    )
    args = parser.parse_args()

    bag_path = args.bag
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    # Validate the pickle file exists
    vicreg_pkl = [file for file in os.listdir(bag_path) if file.endswith("vicreg.pkl")]
    if len(vicreg_pkl) != 1:
        raise FileNotFoundError(f"VICReg pickle file not found in: {bag_path}")
    vicreg_pkl_path = os.path.join(bag_path, vicreg_pkl[0])

    # Validate the pickle file exists
    synced_pkl = [file for file in os.listdir(bag_path) if file.endswith("_synced.pkl")]
    if len(synced_pkl) != 1:
        raise FileNotFoundError(f"VICReg pickle file not found in: {bag_path}")
    synced_pkl_path = os.path.join(bag_path, synced_pkl[0])

    # Validate the pickle file exists
    model_path = os.path.join(bag_path, "models")
    pt = [file for file in os.listdir(model_path) if file.endswith(".pt")]
    if len(pt) != 1:
        raise FileNotFoundError(f"Terrain representation PyTorch model file not found in: {model_path}")
    pt_path = os.path.join(model_path, pt[0])

    # Generate clusters
    cluster = Cluster(
        data_pkl_path=vicreg_pkl_path,
        synced_pkl_path=synced_pkl_path,
        model_path=pt_path,
    )

    #k_values = range(2, 12)
    k_values = 10
    iterations = 1000

    all_cluster_image_indices = cluster.generate_clusters(
        k_values, iterations
    )

    # Render clusters
    rendered_clusters = PatchRenderer.render_clusters(all_cluster_image_indices, cluster.patches)

    for i, cluster in enumerate(rendered_clusters):
        grid_image = PatchRenderer.image_grid(cluster)
        save_path = os.path.join(save_dir, f"cluster_{i}.png")
        # Save the grid image to the specified path
        cv2.imwrite(save_path, grid_image)
