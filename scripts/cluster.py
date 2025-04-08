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
    def render_patch(patch, input_format="RGB", output_format="RGB"):
        """
        Render a single patch image, handling various input formats.

        Args:
            patch (torch.Tensor or np.ndarray): A single patch image (C, H, W) or (H, W, C).
            input_format (str): Input channel order, "RGB" or "BGR" (default "RGB").
            output_format (str): Desired output channel order, "RGB" or "BGR" (default "RGB").

        Returns:
            np.ndarray: Rendered patch as (H, W, 3) uint8 array in specified output_format.
        """
        # Convert to NumPy if it's a tensor
        if isinstance(patch, torch.Tensor):
            patch = patch.cpu().numpy()

        # Handle different input shapes
        if patch.shape[0] in (3, 4):  # (C, H, W)
            patch = patch.transpose(1, 2, 0)  # To (H, W, C)

        # Ensure 3 channels (drop alpha if present)
        if patch.shape[-1] == 4:
            patch = patch[..., :3]

        # Normalize data range to [0, 255] uint8
        if patch.dtype != np.uint8:
            if patch.max() <= 1.0 + 1e-6:  # Allow slight float precision error
                patch = (patch * 255).clip(0, 255)
            else:
                patch = patch.clip(0, 255)
            patch = patch.astype(np.uint8)

        # Convert channel order if needed
        if input_format == "BGR" and output_format == "RGB":
            patch = patch[..., [2, 1, 0]]  # BGR to RGB
        elif input_format == "RGB" and output_format == "BGR":
            patch = patch[..., [2, 1, 0]]  # RGB to BGR

        return patch

    @staticmethod
    def render_clusters(indices, patches, input_format="RGB", output_format="RGB"):
        """
        Render the patches for each cluster.

        Args:
            indices (list): A 2D list of indices to vectors in patches.
            patches (torch.Tensor or list or Dataset): The source of patch data (tensor, list, or lazy-loading dataset).
            input_format (str): Input channel order, "RGB" or "BGR" (default "RGB").
            output_format (str): Desired output channel order, "RGB" or "BGR" (default "RGB").

        Returns:
            list: A 2D list where each row contains rendered patches (NumPy arrays) for a cluster.
        """
        rendered_clusters = []
        for cluster in indices:
            rendered_patches = []
            for index in cluster:
                # Handle different patch sources
                if isinstance(patches, (torch.Tensor, np.ndarray)):
                    single_patch = patches[index]
                elif hasattr(patches, '__getitem__'):  # Dataset-like object
                    single_patch, _, _ = patches[index]  # Assuming (patch1, patch2, inertial)
                else:
                    raise ValueError("Patches must be a tensor, numpy array, or dataset-like object")
                rendered_patch = PatchRenderer.render_patch(single_patch, input_format, output_format)
                rendered_patches.append(rendered_patch)
            rendered_clusters.append(rendered_patches)
        return rendered_clusters

    @staticmethod
    def image_grid(cluster_images, image_size=(64, 64), output_format="RGB"):
        """
        Creates a dynamically sized image grid containing all images in a cluster.

        Args:
            cluster_images (list): A list of images (NumPy arrays) belonging to the cluster.
            image_size (tuple): (width, height) to resize images (default (64, 64)).
            output_format (str): Desired output channel order, "RGB" or "BGR" (default "RGB").

        Returns:
            np.ndarray: Image grid as (H, W, 3) uint8 array in specified output_format.
        """
        if not cluster_images:
            raise ValueError("No images provided for the cluster grid.")

        resized_images = []
        for img in cluster_images:
            if img.dtype != np.uint8:
                img = img.clip(0, 255).astype(np.uint8)
            if img.shape[-1] != 3:
                raise ValueError("Images must have 3 channels")
            # OpenCV expects BGR, so convert if input is RGB
            if output_format == "RGB":
                img = img[..., [2, 1, 0]]  # RGB to BGR for OpenCV
            resized_img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            # Convert back to desired output format
            if output_format == "RGB":
                resized_img = resized_img[..., [2, 1, 0]]  # BGR to RGB
            resized_images.append(resized_img)

        num_images = len(resized_images)
        grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)

        rows = []
        for i in range(grid_rows):
            row_images = resized_images[i * grid_cols : (i + 1) * grid_cols]
            while len(row_images) < grid_cols:
                row_images.append(np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))
            rows.append(np.hstack(row_images))

        grid = np.vstack(rows)
        return grid


class Cluster:
    def __init__(self, vicreg_h5_path, synced_h5_path, model_path, batch_size=256):
        """
        Initialize the Cluster class with paths to .h5 files and a model.

        Args:
            vicreg_h5_path (str): Path to VICReg .h5 file.
            synced_h5_path (str): Path to synced .h5 file.
            model_path (str): Path to pre-trained model weights.
            batch_size (int): Batch size for DataLoader.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validate file paths
        if not os.path.exists(vicreg_h5_path):
            raise FileNotFoundError(f"VICReg .h5 file not found at: {vicreg_h5_path}")
        if not os.path.exists(synced_h5_path):
            raise FileNotFoundError(f"Synced .h5 file not found at: {synced_h5_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load model weights
        self.model = SterlingRepresentation("cpu").to("cpu")
        self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

        # Create dataset and dataloader with lazy loading
        self.dataset = TerrainDataset(
            synced_h5_path=synced_h5_path,
            vicreg_h5_path=vicreg_h5_path,
            incl_orientation=False
        )
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size

    def generate_clusters(
        self,
        k,
        iterations,
        save_model_path="scripts/clusters/kmeans_model.pkl",
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
        # Move model to CPU and compute embeddings lazily
        self.model.cpu()
        self.model.eval()

        num_samples = len(self.dataloader.dataset)
        embedding_size = 2 * self.model.latent_size
        temp_file = "temp_embeddings.dat"
        representation_vectors_np = np.memmap(temp_file, dtype='float32', mode='w+', shape=(num_samples, embedding_size))

        with torch.no_grad():
            start_idx = 0
            for batch in self.dataloader:
                patch1, _, inertial = batch
                patch1 = patch1.cpu()
                inertial = inertial.cpu()

                embeddings = self.model.get_terrain_embedding(patch1, inertial).cpu().numpy()
                batch_size = embeddings.shape[0]
                representation_vectors_np[start_idx:start_idx + batch_size] = embeddings
                start_idx += batch_size

        kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=iterations, n_init=10, random_state=42)
        kmeans.fit(representation_vectors_np)
        cluster_labels = kmeans.labels_

        # Save the K-means model
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(kmeans, save_model_path)

        print("I made (K) clusters: ", k)
        print("Number of items in each cluster:")
        for i in range(k):
            print(f" [Cluster {i}]: {(cluster_labels == i).sum()} items")

        # Organize image indices into clusters
        all_cluster_image_indices = [[] for _ in range(k)]
        for idx, cluster in enumerate(cluster_labels):
            all_cluster_image_indices[cluster].append(idx)  # Assign image index to corresponding cluster

        # Plot clusters
        save_plot_dir = os.path.dirname(save_model_path)
        self.plot_clusters(representation_vectors_np, torch.tensor(cluster_labels), k, save_plot_path=save_plot_dir)
        
        # Clean up temporary file
        del representation_vectors_np
        os.remove(temp_file)

        return all_cluster_image_indices

    def plot_clusters(self, representation_vectors_np, min_indices, k, save_plot_path=None):
        """
        Visualizes the k-means clusters after performing dimensionality reduction using PCA.

        Args:
            representation_vectors (torch.Tensor): Combined embeddings.
            min_indices (torch.Tensor): Cluster labels.
            k (int): Number of clusters.
            save_plot_path (str): Directory to save the plot.
        """

        # Apply PCA for dimensionality reduction (First to 20D, then to 2D)
        pca_high = PCA(n_components=20, random_state=42)
        intermediate_vectors = pca_high.fit_transform(representation_vectors_np)

        pca_final = PCA(n_components=2, whiten=True, random_state=42)
        reduced_vectors = pca_final.fit_transform(intermediate_vectors)

        # Compute centroids in PCA-reduced space
        reduced_centroids = np.array([
            reduced_vectors[min_indices == i].mean(axis=0) for i in range(k)
        ])

        # Plot clusters
        plt.figure(figsize=(8, 6))
        for cluster_idx in range(k):
            cluster_points = reduced_vectors[min_indices == cluster_idx]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}", alpha=0.6)

        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c="black", marker="x", label="Centroids")
        plt.title(f"K-means Clusters with k={k}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)

        # Save the plot
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
