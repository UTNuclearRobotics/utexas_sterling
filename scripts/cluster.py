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
    def __init__(self, data_pkl_path, model_path, batch_size=8192):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        if not os.path.exists(data_pkl_path):
            raise FileNotFoundError(f"Data pickle file not found at: {data_pkl_path}")
        with open(data_pkl_path, "rb") as file:
            data_pkl = pickle.load(file)

        # Load model weights
        self.model = SterlingRepresentation("cpu").to("cpu")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

        # Create dataset and dataloader
        dataset = TerrainDataset(patches=data_pkl)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # TODO: actually batch this so we don't run out of memory
        batch = next(iter(dataloader))
        patch1, _ = batch

        self.patches = patch1.to("cpu")

    def generate_clusters(
        self,
        k,
        iterations,
        save_model_path="scripts/clusters/kmeans_model.pkl",
        save_scaler_path="scripts/clusters/scaler.pkl",
    ):
        """
        Generate clusters using K-means algorithm.
        Args:
            k (int): Number of clusters to generate.
            iterations (int): Number of iterations for K-means.
        """
        # K Means
        representation_vectors = self.model.encode_single_patch(self.patches)
        representation_vectors_np = representation_vectors.detach().cpu().numpy()
        #scaler = MinMaxScaler()
        #representation_vectors_np = scaler.fit_transform(representation_vectors_np)
        #representation_vectors_np = normalize(representation_vectors_np, axis=1, norm='l2')

        # Apply K-means clustering with sklearn
        kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=iterations, n_init=10, random_state=42)
        kmeans.fit(representation_vectors_np)
        min_indices = kmeans.labels_

        # Save the K-means model and scaler
        #joblib.dump(kmeans, save_model_path)
        #joblib.dump(scaler, save_scaler_path)

        print("I made (K) clusters: ", k)
        print("Number of items in each cluster:")
        for i in range(0, k):
            print(" [", i, "]: ", (min_indices == i).sum())

        # Find the K farthest apart vectors for each cluster
        cluster_rep_vectors = []
        for i in range(k):
            cluster = representation_vectors[min_indices == i]
            cluster_size = cluster.size(0)

            # Use pairwise distances to calculate similarities directly
            clusterSim = torch.cdist(cluster, cluster)  # Pairwise distance matrix

            # Store indices of the farthest apart vectors
            cluster_indices = []
            selected = torch.zeros(cluster_size, dtype=torch.bool)  # Track selected indices
            if cluster_size == 0:  # Handle edge case for empty clusters
                cluster_rep_vectors.append(torch.empty(0, cluster.size(1), device=cluster.device))
                continue

            # Start with the first random index
            first_idx = 0
            cluster_indices.append(first_idx)
            selected[first_idx] = True

            # Iteratively select the farthest vector
            while len(cluster_indices) < cluster_size:
                # Compute distances from already selected indices
                distances = clusterSim[~selected][:, selected]
                max_dist_idx = distances.max(dim=0).indices[0].item()  # Find farthest unselected index
                unselected_indices = torch.where(~selected)[0]  # Map to original indices
                next_idx = unselected_indices[max_dist_idx].item()

                cluster_indices.append(next_idx)
                selected[next_idx] = True

            cluster_subtensor = cluster[cluster_indices]
            cluster_rep_vectors.append(cluster_subtensor)

        # Map back to image indices
        all_cluster_image_indices = []
        for cluster in cluster_rep_vectors:
            cluster_image_indices = []
            for row in cluster:
                indices = torch.nonzero(torch.all(representation_vectors == row, dim=1), as_tuple=True)[0]
                if indices.numel() > 0:  # Ensure there is at least one match
                    cluster_image_indices.append(indices[0].item())  # Use the first match
                else:
                    print("Warning: No matching representation vector found for a cluster representative.")
            all_cluster_image_indices.append(cluster_image_indices)

        # Display clusters
        for index, images in enumerate(all_cluster_image_indices):
            print("CLUSTER: ", index)
            print(images)

        # Plot clusters only for the best k
        self.plot_clusters(representation_vectors, torch.tensor(min_indices), k)

        return all_cluster_image_indices

    def iterate_generate_clusters(self, k_values, iterations):
        """
        Iterate over a range of k-values to find the best number of clusters.
        Args:
            k_values (range): Range of k-values to iterate over.
            iterations (int): Number of iterations for K-means.
        """
        representation_vectors = self.model.encode_single_patch(self.patches)
        representation_vectors_np = representation_vectors.detach().cpu().numpy()
        #scaler = MinMaxScaler()
        #representation_vectors_np = scaler.fit_transform(representation_vectors_np)
        #representation_vectors_np = normalize(representation_vectors_np, axis=1, norm='l2')

        silhouette_scores = []
        wcss_values = []
        all_k_cluster_indices = {}

        # Directory to save the plots
        save_dir = os.path.join(script_dir, "clusters")
        os.makedirs(save_dir, exist_ok=True)

        for k in k_values:
            # Initialize and fit KMeans with k-means++
            kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=iterations, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(representation_vectors_np)

            # WCSS Calculation
            wcss = kmeans.inertia_
            wcss_values.append((k, wcss))

            # Silhouette Score Calculation
            silhouette_avg = silhouette_score(representation_vectors_np, cluster_labels)
            silhouette_scores.append((k, silhouette_avg))

            print(f"Clusters formed for k={k}:")
            for cluster_idx in range(k):
                cluster_size = np.sum(cluster_labels == cluster_idx)
                print(f"  Cluster {cluster_idx}: {cluster_size} points")

            print(f"Silhouette Score for k={k}: {silhouette_avg}")

            # Store the cluster indices
            all_cluster_image_indices = self.store_cluster_image_indices(
                torch.tensor(cluster_labels), k, representation_vectors
            )
            all_k_cluster_indices[k] = all_cluster_image_indices

        # Plot the elbow method for WCSS
        ks, wcss = zip(*wcss_values)
        self.plot_wcss(ks, wcss)

        # Plot silhouette scores
        k, scores = zip(*silhouette_scores)
        self.plot_silhouette_scores(k, scores)

        print("Silhouette Scores for all k-values:")
        for k, score in silhouette_scores:
            print(f"  k={k}: {score}")

        best_k = max(silhouette_scores, key=lambda x: x[1])[0]
        # kn = KneeLocator(ks, wcss, curve="convex", direction="decreasing")
        # best_k = kn.knee
        print(f"Best k according to silhouette score: {best_k}")

        # Plot clusters only for the best k
        #self.plot_clusters(representation_vectors, torch.tensor(cluster_labels), best_k)

        return all_k_cluster_indices[best_k], best_k

    def store_cluster_image_indices(self, min_indices, k, representation_vectors):
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

        return all_cluster_image_indices

    def plot_wcss(self, k, wcss):
        kn = KneeLocator(k, wcss, curve="convex", direction="decreasing", interp_method="polynomial")
        plt.figure(figsize=(8, 6))
        plt.plot(k, wcss, marker="o", label="WCSS")
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles="dashed")
        plt.title("Elbow Method: WCSS vs Number of Clusters (k)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("WCSS")
        plt.grid(True)
        plt.legend()
        # Save the plot to the specified directory
        plt.savefig(os.path.join(save_dir, "wcss_vs_k.png"))
        plt.close()  # Close the plot to avoid memory overload
        plt.show()

    def plot_silhouette_scores(self, k, scores):
        plt.plot(k, scores, label="Silhouette Score")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs. k-clusters")
        plt.grid(True)
        # Save the plot to the specified directory
        plt.savefig(os.path.join(save_dir, "silhouette_score_vs_k.png"))
        plt.close()  # Close the plot to avoid memory overload
        plt.show()

    def plot_clusters(self, representation_vectors, min_indices, k):
        """
        Visualizes the k-means clusters after performing dimensionality reduction
        using PCA.
        """
        # Step 1: Normalize the representation vectors
        representation_vectors_np = representation_vectors.detach().cpu().numpy()
        #scaler = MinMaxScaler()
        #representation_vectors_np = scaler.fit_transform(representation_vectors_np)
        #representation_vectors_np = normalize(representation_vectors_np, norm='l2', axis=1)

        # Step 2: Apply PCA for dimensionality reduction (First to 20D, then to 2D)
        pca_high = PCA(n_components=20, random_state=42)
        intermediate_vectors = pca_high.fit_transform(representation_vectors_np)

        pca_final = PCA(n_components=2, whiten=True, random_state=42)
        reduced_vectors = pca_final.fit_transform(intermediate_vectors)

        # Step 3: Compute centroids in PCA-reduced space (Fixing the issue)
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

        # Save the plot
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"clusters_k{k}.png"))
        plt.show()


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
    pkl = [file for file in os.listdir(bag_path) if file.endswith("vicreg.pkl")]
    if len(pkl) != 1:
        raise FileNotFoundError(f"VICReg pickle file not found in: {bag_path}")
    pkl_path = os.path.join(bag_path, pkl[0])

    # Validate the pickle file exists
    model_path = os.path.join(bag_path, "models")
    pt = [file for file in os.listdir(model_path) if file.endswith(".pt")]
    if len(pt) != 1:
        raise FileNotFoundError(f"Terrain representation PyTorch model file not found in: {model_path}")
    pt_path = os.path.join(model_path, pt[0])

    # Generate clusters
    cluster = Cluster(
        data_pkl_path=pkl_path,
        model_path=pt_path,
    )

    #k_values = range(2, 12)
    k_values = 5
    iterations = 1000

    if isinstance(k_values, range):
        k_best_cluster_image_indices, best_k = cluster.iterate_generate_clusters(k_values, iterations)

        # Render clusters
        rendered_clusters = PatchRenderer.render_clusters(k_best_cluster_image_indices, cluster.patches)

        for i, cluster in enumerate(rendered_clusters):
            grid_image = PatchRenderer.image_grid(cluster)
            # Define the save path for the grid
            save_path = os.path.join(save_dir, f"cluster_{i}.png")
            # Save the grid image to the specified path
            cv2.imwrite(save_path, grid_image)

    elif isinstance(k_values, int):
        all_cluster_image_indices = cluster.generate_clusters(
            k_values, iterations
        )

        # Render clusters
        rendered_clusters = PatchRenderer.render_clusters(all_cluster_image_indices, cluster.patches)

        for i, cluster in enumerate(rendered_clusters):
            grid_image = PatchRenderer.image_grid(cluster)
            save_path = os.path.join(save_dir, f"cluster_{i}.png")
            # Save the grid image to the specified path
            #cv2.imwrite(save_path, grid_image)

    else:
        print("k_values is neither a range nor an integer")
