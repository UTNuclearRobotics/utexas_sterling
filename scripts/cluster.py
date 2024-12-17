import os
import pickle

import torch
from PIL import Image
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from train_representation import SterlingRepresentation
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

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

    def generate_clusters(self, k, iterations):
        # K Means
        representation_vectors = self.model.visual_encoder(self.patches)
        row_sums = representation_vectors.sum(dim=1, keepdim=True)
        representation_vectors = representation_vectors / row_sums
        centroids = representation_vectors[torch.randperm(representation_vectors.size(0))[:k]]

        for _ in range(iterations):
            distances = torch.cdist(representation_vectors, centroids)
            min_values, min_indices = torch.min(distances, dim=1)
            new_centroids = torch.stack([representation_vectors[min_indices == i].mean(dim=0) for i in range(k)])

            """
            Silhouette Score
            Cohesion
            """

            shadow = sum(min_values).item()
            # print("shadow:  ", shadow)
            # distances = torch.cdist(representation_vectors, centroids)
            # distances = torch.min(distances, dim=1)
            # print("distances:   ", distances)

            if torch.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        print("I made (K) clusters: ", k)
        print("Number of items in each cluster.")
        for i in range(0, k):
            print(" [", i, "]: ", representation_vectors[min_indices == i].shape[0])

        # Find the K farthest apart vectors for each cluster
        cluster_rep_vectors = []
        for i in range(0, k):
            cluster = representation_vectors[min_indices == i]
            row_norms = torch.norm(cluster, dim=1, keepdim=True)
            normalized_tensor = cluster / row_norms
            clusterT = cluster.transpose(0, 1)
            clusterSim = torch.matmul(cluster, clusterT)

            cluster_indices = []
            while len(cluster_indices) < 25:
                min_value = clusterSim.min()
                min_idx = (clusterSim == min_value).nonzero(as_tuple=False)
                min_row = min_idx[0, 0].item()
                min_col = min_idx[0, 1].item()
                cluster_indices.append(min_row)
                cluster_indices.append(min_col)
                clusterSim[min_row, min_col] = 1
                cluster_indices = list(set(cluster_indices))

            # print("cluster.shape:   ", cluster.shape)
            # print("cluster_indices:   ", cluster_indices)
            cluster_subtensor = cluster[cluster_indices]
            cluster_rep_vectors.append(cluster_subtensor)

        all_cluster_image_indices = []
        for index, cluster in enumerate(cluster_rep_vectors):
            # print("CLUSTER: ", index)
            cluster_image_indices = []
            for row in cluster:
                match = torch.all(representation_vectors == row, dim=1)
                indices = torch.nonzero(match, as_tuple=True)[0]
                if indices.numel() == 1:
                    cluster_image_indices.append(indices.item())
                else:
                    # Handle multiple matches (e.g., take the first match)
                    cluster_image_indices.append(indices[0].item())
            all_cluster_image_indices.append(cluster_image_indices)

        for index, images in enumerate(all_cluster_image_indices):
            print("CLUSTER: ", index)
            print(images)
        
        # Plot clusters only for the best k
        # self.plot_clusters(representation_vectors, min_indices, k)

        return all_cluster_image_indices

    def iterate_generate_clusters(self, k_values, iterations):
        representation_vectors = self.model.visual_encoder(self.patches)
        row_sums = representation_vectors.sum(dim=1, keepdim=True)
        representation_vectors = representation_vectors / row_sums

        silhouette_scores = []
        wcss_values = []
        all_k_cluster_indices = {}

        # Directory to save the plots
        save_dir = os.path.join(script_dir, "clusters")
        os.makedirs(save_dir, exist_ok=True)

        for k in k_values:
            # Initialize centroids using random samples
            centroids = representation_vectors[torch.randperm(representation_vectors.size(0))[:k]]
            
            for _ in range(iterations):
                # Compute the pairwise distance between points and centroids
                distances = torch.cdist(representation_vectors, centroids)
                
                # Assign each point to the nearest centroid
                min_values, min_indices = torch.min(distances, dim=1)
                
                # Efficient centroid update using the indices of assigned clusters
                new_centroids = torch.stack(
                    [representation_vectors[min_indices == i].mean(dim=0) for i in range(k)]
                )

                # Break if centroids have not changed
                if torch.allclose(centroids, new_centroids):
                    break

                centroids = new_centroids

            # WCSS Calculation
            closest_distances = torch.min(distances, dim=1)[0]  # Closest centroid distances for each point
            wcss = torch.sum(closest_distances ** 2).item()  # Sum of squared distances
            wcss_values.append((k, wcss))

            # Silhouette Score Calculation
            silhouette_values = self.calculate_silhouette_scores(representation_vectors, min_indices, k)
            avg_silhouette_score = torch.mean(torch.tensor(silhouette_values)).item()
            silhouette_scores.append((k, avg_silhouette_score))

            print(f"Clusters formed for k={k}:")
            for cluster_idx in range(k):
                cluster_size = representation_vectors[min_indices == cluster_idx].shape[0]
                print(f"  Cluster {cluster_idx}: {cluster_size} points")

            print(f"Silhouette Score for k={k}: {avg_silhouette_score}")

            # Store the cluster indices
            all_cluster_image_indices = self.store_cluster_image_indices(min_indices, k, representation_vectors)
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
        print(f"Best k according to silhouette score: {best_k}")

        # Plot clusters only for the best k
        best_k_indices = all_k_cluster_indices[best_k]
        self.plot_clusters(representation_vectors, min_indices, best_k)

        return all_k_cluster_indices[best_k], silhouette_scores

    def calculate_silhouette_scores(self, representation_vectors, min_indices, k):
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

        return silhouette_values

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
        kn = KneeLocator(k,wcss, curve='convex', direction='decreasing', interp_method ='polynomial')
        plt.figure(figsize=(8, 6))
        plt.plot(k, wcss, marker="o", label="WCSS")
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
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
        plt.plot(k, scores, label='Silhouette Score')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. k-clusters')
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
        min_indices = min_indices.cpu().numpy()
        # Reduce the dimensionality of the representation vectors for visualization
        pca = PCA(n_components=2)  # Use PCA for dimensionality reduction
        reduced_vectors = pca.fit_transform(representation_vectors.detach().cpu().numpy())  # Convert to numpy for PCA        
        # Set up the plot
        plt.figure(figsize=(8, 6))

        # Plot each cluster with a different color
        for cluster_idx in range(k):
            cluster_points = reduced_vectors[min_indices == cluster_idx]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx}', alpha=0.6)

        # Plot centroids
        centroids = torch.stack([representation_vectors[min_indices == i].mean(dim=0) for i in range(k)])
        reduced_centroids = pca.transform(centroids.detach().cpu().numpy())  # Reduce dimensionality of centroids
        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='black', marker='x', label='Centroids')

        plt.title(f"K-means Clusters with k={k}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)

        # Save the plot
        save_dir = os.path.join(script_dir, "clusters")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"clusters_k{k}.png"))
        plt.show()


if __name__ == "__main__":
    # Save directory
    save_dir = os.path.join(script_dir, "clusters")
    os.makedirs(save_dir, exist_ok=True)

    # Generate clusters
    cluster = Cluster(
        data_pkl_path=os.path.join(script_dir, "../datasets/nrg_ahg_courtyard.pkl"),
        model_path=os.path.join(script_dir, "../models/vis_rep.pt"),
    )

    # k_values = range(2, 10)
    k_values = 4
    iterations = 200

    if isinstance(k_values, range):
        k_best_cluster_image_indices, silhouette_scores = cluster.iterate_generate_clusters(k_values, iterations)

        # Render clusters
        rendered_clusters = PatchRenderer.render_clusters(k_best_cluster_image_indices, cluster.patches)

        for i, cluster in enumerate(rendered_clusters):
            grid_image = PatchRenderer.image_grid(cluster)
            grid_image.save(os.path.join(save_dir, f"cluster_{i}.png"))

    elif isinstance(k_values, int):
        all_cluster_image_indices = cluster.generate_clusters(k_values, iterations)

        # Render clusters
        rendered_clusters = PatchRenderer.render_clusters(all_cluster_image_indices, cluster.patches)

        for i, cluster in enumerate(rendered_clusters):
            grid_image = PatchRenderer.image_grid(cluster)
            grid_image.save(os.path.join(save_dir, f"cluster_{i}.png"))

    else:
        print("k_values is neither a range nor an integer")
    
    # image = PatchRenderer.image_streaks(rendered_clusters)
    # image.save(os.path.join(save_dir, "cluster_grid.png"))

