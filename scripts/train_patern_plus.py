import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader, random_split
from models import VisualEncoderModel, ProprioceptionModel, UtilityFuncVisual, UtilityFuncProprioceptive, CostNet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from cluster import PatchRenderer
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import load_bag_h5
import psutil
import h5py
import gc
import tempfile
import joblib

class PaternAdaptation(nn.Module):
    def __init__(self, device, pretrained_weights_path, latent_size=128):
        super(PaternAdaptation, self).__init__()
        self.device = device
        self.latent_size = latent_size
        self.visual_encoder = VisualEncoderModel(latent_size=self.latent_size).to(device)
        self.proprioceptive_encoder = ProprioceptionModel(latent_size=self.latent_size).to(device)
        self.uvis = UtilityFuncVisual(latent_size=self.latent_size).to(device)
        self.upro = UtilityFuncProprioceptive(latent_size=self.latent_size).to(device)
        self.cost_head = CostNet(latent_size=self.latent_size).to(device)

        weight_files = {
            "visual_encoder": "fvis.pt",
            "proprioceptive_encoder": "fpro.pt",
            "uvis": "uvis.pt",
            "upro": "upro.pt",
            "cost_head": "cost_head.pt"
        }
        for name, file in weight_files.items():
            path = os.path.join(pretrained_weights_path, file)
            if os.path.exists(path):
                state_dict = torch.load(path, weights_only=True, map_location=device)
                getattr(self, name).load_state_dict(state_dict)
                print(f"Loaded {name} from {path}")
            else:
                raise FileNotFoundError(f"Missing {file} in {pretrained_weights_path}")

        for param in self.proprioceptive_encoder.parameters():
            param.requires_grad = False
        for param in self.upro.parameters():
            param.requires_grad = False

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

    def forward(self, patches, inertial=None):
        patches = patches.to(self.device)
        phi_vis = self.visual_encoder(patches)
        uvis_pred = self.uvis(phi_vis)
        if inertial is not None:
            inertial = inertial.to(self.device)
            phi_pro = self.proprioceptive_encoder(inertial.float())
            upro_pred = self.upro(phi_pro)
        else:
            phi_pro = torch.zeros_like(phi_vis)
            upro_pred = torch.zeros_like(uvis_pred)
        final_cost = self.cost_head(uvis_pred)
        return phi_vis, phi_pro, uvis_pred, upro_pred, final_cost

    def extract_proprioceptive_features(self, inertial_data):
        self.eval()
        with torch.no_grad():
            phi_pro = self.proprioceptive_encoder(inertial_data.to(self.device).float())
        return phi_pro

    def extrapolate_preferences(self, adaptation_inertial, preadapt_phi_pro, preadapt_preferences, preadapt_labels, max_distance_threshold):
        # Extract features from adaptation inertial data
        adapt_phi_pro = self.extract_proprioceptive_features(adaptation_inertial)
        n_samples = adapt_phi_pro.shape[0]
        if n_samples < 1:
            raise ValueError("Adaptation-set has no samples.")

        preadapt_phi_pro = preadapt_phi_pro.to(self.device)
        preadapt_prefs = preadapt_preferences.to(self.device)
        preadapt_labels = preadapt_labels.to(self.device)

        # Load K-means model centroids
        kmeans_model_path = os.path.join(args.preadapt_bag, "clusters", "kmeans_model.pkl")
        kmeans = joblib.load(kmeans_model_path)
        full_centroids = torch.tensor(kmeans.cluster_centers_, device=self.device)
        
        # Take the second half (proprioceptive features, assuming [visual, proprioceptive] order)
        preadapt_cluster_centers = full_centroids[:, self.latent_size:]  # Shape: [n_clusters, 128]
        n_clusters = preadapt_cluster_centers.shape[0]
        
        # Verify dimensionality
        assert adapt_phi_pro.shape[1] == preadapt_cluster_centers.shape[1], \
            f"Dimensionality mismatch: adapt_phi_pro {adapt_phi_pro.shape[1]} vs centroids {preadapt_cluster_centers.shape[1]}"

        # Compute cluster preferences
        cluster_prefs = torch.zeros(n_clusters, device=self.device)
        cluster_labels = torch.arange(n_clusters, device=self.device)
        for i in range(n_clusters):
            cluster_mask = (preadapt_labels == i)
            cluster_prefs[i] = preadapt_prefs[cluster_mask].mean() if cluster_mask.sum() > 0 else 0.0

        # Compute distances to pre-computed centroids
        distances = torch.cdist(adapt_phi_pro, preadapt_cluster_centers)
        sorted_distances, sorted_indices = distances.sort(dim=1)
        min_distances = sorted_distances[:, 0]
        nearest_cluster_indices = sorted_indices[:, 0]

        extrapolated_prefs = torch.zeros(n_samples, device=self.device)
        extrapolated_labels = cluster_labels[nearest_cluster_indices]
        within_threshold = min_distances <= max_distance_threshold

        extrapolated_prefs[within_threshold] = cluster_prefs[nearest_cluster_indices[within_threshold]]
        outside_threshold = ~within_threshold
        if outside_threshold.sum() > 0:
            nearest_two_indices = sorted_indices[outside_threshold, :2]
            nearest_two_distances = sorted_distances[outside_threshold, :2]
            weights = 1.0 / (nearest_two_distances + 1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True)
            prefs_nearest_two = cluster_prefs[nearest_two_indices]
            extrapolated_prefs[outside_threshold] = (prefs_nearest_two * weights).sum(dim=1)

        return extrapolated_prefs, extrapolated_labels

    def retrain_visual_components(self, train_loader, val_loader, optimizer, scheduler, epochs, initial_weights=None):
        l2_lambda = 0.1  # Hyperparameter for L2 penalty strength (tune as needed)
        
        for epoch in range(epochs):
            self.train()
            total_train_loss = 0

            for batch in train_loader:
                patches, inertial, terrain_labels, preferences = [x.to(self.device) for x in batch]
                preferences = preferences.float()
                batch_size = patches.shape[0]

                scaled_preferences = train_loader.dataset.dataset.get_scaled_preferences(preferences)
                phi_vis, _, uvis_pred, _, final_cost = self.forward(patches, inertial)

                # Triplet loss indices
                labels_expanded = terrain_labels.unsqueeze(1)
                pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                neg_mask = (labels_expanded != labels_expanded.t())

                pos_indices = torch.where(pos_mask)[1]
                neg_indices = torch.where(neg_mask)[1]

                if len(pos_indices) < batch_size or len(neg_indices) < batch_size:
                    pos_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                    neg_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                else:
                    pos_indices = pos_indices[torch.randperm(len(pos_indices), device=self.device)[:batch_size]]
                    neg_indices = neg_indices[torch.randperm(len(neg_indices), device=self.device)[:batch_size]]

                vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
                pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
                pref_diff = scaled_preferences.unsqueeze(1) - scaled_preferences.unsqueeze(0)
                ranking_loss = F.relu(1.0 - (pred_diff / 100.0)[pref_diff > 0]).mean()
                cost_loss = F.mse_loss(final_cost, scaled_preferences)
                task_loss = vis_loss + 0.5 * ranking_loss + cost_loss

                # L2 penalty on weight changes from initial weights
                if initial_weights:
                    l2_penalty = 0
                    for name, param in self.visual_encoder.named_parameters():
                        if name in initial_weights['visual_encoder']:
                            l2_penalty += torch.norm(param - initial_weights['visual_encoder'][name].to(param.device), p=2) ** 2
                    for name, param in self.uvis.named_parameters():
                        if name in initial_weights['uvis']:
                            l2_penalty += torch.norm(param - initial_weights['uvis'][name].to(param.device), p=2) ** 2
                    for name, param in self.cost_head.named_parameters():
                        if name in initial_weights['cost_head']:
                            l2_penalty += torch.norm(param - initial_weights['cost_head'][name].to(param.device), p=2) ** 2
                    total_loss = task_loss + l2_lambda * l2_penalty
                    print(f"Batch - Task Loss: {task_loss.item():.4f}, L2 Penalty: {l2_lambda * l2_penalty.item():.4f}")
                else:
                    total_loss = task_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += total_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation loop (unchanged except for logging)
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    patches, inertial, terrain_labels, preferences = [x.to(self.device) for x in batch]
                    preferences = preferences.float()
                    batch_size = patches.shape[0]

                    scaled_preferences = val_loader.dataset.dataset.get_scaled_preferences(preferences)
                    phi_vis, _, uvis_pred, _, final_cost = self.forward(patches, inertial)

                    labels_expanded = terrain_labels.unsqueeze(1)
                    pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                    neg_mask = (labels_expanded != labels_expanded.t())
                    
                    pos_indices = torch.where(pos_mask)[1]
                    neg_indices = torch.where(neg_mask)[1]
                    
                    if len(pos_indices) < batch_size or len(neg_indices) < batch_size:
                        pos_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                        neg_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                    else:
                        pos_indices = pos_indices[torch.randperm(len(pos_indices), device=self.device)[:batch_size]]
                        neg_indices = neg_indices[torch.randperm(len(neg_indices), device=self.device)[:batch_size]]

                    vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
                    pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
                    pref_diff = scaled_preferences.unsqueeze(1) - scaled_preferences.unsqueeze(0)
                    ranking_loss = F.relu(1.0 - (pred_diff / 100.0)[pref_diff > 0]).mean()
                    cost_loss = F.mse_loss(final_cost, scaled_preferences)
                    total_val_loss += (vis_loss + 0.5 * ranking_loss + cost_loss).item()  # No L2 penalty in validation
            avg_val_loss = total_val_loss / len(val_loader)
            
            scheduler.step()
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    def save_adapted_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for name, model in [("fvis_adapted.pt", self.visual_encoder), ("uvis_adapted.pt", self.uvis), ("cost_head_adapted.pt", self.cost_head)]:
            torch.save(model.state_dict(), os.path.join(save_dir, name))
        print(f"Saved models to {save_dir}")

def visualize_clusters(phi_pro, labels, adapt_phi_pro=None, save_path=None, title="Clusters"):
    pca = PCA(n_components=2, random_state=42)
    phi_pro_2d = pca.fit_transform(phi_pro.cpu().numpy())
    plt.figure(figsize=(10, 8))
    plt.scatter(phi_pro_2d[:, 0], phi_pro_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    if adapt_phi_pro is not None:
        adapt_phi_pro_2d = pca.transform(adapt_phi_pro.cpu().numpy())
        plt.scatter(adapt_phi_pro_2d[:, 0], adapt_phi_pro_2d[:, 1], c='red', marker='x', s=100)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def render_and_save_cluster_patches(patches, labels, save_dir, prefix="cluster"):
    renderer = PatchRenderer()
    unique_labels = np.unique(labels)
    print(f"Unique labels in render: {unique_labels}")
    cluster_indices = [np.where(labels == label)[0].tolist() for label in unique_labels]
    print(f"Cluster sizes: {[len(indices) for indices in cluster_indices]}")
    rendered_clusters = renderer.render_clusters(cluster_indices, patches.cpu().numpy())
    print(f"Number of rendered clusters: {len(rendered_clusters)}")
    os.makedirs(save_dir, exist_ok=True)
    for cluster_id, cluster_patches in enumerate(rendered_clusters):
        print(f"Cluster {cluster_id} has {len(cluster_patches)} patches")
        if cluster_patches:
            grid_image = renderer.image_grid(cluster_patches)
            save_path = os.path.join(save_dir, f"{prefix}_{cluster_id}.png")
            cv2.imwrite(save_path, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
            print(f"Saved {save_path}")

def process_batch_to_disk(loader, cache_dir, device="cpu"):
    print("Extracting adaptation features...")
    adapt_patches_list = []
    adapt_inertial_list = []
    
    # Ensure device is CPU to avoid GPU memory issues
    device = torch.device("cpu")
    
    for i, batch in enumerate(loader):
        patch1, patch2, imu_sample = batch
        # Move to CPU explicitly
        patch1 = patch1.to(device)
        imu_sample = imu_sample.to(device)
        
        # Save each batch to disk immediately
        batch_patch_file = os.path.join(cache_dir, f"patch_{i}.pt")
        batch_inertial_file = os.path.join(cache_dir, f"inertial_{i}.pt")
        
        torch.save(patch1.cpu(), batch_patch_file)
        torch.save(imu_sample.cpu(), batch_inertial_file)
        
        adapt_patches_list.append(batch_patch_file)
        adapt_inertial_list.append(batch_inertial_file)
        
        # Clean up memory
        del patch1, patch2, imu_sample
        
        if i % 10 == 0:
            # Use psutil to monitor system RAM usage instead of GPU
            import psutil
            print(f"Processed {i * args.batch_size} adaptation samples, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    return adapt_patches_list, adapt_inertial_list

def concatenate_cached_files(file_list, output_path, device="cpu"):
    print(f"Saving concatenated features to {output_path}...")
    # Process files one at a time to keep memory usage low
    with torch.no_grad():
        first_file = torch.load(file_list[0], map_location=device)
        total_size = len(file_list) * first_file.shape[0]
        output_shape = (total_size, *first_file.shape[1:])
        
        # Pre-allocate tensor on disk using memory mapping
        result = torch.zeros(output_shape, dtype=first_file.dtype)
        current_idx = 0
        
        for f in file_list:
            batch_data = torch.load(f, map_location=device)
            batch_size = batch_data.shape[0]
            result[current_idx:current_idx + batch_size] = batch_data
            current_idx += batch_size
            del batch_data
        
        torch.save(result, output_path)
    
    # Clean up temporary files
    for f in file_list:
        os.remove(f)

def custom_collate(batch):
    patches, inertial, terrain_labels, preferences = zip(*batch)
    # Replace None with -1 in terrain_labels
    terrain_labels = [label if label is not None else -1 for label in terrain_labels]
    return (torch.stack(patches),
            torch.stack(inertial),
            torch.tensor(terrain_labels, dtype=torch.long),
            torch.stack(preferences))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Preference Extrapolation Training for PATERN")
    parser.add_argument("-preadapt_bag", "-pb", type=str, required=True)
    parser.add_argument("-adapt_bag", "-ab", type=str, required=True)
    parser.add_argument("-batch_size", type=int, default=1024)
    parser.add_argument("-epochs", type=int, default=25)
    parser.add_argument("-val_split", type=float, default=0.2)
    return parser.parse_args()

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_datasets(args):
    print("Loading pre-adaptation data...")
    preadapt_h5_path = os.path.join(args.preadapt_bag, "clusters", "labeled_data.h5")
    preadapt_dataset = TerrainDataset(labeled_dataset=preadapt_h5_path, transform=None)
    n_clusters = len(set(preadapt_dataset.terrain_labels))
    print(f"Found {n_clusters} unique clusters in pre-adaptation data")
    if n_clusters == 0:
        raise ValueError("No unique terrain labels found in pre-adaptation data.")

    print("Loading adaptation data...")
    adapt_vicreg_path = load_bag_h5(args.adapt_bag, "vicreg")
    adapt_synced_path = load_bag_h5(args.adapt_bag, "synced")
    adapt_dataset = TerrainDataset(synced_h5_path=adapt_synced_path, vicreg_h5_path=adapt_vicreg_path, transform=None, train=True)
    print(f"Adaptation dataset size: {len(adapt_dataset)}")
    
    return preadapt_dataset, adapt_dataset, n_clusters

def initialize_model(args, device):
    print("Initializing model...")
    models_dir = os.path.join(args.preadapt_bag, "models")
    return PaternAdaptation(device=device, pretrained_weights_path=models_dir, latent_size=128).to(device)

def create_dataloaders(preadapt_dataset, adapt_dataset, batch_size):
    print("Creating DataLoaders...")
    preadapt_loader = DataLoader(preadapt_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    adapt_loader = DataLoader(adapt_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    return preadapt_loader, adapt_loader

def extract_preadapt_features(model, preadapt_loader, cache_dir, device):
    preadapt_phi_pro_path = os.path.join(cache_dir, "preadapt_phi_pro.pt")
    preadapt_prefs_path = os.path.join(cache_dir, "preadapt_prefs.pt")
    preadapt_labels_path = os.path.join(cache_dir, "preadapt_labels.npy")
    kmeans_model_path = os.path.join(args.preadapt_bag, "clusters", "kmeans_model.pkl")  # Path to saved K-means model
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(preadapt_phi_pro_path) or not os.path.exists(preadapt_labels_path):
        print("Extracting pre-adaptation features and loading K-means labels...")
        preadapt_phi_pro, preadapt_prefs = None, None
        kmeans = joblib.load(kmeans_model_path)  # Load the saved K-means model
        preadapt_labels = kmeans.labels_  # Use pre-computed cluster labels
        
        # Extract features only if not cached
        for i, batch in enumerate(preadapt_loader):
            _, inertial, _, preferences = batch  # Terrain labels ignored since K-means provides them
            phi_pro = model.extract_proprioceptive_features(inertial)
            preferences = preferences.to(device)
            preadapt_phi_pro = phi_pro if preadapt_phi_pro is None else torch.cat([preadapt_phi_pro, phi_pro])
            preadapt_prefs = preferences if preadapt_prefs is None else torch.cat([preadapt_prefs, preferences])
            del phi_pro, preferences
            torch.cuda.empty_cache() if device.type == "cuda" else None
            if i % 10 == 0:
                print(f"Processed {i * args.batch_size} pre-adaptation samples, RAM usage: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB" if device.type == "cuda" else "CPU mode")
        
        print("Saving pre-adaptation features to cache...")
        torch.save(preadapt_phi_pro, preadapt_phi_pro_path)
        torch.save(preadapt_prefs, preadapt_prefs_path)
        np.save(preadapt_labels_path, preadapt_labels)
    else:
        print("Loading pre-adaptation features and K-means labels from cache...")
        preadapt_phi_pro = torch.load(preadapt_phi_pro_path, map_location=device)
        preadapt_prefs = torch.load(preadapt_prefs_path, map_location=device)
        preadapt_labels = np.load(preadapt_labels_path)
    
    print(f"Preadaptation features shape: {preadapt_phi_pro.shape}")
    return preadapt_phi_pro, preadapt_prefs, preadapt_labels

def compute_distance_threshold(preadapt_phi_pro, preadapt_labels, cache_dir, args):
    print("Computing distance threshold using pre-trained K-means centroids...")
    
    # Load the existing K-means model
    kmeans_model_path = os.path.join(args.preadapt_bag, "clusters", "kmeans_model.pkl")
    kmeans = joblib.load(kmeans_model_path)
    full_centroids = torch.tensor(kmeans.cluster_centers_, device=preadapt_phi_pro.device)
    
    # Take the second half (proprioceptive features, assuming [visual, proprioceptive] order)
    latent_size = preadapt_phi_pro.shape[1]  # Assuming preadapt_phi_pro is already the proprioceptive part (128)
    preadapt_cluster_centers = full_centroids[:, latent_size:]  # Shape: [n_clusters, 128]
    
    # Ensure dimensionality matches
    assert preadapt_phi_pro.shape[1] == preadapt_cluster_centers.shape[1], \
        f"Dimensionality mismatch: preadapt_phi_pro {preadapt_phi_pro.shape[1]} vs centroids {preadapt_cluster_centers.shape[1]}"
    
    # Compute distances from each sample to all centroids
    distances = torch.cdist(preadapt_phi_pro, preadapt_cluster_centers)  # Shape: [n_samples, n_clusters]
    
    # Get the distance to the assigned cluster (intra-cluster) and the nearest other cluster (inter-cluster)
    assigned_cluster_distances = distances[range(len(preadapt_labels)), preadapt_labels]  # Intra-cluster distances
    other_distances = distances.clone()
    other_distances[range(len(preadapt_labels)), preadapt_labels] = float('inf')  # Exclude assigned cluster
    nearest_other_distances = other_distances.min(dim=1)[0]  # Min distance to any other cluster (inter-cluster)
    
    # Compute averages
    avg_intra = assigned_cluster_distances.mean().item()
    avg_inter = nearest_other_distances.mean().item()
    
    # Set threshold as the midpoint, with a fallback
    max_distance_threshold = (avg_intra + avg_inter) / 2 if avg_intra > 0 and avg_inter > 0 else 5.0
    
    print(f"Avg Intra-cluster Distance (to centroid): {avg_intra:.4f}")
    print(f"Avg Inter-cluster Distance (to nearest other centroid): {avg_inter:.4f}")
    print(f"Computed Max Distance Threshold: {max_distance_threshold:.4f}")
    
    return max_distance_threshold

def extract_adapt_features(model, adapt_loader, cache_dir):
    device = torch.device("cpu")
    adapt_patches_path = os.path.join(cache_dir, "adapt_patches.pt")
    adapt_inertial_path = os.path.join(cache_dir, "adapt_inertial.pt")

    if not os.path.exists(adapt_patches_path) or not os.path.exists(adapt_inertial_path):
        adapt_patches_list, adapt_inertial_list = process_batch_to_disk(adapt_loader, cache_dir)
        concatenate_cached_files(adapt_patches_list, adapt_patches_path)
        concatenate_cached_files(adapt_inertial_list, adapt_inertial_path)
        adapt_patches = torch.load(adapt_patches_path, map_location=device)
        adapt_inertial = torch.load(adapt_inertial_path, map_location=device)
    else:
        print("Loading adaptation features from cache...")
        adapt_patches = torch.load(adapt_patches_path, map_location=device)
        adapt_inertial = torch.load(adapt_inertial_path, map_location=device)

    print("Extracting proprioceptive features for adaptation data...")
    adapt_phi_pro = model.extract_proprioceptive_features(adapt_inertial)
    return adapt_patches, adapt_inertial, adapt_phi_pro

def extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, preadapt_data, max_distance_threshold, args):
    print("Extrapolating preferences and labels...")
    extrapolated_prefs, extrapolated_labels = model.extrapolate_preferences(
        adapt_inertial, preadapt_data[0], preadapt_data[1], torch.tensor(preadapt_data[2], device=adapt_inertial.device), max_distance_threshold
    )

    adapt_patches_cpu, adapt_inertial_cpu = adapt_patches.cpu(), adapt_inertial.cpu()
    prefs_list, labels_list = extrapolated_prefs.cpu().tolist(), extrapolated_labels.cpu().tolist()

    print("Creating labeled adaptation data...")
    adapt_data_dir = os.path.join(args.preadapt_bag, "adapt_data_cache")
    os.makedirs(adapt_data_dir, exist_ok=True)
    adapt_data_file = os.path.join(adapt_data_dir, "adapt_data.h5")

    batch_size = 1000
    with h5py.File(adapt_data_file, 'w') as f:
        for start_idx in range(0, len(adapt_patches), batch_size):
            end_idx = min(start_idx + batch_size, len(adapt_patches))
            for i in range(start_idx, end_idx):
                group = f.create_group(f"entry_{i}")
                group.create_dataset("patch", data=adapt_patches_cpu[i].numpy(), compression="gzip")
                group.create_dataset("inertial", data=adapt_inertial_cpu[i].numpy(), compression="gzip")
                group.attrs["terrain_label"] = labels_list[i]
                group.attrs["preference"] = prefs_list[i]
            print(f"Processed {end_idx} adaptation data entries, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    del adapt_patches_cpu, adapt_inertial_cpu, prefs_list, labels_list
    gc.collect()
    print(f"RAM usage after clearing adaptation data: {psutil.virtual_memory().used / 1024**2:.2f} MB")
    return adapt_data_file

def create_adapt_dataset(adapt_data_file, args):
    print("Creating adaptation-only dataset and DataLoaders...")
    adapt_data = []
    
    print("Loading adaptation data with extrapolated preferences...")
    with h5py.File(adapt_data_file, 'r') as f:
        total_entries = len(f)
        for i in range(total_entries):
            group = f[f"entry_{i}"]
            terrain_label = -1 if group.attrs["terrain_label"] == -1 else group.attrs["terrain_label"]
            adapt_data.append({
                "patch": torch.from_numpy(group["patch"][()]),
                "inertial": torch.from_numpy(group["inertial"][()]),
                "terrain_label": terrain_label,
                "preference": group.attrs["preference"]
            })
            if (i + 1) % 1000 == 0:
                print(f"Loaded {i + 1}/{total_entries} adaptation entries, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    adapt_dataset = TerrainDataset(labeled_dataset=adapt_data, transform=None)
    print(f"Adaptation-only dataset size: {len(adapt_dataset)}")

    val_size = int(args.val_split * len(adapt_dataset))
    train_size = len(adapt_dataset) - val_size
    train_dataset, val_dataset = random_split(adapt_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=custom_collate)
    
    print(f"Adaptation Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    return train_loader, val_loader

def retrain_model(model, train_loader, val_loader, epochs):
    print("Retraining model with controlled weight updates...")
    
    # Store initial weights for L2 regularization
    initial_weights = {
        'visual_encoder': {k: v.clone() for k, v in model.visual_encoder.state_dict().items()},
        'uvis': {k: v.clone() for k, v in model.uvis.state_dict().items()},
        'cost_head': {k: v.clone() for k, v in model.cost_head.state_dict().items()}
    }
    
    # Lower learning rate and increase weight decay for more regularization
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,  # Reduced from 1e-4 for smaller updates
        weight_decay=1e-4  # Increased from 1e-5 for stronger regularization
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Pass initial weights to retrain_visual_components for L2 penalty
    model.retrain_visual_components(train_loader, val_loader, optimizer, scheduler, epochs, initial_weights=initial_weights)

def concatenate_to_file(file_list, output_path, total_samples, sample_shape, device="cpu"):
    print(f"Concatenating to {output_path}...")
    # Calculate total size
    batch_size = torch.load(file_list[0], map_location="cpu").shape[0]
    output_shape = [total_samples] + list(sample_shape[1:])  # e.g., [99301, 128] or [99301, 3, 64, 64]
    
    # Use HDF5 for large data
    with h5py.File(output_path, 'w') as f:
        dset = f.create_dataset("data", shape=output_shape, dtype=np.float32, compression="gzip")
        current_idx = 0
        for file_path in file_list:
            batch_data = torch.load(file_path, map_location=device).cpu().numpy()
            batch_size = batch_data.shape[0]
            dset[current_idx:current_idx + batch_size] = batch_data
            current_idx += batch_size
            del batch_data
            print(f"Concatenated up to {current_idx} samples, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")
            gc.collect()

def extract_postadapt_features(model, train_dataset, args):
    print("Extracting post-adaptation features...")
    postadapt_cache_dir = os.path.join(args.preadapt_bag, "postadapt_cache")
    os.makedirs(postadapt_cache_dir, exist_ok=True)
    aggregated_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    phi_pro_files, patch_files, postadapt_labels_list = [], [], []
    for i, batch in enumerate(aggregated_loader):
        patches, inertial, terrain_labels, _ = batch
        phi_pro = model.extract_proprioceptive_features(inertial.to(model.device))
        patches = patches.to(model.device)
        phi_pro_file = os.path.join(postadapt_cache_dir, f"phi_pro_{i}.pt")
        patch_file = os.path.join(postadapt_cache_dir, f"patch_{i}.pt")
        torch.save(phi_pro.cpu(), phi_pro_file)
        torch.save(patches.cpu(), patch_file)
        phi_pro_files.append(phi_pro_file)
        patch_files.append(patch_file)
        postadapt_labels_list.extend(terrain_labels)
        del phi_pro, patches, inertial
        if i % 10 == 0:
            print(f"Processed {i * args.batch_size} post-adaptation samples, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")
            gc.collect()

    phi_pro_output = os.path.join(postadapt_cache_dir, "postadapt_phi_pro.h5")
    patch_output = os.path.join(postadapt_cache_dir, "postadapt_patches.h5")
    phi_pro_sample = torch.load(phi_pro_files[0], map_location="cpu")
    patch_sample = torch.load(patch_files[0], map_location="cpu")
    concatenate_to_file(phi_pro_files, phi_pro_output, len(train_dataset), phi_pro_sample.shape)
    concatenate_to_file(patch_files, patch_output, len(train_dataset), patch_sample.shape)

    return phi_pro_output, patch_output, postadapt_labels_list

def visualize_and_render(phi_pro_output, patch_output, postadapt_labels_list, train_dataset, args, n_clusters):
    print("Caching post-adaptation features for visualization and rendering...")
    batch_size = 1000
    phi_pro_cache = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    patch_cache = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    phi_pro_sample = torch.load(os.path.join(args.preadapt_bag, "postadapt_cache", "phi_pro_0.pt"), map_location="cpu")
    patch_sample = torch.load(os.path.join(args.preadapt_bag, "postadapt_cache", "patch_0.pt"), map_location="cpu")

    phi_pro_mmap = np.memmap(phi_pro_cache.name, dtype='float32', mode='w+', shape=(len(train_dataset), phi_pro_sample.shape[1]))
    patch_mmap = np.memmap(patch_cache.name, dtype='float32', mode='w+', shape=(len(train_dataset), *patch_sample.shape[1:]))

    with h5py.File(phi_pro_output, 'r') as f_phi, h5py.File(patch_output, 'r') as f_patch:
        phi_pro_data, patch_data = f_phi["data"], f_patch["data"]
        for start in range(0, len(train_dataset), batch_size):
            end = min(start + batch_size, len(train_dataset))
            phi_pro_mmap[start:end] = phi_pro_data[start:end]
            patch_mmap[start:end] = patch_data[start:end]
            print(f"Cached {end}/{len(train_dataset)} samples, RAM: {psutil.virtual_memory().used / 1024**2:.2f} MB")
        phi_pro_mmap.flush()
        patch_mmap.flush()

    postadapt_labels = np.array([label if label != -1 else -1 for label in postadapt_labels_list])
    phi_pro_mmap_read = np.memmap(phi_pro_cache.name, dtype='float32', mode='r', shape=(len(train_dataset), phi_pro_sample.shape[1]))
    patch_mmap_read = np.memmap(patch_cache.name, dtype='float32', mode='r', shape=(len(train_dataset), *patch_sample.shape[1:]))
    postadapt_phi_pro = torch.from_numpy(phi_pro_mmap_read).cpu()
    postadapt_patches = torch.from_numpy(patch_mmap_read).cpu()

    print("Visualizing post-adaptation clusters...")
    postadapt_plot_path = os.path.join(args.preadapt_bag, "post_adaptation_clusters.png")
    visualize_clusters(postadapt_phi_pro, postadapt_labels, save_path=postadapt_plot_path)
    print(f"Saved cluster visualization to {postadapt_plot_path}")

    print("Rendering and saving post-adaptation patches...")
    postadapt_patch_dir = os.path.join(args.preadapt_bag, "post_adaptation_patches")
    render_and_save_cluster_patches(postadapt_patches, postadapt_labels, postadapt_patch_dir)

    del phi_pro_mmap, patch_mmap, phi_pro_mmap_read, patch_mmap_read, postadapt_phi_pro, postadapt_patches
    os.unlink(phi_pro_cache.name)
    os.unlink(patch_cache.name)
    gc.collect()
    print(f"RAM after cleanup: {psutil.virtual_memory().used / 1024**2:.2f} MB")

def save_models(model, args):
    print("Saving adapted models...")
    save_dir = os.path.join(args.preadapt_bag, "models")
    model.save_adapted_models(save_dir)
    print("Training completed successfully!")

if __name__ == "__main__":
    args = parse_arguments()
    device = setup_device()
    preadapt_dataset, adapt_dataset, n_clusters = load_datasets(args)
    model = initialize_model(args, device)
    preadapt_loader, adapt_loader = create_dataloaders(preadapt_dataset, adapt_dataset, args.batch_size)

    preadapt_cache_dir = os.path.join(args.preadapt_bag, "cache")
    preadapt_data = extract_preadapt_features(model, preadapt_loader, preadapt_cache_dir, device)
    max_distance_threshold = compute_distance_threshold(preadapt_data[0], preadapt_data[2], preadapt_cache_dir, args)

    print("Visualizing pre-adaptation clusters...")
    preadapt_plot_path = os.path.join(args.preadapt_bag, "pre_adaptation_clusters.png")
    adapt_patches, adapt_inertial, adapt_phi_pro = extract_adapt_features(model, adapt_loader, preadapt_cache_dir)
    visualize_clusters(preadapt_data[0], preadapt_data[2], adapt_phi_pro=adapt_phi_pro, save_path=preadapt_plot_path)

    adapt_data_file = extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, preadapt_data, max_distance_threshold, args)
    train_loader, val_loader = create_adapt_dataset(adapt_data_file, args)  # Updated call
    
    retrain_model(model, train_loader, val_loader, args.epochs)
    phi_pro_output, patch_output, postadapt_labels_list = extract_postadapt_features(model, train_loader.dataset, args)
    
    os.remove(adapt_data_file)
    
    visualize_and_render(phi_pro_output, patch_output, postadapt_labels_list, train_loader.dataset, args, n_clusters)
    save_models(model, args)