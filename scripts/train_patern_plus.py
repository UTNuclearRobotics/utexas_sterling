import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader, random_split
from models import VisualEncoderModel, ProprioceptionModel, UtilityFuncVisual, UtilityFuncProprioceptive, CostNet
import pickle
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

    def extrapolate_preferences(self, adaptation_inertial, preadapt_phi_pro, preadapt_preferences, n_clusters, max_distance_threshold):
        # Extract features from adaptation inertial data
        adapt_phi_pro = self.extract_proprioceptive_features(adaptation_inertial)
        n_samples = adapt_phi_pro.shape[0]
        if n_samples < 1:
            raise ValueError("Adaptation-set has no samples to cluster.")
        n_clusters = min(n_clusters, n_samples)
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        # Use standard KMeans to cluster adaptation data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        adapt_clusters = kmeans.fit_predict(adapt_phi_pro.cpu().numpy())  # Shape: [n_samples]
        adapt_cluster_centers = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)  # Shape: [n_clusters, n_features]

        # Move preadaptation data to the correct device
        preadapt_phi_pro = preadapt_phi_pro.to(self.device)
        preadapt_prefs = preadapt_preferences.to(self.device)

        # Compute distances from cluster centers to preadaptation samples
        distances = torch.cdist(adapt_cluster_centers, preadapt_phi_pro)  # Shape: [n_clusters, n_preadapt_samples]
        min_distances, nearest_indices = distances.min(dim=1)             # Shape: [n_clusters], [n_clusters]

        # Select the nearest preference for each cluster
        nearest_prefs = preadapt_prefs[nearest_indices]                   # Shape: [n_clusters, n_prefs]
        
        # Reduce to a scalar preference per cluster (e.g., take the first column)
        if nearest_prefs.shape[1] > 1:
            cluster_prefs = nearest_prefs[:, 0]  # Shape: [n_clusters], select first preference column
        else:
            cluster_prefs = nearest_prefs.squeeze()  # Shape: [n_clusters]

        # Optional: Apply distance-based interpolation
        distance_factor = 1.0 - torch.exp(-min_distances / max_distance_threshold)  # Shape: [n_clusters]
        cluster_prefs = cluster_prefs * (1 - distance_factor) + 255.0 * distance_factor  # Shape: [n_clusters]
        cluster_prefs = torch.clamp(cluster_prefs, 0, 255)

        # Assign each sample the preference of its cluster
        extrapolated_prefs = cluster_prefs[adapt_clusters]  # Shape: [n_samples]

        return extrapolated_prefs

    def retrain_visual_components(self, train_loader, val_loader, optimizer, scheduler, epochs):
        for epoch in range(epochs):
            self.train()
            total_train_loss = 0
            for batch in train_loader:
                patches, inertial, terrain_labels, preferences = [x.to(self.device) for x in batch]
                preferences = preferences.float()

                pref_min, pref_max = preferences.min(), preferences.max()
                scaled_preferences = ((preferences - pref_min) / (pref_max - pref_min + 1e-6)) * 255.0
                phi_vis, _, uvis_pred, _, final_cost = self.forward(patches, inertial)
                uvis_pred = (uvis_pred - uvis_pred.min()) / (uvis_pred.max() - uvis_pred.min() + 1e-6) * 255.0

                # Calculate indices for this specific batch
                batch_size = len(terrain_labels)
                labels_expanded = terrain_labels.unsqueeze(1)
                pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                neg_mask = (labels_expanded != labels_expanded.t())

                # Ensure we have valid indices for triplet loss
                pos_indices = torch.where(pos_mask)[1]
                neg_indices = torch.where(neg_mask)[1]
                
                # Ensure indices match batch size
                if len(pos_indices) < batch_size or len(neg_indices) < batch_size:
                    # If we don't have enough positive/negative pairs, use random sampling with replacement
                    pos_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                    neg_indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                else:
                    # Randomly select batch_size indices
                    pos_indices = pos_indices[torch.randperm(len(pos_indices), device=self.device)[:batch_size]]
                    neg_indices = neg_indices[torch.randperm(len(neg_indices), device=self.device)[:batch_size]]

                # Debug shapes
                # print(f"phi_vis: {phi_vis.shape}, pos: {phi_vis[pos_indices].shape}, neg: {phi_vis[neg_indices].shape}")
                
                vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
                pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
                pref_diff = scaled_preferences.unsqueeze(1) - scaled_preferences.unsqueeze(0)
                ranking_loss = F.relu(1.0 - (pred_diff / 255.0)[pref_diff > 0]).mean()
                cost_loss = F.mse_loss(final_cost, scaled_preferences)
                total_loss = vis_loss + 0.5 * ranking_loss + cost_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += total_loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            # Validation loop
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    patches, inertial, terrain_labels, preferences = [x.to(self.device) for x in batch]
                    preferences = preferences.float()
                    scaled_preferences = ((preferences - preferences.min()) / (preferences.max() - preferences.min() + 1e-6)) * 255.0
                    phi_vis, _, uvis_pred, _, final_cost = self.forward(patches, inertial)

                    # Recalculate indices for validation batch
                    batch_size = len(terrain_labels)
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
                    ranking_loss = F.relu(1.0 - (pred_diff / 255.0)[pref_diff > 0]).mean()
                    cost_loss = F.mse_loss(final_cost, scaled_preferences)
                    total_val_loss += (vis_loss + 0.5 * ranking_loss + cost_loss).item()
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
    cluster_indices = [np.where(labels == label)[0].tolist() for label in unique_labels]
    rendered_clusters = renderer.render_clusters(cluster_indices, patches.cpu().numpy())
    os.makedirs(save_dir, exist_ok=True)
    for cluster_id, cluster_patches in enumerate(rendered_clusters):
        if cluster_patches:
            grid_image = renderer.image_grid(cluster_patches)
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_{cluster_id}.png"), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preference Extrapolation Training for PATERN")
    parser.add_argument("-preadapt_bag", "-pb", type=str, required=True)
    parser.add_argument("-adapt_bag", "-ab", type=str, required=True)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-val_split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-adaptation data (labeled) from HDF5
    print("Loading pre-adaptation data...")
    preadapt_h5_path = os.path.join(args.preadapt_bag, "clusters", "labeled_data.h5")
    preadapt_dataset = TerrainDataset(labeled_dataset=preadapt_h5_path, transform=None)
    unique_labels = set(preadapt_dataset.terrain_labels)
    n_clusters = len(unique_labels)
    print(f"Found {n_clusters} unique clusters in pre-adaptation data")
    if n_clusters == 0:
        raise ValueError("No unique terrain labels found in pre-adaptation data.")

    # Load adaptation data (unlabeled)
    print("Loading adaptation data...")
    adapt_vicreg_path = load_bag_h5(args.adapt_bag, "vicreg")
    adapt_synced_path = load_bag_h5(args.adapt_bag, "synced")
    adapt_dataset = TerrainDataset(
        synced_h5_path=adapt_synced_path,
        vicreg_h5_path=adapt_vicreg_path,
        transform=None,
        train=True
    )
    print(f"Adaptation dataset size: {len(adapt_dataset)}")

    # Initialize model
    print("Initializing model...")
    models_dir = os.path.join(args.preadapt_bag, "models")
    model = PaternAdaptation(device=device, pretrained_weights_path=models_dir, latent_size=128).to(device)

    # DataLoaders
    print("Creating DataLoaders...")
    preadapt_loader = DataLoader(preadapt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    # Extract pre-adaptation features incrementally with caching
    preadapt_cache_dir = os.path.join(args.preadapt_bag, "cache")
    os.makedirs(preadapt_cache_dir, exist_ok=True)
    preadapt_phi_pro_path = os.path.join(preadapt_cache_dir, "preadapt_phi_pro.pt")
    preadapt_prefs_path = os.path.join(preadapt_cache_dir, "preadapt_prefs.pt")
    preadapt_labels_path = os.path.join(preadapt_cache_dir, "preadapt_labels.npy")

    if not os.path.exists(preadapt_phi_pro_path):
        print("Extracting pre-adaptation features...")
        preadapt_phi_pro, preadapt_prefs = None, None
        preadapt_labels_list = []
        for i, batch in enumerate(preadapt_loader):
            _, inertial, terrain_labels, preferences = batch
            phi_pro = model.extract_proprioceptive_features(inertial)
            preferences = preferences.to(device)
            if preadapt_phi_pro is None:
                preadapt_phi_pro = phi_pro
                preadapt_prefs = preferences
            else:
                preadapt_phi_pro = torch.cat([preadapt_phi_pro, phi_pro])
                preadapt_prefs = torch.cat([preadapt_prefs, preferences])
            preadapt_labels_list.extend(terrain_labels)
            del phi_pro, preferences
            torch.cuda.empty_cache() if device.type == "cuda" else None
            if i % 10 == 0:
                print(f"Processed {i * args.batch_size} pre-adaptation samples, RAM usage: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB" if device.type == "cuda" else "CPU mode")
        preadapt_labels = np.array([hash(label) % n_clusters for label in preadapt_labels_list])
        print("Saving pre-adaptation features to cache...")
        torch.save(preadapt_phi_pro, preadapt_phi_pro_path)
        torch.save(preadapt_prefs, preadapt_prefs_path)
        np.save(preadapt_labels_path, preadapt_labels)
    else:
        print("Loading pre-adaptation features from cache...")
        preadapt_phi_pro = torch.load(preadapt_phi_pro_path, map_location=device)
        preadapt_prefs = torch.load(preadapt_prefs_path, map_location=device)
        preadapt_labels = np.load(preadapt_labels_path)
    print(f"Preadaptation features shape: {preadapt_phi_pro.shape}")

    print("Computing distance threshold...")
    batch_size = 512
    distances_cache_dir = os.path.join(preadapt_cache_dir, "distances")
    os.makedirs(distances_cache_dir, exist_ok=True)
    distance_files = []

    for i in range(0, len(preadapt_phi_pro), batch_size):
        batch_phi = preadapt_phi_pro[i:i + batch_size].cpu().numpy()
        # Compute distances only for this batch against the full set incrementally
        batch_dist_file = os.path.join(distances_cache_dir, f"batch_dist_{i}.npy")
        if not os.path.exists(batch_dist_file):
            batch_dist = euclidean_distances(batch_phi, preadapt_phi_pro.cpu().numpy())
            np.save(batch_dist_file, batch_dist)
        distance_files.append(batch_dist_file)
        del batch_phi
        if i % 1000 == 0:
            print(f"Processed {i} samples for distance calculation")

    # Process distances from disk to compute intra/inter distances incrementally
    print("Processing distances to compute intra/inter means...")
    n_samples = len(preadapt_labels)
    batch_size = 512

    # Initialize running sums and counts for means
    intra_sum = 0.0
    intra_count = 0
    inter_sum = 0.0
    inter_count = 0

    # Pre-compute label mask once (if memory allows)
    # If n_samples is too large, we'll stick to batch-wise comparison
    if n_samples * n_samples < 1e9:  # Arbitrary threshold (~30GB for bool array)
        label_mask = preadapt_labels[:, None] == preadapt_labels[None, :]  # Shape: (n_samples, n_samples)
        intra_mask = label_mask & ~np.eye(n_samples, dtype=bool)  # Exclude self-comparisons
        inter_mask = ~label_mask
    else:
        label_mask = None

    for i, dist_file in enumerate(distance_files):
        batch_dist = np.load(dist_file, mmap_mode='r')  # Shape: (batch_size, n_samples)
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_labels = preadapt_labels[start_idx:end_idx]

        if label_mask is not None:
            # Vectorized approach using pre-computed mask
            batch_intra_mask = intra_mask[start_idx:end_idx, :]
            batch_inter_mask = inter_mask[start_idx:end_idx, :]
            
            intra_sum += np.sum(batch_dist * batch_intra_mask)
            intra_count += np.sum(batch_intra_mask)
            inter_sum += np.sum(batch_dist * batch_inter_mask)
            inter_count += np.sum(batch_inter_mask)
        else:
            # Batch-wise vectorized approach
            batch_label_mask = batch_labels[:, None] == preadapt_labels[None, :]  # Shape: (batch_size, n_samples)
            batch_intra_mask = batch_label_mask & ~np.eye(batch_dist.shape[0], n_samples, dtype=bool)
            batch_inter_mask = ~batch_label_mask
            
            intra_sum += np.sum(batch_dist * batch_intra_mask)
            intra_count += np.sum(batch_intra_mask)
            inter_sum += np.sum(batch_dist * batch_inter_mask)
            inter_count += np.sum(batch_inter_mask)

        del batch_dist
        if i % 10 == 0:
            print(f"Processed batch {i}, Intra count: {intra_count}, Inter count: {inter_count}")

    # Compute averages
    avg_intra = intra_sum / intra_count if intra_count > 0 else 0.0
    avg_inter = inter_sum / inter_count if inter_count > 0 else 0.0
    max_distance_threshold = (avg_intra + avg_inter) / 2 if avg_intra > 0 and avg_inter > 0 else 5.0

    print(f"Avg Intra-cluster Distance: {avg_intra:.4f}, Avg Inter-cluster Distance: {avg_inter:.4f}")
    print(f"Computed Max Distance Threshold: {max_distance_threshold:.4f}")

    # Clean up temporary files (optional)
    for dist_file in distance_files:
        os.remove(dist_file)
    os.rmdir(distances_cache_dir)


    device = torch.device("cpu")
    # Extract adaptation features incrementally with CPU-based caching
    adapt_patches_path = os.path.join(preadapt_cache_dir, "adapt_patches.pt")
    adapt_inertial_path = os.path.join(preadapt_cache_dir, "adapt_inertial.pt")

    # Define adapt_patches and adapt_inertial outside the conditional
    adapt_patches = None
    adapt_inertial = None

    if not os.path.exists(adapt_patches_path) or not os.path.exists(adapt_inertial_path):
        adapt_patches_list, adapt_inertial_list = process_batch_to_disk(adapt_loader, preadapt_cache_dir)
        concatenate_cached_files(adapt_patches_list, adapt_patches_path)
        concatenate_cached_files(adapt_inertial_list, adapt_inertial_path)
        # Load the newly created files
        adapt_patches = torch.load(adapt_patches_path, map_location=device)
        adapt_inertial = torch.load(adapt_inertial_path, map_location=device)
    else:
        print("Loading adaptation features from cache...")
        adapt_patches = torch.load(adapt_patches_path, map_location=device)
        adapt_inertial = torch.load(adapt_inertial_path, map_location=device)

    # Now adapt_inertial and adapt_patches are defined regardless of the path taken
    print("Extracting proprioceptive features for adaptation data...")
    adapt_phi_pro = model.extract_proprioceptive_features(adapt_inertial)

    # Visualize pre-adaptation clusters
    print("Visualizing pre-adaptation clusters...")
    preadapt_plot_path = os.path.join(args.preadapt_bag, "pre_adaptation_clusters.png")
    visualize_clusters(preadapt_phi_pro, preadapt_labels, adapt_phi_pro=adapt_phi_pro, save_path=preadapt_plot_path)

    # Extrapolate preferences
    print("Extrapolating preferences...")
    extrapolated_prefs = model.extrapolate_preferences(adapt_inertial, preadapt_phi_pro, preadapt_prefs, n_clusters, max_distance_threshold)

    # Precompute tensors on CPU and convert to list for faster access
    adapt_patches_cpu = adapt_patches.cpu()
    adapt_inertial_cpu = adapt_inertial.cpu()
    prefs_list = extrapolated_prefs.cpu().tolist()  # Convert to list once

    # Create labeled adaptation data incrementally with disk caching using HDF5
    print("Creating labeled adaptation data...")
    adapt_data_dir = os.path.join(args.preadapt_bag, "adapt_data_cache")
    os.makedirs(adapt_data_dir, exist_ok=True)
    adapt_data_file = os.path.join(adapt_data_dir, "adapt_data.h5")  # Single HDF5 file
    adapt_data_files = []  # Keep this for compatibility

    # Batch processing with a reasonable batch size (e.g., 1000)
    batch_size = 1000
    with h5py.File(adapt_data_file, 'w') as f:
        for start_idx in range(0, len(adapt_patches), batch_size):
            end_idx = min(start_idx + batch_size, len(adapt_patches))
            batch_range = range(start_idx, end_idx)
            
            # Save batch data directly to HDF5
            for i in batch_range:
                group = f.create_group(f"entry_{i}")
                group.create_dataset("patch", data=adapt_patches_cpu[i].numpy(), compression="gzip")
                group.create_dataset("inertial", data=adapt_inertial_cpu[i].numpy(), compression="gzip")
                group.attrs["terrain_label"] = "none"  # Placeholder for None
                group.attrs["preference"] = prefs_list[i]
            
            # Progress update
            print(f"Processed {end_idx} adaptation data entries, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    # Store the single HDF5 file path in adapt_data_files
    adapt_data_files = [adapt_data_file]

    # Free memory after saving to HDF5
    del adapt_patches_cpu, adapt_inertial_cpu, prefs_list
    del adapt_phi_pro, preadapt_phi_pro, preadapt_prefs  # Clear intermediate results
    gc.collect()  # Force garbage collection
    print(f"RAM usage after clearing adaptation data: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    # Aggregate and split datasets
    print("Aggregating and splitting datasets...")
    preadapt_data_list = list(preadapt_dataset)

    # Load adaptation data from HDF5 incrementally to minimize memory
    adapt_data = []
    label_map = {}  # Dictionary to map string labels to integers

    # First pass: Build label map from preadaptation data (if available) to ensure consistency
    print("Building label map from preadaptation data...")
    for item in preadapt_data_list:
        if isinstance(item, tuple) and len(item) == 4:
            _, _, terrain_label, _ = item
            if isinstance(terrain_label, str) and terrain_label not in label_map:
                label_map[terrain_label] = len(label_map)
        elif isinstance(item, dict) and "terrain_label" in item:
            terrain_label = item["terrain_label"]
            if isinstance(terrain_label, str) and terrain_label not in label_map:
                label_map[terrain_label] = len(label_map)

    # Load and normalize adaptation data
    print("Loading and normalizing adaptation data...")
    with h5py.File(adapt_data_file, 'r') as f:
        total_entries = len(adapt_patches)
        for i in range(total_entries):
            group = f[f"entry_{i}"]
            terrain_label = group.attrs["terrain_label"]
            
            # Normalize terrain_label
            if terrain_label == "none":
                terrain_label = -1  # Sentinel for None/unlabeled
            elif isinstance(terrain_label, str):
                if terrain_label not in label_map:
                    label_map[terrain_label] = len(label_map)  # Add new labels dynamically
                terrain_label = label_map[terrain_label]
            # If terrain_label is already an integer, keep it as-is
            
            entry = {
                "patch": torch.from_numpy(group["patch"][()]),
                "inertial": torch.from_numpy(group["inertial"][()]),
                "terrain_label": terrain_label,
                "preference": group.attrs["preference"]
            }
            adapt_data.append(entry)
            
            # Optionally clear memory periodically
            if (i + 1) % batch_size == 0:
                print(f"Loaded {i + 1}/{total_entries} adaptation entries, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    del adapt_patches, adapt_inertial, extrapolated_prefs  # Clear original tensors if still in scope

    # Ensure preadapt_data_list matches the expected dictionary format and normalize labels
    print("Normalizing preadaptation data...")
    normalized_preadapt_data = []
    for item in preadapt_data_list:
        if isinstance(item, dict) and all(k in item for k in ["patch", "inertial", "terrain_label", "preference"]):
            terrain_label = item["terrain_label"]
            if isinstance(terrain_label, str):
                terrain_label = label_map[terrain_label]  # Use pre-built mapping
            elif terrain_label is None:
                terrain_label = -1
            # If already an integer, keep it
            normalized_item = {
                "patch": item["patch"],
                "inertial": item["inertial"],
                "terrain_label": terrain_label,
                "preference": item["preference"]
            }
            normalized_preadapt_data.append(normalized_item)
        elif isinstance(item, tuple) and len(item) == 4:
            # Convert tuple to dictionary
            patch, inertial, terrain_label, preference = item
            if isinstance(terrain_label, str):
                terrain_label = label_map[terrain_label]  # Use pre-built mapping
            elif terrain_label is None:
                terrain_label = -1
            # If already an integer, keep it
            normalized_item = {
                "patch": patch,
                "inertial": inertial,
                "terrain_label": terrain_label,
                "preference": preference
            }
            normalized_preadapt_data.append(normalized_item)
        else:
            raise ValueError(f"Unexpected preadapt_data_list item format: {type(item)}, content: {item}")

    # Combine data
    print("Combining normalized data...")
    aggregated_data = normalized_preadapt_data + adapt_data

    # Free memory before creating TerrainDataset
    del preadapt_data_list, adapt_data, normalized_preadapt_data
    gc.collect()  # Force garbage collection again
    print(f"RAM usage before TerrainDataset: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    # Create dataset
    try:
        aggregated_dataset = TerrainDataset(labeled_dataset=aggregated_data, transform=None)
        print(f"Successfully created TerrainDataset with {len(aggregated_dataset)} samples")
    except ValueError as e:
        print(f"Error creating TerrainDataset: {e}")
        raise

    val_size = int(args.val_split * len(aggregated_dataset))
    train_size = len(aggregated_dataset) - val_size
    train_dataset, val_dataset = random_split(aggregated_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=custom_collate)

    for f in adapt_data_files:
        os.remove(f)
    del aggregated_data
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Retrain
    print("Retraining model...")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    model.retrain_visual_components(train_loader, val_loader, optimizer, scheduler, args.epochs)

    # Post-adaptation extraction incrementally with disk caching
    print("Extracting post-adaptation features...")
    postadapt_cache_dir = os.path.join(args.preadapt_bag, "postadapt_cache")
    os.makedirs(postadapt_cache_dir, exist_ok=True)
    aggregated_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    phi_pro_files = []
    patch_files = []
    postadapt_labels_list = []

    for i, batch in enumerate(aggregated_loader):
        patches, inertial, terrain_labels, _ = batch
        phi_pro = model.extract_proprioceptive_features(inertial.to(device))
        patches = patches.to(device)
        
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

    postadapt_phi_pro = torch.cat([torch.load(f, map_location=device) for f in phi_pro_files])
    postadapt_patches = torch.cat([torch.load(f, map_location=device) for f in patch_files])
    postadapt_labels = np.array([-1 if label is None else hash(label) % n_clusters for label in postadapt_labels_list])

    for f in phi_pro_files + patch_files:
        os.remove(f)

    # Visualize and save post-adaptation
    print("Visualizing post-adaptation clusters...")
    postadapt_plot_path = os.path.join(args.preadapt_bag, "post_adaptation_clusters.png")
    visualize_clusters(postadapt_phi_pro, postadapt_labels, save_path=postadapt_plot_path)

    print("Rendering and saving post-adaptation patches...")
    postadapt_patch_dir = os.path.join(args.preadapt_bag, "post_adaptation_patches")
    render_and_save_cluster_patches(postadapt_patches, postadapt_labels, postadapt_patch_dir)

    # Save models
    print("Saving adapted models...")
    save_dir = os.path.join(args.preadapt_bag, "models")
    model.save_adapted_models(save_dir)
    print("Training completed successfully!")