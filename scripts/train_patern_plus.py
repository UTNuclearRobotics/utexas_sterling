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
        adapt_phi_pro = self.extract_proprioceptive_features(adaptation_inertial)
        n_samples = adapt_phi_pro.shape[0]
        if n_samples < 1:
            raise ValueError("Adaptation-set has no samples to cluster.")
        n_clusters = min(n_clusters, n_samples)
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        # Use standard KMeans instead of MiniBatchKMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        adapt_clusters = kmeans.fit_predict(adapt_phi_pro.cpu().numpy())
        adapt_cluster_centers = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)

        preadapt_phi_pro = preadapt_phi_pro.to(self.device)
        preadapt_prefs = preadapt_preferences.to(self.device)
        distances = torch.cdist(adapt_cluster_centers, preadapt_phi_pro)
        min_distances, nearest_indices = distances.min(dim=1)
        nearest_prefs = preadapt_prefs[nearest_indices]

        distance_factor = 1.0 - torch.exp(-min_distances / max_distance_threshold)
        cluster_prefs = nearest_prefs * (1 - distance_factor) + 255.0 * distance_factor
        cluster_prefs = torch.clamp(cluster_prefs, 0, 255)
        extrapolated_prefs = cluster_prefs[adapt_clusters]
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

                terrain_labels_tensor = torch.tensor([label if label is not None else -1 for label in terrain_labels], 
                                                    device=self.device, dtype=torch.long)
                batch_size = len(terrain_labels)
                labels_expanded = terrain_labels_tensor.unsqueeze(1)
                pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                neg_mask = (labels_expanded != labels_expanded.t())

                pos_indices = torch.where(pos_mask)[1]
                neg_indices = torch.where(neg_mask)[1]
                if len(pos_indices) > batch_size:
                    pos_indices = pos_indices[torch.randperm(len(pos_indices), device=self.device)[:batch_size]]
                    neg_indices = neg_indices[torch.randperm(len(neg_indices), device=self.device)[:batch_size]]
                else:
                    pos_indices = torch.arange(batch_size, device=self.device)
                    neg_indices = torch.arange(batch_size, device=self.device)

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

            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    patches, inertial, terrain_labels, preferences = [x.to(self.device) for x in batch]
                    preferences = preferences.float()
                    scaled_preferences = ((preferences - preferences.min()) / (preferences.max() - preferences.min() + 1e-6)) * 255.0
                    phi_vis, _, uvis_pred, _, final_cost = self.forward(patches, inertial)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preference Extrapolation Training for PATERN")
    parser.add_argument("-bag", "-b", type=str, required=True)
    parser.add_argument("-adapt_bag", type=str, required=True)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-val_split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-adaptation data (labeled)
    with open(os.path.join(args.bag, "clusters", "labeled_dataset.pkl"), 'rb') as f:
        preadapt_data = pickle.load(f)
    unique_labels = set(data["terrain_label"] for data in preadapt_data if "terrain_label" in data)
    n_clusters = len(unique_labels)
    if n_clusters == 0:
        raise ValueError("No unique terrain labels found in pre-adaptation data.")
    preadapt_dataset = TerrainDataset(labeled_data=preadapt_data, transform=None)

    # Load adaptation data (unlabeled)
    with open(os.path.join(args.adapt_bag, "clusters", "labeled_dataset.pkl"), 'rb') as f:
        adapt_data = pickle.load(f)
    # Ensure adapt_data has inertial and patches, but no labels
    for data in adapt_data:
        if "terrain_label" not in data:
            data["terrain_label"] = None  # Add placeholder for compatibility
    adapt_dataset = TerrainDataset(labeled_data=adapt_data, transform=None)

    # Initialize model
    models_dir = os.path.join(args.bag, "models")
    model = PaternAdaptation(device=device, pretrained_weights_path=models_dir, latent_size=128).to(device)

    # DataLoaders
    preadapt_loader = DataLoader(preadapt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Extract pre-adaptation features
    preadapt_phi_pro_list, preadapt_prefs_list, preadapt_labels_list = [], [], []
    for batch in preadapt_loader:
        _, inertial, terrain_labels, preferences = batch
        phi_pro = model.extract_proprioceptive_features(inertial)
        preadapt_phi_pro_list.append(phi_pro)
        preadapt_prefs_list.append(preferences.to(device))
        preadapt_labels_list.extend(terrain_labels)
    preadapt_phi_pro = torch.cat(preadapt_phi_pro_list)
    preadapt_prefs = torch.cat(preadapt_prefs_list)
    preadapt_labels = np.array([hash(label) % n_clusters for label in preadapt_labels_list])

    # Compute distance threshold
    distances = euclidean_distances(preadapt_phi_pro.cpu().numpy())
    mask = (preadapt_labels[:, None] == preadapt_labels[None, :])
    intra_distances = distances[mask & ~np.eye(len(preadapt_labels), dtype=bool)]
    inter_distances = distances[~mask]
    avg_intra = np.mean(intra_distances) if intra_distances.size > 0 else 0.0
    avg_inter = np.mean(inter_distances) if inter_distances.size > 0 else 0.0
    max_distance_threshold = (avg_intra + avg_inter) / 2 if avg_intra > 0 and avg_inter > 0 else 5.0
    print(f"Avg Intra-cluster Distance: {avg_intra:.4f}, Avg Inter-cluster Distance: {avg_inter:.4f}")
    print(f"Computed Max Distance Threshold: {max_distance_threshold:.4f}")

    # Extract adaptation features
    adapt_inertial = torch.cat([batch[1] for batch in adapt_loader])
    adapt_phi_pro = model.extract_proprioceptive_features(adapt_inertial)

    # Visualize pre-adaptation clusters
    preadapt_plot_path = os.path.join(args.bag, "pre_adaptation_clusters.png")
    visualize_clusters(preadapt_phi_pro, preadapt_labels, adapt_phi_pro=adapt_phi_pro, save_path=preadapt_plot_path)

    # Extrapolate preferences using pre-adaptation n_clusters
    extrapolated_prefs = model.extrapolate_preferences(adapt_inertial, preadapt_phi_pro, preadapt_prefs, n_clusters, max_distance_threshold)

    # Update adapt_data with extrapolated preferences
    for i, data in enumerate(adapt_data):
        data["preference"] = extrapolated_prefs[i % len(extrapolated_prefs)].item()

    # Aggregate and split datasets
    aggregated_data = preadapt_data + adapt_data
    aggregated_dataset = TerrainDataset(labeled_data=aggregated_data, transform=None)
    val_size = int(args.val_split * len(aggregated_dataset))
    train_size = len(aggregated_dataset) - val_size
    train_dataset, val_dataset = random_split(aggregated_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Retrain
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    model.retrain_visual_components(train_loader, val_loader, optimizer, scheduler, args.epochs)

    # Post-adaptation extraction
    aggregated_loader = DataLoader(aggregated_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    postadapt_phi_pro_list, postadapt_patches_list, postadapt_labels_list = [], [], []
    for batch in aggregated_loader:
        patches, inertial, terrain_labels, _ = batch
        phi_pro = model.extract_proprioceptive_features(inertial)
        postadapt_phi_pro_list.append(phi_pro)
        postadapt_patches_list.append(patches.to(device))
        postadapt_labels_list.extend(terrain_labels)
    postadapt_phi_pro = torch.cat(postadapt_phi_pro_list)
    postadapt_patches = torch.cat(postadapt_patches_list)
    postadapt_labels = np.array([-1 if label is None else hash(label) % n_clusters for label in postadapt_labels_list])

    # Visualize and save post-adaptation
    postadapt_plot_path = os.path.join(args.adapt_bag, "post_adaptation_clusters.png")
    visualize_clusters(postadapt_phi_pro, postadapt_labels, save_path=postadapt_plot_path)
    postadapt_patch_dir = os.path.join(args.adapt_bag, "post_adaptation_patches")
    render_and_save_cluster_patches(postadapt_patches, postadapt_labels, postadapt_patch_dir)

    # Save models
    save_dir = os.path.join(args.adapt_bag, "models")
    model.save_adapted_models(save_dir)
