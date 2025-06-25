import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader, random_split
from models import VisualEncoderModel, ProprioceptionModel, UtilityFuncVisual, UtilityFuncProprioceptive, CostNet
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from cluster import PatchRenderer
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import load_bag_h5
import psutil
import h5py
import tempfile
import shutil
import gc
import gi
from gi.repository import GLib, Gtk, GdkPixbuf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import yaml

gi.require_version("Gtk", "4.0")

script_dir = os.path.dirname(os.path.realpath(__file__))

# GTK-based GUI for labeling outlier groups
class OutlierLabelUI(Gtk.Application):
    def __init__(self, patches=None, extrapolated_prefs=None, outlier_indices=None, adapt_phi_pro=None, save_dir=None, default_labels=None, preadapt_name_to_id=None):
        super().__init__(application_id="com.example.OutlierLabelUI")
        GLib.set_application_name("Outlier Label UI")
        self.patches = patches
        self.extrapolated_prefs = extrapolated_prefs
        self.outlier_indices = outlier_indices
        self.adapt_phi_pro = adapt_phi_pro
        self.save_dir = save_dir or os.path.join(script_dir, "outlier_patches")
        self.default_labels = default_labels
        self.preadapt_name_to_id = preadapt_name_to_id or {}
        self.user_labels_dict = {}
        self.user_cluster_ids_dict = {}
        self.user_prefs_dict = {}
        self.current_group = 0
        self.rendered_clusters = None
        self.cluster_indices = None
        self.base_cluster_id = max(int(v) for v in self.preadapt_name_to_id.values()) + 1 if self.preadapt_name_to_id else 0
        self.cluster_id_label = None  # Initialize here to avoid attribute error

    def do_activate(self):
        if self.patches is None or self.extrapolated_prefs is None or self.outlier_indices is None or self.adapt_phi_pro is None:
            window = Gtk.ApplicationWindow(application=self, title="Outlier Label UI - Error")
            window.set_default_size(400, 200)
            label = Gtk.Label(label="Error: OutlierLabelUI requires patches, preferences, indices, and phi_pro data.")
            label.set_margin_top(20)
            label.set_margin_bottom(20)
            label.set_margin_start(20)
            label.set_margin_end(20)
            window.set_child(label)
            window.present()
            return

        self.base_cluster_id = torch.max(self.default_labels).item() + 1 if self.default_labels is not None else 0

        outlier_indices_cpu = self.outlier_indices.cpu()
        outlier_phi_pro = self.adapt_phi_pro[outlier_indices_cpu].cpu().numpy()
        outlier_patches = self.patches[outlier_indices_cpu].cpu().numpy()
        outlier_prefs = self.extrapolated_prefs[outlier_indices_cpu].cpu().numpy()
        outlier_default_labels = self.default_labels[outlier_indices_cpu].cpu().numpy() if self.default_labels is not None else None
        
        max_possible_clusters = min(len(outlier_indices_cpu), 10)
        if max_possible_clusters <= 1:
            group_labels = np.zeros(len(outlier_indices_cpu), dtype=int)
        else:
            best_n_clusters = 1
            best_score = -1
            for n in range(2, min(max_possible_clusters + 1, len(outlier_indices_cpu) // 2 + 1)):
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(outlier_phi_pro)
                score = silhouette_score(outlier_phi_pro, labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            group_labels = kmeans.fit_predict(outlier_phi_pro)
            print(f"Automatically selected {best_n_clusters} clusters for outliers based on silhouette score: {best_score:.3f}")
        
        renderer = PatchRenderer()
        os.makedirs(self.save_dir, exist_ok=True)
        unique_groups = np.unique(group_labels)
        self.cluster_indices = [np.where(group_labels == group_id)[0].tolist() for group_id in unique_groups]
        self.rendered_clusters = renderer.render_clusters(self.cluster_indices, outlier_patches)
        self.outlier_prefs = outlier_prefs
        self.outlier_default_labels = outlier_default_labels
        
        self.show_next_group()

    # Helper function to convert NumPy array to GdkPixbuf
    def numpy_to_pixbuf(self, array):
        """Convert a NumPy array (H, W, 3) RGB to GdkPixbuf."""
        height, width, channels = array.shape
        if channels not in (3, 4):
            raise ValueError("Array must have 3 (RGB) or 4 (RGBA) channels")
        data = array.tobytes()
        rowstride = width * channels
        return GdkPixbuf.Pixbuf.new_from_data(
            data, GdkPixbuf.Colorspace.RGB, channels == 4, 8, width, height, rowstride
        )

    def show_next_group(self):
        if self.current_group >= len(self.rendered_clusters):
            self.quit()
            return
        
        cluster_patches = self.rendered_clusters[self.current_group]
        indices = self.cluster_indices[self.current_group]
        if not cluster_patches:
            self.current_group += 1
            self.show_next_group()
            return
        
        window = Gtk.ApplicationWindow(application=self, title=f"Outlier Group {self.current_group} ({len(cluster_patches)} patches)")
        window.set_default_size(1200, 800)
        
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        window.set_child(vbox)
        
        grid_image = PatchRenderer().image_grid(cluster_patches)
        pixbuf = self.numpy_to_pixbuf(grid_image)
        image_widget = Gtk.Image.new_from_pixbuf(pixbuf)
        
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_child(image_widget)
        scrolled_window.set_hexpand(True)
        scrolled_window.set_vexpand(True)
        vbox.append(scrolled_window)
        
        group_prefs = self.outlier_prefs[indices]
        mean_extrapolated_pref = np.mean(group_prefs)
        vbox.append(Gtk.Label(label=f"Mean Extrapolated Preference: {mean_extrapolated_pref:.4f}"))
        
        label_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        label_hbox.append(Gtk.Label(label="Label (string):"))
        label_entry = Gtk.Entry()
        label_entry.set_placeholder_text("Enter a descriptive label...")
        label_entry.set_text(f"outlier_group_{self.current_group}")
        label_hbox.append(label_entry)
        vbox.append(label_hbox)
        
        default_cluster_id = self.base_cluster_id + self.current_group
        self.cluster_id_label = Gtk.Label(label=f"Assigned Cluster ID: {default_cluster_id}")
        vbox.append(self.cluster_id_label)
        
        pref_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        pref_hbox.append(Gtk.Label(label="Preference:"))
        pref_entry = Gtk.Entry()
        pref_entry.set_text(f"{mean_extrapolated_pref:.4f}")
        pref_hbox.append(pref_entry)
        vbox.append(pref_hbox)
        
        submit_button = Gtk.Button(label="Submit")
        submit_button.connect("clicked", self.on_submit_clicked, window, label_entry, pref_entry, indices, default_cluster_id)
        vbox.append(submit_button)
        
        window.present()

    def on_submit_clicked(self, button, window, label_entry, pref_entry, indices, default_cluster_id):
        try:
            label = label_entry.get_text().strip()
            pref = float(pref_entry.get_text())
            
            if label in self.preadapt_name_to_id:
                assigned_cluster_id = self.preadapt_name_to_id[label]
                print(f"Matched '{label}' to pre-adaptation cluster ID {assigned_cluster_id}")
            else:
                assigned_cluster_id = default_cluster_id
                self.preadapt_name_to_id[label] = assigned_cluster_id  # Add new label to preadapt_name_to_id
                print(f"Assigned new cluster ID {assigned_cluster_id} to '{label}'")
            
            self.cluster_id_label.set_text(f"Assigned Cluster ID: {assigned_cluster_id}")
            
            for idx in indices:
                original_idx = self.outlier_indices[idx].item()
                self.user_labels_dict[original_idx] = label
                self.user_cluster_ids_dict[original_idx] = assigned_cluster_id
                self.user_prefs_dict[original_idx] = pref
            window.destroy()
            self.current_group += 1
            self.show_next_group()
        except ValueError:
            error_dialog = Gtk.MessageDialog(
                transient_for=window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Invalid input: Preference must be a number."
            )
            error_dialog.connect("response", lambda dialog, response: dialog.destroy())
            error_dialog.show()

    def get_results(self):
        user_labels = []
        user_cluster_ids = []
        user_prefs = []
        default_labels = self.default_labels.cpu().numpy() if self.default_labels is not None else np.zeros(len(self.outlier_indices), dtype=int)
        default_prefs = self.extrapolated_prefs.cpu().numpy()

        for i, idx in enumerate(self.outlier_indices):
            idx_item = idx.item()
            if idx_item in self.user_labels_dict:
                user_labels.append(self.user_labels_dict[idx_item])
                user_cluster_ids.append(self.user_cluster_ids_dict[idx_item])
                user_prefs.append(self.user_prefs_dict[idx_item])
            else:
                user_labels.append(f"unlabeled_outlier_{i}")
                user_cluster_ids.append(default_labels[i])
                user_prefs.append(default_prefs[i])
                print(f"Warning: Outlier index {idx_item} not labeled; using default cluster ID {default_labels[i]} and preference {default_prefs[i]}")
        
        device = self.extrapolated_prefs.device
        return (list(zip(user_labels, user_cluster_ids)),
                torch.tensor(user_prefs, dtype=torch.float, device=device))

class PaternAdaptation(nn.Module):
    def __init__(self, device, pretrained_weights_path, latent_size=128):
        super().__init__()
        self.device = device
        self.latent_size = latent_size
        self.models = nn.ModuleDict({
            "visual_encoder": VisualEncoderModel(latent_size=latent_size).to(device),
            "proprioceptive_encoder": ProprioceptionModel(latent_size=latent_size).to(device),
            "uvis": UtilityFuncVisual(latent_size=latent_size).to(device),
            "upro": UtilityFuncProprioceptive(latent_size=latent_size).to(device),
            "cost_head": CostNet(latent_size=latent_size).to(device)
        })

        weight_files = {
            "visual_encoder": "fvis.pt",
            "proprioceptive_encoder": "fpro.pt",
            "uvis": "uvis.pt",
            "upro": "upro.pt",
            "cost_head": "cost_head.pt"
        }
        for name, file in weight_files.items():
            path = os.path.join(pretrained_weights_path, file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {file}")
            state_dict = torch.load(path, weights_only=True, map_location=device)
            self.models[name].load_state_dict(state_dict)

        for param in self.models["proprioceptive_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["upro"].parameters():
            param.requires_grad = False

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

    def forward(self, patches, inertial=None):
        patches = patches.to(self.device)
        phi_vis = self.models["visual_encoder"](patches)
        uvis_pred = self.models["uvis"](phi_vis)
        phi_pro = self.models["proprioceptive_encoder"](inertial.float().to(self.device)) if inertial is not None else torch.zeros_like(phi_vis)
        upro_pred = self.models["upro"](phi_pro) if inertial is not None else torch.zeros_like(uvis_pred)
        final_cost = self.models["cost_head"](uvis_pred)
        return phi_vis, phi_pro, uvis_pred, upro_pred, final_cost

    def extract_proprioceptive_features(self, inertial_data):
        self.eval()
        with torch.no_grad():
            return self.models["proprioceptive_encoder"](inertial_data.to(self.device).float())

    def compute_preadapt_encodings(self, preadapt_loader, name_to_id):
        """
        Compute ground truth visual encodings and proprioceptive cluster centers.
        """
        self.eval()
        all_phi_vis, all_phi_pro, all_labels = [], [], []
        with torch.no_grad():
            for batch in preadapt_loader:
                patches, inertial, terrain_labels, _ = batch
                patches = patches.to(self.device)
                inertial = inertial.to(self.device)
                
                # Convert terrain_labels to numerical tensor using name_to_id
                if isinstance(terrain_labels, torch.Tensor):
                    terrain_labels = terrain_labels.to(self.device)
                else:
                    # Handle list of strings or single string
                    if isinstance(terrain_labels, (list, tuple)):
                        terrain_labels = [name_to_id[label] for label in terrain_labels]
                    else:
                        terrain_labels = [name_to_id[terrain_labels]]
                    terrain_labels = torch.tensor(terrain_labels, device=self.device, dtype=torch.long)
                
                all_phi_vis.append(self.models["visual_encoder"](patches))
                all_phi_pro.append(self.extract_proprioceptive_features(inertial))
                all_labels.append(terrain_labels)
            
        all_phi_vis = torch.cat(all_phi_vis)
        all_phi_pro = torch.cat(all_phi_pro)
        all_labels = torch.cat(all_labels)
        
        unique_labels, inverse = torch.unique(all_labels, return_inverse=True)
        n_clusters = len(unique_labels)
        if n_clusters < 1:
            raise ValueError("No clusters found")
        
        cluster_phi_vis = {}
        cluster_centers = torch.zeros(n_clusters, self.latent_size, device=self.device)
        
        for i, label in enumerate(unique_labels):
            mask = inverse == i
            cluster_phi_vis[label.item()] = all_phi_vis[mask].mean(dim=0)
            cluster_centers[i] = all_phi_pro[mask].mean(dim=0)
        
        return cluster_phi_vis, cluster_centers

    def compute_triplet_loss(self, phi_vis, phi_pro, terrain_labels, cluster_phi_vis, cluster_centers, max_distance_threshold):
        """
        Compute triplet loss with ground truth alignment for proprioceptively similar patches.
        """
        batch_size = phi_vis.shape[0]
        max_existing_label = max(cluster_phi_vis.keys()) if cluster_phi_vis else -1
        
        distances = torch.cdist(phi_pro, cluster_centers)
        min_distances, nearest_indices = distances.min(dim=1)
        within_threshold = min_distances <= max_distance_threshold
        
        cluster_labels = torch.tensor(list(cluster_phi_vis.keys()), device=self.device)
        valid_mask = (within_threshold & (terrain_labels <= max_existing_label)).to(torch.bool)
        
        if valid_mask.any():
            anchors = torch.stack([cluster_phi_vis[cluster_labels[nearest_indices[i]].item()] for i in range(batch_size) if valid_mask[i]])
            positives = phi_vis[valid_mask]
            neg_indices = (nearest_indices[valid_mask] + 1) % len(cluster_labels)
            negatives = torch.stack([cluster_phi_vis[cluster_labels[neg_indices[i]].item()] for i in range(len(anchors))])
            triplet_loss = self.triplet_loss(anchors, positives, negatives)
            triplet_loss = triplet_loss * valid_mask.sum() / batch_size
        else:
            pos_mask = (terrain_labels.unsqueeze(1) == terrain_labels.unsqueeze(0)) & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
            neg_mask = ~pos_mask
            if pos_mask.any() and neg_mask.any():
                pos_indices = pos_mask.nonzero(as_tuple=True)[1].reshape(batch_size, -1)[:, 0]
                neg_indices = neg_mask.nonzero(as_tuple=True)[1].reshape(batch_size, -1)[:, 0]
                triplet_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
            else:
                print("Warning: No valid triplets found in batch. Returning zero loss.")
                triplet_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return triplet_loss

    def compute_contrastive_loss(self, phi_vis, terrain_labels, margin=1.0):
        """
        Compute contrastive loss for novel terrain clusters.
        """
        batch_size = phi_vis.shape[0]
        distances = torch.cdist(phi_vis, phi_vis)
        pos_mask = (terrain_labels.unsqueeze(1) == terrain_labels.unsqueeze(0)) & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
        neg_mask = ~pos_mask
        
        loss = 0.0
        if pos_mask.any():
            loss += distances[pos_mask].mean()
        if neg_mask.any():
            loss += F.relu(margin - distances[neg_mask]).mean()
        
        return loss if loss > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)

    def extrapolate_preferences(self, adaptation_inertial, preadapt_phi_pro, preadapt_preferences, preadapt_labels, max_distance_threshold):
        # Extract features and ensure device consistency
        adapt_phi_pro = self.extract_proprioceptive_features(adaptation_inertial).to(self.device)
        n_samples = adapt_phi_pro.shape[0]
        if n_samples < 1:
            raise ValueError("Empty adaptation set")
        preadapt_phi_pro, preadapt_prefs, preadapt_labels = [
            x.to(self.device) for x in [preadapt_phi_pro, preadapt_preferences, preadapt_labels]
        ]

        # Initialize cluster statistics
        unique_clusters, inverse = torch.unique(preadapt_labels, return_inverse=True)
        n_clusters = len(unique_clusters)
        if n_clusters < 1:
            raise ValueError("No clusters found")
        cluster_centers = torch.zeros((n_clusters, preadapt_phi_pro.shape[1]), device=self.device)
        cluster_prefs = torch.zeros(n_clusters, device=self.device)
        cluster_std = torch.zeros(n_clusters, device=self.device)

        # Compute cluster statistics
        for i in range(n_clusters):
            mask = inverse == i
            points = preadapt_phi_pro[mask]
            cluster_centers[i] = points.mean(dim=0)
            cluster_prefs[i] = preadapt_prefs[mask].mean()
            std = torch.std(points, dim=0, unbiased=False) if points.shape[0] > 1 else torch.tensor(0.0, device=self.device)
            cluster_std[i] = std.mean() if std.numel() > 0 else 0.0

        # Compute distances and find nearest clusters
        distances = torch.cdist(adapt_phi_pro, cluster_centers)
        min_distances, nearest_indices = distances.min(dim=1)
        dynamic_thresholds = cluster_std[nearest_indices] * 2.0
        max_threshold = torch.tensor(max_distance_threshold, device=self.device)
        within_threshold = min_distances <= torch.minimum(dynamic_thresholds, max_threshold)

        extrapolated_prefs = torch.zeros(n_samples, device=self.device)
        extrapolated_labels = unique_clusters[nearest_indices]
        extrapolated_prefs[within_threshold] = cluster_prefs[nearest_indices[within_threshold]]

        # Handle points outside threshold with inverse distance weighting
        outside_indices = torch.where(~within_threshold)[0]
        if outside_indices.numel() > 0:
            nearest_two = distances[outside_indices].topk(2, largest=False, dim=1)
            weights = 1.0 / (nearest_two.values + 1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True)
            extrapolated_prefs[outside_indices] = (cluster_prefs[nearest_two.indices] * weights).sum(dim=1)

        return extrapolated_prefs, extrapolated_labels, outside_indices, adapt_phi_pro

    def retrain_visual_components(self, train_loader, val_loader, optimizer, scheduler, epochs, preadapt_loader, max_distance_threshold, name_to_id):
        """
        Retrain visual encoder, utility function, and cost network.
        """
        cluster_phi_vis, cluster_centers = self.compute_preadapt_encodings(preadapt_loader, name_to_id)
        max_existing_label = max(cluster_phi_vis.keys()) if cluster_phi_vis else -1

        for epoch in range(epochs):
            for training, loader in [(True, train_loader), (False, val_loader)]:
                if training:
                    self.train()
                else:
                    self.eval()
                    total_cosine_sim, total_valid_samples, total_contrastive_loss, total_novel_samples = 0, 0, 0, 0
                
                total_loss = 0
                for batch in loader:
                    patches, inertial, terrain_labels, preferences = [x.to(self.device) for x in batch]
                    preferences = preferences.float()
                    batch_size = patches.shape[0]

                    scaled_preferences = loader.dataset.dataset.get_scaled_preferences(preferences)
                    phi_vis, phi_pro, uvis_pred, _, final_cost = self.forward(patches, inertial)

                    vis_loss = self.compute_triplet_loss(phi_vis, phi_pro, terrain_labels, cluster_phi_vis, cluster_centers, max_distance_threshold)
                    contrastive_loss = self.compute_contrastive_loss(phi_vis, terrain_labels) if (terrain_labels > max_existing_label).any() else 0
                    pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
                    pref_diff = scaled_preferences.unsqueeze(1) - scaled_preferences.unsqueeze(0)
                    ranking_loss = F.relu(1.0 - pred_diff[pref_diff > 0]).mean() if pref_diff.gt(0).any() else 0
                    cost_loss = F.smooth_l1_loss(final_cost, scaled_preferences, beta=1.0)
                    task_loss = vis_loss + ranking_loss + 2.0 * cost_loss + 0.5 * contrastive_loss

                    if training:
                        optimizer.zero_grad()
                        task_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    total_loss += task_loss.item()

                    if not training:
                        distances = torch.cdist(phi_pro, cluster_centers)
                        min_distances, nearest_indices = distances.min(dim=1)
                        valid_mask = (min_distances <= max_distance_threshold) & (terrain_labels <= max_existing_label)
                        if valid_mask.any():
                            cluster_labels = torch.tensor(list(cluster_phi_vis.keys()), device=self.device)
                            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                            total_cosine_sim += F.cosine_similarity(
                                phi_vis[valid_indices], 
                                torch.stack([cluster_phi_vis[cluster_labels[nearest_indices[i]].item()] for i in valid_indices])
                            ).sum().item()
                            total_valid_samples += valid_mask.sum().item()
                        if (terrain_labels > max_existing_label).any():
                            total_contrastive_loss += contrastive_loss.item()
                            total_novel_samples += (terrain_labels > max_existing_label).sum().item()

                avg_loss = total_loss / len(loader)
                print(f"Epoch [{epoch+1}/{epochs}], {'Train' if training else 'Val'} Loss: {avg_loss:.4f}")
                if not training and total_valid_samples > 0:
                    print(f"Known Terrain Metrics: Cosine Similarity: {total_cosine_sim / total_valid_samples:.4f}")
                if not training and total_novel_samples > 0:
                    print(f"Novel Terrain Metrics: Contrastive Loss: {total_contrastive_loss / (len(val_loader) * batch_size / total_novel_samples):.4f}")
            
            scheduler.step()

    def save_adapted_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saved models to {save_dir}")
        for name, file in [
            ("visual_encoder", "fvis_adapted.pt"),
            ("uvis", "uvis_adapted.pt"),
            ("cost_head", "cost_head_adapted.pt")
        ]:
            torch.save(self.models[name].state_dict(), os.path.join(save_dir, file))

def visualize_clusters(phi_pro, labels, adapt_phi_pro=None, save_path=None, title="Clusters", label_to_name=None):
    pca = PCA(n_components=2, random_state=42)
    phi_pro_2d = pca.fit_transform(phi_pro.cpu().numpy())
    plt.figure(figsize=(10, 8))

    # Ensure labels are integers
    if not np.issubdtype(labels.dtype, np.integer):
        if label_to_name:
            # Map string labels to integers using label_to_name's inverse
            label_map = {v: k for k, v in label_to_name.items()}
            labels = np.array([label_map.get(label, -1) if isinstance(label, str) else int(label) for label in labels])
        else:
            labels = np.array([int(label) if label != -1 else -1 for label in labels])

    for label in np.unique(labels):
        mask = labels == label
        if not mask.any():
            continue
        label_name = label_to_name.get(label, f"Cluster_{label}" if label != -1 else "Outliers")
        plt.scatter(phi_pro_2d[mask, 0], phi_pro_2d[mask, 1], color=plt.cm.tab10(int(label) % 10), alpha=0.6, label=label_name)

    if adapt_phi_pro is not None:
        adapt_phi_pro_2d = pca.transform(adapt_phi_pro.cpu().numpy())
        plt.scatter(adapt_phi_pro_2d[:, 0], adapt_phi_pro_2d[:, 1], color='black', marker='x', s=50, label='Adaptation Samples')

    plt.title(f"{title} (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
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

def process_batch_to_disk(loader, cache_dir):
    device = torch.device("cpu")
    os.makedirs(cache_dir, exist_ok=True)
    patch_file = os.path.join(cache_dir, "patches.h5")
    inertial_file = os.path.join(cache_dir, "inertial.h5")
    print("Extracting adaptation features...")

    with h5py.File(patch_file, 'w') as pf, h5py.File(inertial_file, 'w') as inf:
        for i, batch in enumerate(loader):
            patch1, _, imu_sample = batch
            patch1, imu_sample = patch1.to(device), imu_sample.to(device)
            pf.create_dataset(f"batch_{i}", data=patch1.cpu().numpy(), compression="gzip")
            inf.create_dataset(f"batch_{i}", data=imu_sample.cpu().numpy(), compression="gzip")
            if i % 10 == 0:
                print(f"Processed {i * loader.batch_size} samples, RAM: {psutil.virtual_memory().used / 1024**2:.2f} MB")
            del patch1, imu_sample
            gc.collect()

    return patch_file, inertial_file

def load_cached_h5(file_path, device="cpu"):
    with h5py.File(file_path, 'r') as f:
        data = np.concatenate([f[key][()] for key in f.keys()], axis=0)
    return torch.from_numpy(data).to(device)

def custom_collate(batch):
    patches, inertial, terrain_labels, preferences = zip(*batch)
    terrain_labels = [-1 if label is None else label for label in terrain_labels]
    return (
        torch.stack(patches),
        torch.stack(inertial),
        torch.tensor(terrain_labels, dtype=torch.long),
        torch.stack(preferences)
    )

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_datasets(args):
    print("Loading pre-adaptation data...")
    preadapt_h5_path = os.path.join(args.preadapt_bag, "clusters", "labeled_data.h5")
    preadapt_dataset = TerrainDataset(labeled_dataset=preadapt_h5_path, transform=None)

    config_path = os.path.join(script_dir, "homography", "config.yaml")
    name_to_id = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
            if 'terrains' in config:
                name_to_id = {terrain['name']: terrain['label'] for terrain in config['terrains']}
                print(f"Loaded name-to-ID mapping: {name_to_id}")
    else:
        name_to_id = {label: i for i, label in enumerate(set(preadapt_dataset.terrain_labels))}

    if not name_to_id:
        raise ValueError("No terrain labels found")

    print("Loading adaptation data...")
    adapt_vicreg_path = load_bag_h5(args.adapt_bag, "vicreg")
    adapt_synced_path = load_bag_h5(args.adapt_bag, "synced")
    adapt_dataset = TerrainDataset(synced_h5_path=adapt_synced_path, vicreg_h5_path=adapt_vicreg_path, transform=None, train=True)
    print(f"Adaptation dataset size: {len(adapt_dataset)}")

    return preadapt_dataset, adapt_dataset, len(name_to_id), name_to_id

def initialize_model(args, device):
    print("Initializing model...")
    pretrained_weights_path = os.path.join(args.preadapt_bag, "models")
    model = PaternAdaptation(device=device, pretrained_weights_path=pretrained_weights_path).to(device)

    adapted_files = {
        "visual_encoder": "fvis_adapted.pt",
        "uvis": "uvis_adapted.pt",
        "cost_head": "cost_head_adapted.pt"
    }
    adapted_found = False
    for name, file in adapted_files.items():
        path = os.path.join(pretrained_weights_path, file)
        if os.path.exists(path):
            state_dict = torch.load(path, weights_only=True, map_location=device)
            model.models[name].load_state_dict(state_dict)
            print(f"Loaded adapted {name} from {path}")
            adapted_found = True

    return model, adapted_found

def create_dataloaders(preadapt_dataset, adapt_dataset, batch_size):
    print("Creating DataLoaders...")
    return (
        DataLoader(preadapt_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False),
        DataLoader(adapt_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False))

def extract_preadapt_features(model, preadapt_loader, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    paths = {
        "phi_pro": os.path.join(cache_dir, "preadapt_phi_pro.h5"),
        "prefs": os.path.join(cache_dir, "preadapt_prefs.pt"),
        "labels": os.path.join(cache_dir, "preadapt_labels.npy"),
        "stats": os.path.join(cache_dir, "preadapt_inertial_stats.pt")
    }

    if not all(os.path.exists(p) for p in paths.values()):
        inertial_data, preadapt_prefs, preadapt_labels = [], None, []
        for batch in preadapt_loader:
            _, inertial, terrain_labels, preferences = batch
            inertial_data.append(inertial)
            preferences = preferences.to(device)
            preadapt_prefs = preferences if preadapt_prefs is None else torch.cat([preadapt_prefs, preferences])
            if isinstance(terrain_labels, (torch.Tensor, np.ndarray)):
                preadapt_labels.extend(terrain_labels.tolist())
            else:
                preadapt_labels.extend(terrain_labels)

        inertial_data = torch.cat(inertial_data)
        inertial_mean = inertial_data.mean(dim=0)
        inertial_std = inertial_data.std(dim=0, unbiased=True) + 1e-6
        torch.save({'mean': inertial_mean, 'std': inertial_std}, paths["stats"])

        with h5py.File(paths["phi_pro"], 'w') as f:
            dset = None
            current_idx = 0
            for i, batch in enumerate(preadapt_loader):
                _, inertial, _, _ = batch
                normalized_inertial = (inertial.to(device) - inertial_mean.to(device)) / inertial_std.to(device)
                phi_pro = model.extract_proprioceptive_features(normalized_inertial)
                if dset is None:
                    dset = f.create_dataset("data", shape=(len(preadapt_loader.dataset), *phi_pro.shape[1:]), dtype=np.float32, compression="gzip")
                batch_size = inertial.shape[0]
                dset[current_idx:current_idx + batch_size] = phi_pro.cpu().numpy()
                current_idx += batch_size
                del phi_pro
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        torch.save(preadapt_prefs, paths["prefs"])
        np.save(paths["labels"], np.array(preadapt_labels))

    preadapt_phi_pro = load_cached_h5(paths["phi_pro"], device)
    preadapt_prefs = torch.load(paths["prefs"], map_location=device)
    preadapt_labels = np.load(paths["labels"])
    inertial_stats = torch.load(paths["stats"], map_location=device)
    print(f"Preadaptation phi_pro range: min={preadapt_phi_pro.min().item():.4f}, max={preadapt_phi_pro.max().item():.4f}, mean={preadapt_phi_pro.mean().item():.4f}, std={preadapt_phi_pro.std().item():.4f}")

    return preadapt_phi_pro, preadapt_prefs, preadapt_labels, inertial_stats

def compute_distance_threshold(preadapt_phi_pro, preadapt_labels, cache_dir):
    print("Computing distance threshold...")
    batch_size = 2048
    n_samples = len(preadapt_phi_pro)
    distances_cache = os.path.join(cache_dir, "distances.h5")
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(distances_cache):
        with h5py.File(distances_cache, 'w') as f:
            dset = f.create_dataset("distances", (n_samples, n_samples), dtype=np.float32, compression="gzip")
            for i in range(0, n_samples, batch_size):
                batch_phi = preadapt_phi_pro[i:i + batch_size].cpu().numpy()
                batch_dist = torch.cdist(preadapt_phi_pro[i:i + batch_size], preadapt_phi_pro).cpu().numpy()
                dset[i:i + batch_size] = batch_dist

    intra_sum, intra_count, inter_sum, inter_count = 0.0, 0, 0.0, 0
    with h5py.File(distances_cache, 'r') as f:
        dist = f["distances"]
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_labels = preadapt_labels[i:end]
            batch_dist = dist[i:end]
            mask = batch_labels[:, None] == preadapt_labels[None, :]
            intra_mask = mask & ~np.eye(end - i, n_samples, dtype=bool)
            inter_mask = ~mask
            intra_sum += np.sum(batch_dist * intra_mask)
            intra_count += np.sum(intra_mask)
            inter_sum += np.sum(batch_dist * inter_mask)
            inter_count += np.sum(inter_mask)
            if i % 1000 == 0:
                print(f"Processed {i} samples for distance calculation")

    avg_intra = intra_sum / intra_count if intra_count > 0 else 0.0
    avg_inter = inter_sum / inter_count if inter_count > 0 else 0.0
    max_distance_threshold = (avg_intra + avg_inter) / 2 if avg_intra > 0 and avg_inter > 0 else 5.0
    print(f"Avg Intra-cluster Distance: {avg_intra:.4f}, Avg Inter-cluster Distance: {avg_inter:.4f}")
    print(f"Computed Max Distance Threshold: {max_distance_threshold:.4f}")
    os.remove(distances_cache)
    return max_distance_threshold

def extract_adapt_features(model, adapt_loader, cache_dir, inertial_stats):
    device = torch.device("cpu")
    patch_file, inertial_file = process_batch_to_disk(adapt_loader, cache_dir)
    adapt_patches = load_cached_h5(patch_file, device)
    adapt_inertial = load_cached_h5(inertial_file, device)
    normalized_inertial = (adapt_inertial.to(model.device) - inertial_stats['mean'].to(model.device)) / inertial_stats['std'].to(model.device)
    adapt_phi_pro = model.extract_proprioceptive_features(normalized_inertial)

    print(f"Adaptation phi_pro range: min={adapt_phi_pro.min().item():.4f}, max={adapt_phi_pro.max().item():.4f}, mean={adapt_phi_pro.mean().item():.4f}, std={adapt_phi_pro.std().item():.4f}")
    os.remove(patch_file)
    os.remove(inertial_file)
    return adapt_patches, adapt_inertial, adapt_phi_pro

def extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, preadapt_data, max_distance_threshold, args, name_to_id):
    """
    Extrapolate preferences and labels, cache results, and handle outliers with user input.
    """
    print("Extrapolating preferences and labels...")
    numerical_labels = np.array([name_to_id[label] for label in preadapt_data[2]])
    numerical_labels_tensor = torch.tensor(numerical_labels, device=adapt_inertial.device)
    extrapolated_prefs, extrapolated_labels, outlier_indices, adapt_phi_pro = model.extrapolate_preferences(
        adapt_inertial, preadapt_data[0], preadapt_data[1], numerical_labels_tensor, max_distance_threshold
    )
    label_to_name = {int(v): k for k, v in name_to_id.items()}

    max_existing_label = numerical_labels_tensor.max().item() if numerical_labels_tensor.numel() > 0 else -1
    new_clusters = torch.unique(extrapolated_labels[extrapolated_labels > max_existing_label])
    for cluster_id in new_clusters:
        if cluster_id.item() not in label_to_name:
            label_to_name[cluster_id.item()] = f"Novel_{cluster_id.item()}"

    if len(outlier_indices) > 0:
        print(f"Found {len(outlier_indices)} data points outside cluster thresholds.")
        outlier_dir = os.path.join(args.preadapt_bag, "outlier_patches")
        app = OutlierLabelUI(
            patches=adapt_patches,
            extrapolated_prefs=extrapolated_prefs,
            outlier_indices=outlier_indices,
            adapt_phi_pro=adapt_phi_pro,
            save_dir=outlier_dir,
            default_labels=extrapolated_labels,
            preadapt_name_to_id=name_to_id
        )
        app.run()
        user_labels_and_ids, user_prefs = app.get_results()

        for idx, (string_label, cluster_id), pref in zip(outlier_indices.tolist(), user_labels_and_ids, user_prefs):
            extrapolated_labels[idx] = cluster_id
            extrapolated_prefs[idx] = pref
            if string_label not in label_to_name.values() and cluster_id not in label_to_name:
                label_to_name[cluster_id] = string_label
        shutil.rmtree(outlier_dir, ignore_errors=True)

    adapt_data_file = os.path.join(args.preadapt_bag, "adapt_data_cache", "adapt_data.h5")
    os.makedirs(os.path.dirname(adapt_data_file), exist_ok=True)

    with h5py.File(adapt_data_file, 'w') as f:
        for i in range(len(adapt_patches)):
            group = f.create_group(f"entry_{i}")
            group.create_dataset("patch", data=adapt_patches[i].numpy(), compression="gzip")
            group.create_dataset("inertial", data=adapt_inertial[i].numpy(), compression="gzip")
            group.attrs["terrain_label"] = extrapolated_labels[i].item()
            group.attrs["preference"] = extrapolated_prefs[i].item()

    return adapt_data_file, label_to_name

def aggregate_and_split_datasets(preadapt_dataset, adapt_data_file, args):
    preadapt_data = list(preadapt_dataset)
    adapt_data, label_map = [], {}

    for item in preadapt_data:
        terrain_label = item[2] if isinstance(item, tuple) else item.get("terrain_label")
        if isinstance(terrain_label, str) and terrain_label not in label_map:
            label_map[terrain_label] = len(label_map)

    with h5py.File(adapt_data_file, 'r') as f:
        for i in range(len(f)):
            group = f[f"entry_{i}"]
            terrain_label = -1 if group.attrs["terrain_label"] == -1 else group.attrs["terrain_label"]
            adapt_data.append({
                "patch": torch.from_numpy(group["patch"][()]),
                "inertial": torch.from_numpy(group["inertial"][()]),
                "terrain_label": terrain_label,
                "preference": group.attrs["preference"]
            })

    normalized_preadapt_data = [
        {
            "patch": item[0] if isinstance(item, tuple) else item["patch"],
            "inertial": item[1] if isinstance(item, tuple) else item["inertial"],
            "terrain_label": label_map.get(item[2], -1) if isinstance(item, tuple) else label_map.get(item.get("terrain_label"), -1),
            "preference": item[3] if isinstance(item, tuple) else item["preference"]
        }
        for item in preadapt_data
    ]

    aggregated_dataset = TerrainDataset(labeled_dataset=normalized_preadapt_data + adapt_data, transform=None)
    train_dataset, val_dataset = random_split(aggregated_dataset, [len(aggregated_dataset) - int(args.val_split * len(aggregated_dataset)), int(args.val_split * len(aggregated_dataset))])

    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=custom_collate),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=custom_collate),
        [adapt_data_file]
    )

def retrain_model(model, train_loader, val_loader, preadapt_loader, epochs, max_distance_threshold, name_to_id):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    model.retrain_visual_components(train_loader, val_loader, optimizer, scheduler, epochs, preadapt_loader, max_distance_threshold, name_to_id)

def concatenate_to_file(file_list, output_path, total_samples, sample_shape):
    with h5py.File(output_path, 'w') as f:
        dset = f.create_dataset("data", shape=[total_samples] + list(sample_shape[1:]), dtype=np.float32, compression="gzip")
        current_idx = 0
        for file_path in file_list:
            batch_data = torch.load(file_path, map_location="cpu").numpy()
            dset[current_idx:current_idx + batch_data.shape[0]] = batch_data
            current_idx += batch_data.shape[0]
            os.remove(file_path)

def extract_postadapt_features(model, train_dataset, args):
    print("Extracting features utilized during training...")
    postadapt_cache_dir = os.path.join(args.preadapt_bag, "postadapt_cache")
    os.makedirs(postadapt_cache_dir, exist_ok=True)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    phi_pro_file = os.path.join(postadapt_cache_dir, "phi_pro.h5")
    patch_file = os.path.join(postadapt_cache_dir, "patches.h5")
    labels = []

    with torch.no_grad():
        with h5py.File(phi_pro_file, 'w') as pf, h5py.File(patch_file, 'w') as patch_f:
            # Initialize datasets
            phi_dset = None
            patch_dset = None
            dataset_size = len(train_dataset)

            # Process all batches
            for i, batch in enumerate(loader):
                patches, inertial, terrain_labels, _ = batch
                phi_pro = model.extract_proprioceptive_features(inertial.to(model.device))
                if i == 0:
                    phi_dset = pf.create_dataset("data", shape=(dataset_size, *phi_pro.shape[1:]), dtype=np.float32, compression="gzip", chunks=(args.batch_size, *phi_pro.shape[1:]))

                    patch_dset = patch_f.create_dataset("data", shape=(dataset_size, *patches.shape[1:]), dtype=np.float32, compression="gzip", chunks=(args.batch_size, *patches.shape[1:]))
                
                # Compute indices and actual batch size
                start_idx = i * args.batch_size
                end_idx = min(start_idx + args.batch_size, dataset_size)
                batch_size = end_idx - start_idx  # Actual number of samples in this batch
                phi_dset[start_idx:end_idx] = phi_pro.cpu().numpy()[:batch_size]
                patch_dset[start_idx:end_idx] = patches.cpu().numpy()[:batch_size]
                labels.extend(terrain_labels.tolist())
                del phi_pro, patches, inertial, terrain_labels
                gc.collect()

    return phi_pro_file, patch_file, labels

def visualize_and_render(phi_pro_output, patch_output, labels, train_dataset, args, label_to_name=None):
    print("Creating PCA plot for post-adaptation clusters...")
    labels = np.array([-1 if label is None else label for label in labels])
    with h5py.File(phi_pro_output, 'r') as pf, h5py.File(patch_output, 'r') as patch_f:
        postadapt_phi_pro = torch.from_numpy(pf["data"][()]).cpu()
        postadapt_patches = torch.from_numpy(patch_f["data"][()]).cpu()

    plot_path = os.path.join(args.preadapt_bag, "post_adaptation_clusters.png")
    visualize_clusters(postadapt_phi_pro, labels, save_path=plot_path, title="Post-adaptation Clusters", label_to_name=label_to_name)

    print("Rendering patches in post-adaptation clusters...")
    patch_dir = os.path.join(args.preadapt_bag, "post_adaptation_patches")
    render_and_save_cluster_patches(postadapt_patches, labels, patch_dir)

def save_models(model, args):
    print("Patern adaptation training completed. Saving models...")
    model.save_adapted_models(os.path.join(args.preadapt_bag, "models"))
    print("Models saved successfully.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="PATERN Preference Extrapolation")
    parser.add_argument("-preadapt_bag", "-pb", type=str, required=True)
    parser.add_argument("-adapt_bag", "-ab", type=str, required=True)
    parser.add_argument("-batch_size", type=int, default=2048)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-val_split", type=float, default=0.2)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    device = setup_device()
    preadapt_dataset, adapt_dataset, n_clusters, name_to_id = load_datasets(args)
    model, adapted_found = initialize_model(args, device)
    preadapt_loader, adapt_loader = create_dataloaders(preadapt_dataset, adapt_dataset, args.batch_size)

    cache_dir = os.path.join(args.preadapt_bag, "cache")
    preadapt_phi_pro, preadapt_prefs, preadapt_labels, inertial_stats = extract_preadapt_features(model, preadapt_loader, cache_dir, device)
    max_distance_threshold = compute_distance_threshold(preadapt_phi_pro, preadapt_labels, cache_dir)
    #max_distance_threshold = 1.0

    plot_path = os.path.join(args.preadapt_bag, "pre_adaptation_clusters.png")
    adapt_patches, adapt_inertial, adapt_phi_pro = extract_adapt_features(model, adapt_loader, cache_dir, inertial_stats)
    visualize_clusters(preadapt_phi_pro, preadapt_labels, adapt_phi_pro, plot_path, "Pre-adaptation Clusters", label_to_name={v: k for k, v in name_to_id.items()} if name_to_id else None)

    adapt_data_file, label_to_name = extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, [preadapt_phi_pro, preadapt_prefs, preadapt_labels], max_distance_threshold, args, name_to_id)
    train_loader, val_loader, adapt_data_files = aggregate_and_split_datasets(preadapt_dataset, adapt_data_file, args)

    retrain_model(model, train_loader, val_loader, preadapt_loader, args.epochs, max_distance_threshold, name_to_id)
    phi_pro_output, patch_output, labels = extract_postadapt_features(model, train_loader.dataset, args)

    for f in adapt_data_files:
        os.remove(f)

    visualize_and_render(phi_pro_output, patch_output, labels, train_loader.dataset, args, label_to_name)
    save_models(model, args)


    print("Removing temporary directories...")
    for dir_path in [
        os.path.join(args.preadapt_bag, "postadapt_cache"),
        cache_dir,
        os.path.join(args.preadapt_bag, "adapt_data_cache")
    ]:
        shutil.rmtree(dir_path, ignore_errors=True)