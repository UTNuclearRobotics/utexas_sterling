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
import glob
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, GdkPixbuf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import yaml

script_dir = os.path.dirname(os.path.realpath(__file__))

# GTK-based GUI for labeling outlier groups
class OutlierLabelUI(Gtk.Application):
    def __init__(self, patches=None, extrapolated_prefs=None, outlier_indices=None,
                 within_indices=None, adapt_phi_pro=None, save_dir=None,
                 default_labels=None, preadapt_name_to_id=None):
        super().__init__(application_id="com.example.OutlierLabelUI")
        GLib.set_application_name("Outlier Label UI")
        self.patches = patches
        self.extrapolated_prefs = extrapolated_prefs
        self.outlier_indices = outlier_indices if outlier_indices is not None else torch.tensor([], dtype=torch.long)
        self.within_indices = within_indices if within_indices is not None else torch.tensor([], dtype=torch.long)
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
        self.is_outlier_group = None
        self.next_label_id = max(int(v) for v in self.preadapt_name_to_id.values()) + 1 if self.preadapt_name_to_id else 0
        self.cluster_id_label = None  # Initialize here to avoid attribute error
        self.config_path = os.path.join(script_dir, "homography", "config.yaml")
        self.cluster_split_counts = {}

    # 1. convert NumPy image → GdkPixbuf
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

    # Update config.yaml with new terrain label
    def update_config_yaml(self, label, cluster_id, preference):
        """Update config.yaml with a new terrain label and cluster ID."""
        try:
            # Load existing config
            config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file) or {}
            
            # Ensure 'terrains' exists
            if 'terrains' not in config:
                config['terrains'] = []
            
            # Check if label already exists to avoid duplicates
            for terrain in config['terrains']:
                if terrain['name'] == label:
                    print(f"Terrain '{label}' already exists in config.yaml with label {terrain['label']}")
                    return
            
            # Append new terrain
            config['terrains'].append({'name': label, 'label': cluster_id, 'preference': preference})
            print(f"Adding new terrain to config.yaml: name='{label}', label={cluster_id}, preference={preference}")
            
            # Write back to file
            with open(self.config_path, 'w') as file:
                yaml.safe_dump(config, file, default_flow_style=None)
            
            print(f"Updated config.yaml successfully")
        except Exception as e:
            print(f"Error updating config.yaml: {e}")

    def split_cluster(self, cluster_idx, n_clusters):
        """Re-cluster the patches in the specified cluster into n_clusters."""
        indices = self.cluster_indices[cluster_idx]
        phi_pro = self.adapt_phi_pro[indices].cpu().numpy()
        max_possible_clusters = min(len(indices), 10)

        if n_clusters > max_possible_clusters:
            n_clusters = max_possible_clusters
            print(f"Capped number of clusters to {max_possible_clusters} due to sample size.")

        if n_clusters <= 1:
            print("Cannot split into fewer than 2 clusters.")
            return

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(phi_pro)
        print(f"Split cluster {cluster_idx} into {n_clusters} sub-clusters.")

        new_cluster_indices = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            new_cluster_indices[label].append(i)

        new_cluster_indices = [indices_local for indices_local in new_cluster_indices if indices_local]

        renderer = PatchRenderer()
        patches_np = self.patches[torch.tensor(indices, device=self.patches.device)].cpu().numpy()
        new_rendered_clusters = renderer.render_clusters(new_cluster_indices, patches_np, input_format="RGB", output_format="RGB")
        import gc
        gc.collect()

        global_new_cluster_indices = [[indices[i] for i in indices_local] for indices_local in new_cluster_indices]

        self.rendered_clusters.pop(cluster_idx)
        self.cluster_indices.pop(cluster_idx)
        is_outlier = self.is_outlier_group.pop(cluster_idx)

        self.rendered_clusters[cluster_idx:cluster_idx] = new_rendered_clusters
        self.cluster_indices[cluster_idx:cluster_idx] = global_new_cluster_indices
        self.is_outlier_group[cluster_idx:cluster_idx] = [is_outlier] * len(new_rendered_clusters)

        original_cluster_id = cluster_idx
        self.cluster_split_counts[original_cluster_id] = self.cluster_split_counts.get(original_cluster_id, 1) + 1

    def on_split_clicked(self, button, window, cluster_idx):
        """Handle the 'Split Cluster' button click."""
        is_outlier = self.is_outlier_group[cluster_idx]
        if not is_outlier:
            confirm_dialog = Gtk.MessageDialog(
                parent=window,
                modal=True,
                message_type=Gtk.MessageType.QUESTION,
                buttons=Gtk.ButtonsType.YES_NO,
                text="This is an in-threshold cluster. Are you sure you want to split it?"
            )
            response = confirm_dialog.run()
            confirm_dialog.destroy()
            if response != Gtk.ResponseType.YES:
                return
        
        dialog = Gtk.Dialog(
            title="Select Number of Sub-Clusters",
            parent=window,
            modal=True
        )
        dialog.set_default_size(300, 100)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK
        )
        box = dialog.get_content_area()
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        box.set_margin_start(10)
        box.set_margin_end(10)

        label = Gtk.Label(label="Number of sub-clusters (2-10):")
        box.pack_start(label, False, False, 5)

        spin_button = Gtk.SpinButton.new_with_range(2, 10, 1)
        spin_button.set_value(2)
        box.pack_start(spin_button, False, False, 5)

        dialog.show_all()

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            n_clusters = int(spin_button.get_value())
            self.split_cluster(cluster_idx, n_clusters)
            window.destroy()
            self.show_next_group()
        dialog.destroy()

    # do_activate – build outlier + in-threshold groups
    def do_activate(self):
        if self.patches is None or self.extrapolated_prefs is None or self.adapt_phi_pro is None:
            window = Gtk.ApplicationWindow(application=self, title="Outlier Label UI - Error")
            window.set_default_size(400, 200)
            label = Gtk.Label(label="Error: OutlierLabelUI requires patches, preferences, and phi_pro data.")
            label.set_margin_top(20)
            label.set_margin_bottom(20)
            label.set_margin_start(20)
            label.set_margin_end(20)
            window.add(label)
            window.show_all()
            window.present()
            return

        self.base_cluster_id = len(self.preadapt_name_to_id) if self.preadapt_name_to_id else 0

        # ALL adaptation points
        all_indices = torch.arange(len(self.patches), device=self.patches.device)
        patches_np = self.patches[all_indices].cpu().numpy()
        prefs_np = self.extrapolated_prefs[all_indices].cpu().numpy()
        default_labels_np = (self.default_labels[all_indices].cpu().numpy()
                             if self.default_labels is not None else None)

        renderer = PatchRenderer()
        os.makedirs(self.save_dir, exist_ok=True)

        # Outlier clusters (K-means)
        outlier_phi_pro = self.adapt_phi_pro[self.outlier_indices].cpu().numpy()
        n_out = len(outlier_phi_pro)

        if n_out <= 3:
            outlier_group_labels = np.zeros(n_out, dtype=int)
            print(f"Too few outliers ({n_out}), assigning all to group 0")
        else:
            max_possible_clusters = min(10, n_out - 1)
            best_labels = np.zeros(n_out, dtype=int)
            best_n_clusters = 1
            best_score = -1.0

            for n in range(2, max_possible_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = kmeans.fit_predict(outlier_phi_pro)
                
                # Critical: check actual number of unique clusters after fitting
                n_unique = len(np.unique(labels))
                if n_unique < 2:
                    continue  # skip – useless for silhouette

                try:
                    score = silhouette_score(outlier_phi_pro, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n
                        best_labels = labels.copy()  # save them
                except ValueError as e:
                    if "Number of labels is 1" in str(e):
                        continue
                    raise

            if best_score > -1:  # we found at least one valid clustering
                outlier_group_labels = best_labels
                print(f"Selected {best_n_clusters} clusters for {n_out} outliers (silhouette: {best_score:.3f})")
            else:
                outlier_group_labels = np.zeros(n_out, dtype=int)
                print(f"No valid multi-cluster solution for {n_out} outliers → using 1 group")

        unique_outlier_groups = np.unique(outlier_group_labels)
        outlier_cluster_indices = [
            self.outlier_indices[torch.where(
                torch.from_numpy(outlier_group_labels) == g)[0]].tolist()
            for g in unique_outlier_groups
        ]
        rendered_outlier_clusters = renderer.render_clusters(
            outlier_cluster_indices, patches_np, input_format="RGB", output_format="RGB")

        # In-threshold clusters – group by extrapolated label
        within_labels = self.default_labels[self.within_indices].cpu().numpy()
        unique_within_labels = np.unique(within_labels)
        within_cluster_indices = [
            self.within_indices[torch.where(
                torch.from_numpy(within_labels) == lbl)[0]].tolist()
            for lbl in unique_within_labels
        ]
        rendered_within_clusters = renderer.render_clusters(
            within_cluster_indices, patches_np, input_format="RGB", output_format="RGB")

        self.rendered_clusters = rendered_outlier_clusters + rendered_within_clusters
        self.cluster_indices = outlier_cluster_indices + within_cluster_indices
        self.is_outlier_group = [True] * len(rendered_outlier_clusters) + \
                               [False] * len(rendered_within_clusters)
        self.outlier_prefs = prefs_np
        self.outlier_default_labels = default_labels_np

        self.show_next_group()

    def show_next_group(self):
        if not self.rendered_clusters or self.current_group >= len(self.rendered_clusters):
            self.quit()
            return

        cluster_patches = self.rendered_clusters[self.current_group]
        indices = self.cluster_indices[self.current_group]
        is_outlier = self.is_outlier_group[self.current_group]
        if not cluster_patches:
            self.current_group += 1
            self.show_next_group()
            return

        window = Gtk.ApplicationWindow(
            application=self,
            title=f"{'Outlier' if is_outlier else 'Assigned'} Group {self.current_group} ({len(cluster_patches)} patches)"
        )
        window.set_default_size(1200, 800)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        grid_image = PatchRenderer().image_grid(cluster_patches)
        pixbuf = self.numpy_to_pixbuf(grid_image)
        image_widget = Gtk.Image.new_from_pixbuf(pixbuf)

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.add(image_widget)
        scrolled_window.set_hexpand(True)
        scrolled_window.set_vexpand(True)
        vbox.pack_start(scrolled_window, True, True, 0)

        group_prefs = self.outlier_prefs[indices]
        mean_extrapolated_pref = np.mean(group_prefs)
        vbox.pack_start(Gtk.Label(label=f"Mean Extrapolated Preference: {mean_extrapolated_pref:.4f}"), False, False, 0)

        current_label_id = int(self.default_labels[indices[0]].item())
        current_label_str = next(
            (name for name, cid in self.preadapt_name_to_id.items()
             if cid == current_label_id), f"cluster_{current_label_id}")

        label_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        label_hbox.pack_start(Gtk.Label(label="Label (string):"), False, False, 0)
        label_entry = Gtk.Entry()
        label_entry.set_placeholder_text("Enter a descriptive label...")
        label_entry.set_text(current_label_str)
        label_hbox.pack_start(label_entry, True, True, 0)
        vbox.pack_start(label_hbox, False, False, 0)

        temp_cluster_id = self.current_group
        self.cluster_id_label = Gtk.Label(label=f"Assigned Cluster ID: {temp_cluster_id} (pending final label)")
        vbox.pack_start(self.cluster_id_label, False, False, 0)

        pref_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        pref_hbox.pack_start(Gtk.Label(label="Preference:"), False, False, 0)
        pref_entry = Gtk.Entry()
        pref_entry.set_text(f"{mean_extrapolated_pref:.4f}")
        pref_hbox.pack_start(pref_entry, True, True, 0)
        vbox.pack_start(pref_hbox, False, False, 0)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        submit_button = Gtk.Button(label="Submit")
        submit_button.connect("clicked", self.on_submit_clicked,
                             window, label_entry, pref_entry,
                             indices, current_label_id, is_outlier)
        button_box.pack_start(submit_button, False, False, 0)

        split_button = Gtk.Button(label="Split Cluster")
        split_button.connect("clicked", self.on_split_clicked, window, self.current_group)
        button_box.pack_start(split_button, False, False, 0)

        vbox.pack_start(button_box, False, False, 0)
        window.add(vbox)
        window.show_all()
        window.present()

        window.connect("destroy", lambda w: window.remove(vbox))

    def on_submit_clicked(self, button, window, label_entry, pref_entry,
                          indices, auto_cluster_id, is_outlier):
        try:
            label = label_entry.get_text().strip()
            pref = float(pref_entry.get_text())

            if label in self.preadapt_name_to_id:
                assigned_cluster_id = self.preadapt_name_to_id[label]
                print(f"Matched '{label}' to pre-adaptation cluster ID {assigned_cluster_id}")
            else:
                assigned_cluster_id = self.base_cluster_id + self.current_group
                self.preadapt_name_to_id[label] = assigned_cluster_id
                print(f"Assigned new cluster ID {assigned_cluster_id} to '{label}'")
                self.update_config_yaml(label, assigned_cluster_id, pref)  # <-- uses the method above
            
            self.cluster_id_label.set_text(f"Assigned Cluster ID: {assigned_cluster_id}")
            
            for idx in indices:
                original_idx = idx
                self.user_labels_dict[original_idx] = label
                self.user_cluster_ids_dict[original_idx] = assigned_cluster_id
                self.user_prefs_dict[original_idx] = pref
            window.destroy()
            self.current_group += 1
            self.show_next_group()
        except ValueError:
            error_dialog = Gtk.MessageDialog(
                parent=window,
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
        default_labels = self.default_labels.cpu().numpy() if self.default_labels is not None else np.zeros(len(self.patches), dtype=int)
        default_prefs = self.extrapolated_prefs.cpu().numpy()

        for i, idx in enumerate(torch.arange(len(self.patches))):
            idx_item = idx.item()
            if idx_item in self.user_labels_dict:
                user_labels.append(self.user_labels_dict[idx_item])
                user_cluster_ids.append(self.user_cluster_ids_dict[idx_item])
                user_prefs.append(self.user_prefs_dict[idx_item])
            else:
                user_labels.append(f"auto_{default_labels[i]}")
                user_cluster_ids.append(default_labels[i])
                user_prefs.append(default_prefs[i])
        
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
        cluster_angular_std = torch.zeros(n_clusters, device=self.device)
        # Store the representative terrain label for each cluster (mode of preadapt_labels)
        cluster_terrain_label = torch.zeros(n_clusters, dtype=torch.long, device=self.device)

        # Compute cluster statistics
        for i in range(n_clusters):
            mask = inverse == i
            points = preadapt_phi_pro[mask]
            if points.shape[0] == 0:
                continue

            # Normalize all points in cluster
            points_norm = torch.nn.functional.normalize(points, p=2, dim=1)
            center = points_norm.mean(dim=0)
            cluster_centers[i] = center

            # Mean preference
            cluster_prefs[i] = preadapt_prefs[mask].mean()

            # Angular std: mean cosine similarity to center → 1 - mean_cos_sim
            if points.shape[0] > 1:
                cos_sim_to_center = torch.mm(points_norm, center.unsqueeze(1)).squeeze(1)
                mean_cos_sim = cos_sim_to_center.mean()
                angular_std = torch.acos(torch.clamp(mean_cos_sim, -1.0, 1.0))  # radians
                cluster_angular_std[i] = angular_std
            else:
                cluster_angular_std[i] = 0.0

            # Terrain label: mode of preadapt_labels in this cluster
            cluster_terrain_label[i] = preadapt_labels[mask].mode().values

        # Normalize adaptation features
        adapt_norm = torch.nn.functional.normalize(adapt_phi_pro, p=2, dim=1)
        centers_norm = torch.nn.functional.normalize(cluster_centers, p=2, dim=1)

        # Compute distances and find nearest clusters
        cos_sim = torch.mm(adapt_norm, centers_norm.t())
        distances = 1.0 - cos_sim  # [0, 2]

        min_distances, nearest_indices = distances.min(dim=1)

        # Dynamic threshold in *cosine distance* space
        # Use angular_std → convert to cosine distance: 1 - cos(θ)
        dynamic_thresholds = 1.0 - torch.cos(cluster_angular_std[nearest_indices] * 2.0)
        max_threshold = torch.tensor(max_distance_threshold, device=self.device)
        within_threshold = min_distances <= torch.minimum(dynamic_thresholds, max_threshold)

        # === PRINT: Cluster Coverage with Real Terrain Labels ===
        print("\n=== Cluster Coverage Statistics ===")
        print(f"Total adaptation datapoints: {n_samples}")
        print(f"Within threshold           : {within_threshold.sum().item()}")
        print(f"Using IDW (outside)        : {(~within_threshold).sum().item()}\n")
        print("Cluster ID | Terrain Label | Points in Threshold | Coverage %")
        print("-" * 60)

        for cid in unique_clusters:
            cid_int = cid.item()
            nearest_to_cid = (nearest_indices == cid)
            within_this = nearest_to_cid & within_threshold
            count_within = within_this.sum().item()
            total_assigned = nearest_to_cid.sum().item()
            coverage_pct = (count_within / total_assigned * 100) if total_assigned > 0 else 0.0
            terrain_label = cluster_terrain_label[cid_int].item()

            print(f"{cid_int:10d} | {terrain_label:13d} | {count_within:19d} | {coverage_pct:9.1f}%")

        # === End of Print Section ===

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

def render_and_save_cluster_patches(patches, labels, save_dir, prefix="cluster", label_to_name=None):
    renderer = PatchRenderer()
    unique_labels = np.unique(labels)
    cluster_indices = [np.where(labels == label)[0].tolist() for label in unique_labels]
    rendered_clusters = renderer.render_clusters(cluster_indices, patches.cpu().numpy())
    os.makedirs(save_dir, exist_ok=True)
    
    for cluster_id, cluster_patches in enumerate(rendered_clusters):
        if cluster_patches:
            label_idx = unique_labels[cluster_id]
            
            # Determine filename based on label_to_name or fallback to cluster ID
            if label_to_name is not None and label_idx in label_to_name:
                terrain_name = label_to_name[label_idx]
                # Sanitize filename (remove invalid characters)
                safe_name = "".join(c for c in terrain_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{prefix}_{safe_name.replace(' ', '_')}.png"
            else:
                # Fallback to cluster ID for unknown labels
                filename = f"{prefix}_{int(label_idx)}.png"
            
            grid_image = renderer.image_grid(cluster_patches)
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
            print(f"Saved: {filepath} ({len(cluster_patches)} patches)")

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
    adapt_dataset = TerrainDataset(synced_h5_path=adapt_synced_path, vicreg_h5_path=adapt_vicreg_path, transform=None)
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
        DataLoader(preadapt_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False),
        DataLoader(adapt_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False))

def extract_preadapt_features(model, preadapt_loader, cache_dir, device, adapt_bag_path):
    os.makedirs(cache_dir, exist_ok=True)
    paths = {
        "phi_pro": os.path.join(cache_dir, "preadapt_phi_pro.h5"),
        "prefs": os.path.join(cache_dir, "preadapt_prefs.pt"),
        "labels": os.path.join(cache_dir, "preadapt_labels.npy")
    }

    # Derive imu_stats_dir from adapt_bag_path (points to /bags/<bag_name>/)
    imu_stats_dir = os.path.dirname(adapt_bag_path)  # e.g., /bags/
    # Search for file ending with _imu_stats.pt
    imu_stats_files = glob.glob(os.path.join(imu_stats_dir, "*_imu_stats.pt"))
    if not imu_stats_files:
        raise FileNotFoundError(f"No IMU stats file found in {imu_stats_dir} matching '*_imu_stats.pt'")
    if len(imu_stats_files) > 1:
        print(f"Warning: Multiple IMU stats files found in {imu_stats_dir}: {imu_stats_files}. Using the first one.")
    imu_stats_path = imu_stats_files[0]
    print(f"Loading IMU stats from: {imu_stats_path}")

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
        # Do not compute new stats; load from imu_stats_path below

        with h5py.File(paths["phi_pro"], 'w') as f:
            dset = None
            current_idx = 0
            for i, batch in enumerate(preadapt_loader):
                _, inertial, _, _ = batch
                # Use pre-normalized inertial data directly
                phi_pro = model.extract_proprioceptive_features(inertial.to(device))
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
    # Load inertial stats from imu_stats_path
    inertial_stats = torch.load(imu_stats_path, map_location=device)
    if 'mean' not in inertial_stats or 'std' not in inertial_stats:
        raise ValueError(f"IMU stats file {imu_stats_path} missing 'imu_mean' or 'imu_std' keys")
    print(f"Preadaptation phi_pro range: min={preadapt_phi_pro.min().item():.4f}, max={preadapt_phi_pro.max().item():.4f}, mean={preadapt_phi_pro.mean().item():.4f}, std={preadapt_phi_pro.std().item():.4f}")

    return preadapt_phi_pro, preadapt_prefs, preadapt_labels, inertial_stats

def compute_distance_threshold(preadapt_phi_pro, preadapt_labels, cache_dir):
    print("Computing distance threshold (cosine + L2 normalized)...")
    
    # L2 NORMALIZE FIRST
    preadapt_phi_pro = torch.nn.functional.normalize(preadapt_phi_pro, p=2, dim=1)
    
    batch_size = 2048
    n_samples = len(preadapt_phi_pro)
    distances_cache = os.path.join(cache_dir, "distances.h5")
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(distances_cache):
        with h5py.File(distances_cache, 'w') as f:
            dset = f.create_dataset("distances", (n_samples, n_samples), dtype=np.float32, compression="gzip")
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch_phi = preadapt_phi_pro[i:end]
                # Cosine distance = 1 - cosine similarity
                cos_sim = torch.mm(batch_phi, preadapt_phi_pro.t())
                batch_dist = 1.0 - cos_sim
                dset[i:end] = batch_dist.cpu().numpy()

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
    max_distance_threshold = (avg_intra + avg_inter) / 2 if avg_intra > 0 and avg_inter > 0 else 1.0
    print(f"Avg Intra-cluster Cosine Distance: {avg_intra:.4f}, Avg Inter-cluster: {avg_inter:.4f}")
    print(f"Computed Max Distance Threshold (cosine): {max_distance_threshold:.4f}")
    os.remove(distances_cache)
    return max_distance_threshold

def extract_adapt_features(model, adapt_loader, cache_dir, inertial_stats):
    device = torch.device("cpu")
    patch_file, inertial_file = process_batch_to_disk(adapt_loader, cache_dir)
    adapt_patches = load_cached_h5(patch_file, device)
    adapt_inertial = load_cached_h5(inertial_file, device)
    # Use adapt_inertial directly (already normalized by TerrainDataset)
    adapt_phi_pro = model.extract_proprioceptive_features(adapt_inertial.to(model.device))

    print(f"Adaptation phi_pro range: min={adapt_phi_pro.min().item():.4f}, max={adapt_phi_pro.max().item():.4f}, mean={adapt_phi_pro.mean().item():.4f}, std={adapt_phi_pro.std().item():.4f}")
    os.remove(patch_file)
    os.remove(inertial_file)
    return adapt_patches, adapt_inertial, adapt_phi_pro

def extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, preadapt_data, max_distance_threshold, args, name_to_id):
    print("Extrapolating preferences and labels...")
    numerical_labels = np.array([name_to_id[label] for label in preadapt_data[2]])
    numerical_labels_tensor = torch.tensor(numerical_labels, device=model.device)
    extrapolated_prefs, extrapolated_labels, outlier_indices, adapt_phi_pro = model.extrapolate_preferences(
        adapt_inertial, preadapt_data[0], preadapt_data[1], numerical_labels_tensor, max_distance_threshold
    )
    label_to_name = {int(v): k for k, v in name_to_id.items()}

    max_existing_label = numerical_labels_tensor.max().item() if numerical_labels_tensor.numel() > 0 else -1
    new_clusters = torch.unique(extrapolated_labels[extrapolated_labels > max_existing_label])
    for cluster_id in new_clusters:
        if cluster_id.item() not in label_to_name:
            label_to_name[cluster_id.item()] = f"Novel_{cluster_id.item()}"

    within_indices = torch.where(~(torch.isin(
        torch.arange(len(adapt_patches), device=model.device), outlier_indices)))[0]

    if len(adapt_patches) > 0:
        print(f"Launching UI for {len(adapt_patches)} adaptation samples "
              f"({len(within_indices)} in-threshold, {len(outlier_indices)} outliers).")
        outlier_dir = os.path.join(args.preadapt_bag, "outlier_patches")
        app = OutlierLabelUI(
            patches=adapt_patches,
            extrapolated_prefs=extrapolated_prefs,
            outlier_indices=outlier_indices,           # <-- true outliers
            within_indices=within_indices,             # <-- in-threshold
            adapt_phi_pro=adapt_phi_pro,
            save_dir=outlier_dir,
            default_labels=extrapolated_labels,
            preadapt_name_to_id=name_to_id
        )
        app.run()
        user_labels_and_ids, user_prefs = app.get_results()

        # overwrite whatever the user changed
        for idx, (string_label, cluster_id), pref in zip(
                range(len(adapt_patches)), user_labels_and_ids, user_prefs):
            extrapolated_labels[idx] = cluster_id
            extrapolated_prefs[idx] = pref
            if string_label not in name_to_id:
                name_to_id[string_label] = cluster_id
            if cluster_id not in label_to_name:
                label_to_name[cluster_id] = string_label
        shutil.rmtree(outlier_dir, ignore_errors=True)

    adapt_data_file = os.path.join(args.preadapt_bag, "clusters", "labeled_data.h5")

    # Load existing data
    with h5py.File(adapt_data_file, 'r') as f:
        existing_patches = f['patches'][:] if 'patches' in f else np.empty((0, 3, 128, 128), dtype=np.float32)
        existing_inertial = f['inertial'][:] if 'inertial' in f else np.empty((0, 115), dtype=np.float32)
        existing_labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in f['terrain_labels'][:]]
        existing_prefs = f['preferences'][:] if 'preferences' in f else np.empty((0,), dtype=np.float32)

    # Prepare new data
    new_patches = adapt_patches.cpu().numpy()
    new_inertial = adapt_inertial.cpu().numpy()
    new_labels = []
    for i in range(len(adapt_patches)):
        label_id = extrapolated_labels[i].item()
        label_str = label_to_name.get(label_id, f"unknown_{label_id}")
        new_labels.append(label_str)
    new_prefs = extrapolated_prefs.cpu().numpy()

    # Combine
    combined_patches = np.concatenate((existing_patches, new_patches))
    combined_inertial = np.concatenate((existing_inertial, new_inertial))
    combined_prefs = np.concatenate((existing_prefs, new_prefs))
    combined_labels = existing_labels + new_labels

    # Overwrite the file with combined data
    with h5py.File(adapt_data_file, 'w') as f:
        f.create_dataset('patches', data=combined_patches, compression="gzip")
        f.create_dataset('inertial', data=combined_inertial, compression="gzip")
        f.create_dataset('preferences', data=combined_prefs, compression="gzip")
        f.create_dataset('terrain_labels', data=combined_labels, dtype=h5py.string_dtype('utf-8'), compression="gzip")

    return adapt_data_file, label_to_name

def aggregate_and_split_datasets(preadapt_dataset, adapt_data_file, args):
    aggregated_dataset = TerrainDataset(labeled_dataset=adapt_data_file, transform=None)
    train_size = len(aggregated_dataset) - int(args.val_split * len(aggregated_dataset))
    val_size = len(aggregated_dataset) - train_size
    train_dataset, val_dataset = random_split(aggregated_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=custom_collate)
    return train_loader, val_loader, []

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

def extract_postadapt_features(model, train_dataset, args, name_to_id=None):
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
                batch_size = end_idx - start_idx
                phi_dset[start_idx:end_idx] = phi_pro.cpu().numpy()[:batch_size]
                patch_dset[start_idx:end_idx] = patches.cpu().numpy()[:batch_size]
                
                if name_to_id is not None:
                    mapped_labels = []
                    for label in terrain_labels:
                        if label is None:
                            mapped_labels.append(-1)
                        elif isinstance(label, (str, bytes)):
                            # Handle both string and bytes (from HDF5)
                            label_str = label.decode('utf-8') if isinstance(label, bytes) else label
                            mapped_labels.append(name_to_id.get(label_str, -1))
                        else:
                            mapped_labels.append(int(label))
                    labels.extend(mapped_labels)
                else:
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
    visualize_clusters(postadapt_phi_pro, labels, save_path=plot_path, title="Post-adaptation Clusters", 
                      label_to_name=label_to_name)

    print("Rendering patches in post-adaptation clusters...")
    patch_dir = os.path.join(args.preadapt_bag, "post_adaptation_patches")
    render_and_save_cluster_patches(postadapt_patches, labels, patch_dir, prefix="post_adapt", 
                                   label_to_name=label_to_name)

def save_models(model, args):
    print("Patern adaptation training completed. Saving models...")
    model.save_adapted_models(os.path.join(args.preadapt_bag, "models"))
    print("Models saved successfully.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="PATERN Preference Extrapolation")
    parser.add_argument("-preadapt_bag", "-pb", type=str, required=True)
    parser.add_argument("-adapt_bag", "-ab", type=str, required=True)
    parser.add_argument("-batch_size", type=int, default=4096)
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
    # Pass the adaptation bag path (points to /bags/<bag_name>/)
    adapt_bag_path = args.adapt_bag  # e.g., /bags/bag1/
    preadapt_phi_pro, preadapt_prefs, preadapt_labels, inertial_stats = extract_preadapt_features(model, preadapt_loader, cache_dir, device, adapt_bag_path)
    max_distance_threshold = compute_distance_threshold(preadapt_phi_pro, preadapt_labels, cache_dir)

    plot_path = os.path.join(args.preadapt_bag, "pre_adaptation_clusters.png")
    adapt_patches, adapt_inertial, adapt_phi_pro = extract_adapt_features(model, adapt_loader, cache_dir, inertial_stats)
    visualize_clusters(preadapt_phi_pro, preadapt_labels, adapt_phi_pro, plot_path, "Pre-adaptation Clusters", 
                      label_to_name={v: k for k, v in name_to_id.items()} if name_to_id else None)

    adapt_data_file, label_to_name = extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, 
                                                                     [preadapt_phi_pro, preadapt_prefs, preadapt_labels], 
                                                                     max_distance_threshold, args, name_to_id)
    
    config_path = os.path.join(script_dir, "homography", "config.yaml")
    name_to_id = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
            if 'terrains' in config:
                name_to_id = {terrain['name']: terrain['label'] for terrain in config['terrains']}
                print(f"Reloaded updated name-to-ID mapping: {name_to_id}")

    def custom_collate(batch):
        patches, inertial, terrain_labels, preferences = zip(*batch)
        terrain_labels_num = []
        for label in terrain_labels:
            if label is None:
                terrain_labels_num.append(-1)
            elif isinstance(label, str):
                terrain_labels_num.append(name_to_id.get(label, -1))
            else:
                terrain_labels_num.append(label)
        return torch.stack(patches), torch.stack(inertial), torch.tensor(terrain_labels_num, dtype=torch.long), torch.stack(preferences)
    
    train_loader, val_loader, adapt_data_files = aggregate_and_split_datasets(preadapt_dataset, adapt_data_file, args)
    retrain_model(model, train_loader, val_loader, preadapt_loader, args.epochs, max_distance_threshold, name_to_id)
    phi_pro_output, patch_output, correct_labels = extract_postadapt_features(
        model, train_loader.dataset, args, name_to_id=name_to_id  # Pass the updated mapping
    )

    for f in adapt_data_files:
        os.remove(f)

    label_to_name_updated = {v: k for k, v in name_to_id.items()}
    visualize_and_render(phi_pro_output, patch_output, correct_labels, train_loader.dataset, args, label_to_name_updated)
    save_models(model, args)


    print("Removing temporary directories...")
    for dir_path in [
        os.path.join(args.preadapt_bag, "postadapt_cache"),
        cache_dir,
        os.path.join(args.preadapt_bag, "adapt_data_cache")
    ]:
        shutil.rmtree(dir_path, ignore_errors=True)