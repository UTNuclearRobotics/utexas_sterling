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
        self.base_cluster_id = None
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
            n_clusters = 1
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
        window.set_default_size(800, 600)
        
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
        adapt_phi_pro = self.extract_proprioceptive_features(adaptation_inertial)
        n_samples = adapt_phi_pro.shape[0]
        if n_samples < 1:
            raise ValueError("Adaptation-set has no samples.")

        preadapt_phi_pro = preadapt_phi_pro.to(self.device)
        preadapt_prefs = preadapt_preferences.to(self.device)
        preadapt_labels = preadapt_labels.to(self.device)

        unique_clusters = torch.unique(preadapt_labels)
        n_clusters = len(unique_clusters)
        if n_clusters < 1:
            raise ValueError("No clusters found in pre-adaptation labels.")

        preadapt_cluster_centers = torch.zeros((n_clusters, preadapt_phi_pro.shape[1]), device=self.device)
        cluster_prefs = torch.zeros(n_clusters, device=self.device)
        cluster_labels = torch.zeros(n_clusters, dtype=torch.long, device=self.device)
        cluster_std = torch.zeros(n_clusters, device=self.device)

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = (preadapt_labels == cluster_id)
            cluster_points = preadapt_phi_pro[cluster_mask]
            preadapt_cluster_centers[i] = cluster_points.mean(dim=0)
            cluster_prefs[i] = preadapt_prefs[cluster_mask].mean()
            cluster_labels[i] = cluster_id
            cluster_std[i] = torch.std(cluster_points, dim=0).mean()

        distances = torch.cdist(adapt_phi_pro, preadapt_cluster_centers)
        sorted_distances, sorted_indices = distances.sort(dim=1)
        min_distances = sorted_distances[:, 0]
        nearest_cluster_indices = sorted_indices[:, 0]

        dynamic_thresholds = cluster_std[nearest_cluster_indices] * 2.0

        extrapolated_prefs = torch.zeros(n_samples, device=self.device)
        extrapolated_labels = cluster_labels[nearest_cluster_indices]
        within_threshold = min_distances <= dynamic_thresholds

        extrapolated_prefs[within_threshold] = cluster_prefs[nearest_cluster_indices[within_threshold]]

        outside_threshold = ~within_threshold
        outlier_indices = torch.where(outside_threshold)[0]
        
        if outside_threshold.sum() > 0:
            nearest_two_indices = sorted_indices[outside_threshold, :2]
            nearest_two_distances = sorted_distances[outside_threshold, :2]
            weights = 1.0 / (nearest_two_distances + 1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True)
            prefs_nearest_two = cluster_prefs[nearest_two_indices]
            extrapolated_prefs[outside_threshold] = (prefs_nearest_two * weights).sum(dim=1)

        return extrapolated_prefs, extrapolated_labels, outlier_indices, adapt_phi_pro

    def retrain_visual_components(self, train_loader, val_loader, optimizer, scheduler, epochs, initial_weights=None):
        l2_lambda = 0.001  # Hyperparameter for L2 penalty strength (tune as needed)
        
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
                task_loss = vis_loss + 0.5 * ranking_loss + 2.0 * cost_loss

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
                    #print(f"Batch - Task Loss: {task_loss.item():.4f}, L2 Penalty: {l2_lambda * l2_penalty.item():.4f}")
                else:
                    total_loss = task_loss

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
    pretrained_weights_path = os.path.join(args.preadapt_bag, "models")
    save_dir = pretrained_weights_path  # Assuming adapted models are in the same directory

    # Initialize the model with pretrained weights
    model = PaternAdaptation(device=device, pretrained_weights_path=pretrained_weights_path, latent_size=128).to(device)

    # Check for existing adapted models
    adapted_weight_files = {
        "visual_encoder": "fvis_adapted.pt",
        "uvis": "uvis_adapted.pt",
        "cost_head": "cost_head_adapted.pt"
    }
    
    adapted_models_found = False
    all_adapted_exist = True
    for name, file in adapted_weight_files.items():
        adapted_path = os.path.join(save_dir, file)
        if os.path.exists(adapted_path):
            state_dict = torch.load(adapted_path, weights_only=True, map_location=device)
            getattr(model, name).load_state_dict(state_dict)
            print(f"Loaded adapted {name} from {adapted_path}")
            adapted_models_found = True
        else:
            all_adapted_exist = False
            print(f"Adapted model {file} not found in {save_dir}. Using pretrained weights for {name}.")

    if adapted_models_found and all_adapted_exist:
        print("All adapted models found and loaded successfully.")
    elif adapted_models_found:
        print("Some adapted models found and loaded; missing ones use pretrained weights.")
    else:
        print("No adapted models found. Proceeding with pretrained weights only.")

    return model, adapted_models_found

def create_dataloaders(preadapt_dataset, adapt_dataset, batch_size):
    print("Creating DataLoaders...")
    preadapt_loader = DataLoader(preadapt_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    adapt_loader = DataLoader(adapt_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    return preadapt_loader, adapt_loader

def extract_preadapt_features(model, preadapt_loader, cache_dir, device):
    preadapt_phi_pro_path = os.path.join(cache_dir, "preadapt_phi_pro.pt")
    preadapt_prefs_path = os.path.join(cache_dir, "preadapt_prefs.pt")
    preadapt_labels_path = os.path.join(cache_dir, "preadapt_labels.npy")
    preadapt_stats_path = os.path.join(cache_dir, "preadapt_inertial_stats.pt")
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(preadapt_phi_pro_path) or not os.path.exists(preadapt_stats_path):
        print("Extracting pre-adaptation features and computing inertial statistics...")
        preadapt_phi_pro, preadapt_prefs, preadapt_labels_list = None, None, []
        inertial_data = []
        for i, batch in enumerate(preadapt_loader):
            _, inertial, terrain_labels, preferences = batch
            inertial_data.append(inertial)
            preferences = preferences.to(device)
            preadapt_prefs = preferences if preadapt_prefs is None else torch.cat([preadapt_prefs, preferences])
            preadapt_labels_list.extend(terrain_labels)
            if i % 10 == 0:
                print(f"Processed {i * args.batch_size} pre-adaptation samples, RAM usage: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB" if device.type == "cuda" else "CPU mode")

        # Compute inertial statistics
        inertial_data = torch.cat(inertial_data)
        inertial_mean = inertial_data.mean(dim=0)
        inertial_std = inertial_data.std(dim=0, unbiased=True) + 1e-6  # Avoid division by zero
        torch.save({'mean': inertial_mean, 'std': inertial_std}, preadapt_stats_path)
        print(f"Saved inertial statistics to {preadapt_stats_path}")

        # Normalize inertial data and extract features
        for i, batch in enumerate(preadapt_loader):
            _, inertial, _, _ = batch
            normalized_inertial = (inertial.to(device) - inertial_mean.to(device)) / inertial_std.to(device)
            phi_pro = model.extract_proprioceptive_features(normalized_inertial)
            preadapt_phi_pro = phi_pro if preadapt_phi_pro is None else torch.cat([preadapt_phi_pro, phi_pro])
            del phi_pro
            torch.cuda.empty_cache() if device.type == "cuda" else None

        preadapt_labels = np.array([hash(label) % n_clusters for label in preadapt_labels_list])
        print("Saving pre-adaptation features to cache...")
        torch.save(preadapt_phi_pro, preadapt_phi_pro_path)
        torch.save(preadapt_prefs, preadapt_prefs_path)
        np.save(preadapt_labels_path, preadapt_labels)
    else:
        print("Loading pre-adaptation features and inertial statistics from cache...")
        preadapt_phi_pro = torch.load(preadapt_phi_pro_path, map_location=device)
        preadapt_prefs = torch.load(preadapt_prefs_path, map_location=device)
        preadapt_labels = np.load(preadapt_labels_path)
        inertial_stats = torch.load(preadapt_stats_path, map_location=device)
        inertial_mean, inertial_std = inertial_stats['mean'], inertial_stats['std']

    # Log feature ranges
    print(f"Preadaptation phi_pro range: min={preadapt_phi_pro.min().item():.4f}, max={preadapt_phi_pro.max().item():.4f}, mean={preadapt_phi_pro.mean().item():.4f}, std={preadapt_phi_pro.std().item():.4f}")
    print(f"Preadaptation phi_pro shape: {preadapt_phi_pro.shape}")
    return preadapt_phi_pro, preadapt_prefs, preadapt_labels, {'mean': inertial_mean, 'std': inertial_std}

def compute_distance_threshold(preadapt_phi_pro, preadapt_labels, cache_dir):
    print("Computing distance threshold...")
    batch_size = 512
    distances_cache_dir = os.path.join(cache_dir, "distances")
    os.makedirs(distances_cache_dir, exist_ok=True)
    distance_files = []

    for i in range(0, len(preadapt_phi_pro), batch_size):
        batch_phi = preadapt_phi_pro[i:i + batch_size].cpu().numpy()
        batch_dist_file = os.path.join(distances_cache_dir, f"batch_dist_{i}.npy")
        if not os.path.exists(batch_dist_file):
            batch_dist = euclidean_distances(batch_phi, preadapt_phi_pro.cpu().numpy())
            np.save(batch_dist_file, batch_dist)
        distance_files.append(batch_dist_file)
        del batch_phi
        if i % 1000 == 0:
            print(f"Processed {i} samples for distance calculation")

    print("Processing distances to compute intra/inter means...")
    n_samples = len(preadapt_labels)
    intra_sum, intra_count, inter_sum, inter_count = 0.0, 0, 0.0, 0
    label_mask = preadapt_labels[:, None] == preadapt_labels[None, :] if n_samples * n_samples < 1e9 else None
    intra_mask = label_mask & ~np.eye(n_samples, dtype=bool) if label_mask is not None else None
    inter_mask = ~label_mask if label_mask is not None else None

    for i, dist_file in enumerate(distance_files):
        batch_dist = np.load(dist_file, mmap_mode='r')
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, n_samples)
        batch_labels = preadapt_labels[start_idx:end_idx]
        if label_mask is not None:
            intra_sum += np.sum(batch_dist * intra_mask[start_idx:end_idx, :])
            intra_count += np.sum(intra_mask[start_idx:end_idx, :])
            inter_sum += np.sum(batch_dist * inter_mask[start_idx:end_idx, :])
            inter_count += np.sum(inter_mask[start_idx:end_idx, :])
        else:
            batch_label_mask = batch_labels[:, None] == preadapt_labels[None, :]
            batch_intra_mask = batch_label_mask & ~np.eye(batch_dist.shape[0], n_samples, dtype=bool)
            batch_inter_mask = ~batch_label_mask
            intra_sum += np.sum(batch_dist * batch_intra_mask)
            intra_count += np.sum(batch_intra_mask)
            inter_sum += np.sum(batch_dist * batch_inter_mask)
            inter_count += np.sum(batch_inter_mask)
        del batch_dist
        if i % 10 == 0:
            print(f"Processed batch {i}, Intra count: {intra_count}, Inter count: {inter_count}")

    avg_intra = intra_sum / intra_count if intra_count > 0 else 0.0
    avg_inter = inter_sum / inter_count if inter_count > 0 else 0.0
    max_distance_threshold = (avg_intra + avg_inter) / 2 if avg_intra > 0 and avg_inter > 0 else 5.0
    print(f"Avg Intra-cluster Distance: {avg_intra:.4f}, Avg Inter-cluster Distance: {avg_inter:.4f}")
    print(f"Computed Max Distance Threshold: {max_distance_threshold:.4f}")

    for dist_file in distance_files:
        os.remove(dist_file)
    os.rmdir(distances_cache_dir)
    return max_distance_threshold

def extract_adapt_features(model, adapt_loader, cache_dir, inertial_stats):
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

    print("Extracting proprioceptive features for adaptation data with normalization...")
    # Normalize inertial data using pre-adaptation statistics
    model_device = next(model.parameters()).device
    normalized_inertial = (adapt_inertial.to(model_device) - inertial_stats['mean'].to(model_device)) / inertial_stats['std'].to(model_device)
    adapt_phi_pro = model.extract_proprioceptive_features(normalized_inertial)
    
    # Log feature ranges
    print(f"Adaptation phi_pro range: min={adapt_phi_pro.min().item():.4f}, max={adapt_phi_pro.max().item():.4f}, mean={adapt_phi_pro.mean().item():.4f}, std={adapt_phi_pro.std().item():.4f}")
    print(f"Adaptation phi_pro shape: {adapt_phi_pro.shape}")
    
    return adapt_patches, adapt_inertial, adapt_phi_pro

# Updated extrapolate_and_cache_adapt_data function
def extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, preadapt_data, max_distance_threshold, args):
    print("Extrapolating preferences and labels...")
    extrapolated_prefs, extrapolated_labels, outlier_indices, adapt_phi_pro = model.extrapolate_preferences(
        adapt_inertial, preadapt_data[0], preadapt_data[1], torch.tensor(preadapt_data[2], device=adapt_inertial.device), max_distance_threshold
    )

    if len(outlier_indices) > 0:
        print(f"Found {len(outlier_indices)} data points outside cluster thresholds.")
        outlier_dir = os.path.join(args.preadapt_bag, "outlier_patches")
        # Load pre-adaptation config to get name-to-ID mapping
        config_path = os.path.join(script_dir, "homography", "config.yaml")
        preadapt_name_to_id = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file) or {}
                if 'terrains' in config:
                    preadapt_name_to_id = {terrain['name']: terrain['label'] for terrain in config['terrains']}
                    print(f"Loaded pre-adaptation name-to-ID mapping: {preadapt_name_to_id}")
        else:
            print(f"Warning: Config file {config_path} not found. New labels will get incremental IDs.")
        
        app = OutlierLabelUI(
            patches=adapt_patches,
            extrapolated_prefs=extrapolated_prefs,
            outlier_indices=outlier_indices,
            adapt_phi_pro=adapt_phi_pro,
            save_dir=outlier_dir,
            default_labels=extrapolated_labels,
            preadapt_name_to_id=preadapt_name_to_id
        )
        app.run()
        user_labels_and_ids, user_prefs = app.get_results()

        for idx, (string_label, cluster_id), pref in zip(outlier_indices, user_labels_and_ids, user_prefs):
            extrapolated_labels[idx] = cluster_id
            extrapolated_prefs[idx] = pref
        
        shutil.rmtree(outlier_dir, ignore_errors=True)

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
                group.attrs["terrain_label"] = labels_list[i]  # Numeric cluster ID
                group.attrs["preference"] = prefs_list[i]
            print(f"Processed {end_idx} adaptation data entries, RAM usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    del adapt_patches_cpu, adapt_inertial_cpu, prefs_list, labels_list
    gc.collect()
    print(f"RAM usage after clearing adaptation data: {psutil.virtual_memory().used / 1024**2:.2f} MB")
    return adapt_data_file

def aggregate_and_split_datasets(preadapt_dataset, adapt_data_file, args):
    print("Aggregating and splitting datasets...")
    preadapt_data_list = list(preadapt_dataset)
    adapt_data, label_map = [], {}

    print("Building label map from preadaptation data...")
    for item in preadapt_data_list:
        terrain_label = item[2] if isinstance(item, tuple) else item.get("terrain_label")
        if isinstance(terrain_label, str) and terrain_label not in label_map:
            label_map[terrain_label] = len(label_map)

    print("Loading and normalizing adaptation data...")
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

    print("Normalizing preadaptation data...")
    normalized_preadapt_data = []
    for item in preadapt_data_list:
        if isinstance(item, tuple):
            patch, inertial, terrain_label, preference = item
        else:
            patch, inertial, terrain_label, preference = item.values()
        terrain_label = label_map.get(terrain_label, -1) if isinstance(terrain_label, str) else terrain_label or -1
        normalized_preadapt_data.append({"patch": patch, "inertial": inertial, "terrain_label": terrain_label, "preference": preference})

    # Calculate and print preference ranges before aggregation
    preadapt_prefs = torch.tensor([d["preference"] for d in normalized_preadapt_data], dtype=torch.float32)
    adapt_prefs = torch.tensor([d["preference"] for d in adapt_data], dtype=torch.float32)
    print(f"Pre-adaptation data preference range: min={preadapt_prefs.min():.2f}, max={preadapt_prefs.max():.2f}")
    print(f"Adaptation data preference range: min={adapt_prefs.min():.2f}, max={adapt_prefs.max():.2f}")

    aggregated_data = normalized_preadapt_data + adapt_data

    del preadapt_data_list, adapt_data, normalized_preadapt_data
    gc.collect()
    print(f"RAM usage before TerrainDataset: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    aggregated_dataset = TerrainDataset(labeled_dataset=aggregated_data, transform=None)
    print(f"Successfully created TerrainDataset with {len(aggregated_dataset)} samples")

    val_size = int(args.val_split * len(aggregated_dataset))
    train_size = len(aggregated_dataset) - val_size
    train_dataset, val_dataset = random_split(aggregated_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=custom_collate)
    
    del aggregated_data
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    return train_loader, val_loader, [adapt_data_file]

def retrain_model(model, train_loader, val_loader, epochs):
    print("Retraining model...")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    model.retrain_visual_components(train_loader, val_loader, optimizer, scheduler, epochs)

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

    postadapt_labels = np.array([label if label is not None else -1 for label in postadapt_labels_list])
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
    model, adapted_models_found = initialize_model(args, device)
    preadapt_loader, adapt_loader = create_dataloaders(preadapt_dataset, adapt_dataset, args.batch_size)

    preadapt_cache_dir = os.path.join(args.preadapt_bag, "cache")
    preadapt_data = extract_preadapt_features(model, preadapt_loader, preadapt_cache_dir, device)[:3]
    inertial_stats = extract_preadapt_features(model, preadapt_loader, preadapt_cache_dir, device)[3]
    max_distance_threshold = compute_distance_threshold(preadapt_data[0], preadapt_data[2], preadapt_cache_dir)

    print("Visualizing pre-adaptation clusters...")
    preadapt_plot_path = os.path.join(args.preadapt_bag, "pre_adaptation_clusters.png")
    adapt_patches, adapt_inertial, adapt_phi_pro = extract_adapt_features(model, adapt_loader, preadapt_cache_dir, inertial_stats)
    visualize_clusters(preadapt_data[0], preadapt_data[2], adapt_phi_pro=adapt_phi_pro, save_path=preadapt_plot_path)

    adapt_data_file = extrapolate_and_cache_adapt_data(model, adapt_patches, adapt_inertial, preadapt_data, max_distance_threshold, args)
    train_loader, val_loader, adapt_data_files = aggregate_and_split_datasets(preadapt_dataset, adapt_data_file, args)
    
    # Retrain regardless of whether adapted models were found, but log the context
    if adapted_models_found:
        print("Adapted models detected. Retraining on top of adapted weights...")
    else:
        print("No adapted models detected. Training from pretrained weights...")
    
    retrain_model(model, train_loader, val_loader, args.epochs)

    phi_pro_output, patch_output, postadapt_labels_list = extract_postadapt_features(model, train_loader.dataset, args)
    
    for f in adapt_data_files:
        os.remove(f)
    
    visualize_and_render(phi_pro_output, patch_output, postadapt_labels_list, train_loader.dataset, args, n_clusters)
    save_models(model, args)

    # Cleanup temprorary directories
    postadapt_cache_dir = os.path.join(args.preadapt_bag, "postadapt_cache")
    cache_dir = os.path.join(args.preadapt_bag, "cache")
    adapt_data_cache_dir = os.path.join(args.preadapt_bag, "adapt_data_cache")
    try:
        shutil.rmtree(postadapt_cache_dir)
        shutil.rmtree(cache_dir)
        shutil.rmtree(adapt_data_cache_dir)
        print(f"Successfully deleted {postadapt_cache_dir}, {cache_dir}, {adapt_data_cache_dir} and all its contents.")
    except OSError as e:
        print(f"Error deleting {postadapt_cache_dir}: {e.strerror}")