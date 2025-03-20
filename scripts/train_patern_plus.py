import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader, random_split
from scripts.models import VisualEncoderModel, ProprioceptionModel, UtilityFuncVisual, UtilityFuncProprioceptive, CostNet
import pickle
from sklearn.cluster import KMeans
import numpy as np

class PaternAdaptation(nn.Module):
    def __init__(self, device, pretrained_weights_path, latent_size=128):
        super(PaternAdaptation, self).__init__()
        self.device = device
        self.latent_size = latent_size
        self.visual_encoder = VisualEncoderModel(latent_size=self.latent_size)
        self.proprioceptive_encoder = ProprioceptionModel(latent_size=self.latent_size)
        self.uvis = UtilityFuncVisual(latent_size=self.latent_size)
        self.upro = UtilityFuncProprioceptive(latent_size=self.latent_size)
        self.cost_head = CostNet(latent_size=self.latent_size)

        # Load pre-trained weights
        weight_files = {
            "visual_encoder": "fvis.pt",
            "proprioceptive_encoder": "fpro.pt",
            "uvis": "uvis.pt",
            "upro": "upro.pt",
            "cost_head": "cost_head.pt"
        }
        for submodule_name, file_name in weight_files.items():
            file_path = os.path.join(pretrained_weights_path, file_name)
            if os.path.exists(file_path):
                state_dict = torch.load(file_path, weights_only=True, map_location=device)
                getattr(self, submodule_name).load_state_dict(state_dict)
                print(f"Loaded {submodule_name} from {file_path}")
            else:
                raise FileNotFoundError(f"Missing {file_name} in {pretrained_weights_path}")

        # Freeze fpro and upro
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
        return phi_pro.cpu().numpy()

    def extrapolate_preferences(self, adaptation_inertial, preadapt_phi_pro, preadapt_preferences, threshold=1.0):
        adapt_phi_pro = self.extract_proprioceptive_features(adaptation_inertial)
        kmeans = KMeans(n_clusters=min(5, len(adapt_phi_pro)), random_state=42)
        adapt_clusters = kmeans.fit_predict(adapt_phi_pro)
        adapt_cluster_centers = kmeans.cluster_centers_
        preadapt_phi_pro_np = preadapt_phi_pro.cpu().numpy() if torch.is_tensor(preadapt_phi_pro) else preadapt_phi_pro
        distances = np.linalg.norm(adapt_cluster_centers[:, np.newaxis] - preadapt_phi_pro_np[np.newaxis, :], axis=2)
        nearest_indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(adapt_cluster_centers)), nearest_indices]
        extrapolated_prefs = [preadapt_preferences[idx].item() if dist <= threshold else 0.0 for dist, idx in zip(min_distances, nearest_indices)]
        return torch.tensor(extrapolated_prefs, dtype=torch.float32, device=self.device)

    def retrain_visual_components(self, train_loader, val_loader, optimizer, scheduler, epochs):
        for epoch in range(epochs):
            self.train()
            total_train_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                patches, inertial, terrain_labels, preferences = batch
                preferences = preferences.to(self.device).float()
                pref_min, pref_max = preferences.min(), preferences.max()
                scaled_preferences = ((preferences - pref_min) / (pref_max - pref_min)) * 255.0 if pref_max > pref_min else preferences * 0.0

                phi_vis, phi_pro, uvis_pred, upro_pred, final_cost = self.forward(patches, inertial)
                if uvis_pred.max() > uvis_pred.min():
                    uvis_pred = ((uvis_pred - uvis_pred.min()) / (uvis_pred.max() - uvis_pred.min())) * 255.0

                terrain_labels_tensor = torch.tensor([hash(label) for label in terrain_labels], dtype=torch.long, device=self.device)
                batch_size = len(terrain_labels)
                labels_expanded = terrain_labels_tensor.unsqueeze(1)
                pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                neg_mask = (labels_expanded != labels_expanded.t())
                pos_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                neg_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                for i in range(batch_size):
                    pos_candidates = pos_mask[i].nonzero(as_tuple=False).flatten()
                    neg_candidates = neg_mask[i].nonzero(as_tuple=False).flatten()
                    pos_indices[i] = pos_candidates[torch.randint(0, len(pos_candidates), (1,), device=self.device)] if len(pos_candidates) > 0 else i
                    neg_indices[i] = neg_candidates[torch.randint(0, len(neg_candidates), (1,), device=self.device)] if len(neg_candidates) > 0 else i

                vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
                ranking_loss = F.relu(1.0 - ((uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)) / 255.0)[(scaled_preferences.unsqueeze(1) > scaled_preferences.unsqueeze(0))]).mean()
                cost_loss = F.mse_loss(final_cost, scaled_preferences)
                total_loss = 1.0 * vis_loss + 0.5 * ranking_loss + 1.0 * cost_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += total_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    patches, inertial, terrain_labels, preferences = batch
                    preferences = preferences.to(self.device).float()
                    scaled_preferences = ((preferences - preferences.min()) / (preferences.max() - preferences.min())) * 255.0 if preferences.max() > preferences.min() else preferences * 0.0
                    phi_vis, _, uvis_pred, _, final_cost = self.forward(patches, inertial)
                    vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
                    ranking_loss = F.relu(1.0 - ((uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)) / 255.0)[(scaled_preferences.unsqueeze(1) > scaled_preferences.unsqueeze(0))]).mean()
                    cost_loss = F.mse_loss(final_cost, scaled_preferences)
                    total_val_loss += (vis_loss + 0.5 * ranking_loss + cost_loss).item()
            avg_val_loss = total_val_loss / len(val_loader)

            scheduler.step()
            print(f"Adaptation Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    def save_adapted_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.visual_encoder.state_dict(), os.path.join(save_dir, "fvis_adapted.pt"))
        torch.save(self.uvis.state_dict(), os.path.join(save_dir, "uvis_adapted.pt"))
        print(f"Saved PATERN+ models to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Preference Extrapolation Training for PATERN")
    parser.add_argument("-bag", "-b", type=str, required=True, help="Base bag directory with pre-trained models")
    parser.add_argument("-adapt_bag", type=str, required=True, help="Adaptation-set bag directory")
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-val_split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-adaptation data
    preadapt_pkl_path = os.path.join(args.bag, "clusters", "labeled_dataset.pkl")
    with open(preadapt_pkl_path, 'rb') as f:
        preadapt_data = pickle.load(f)
    preadapt_dataset = TerrainDataset(labeled_data=preadapt_data, transform=None)

    # Load adaptation-set data
    adapt_pkl_path = os.path.join(args.adapt_bag, "clusters", "labeled_dataset.pkl")
    with open(adapt_pkl_path, 'rb') as f:
        adapt_data = pickle.load(f)
    adapt_dataset = TerrainDataset(labeled_data=adapt_data, transform=None)

    # Initialize model with pre-trained weights
    models_dir = os.path.join(args.bag, "models")
    model = PaternAdaptation(device=device, pretrained_weights_path=models_dir, latent_size=128).to(device)

    # Extract pre-adaptation ϕpro and preferences
    preadapt_loader = DataLoader(preadapt_dataset, batch_size=args.batch_size, shuffle=False)
    preadapt_phi_pro_list, preadapt_prefs_list = [], []
    for batch in preadapt_loader:
        _, inertial, _, preferences = batch
        phi_pro = model.extract_proprioceptive_features(inertial)
        preadapt_phi_pro_list.append(phi_pro)
        preadapt_prefs_list.append(preferences)
    preadapt_phi_pro = torch.tensor(np.concatenate(preadapt_phi_pro_list), device=device)
    preadapt_prefs = torch.cat(preadapt_prefs_list)

    # Extrapolate preferences for adaptation-set
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.batch_size, shuffle=False)
    adapt_inertial_list = [batch[1] for batch in adapt_loader]
    adapt_inertial = torch.cat(adapt_inertial_list)
    extrapolated_prefs = model.extrapolate_preferences(adapt_inertial, preadapt_phi_pro, preadapt_prefs)

    # Update adapt_dataset with extrapolated preferences
    for i, data in enumerate(adapt_data):
        data["preference"] = extrapolated_prefs[i % len(extrapolated_prefs)].item()

    # Aggregate datasets
    aggregated_data = preadapt_data + adapt_data
    aggregated_dataset = TerrainDataset(labeled_data=aggregated_data, transform=None)
    val_size = int(args.val_split * len(aggregated_dataset))
    train_size = len(aggregated_dataset) - val_size
    train_dataset, val_dataset = random_split(aggregated_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Retrain visual components
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    model.retrain_visual_components(train_loader, val_loader, optimizer, scheduler, args.epochs)

    # Save adapted models
    save_dir = os.path.join(args.adapt_bag, "models")
    model.save_adapted_models(save_dir)

if __name__ == "__main__":
    main()