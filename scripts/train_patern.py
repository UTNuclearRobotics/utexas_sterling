import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader, random_split
from utils import load_bag_pkl, load_bag_pt_model
from visual_encoder_model import VisualEncoderModel
from proprioception_model import ProprioceptionModel
import pickle
import sys


class PaternPreAdaptation(nn.Module):
    def __init__(self, device, pretrained_weights_path=None, latent_size=128):
        super(PaternPreAdaptation, self).__init__()
        self.device = device
        self.latent_size = latent_size  # Fixed at 128D

        # Initialize encoders
        self.visual_encoder = VisualEncoderModel(latent_size=self.latent_size)
        self.proprioceptive_encoder = ProprioceptionModel(latent_size=self.latent_size)

        # Load pre-trained weights if provided
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            full_state_dict = torch.load(pretrained_weights_path, weights_only=True)
            visual_state_dict = {k.replace('visual_encoder.', ''): v for k, v in full_state_dict.items() if k.startswith('visual_encoder.')}
            self.visual_encoder.load_state_dict(visual_state_dict, strict=False)
            print(f"Loaded visual encoder weights from {pretrained_weights_path}")
            proprio_state_dict = {k.replace('proprioceptive_encoder.', ''): v for k, v in full_state_dict.items() if k.startswith('proprioceptive_encoder.')}
            self.proprioceptive_encoder.load_state_dict(proprio_state_dict, strict=False)
            print(f"Loaded proprioception encoder weights from {pretrained_weights_path}")
        elif pretrained_weights_path:
            raise FileNotFoundError(f"Pre-trained weights file not found at: {pretrained_weights_path}")

        # Utility functions (2-layer MLP on 128D vectors)
        self.uvis = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.upro = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

    def forward(self, patches, inertial):
        patches = patches.to(self.device)
        inertial = inertial.to(self.device)
        phi_vis = self.visual_encoder(patches)
        phi_pro = self.proprioceptive_encoder(inertial.float())
        uvis_pred = self.uvis(phi_vis)
        upro_pred = self.upro(phi_pro)
        return phi_vis, phi_pro, uvis_pred, upro_pred

    def training_step(self, batch, batch_idx):
        patches, inertial, terrain_labels, preferences = batch
        preferences = preferences.to(self.device)

        terrain_labels_tensor = torch.tensor([hash(label) for label in terrain_labels], dtype=torch.long, device=self.device)
        phi_vis, phi_pro, uvis_pred, upro_pred = self.forward(patches, inertial)

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
        pro_loss = self.triplet_loss(phi_pro, phi_pro[pos_indices], phi_pro[neg_indices])

        pref_diff = preferences.unsqueeze(1) - preferences.unsqueeze(0)
        pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
        ranking_mask = pref_diff > 0
        ranking_loss = F.relu(1.0 - pred_diff[ranking_mask]).mean() if ranking_mask.any() else torch.tensor(0.0, device=self.device)

        mse_loss = F.mse_loss(uvis_pred.detach(), upro_pred)

        total_loss = vis_loss + pro_loss + ranking_loss + mse_loss
        #print(f"Train Batch {batch_idx}: vis_loss={vis_loss.item():.4f}, pro_loss={pro_loss.item():.4f}, ranking_loss={ranking_loss.item():.4f}, mse_loss={mse_loss.item():.4f}, total_loss={total_loss.item():.4f}")
        #print(f"uvis_pred norm: {torch.norm(uvis_pred).item():.4f}, upro_pred norm: {torch.norm(upro_pred).item():.4f}")
        return total_loss

    def validation_step(self, batch, batch_idx):
        patches, inertial, terrain_labels, preferences = batch
        preferences = preferences.to(self.device)

        terrain_labels_tensor = torch.tensor([hash(label) for label in terrain_labels], dtype=torch.long, device=self.device)
        phi_vis, phi_pro, uvis_pred, upro_pred = self.forward(patches, inertial)

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
        pro_loss = self.triplet_loss(phi_pro, phi_pro[pos_indices], phi_pro[neg_indices])

        pref_diff = preferences.unsqueeze(1) - preferences.unsqueeze(0)
        pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
        ranking_mask = pref_diff > 0
        ranking_loss = F.relu(1.0 - pred_diff[ranking_mask]).mean() if ranking_mask.any() else torch.tensor(0.0, device=self.device)

        mse_loss = F.mse_loss(uvis_pred.detach(), upro_pred)

        total_loss = vis_loss + pro_loss + ranking_loss + mse_loss
        #print(f"Val Batch {batch_idx}: vis_loss={vis_loss.item():.4f}, pro_loss={pro_loss.item():.4f}, ranking_loss={ranking_loss.item():.4f}, mse_loss={mse_loss.item():.4f}, total_loss={total_loss.item():.4f}")
        return total_loss

    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.visual_encoder.state_dict(), os.path.join(save_dir, "fvis.pt"))
        torch.save(self.proprioceptive_encoder.state_dict(), os.path.join(save_dir, "fpro.pt"))
        torch.save(self.uvis.state_dict(), os.path.join(save_dir, "uvis.pt"))
        torch.save(self.upro.state_dict(), os.path.join(save_dir, "upro.pt"))
        print(f"Saved PATERN− models to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-Adaptation Training for PATERN with 128D")
    parser.add_argument("-bag_path", type=str, required=True, help="Base bag directory (e.g., bags/agh_courtyard_2)")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs for training")
    args = parser.parse_args()

    # Load labeled dataset
    labeled_pkl_path = os.path.join(args.bag_path, "clusters", "labeled_dataset.pkl")
    print(f"Attempting to load: {labeled_pkl_path}")
    try:
        with open(labeled_pkl_path, 'rb') as f:
            labeled_data = pickle.load(f)
        print("Successfully loaded labeled_data with pickle")
    except Exception as e:
        print(f"Failed to load {labeled_pkl_path}: {e}")
        sys.exit(1)
    print(f"labeled_data type: {type(labeled_data)}, length or attributes: {len(labeled_data) if isinstance(labeled_data, (list, tuple)) else dir(labeled_data)[:10]}")
    if not labeled_data:
        print("Warning: labeled_data is empty")
        sys.exit(1)

    # Search for pre-trained weights
    models_dir = os.path.join(args.bag_path, "models")
    save_dir = models_dir
    pretrained_weights_path = None
    if os.path.exists(models_dir):
        for file_name in os.listdir(models_dir):
            if file_name.endswith("terrain_rep.pt"):
                pretrained_weights_path = os.path.join(models_dir, file_name)
                break
    print(f"Pre-trained weights: {pretrained_weights_path or 'None'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    print("Creating TerrainDataset instance")
    try:
        dataset = TerrainDataset(labeled_data=labeled_data, transform=None)
        print("TerrainDataset created successfully")
    except Exception as e:
        print(f"Failed to create TerrainDataset: {e}")
        sys.exit(1)
    train_size = int(0.75 * len(dataset))  # 75% for training
    val_size = len(dataset) - train_size  # 25% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PaternPreAdaptation(device=device, pretrained_weights_path=pretrained_weights_path, latent_size=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True)  # Lowered LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    print("Starting training")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                val_loss = model.validation_step(batch, batch_idx)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    model.save_models(save_dir)