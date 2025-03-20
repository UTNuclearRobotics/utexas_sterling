import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader, random_split
from utils import load_bag_pkl, load_bag_pt_model
from scripts.models import VisualEncoderModel, ProprioceptionModel, UtilityFuncVisual, UtilityFuncProprioceptive, CostNet
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
        
        # Utility functions (2-layer MLP on 128D vectors with scaling to 0-255)
        self.uvis = UtilityFuncVisual(latent_size=self.latent_size)
        self.upro = UtilityFuncProprioceptive(latent_size=self.latent_size)
        self.cost_head = CostNet(latent_size=self.latent_size)

        # Load pre-trained weights if provided
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            weight_files = {
                "visual_encoder": "fvis.pt",
                "proprioceptive_encoder": "fpro.pt",
                "uvis": "uvis.pt",
                "upro": "upro.pt",
                "cost_head": "cost_head.pt"
            }
            all_files_exist = all(os.path.exists(os.path.join(pretrained_weights_path, file_name)) for file_name in weight_files.values())
            if all_files_exist:
                for submodule_name, file_name in weight_files.items():
                    file_path = os.path.join(pretrained_weights_path, file_name)
                    state_dict = torch.load(file_path, weights_only=True, map_location=device)
                    submodule = getattr(self, submodule_name)
                    submodule.load_state_dict(state_dict)
                    print(f"Loaded {submodule_name} weights from {file_path} for fine-tuning")
            else:
                print(f"Warning: Not all required weight files found in {pretrained_weights_path}. Initializing from scratch.")
        else:
            print(f"No pre-trained weights directory found at {pretrained_weights_path}. Initializing from scratch.")

        # Initialize weights and biases to encourage positive outputs
        nn.init.kaiming_normal_(self.cost_head[0].weight, mode='fan_in', nonlinearity='relu')
        self.cost_head[0].bias.data.fill_(1.0)  # Positive bias to avoid ReLU zeroing out
        nn.init.kaiming_normal_(self.cost_head[2].weight, mode='fan_in', nonlinearity='relu')
        self.cost_head[2].bias.data.fill_(1.0)
        nn.init.kaiming_normal_(self.cost_head[4].weight, mode='fan_in', nonlinearity='relu')
        self.cost_head[4].bias.data.fill_(1.0)

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

    def forward(self, patches, inertial=None):
        patches = patches.to(self.device)
        phi_vis = self.visual_encoder(patches)
        uvis_pred = self.uvis(phi_vis)

        # Optionally process inertial data if provided
        if inertial is not None:
            inertial = inertial.to(self.device)
            phi_pro = self.proprioceptive_encoder(inertial.float())
            upro_pred = self.upro(phi_pro)
        else:
            phi_pro = torch.zeros_like(phi_vis)  # Dummy for consistency
            upro_pred = torch.zeros_like(uvis_pred)  # Dummy for consistency

        # Use only uvis_pred for final cost
        final_cost = self.cost_head(uvis_pred)
        
        return phi_vis, phi_pro, uvis_pred, upro_pred, final_cost

    def training_step(self, batch, batch_idx):
        patches, inertial, terrain_labels, preferences = batch
        preferences = preferences.to(self.device).float()
        
        # Normalize preferences to 0-255 range
        pref_min = preferences.min()
        pref_max = preferences.max()
        if pref_max > pref_min:  # Avoid division by zero
            scaled_preferences = ((preferences - pref_min) / (pref_max - pref_min)) * 255.0
        else:
            scaled_preferences = preferences * 0.0  # If all preferences are same, set to 0

        phi_vis, phi_pro, uvis_pred, upro_pred, final_cost = self.forward(patches, inertial)

        # Scale predictions to 0-255 based on their own range
        if uvis_pred.max() > uvis_pred.min():
            uvis_pred = ((uvis_pred - uvis_pred.min()) / (uvis_pred.max() - uvis_pred.min())) * 255.0
        else:
            uvis_pred = uvis_pred * 0.0

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
        pro_loss = self.triplet_loss(phi_pro, phi_pro[pos_indices], phi_pro[neg_indices])

        pref_diff = scaled_preferences.unsqueeze(1) - scaled_preferences.unsqueeze(0)  # Use scaled preferences
        pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
        ranking_mask = pref_diff > 0
        ranking_loss = F.relu(1.0 - (pred_diff / 255.0)[ranking_mask]).mean() if ranking_mask.any() else torch.tensor(0.0, device=self.device)

        modality_mse_loss = F.mse_loss(uvis_pred.detach(), upro_pred)
        cost_loss = F.mse_loss(final_cost, scaled_preferences)

        #total_loss = 1.0 * (vis_loss + 0.1*pro_loss) + 0.5 * ranking_loss + 0.5 * modality_mse_loss + 1.0 * cost_loss
        total_loss = 1.0 * (vis_loss + pro_loss) + 0.5 * ranking_loss + 0.5 * modality_mse_loss + 1.0 * cost_loss

        #print(f"Train Batch {batch_idx}: vis_loss={vis_loss.item():.4f}, pro_loss={pro_loss.item():.4f}, "
        #      f"ranking_loss={ranking_loss.item():.4f}, modality_mse_loss={modality_mse_loss.item():.4f}, "
        #      f"cost_loss={cost_loss.item():.4f}, total_loss={total_loss.item():.4f}")
        print(f"uvis_pred range: {uvis_pred.min().item():.4f} to {uvis_pred.max().item():.4f}")
        print(f"final_cost range: {final_cost.min().item():.4f} to {final_cost.max().item():.4f}")
        print(f"scaled_preferences range: {scaled_preferences.min().item():.4f} to {scaled_preferences.max().item():.4f}")
        return total_loss

    def validation_step(self, batch, batch_idx):
        patches, inertial, terrain_labels, preferences = batch
        preferences = preferences.to(self.device).float()
        
        # Normalize preferences to 0-255 range
        pref_min = preferences.min()
        pref_max = preferences.max()
        if pref_max > pref_min:
            scaled_preferences = ((preferences - pref_min) / (pref_max - pref_min)) * 255.0
        else:
            scaled_preferences = preferences * 0.0

        phi_vis, phi_pro, uvis_pred, upro_pred, final_cost = self.forward(patches, inertial)

        # Scale predictions to 0-255 based on their own range
        if uvis_pred.max() > uvis_pred.min():
            uvis_pred = ((uvis_pred - uvis_pred.min()) / (uvis_pred.max() - uvis_pred.min())) * 255.0
        else:
            uvis_pred = uvis_pred * 0.0
            
        if final_cost.max() > final_cost.min():
            final_cost = ((final_cost - final_cost.min()) / (final_cost.max() - final_cost.min())) * 255.0
        else:
            final_cost = final_cost * 0.0

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
        pro_loss = self.triplet_loss(phi_pro, phi_pro[pos_indices], phi_pro[neg_indices])

        pref_diff = scaled_preferences.unsqueeze(1) - scaled_preferences.unsqueeze(0)
        pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
        ranking_mask = pref_diff > 0
        ranking_loss = F.relu(1.0 - (pred_diff / 255.0)[ranking_mask]).mean() if ranking_mask.any() else torch.tensor(0.0, device=self.device)

        modality_mse_loss = F.mse_loss(uvis_pred.detach(), upro_pred)
        cost_loss = F.mse_loss(final_cost, scaled_preferences)

        #total_loss = 1.0 * (vis_loss + 0.1*pro_loss) + 0.5 * ranking_loss + 0.5 * modality_mse_loss + 1.0 * cost_loss
        total_loss = 1.0 * (vis_loss + pro_loss) + 0.5 * ranking_loss + 0.5 * modality_mse_loss + 1.0 * cost_loss
        return total_loss
    
    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.visual_encoder.state_dict(), os.path.join(save_dir, "fvis.pt"))
        torch.save(self.proprioceptive_encoder.state_dict(), os.path.join(save_dir, "fpro.pt"))
        torch.save(self.uvis.state_dict(), os.path.join(save_dir, "uvis.pt"))
        torch.save(self.upro.state_dict(), os.path.join(save_dir, "upro.pt"))
        torch.save(self.cost_head.state_dict(), os.path.join(save_dir, "cost_head.pt"))
        print(f"Saved PATERN− models to {save_dir}")

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    for epoch in range(epochs):
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
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-Adaptation Training for PATERN with 128D")
    parser.add_argument("-bag","-b", type=str, required=True, help="Base bag directory (e.g., bags/agh_courtyard_2)")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("-val_split", type=float, default=0.2, help="Fraction of dataset to use for validation (0.0 to 1.0)")
    args = parser.parse_args()

    # Load labeled dataset
    labeled_pkl_path = os.path.join(args.bag, "clusters", "labeled_dataset.pkl")
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
    models_dir = os.path.join(args.bag, "models")
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
    # Split dataset into training and validation
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = PaternPreAdaptation(device=device, pretrained_weights_path=models_dir, latent_size=128).to(device)

    # Check if weights were loaded (you can add a flag in PaternPreAdaptation)
    weights_loaded = False
    if os.path.exists(models_dir):
        weight_files = ["fvis.pt", "fpro.pt", "uvis.pt", "upro.pt", "cost_head.pt"]
        weights_loaded = all(os.path.exists(os.path.join(models_dir, file_name)) for file_name in weight_files)

    # Set learning rate based on whether weights were loaded
    lr = 1e-4 if weights_loaded else 1e-3  # Higher LR for training from scratch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # Optionally freeze encoders for initial epochs (only if fine-tuning)
    freeze_epochs = 5 if weights_loaded else 0  # No freezing if training from scratch
    if freeze_epochs > 0:
        for param in model.visual_encoder.parameters():
            param.requires_grad = False
        for param in model.proprioceptive_encoder.parameters():
            param.requires_grad = False

    print("Starting training")
    train_model(model, train_loader, val_loader, optimizer, scheduler, args.epochs, device)

    # Unfreeze encoders if they were frozen
    if freeze_epochs > 0 and args.epochs > freeze_epochs:
        for param in model.visual_encoder.parameters():
            param.requires_grad = True
        for param in model.proprioceptive_encoder.parameters():
            param.requires_grad = True
        train_model(model, train_loader, val_loader, optimizer, scheduler, args.epochs - freeze_epochs, device)

    model.save_models(save_dir)