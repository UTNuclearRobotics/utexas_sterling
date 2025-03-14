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
        patches = batch['patch']
        inertial = batch['inertial']
        terrain_labels = batch['terrain_label']
        preferences = batch['preference'].to(self.device)

        terrain_labels_tensor = torch.tensor([hash(label) for label in terrain_labels], dtype=torch.long).to(self.device)
        phi_vis, phi_pro, uvis_pred, upro_pred = self.forward(patches, inertial)

        anchor_idx = torch.arange(len(terrain_labels), device=self.device)
        pos_mask = torch.eq(terrain_labels_tensor.unsqueeze(1), terrain_labels_tensor.unsqueeze(0)) & ~torch.eye(len(terrain_labels), dtype=torch.bool, device=self.device)
        neg_mask = ~torch.eq(terrain_labels_tensor.unsqueeze(1), terrain_labels_tensor.unsqueeze(0))

        pos_idx = [torch.where(pos_mask[i])[0][0] if pos_mask[i].any() else i for i in range(len(terrain_labels))]
        neg_idx = [torch.where(neg_mask[i])[0][0] for i in range(len(terrain_labels))]
        pos_idx = torch.tensor(pos_idx, device=self.device)
        neg_idx = torch.tensor(neg_idx, device=self.device)

        vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_idx], phi_vis[neg_idx])
        pro_loss = self.triplet_loss(phi_pro, phi_pro[pos_idx], phi_pro[neg_idx])

        ranking_loss = 0
        count = 0
        for i in range(len(preferences)):
            for j in range(len(preferences)):
                if preferences[i] > preferences[j]:
                    ranking_loss += F.relu(1.0 - (uvis_pred[i] - uvis_pred[j]))
                    count += 1
        ranking_loss = ranking_loss / (count + 1e-6) if count > 0 else torch.tensor(0.0, device=self.device)

        mse_loss = F.mse_loss(uvis_pred.detach(), upro_pred)
        total_loss = vis_loss + pro_loss + ranking_loss + mse_loss
        return total_loss

    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.visual_encoder.state_dict(), os.path.join(save_dir, "fvis.pt"))
        torch.save(self.proprioceptive_encoder.state_dict(), os.path.join(save_dir, "fpro.pt"))
        torch.save(self.uvis.state_dict(), os.path.join(save_dir, "uvis.pt"))
        torch.save(self.upro.state_dict(), os.path.join(save_dir, "upro.pt"))
        print(f"Saved PATERN− models to {save_dir}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pre-Adaptation Training for PATERN with 128D")
    parser.add_argument("-bag_path", type=str, required=True, help="Base bag directory (e.g., bags/agh_courtyard_2)")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("-batch_size", type=int, default=256, help="Batch size for training")
    args = parser.parse_args()

    # Load labeled dataset
    labeled_pkl_path = os.path.join(args.bag_path, "clusters")
    labeled_data = load_bag_pkl(labeled_pkl_path, "labeled_dataset")  # Adjust identifier as needed
    if not labeled_data:
        raise FileNotFoundError(f"Failed to load labeled dataset from {labeled_pkl_path}")
    
    # Search for pre-trained weights
    models_dir = os.path.join(args.bag_path, "models")
    save_dir = models_dir
    pretrained_weights_path = None
    if os.path.exists(models_dir):
        for file_name in os.listdir(models_dir):
            if file_name.endswith("terrain_rep.pt"):
                pretrained_weights_path = os.path.join(models_dir, file_name)
                break

    # Verify file existence
    if pretrained_weights_path is None:
        print(f"Warning: No file ending with 'terrain_rep.pt' found in {models_dir}, proceeding without loading weights")
    else:
        print(f"Found pre-trained weights at: {pretrained_weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = TerrainDataset(labeled_data=labeled_data, transform=None)
    train_size = int(0.75 * len(dataset))  # 75-25 split
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = PaternPreAdaptation(
        device=device,
        pretrained_weights_path=pretrained_weights_path,
        latent_size=128
    ).to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5, amsgrad=True)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    # Save models
    model.save_models(save_dir)