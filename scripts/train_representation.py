import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset, worker_init_fn
from torch.utils.data import DataLoader, Subset
from utils import load_bag_pt_model, load_bag_h5
from vicreg import VICRegLoss
from models import VisualEncoderModel, ProprioceptionModel
import torchvision.transforms.v2 as v2
import os
from torch.utils.data import random_split
from tqdm import tqdm
from homography_params import get_homography_params


class SterlingRepresentation(nn.Module):
    def __init__(self, device, pretrained_weights_dir=None, latent_size=128):
        super(SterlingRepresentation, self).__init__()
        self.device = device
        self.latent_size = latent_size
        self.rep_size = self.latent_size

        # Initialize encoders
        self.visual_encoder = VisualEncoderModel(self.latent_size)
        self.proprioceptive_encoder = ProprioceptionModel(self.latent_size)

        # Load pre-trained weights if provided (files ending with terrain_rep.pt)
        if pretrained_weights_dir and os.path.exists(pretrained_weights_dir):
            terrain_rep_files = [file for file in os.listdir(pretrained_weights_dir) if file.endswith("terrain_rep.pt")]
            if terrain_rep_files:
                file_path = os.path.join(pretrained_weights_dir, terrain_rep_files[0])  # Load the first matching file
                state_dict = torch.load(file_path, weights_only=True, map_location=self.device)
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded full model weights from {file_path} for fine-tuning")
            else:
                print(f"Warning: No files ending with terrain_rep.pt found in {pretrained_weights_dir}. Initializing from scratch.")
        else:
            print(f"No pre-trained weights directory found at {pretrained_weights_dir}. Initializing from scratch.")

        # Projector for VICReg
        self.projector = nn.Sequential(
            nn.Linear(self.rep_size, self.latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size, self.latent_size),
        )

        self.vicreg_loss = VICRegLoss()
        self.l1_coeff = 0.5

    def forward(self, patch1, patch2, inertial_data):
        patch1 = patch1.to(self.device)
        patch2 = patch2.to(self.device)
        inertial_data = inertial_data.to(self.device)
        v_encoded_1 = self.visual_encoder(patch1)
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2)
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)
        i_encoded = self.proprioceptive_encoder(inertial_data.float())

        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)

        return zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded

    def encode_single_patch(self, patch):
        patch = patch.to(self.device)
        v_encoded = self.visual_encoder(patch)
        v_encoded = F.normalize(v_encoded, dim=-1)
        return v_encoded

    def encode_IMU(self, inertial_data):
        inertial_data = inertial_data.to(self.device)
        i_encoded = self.proprioceptive_encoder(inertial_data)
        return i_encoded

    def get_terrain_embedding(self, patch, inertial_data):
        patch = patch.to(self.device)
        inertial_data = inertial_data.to(self.device)
        v_encoded = self.encode_single_patch(patch)
        i_encoded = self.proprioceptive_encoder(inertial_data.float())
        zv = self.projector(v_encoded)
        zi = self.projector(i_encoded)
        combined_embedding = torch.cat((zv, zi), dim=-1)
        return combined_embedding

    def training_step(self, batch, batch_idx):
        patch1, patch2, inertial = batch
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial)
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi
        return loss

    def validation_step(self, batch, batch_idx):
        patch1, patch2, inertial = batch
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial)
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi
        return loss

def custom_collate(batch):
    patch1s, patch2s, imus = zip(*batch)
    patch1s = torch.stack(patch1s)
    patch2s = torch.stack(patch2s)
    imus = torch.stack(imus)
    return patch1s, patch2s, imus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sterling Representation Model")
    parser.add_argument("-bag", "-b", type=str, required=True)
    parser.add_argument("-batch_size", "-batch", type=int, default=4096)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-val_split", type=float, default=0.2)
    args = parser.parse_args()

    px_meter = get_homography_params().px_meter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vicreg_h5_path = load_bag_h5(args.bag, "vicreg")
    synced_h5_path = load_bag_h5(args.bag, "synced")

    augment_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
    ]).to(device)  # Move to GPU

    dataset = TerrainDataset(synced_h5_path=synced_h5_path, vicreg_h5_path=vicreg_h5_path, incl_orientation=True, train=True, patch_size=128)
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, pin_memory=False, collate_fn=custom_collate, 
        worker_init_fn=worker_init_fn  # Use the standalone function
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, pin_memory=False, collate_fn=custom_collate, 
        worker_init_fn=worker_init_fn  # Use the standalone function
    )

    # Initialize model with pre-trained weights
    models_dir = os.path.join(args.bag, "models")
    model = SterlingRepresentation(device, pretrained_weights_dir=models_dir).to(device)
    save_path = load_bag_pt_model(args.bag, "terrain_rep", model)

    # Check if weights were loaded
    weights_loaded = False
    if os.path.exists(models_dir):
        weight_files = [file for file in os.listdir(models_dir) if file.endswith("terrain_rep.pt")]
        weights_loaded = any(os.path.exists(os.path.join(models_dir, file_name)) 
                            for file_name in weight_files)

    # Define optimizer with a lower learning rate for fine-tuning
    lr = 1e-4 if weights_loaded else 3e-4  # Lower LR if fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )

    # Optionally freeze encoders for initial epochs
    freeze_epochs = 10 if weights_loaded else 0
    if freeze_epochs > 0:
        for param in model.visual_encoder.parameters():
            param.requires_grad = False
        for param in model.proprioceptive_encoder.parameters():
            param.requires_grad = False

    # Training and validation loop
    best_val_loss = float('inf')
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs - 1 and freeze_epochs > 0:
            for param in model.visual_encoder.parameters():
                param.requires_grad = True
            for param in model.proprioceptive_encoder.parameters():
                param.requires_grad = True
            print("Unfrozen visual and proprioceptive encoders for full fine-tuning.")

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                val_loss = model.validation_step(batch, batch_idx)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved model with validation loss: {best_val_loss:.4f}")

    dataset.__del__()