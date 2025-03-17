import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from terrain_dataset import TerrainDataset
from torch.utils.data import DataLoader
from utils import load_bag_pkl, load_bag_pt_model
from vicreg import VICRegLoss
from visual_encoder_model import VisualEncoderModel
from proprioception_model import ProprioceptionModel
from torchvision import transforms
import torchvision.transforms.v2 as v2
import os

class SterlingRepresentation(nn.Module):
    def __init__(self, device, pretrained_weights_dir=None):
        super(SterlingRepresentation, self).__init__()
        self.device = device
        self.latent_size = 128
        self.rep_size = self.latent_size

        # Initialize encoders
        self.visual_encoder = VisualEncoderModel(self.latent_size)
        self.proprioceptive_encoder = ProprioceptionModel(self.latent_size)

        # Load pre-trained weights if provided
        if pretrained_weights_dir and os.path.exists(pretrained_weights_dir):
            weight_files = {
                "visual_encoder": "fvis.pt",
                "proprioceptive_encoder": "fpro.pt"
            }
            for submodule_name, file_name in weight_files.items():
                file_path = os.path.join(pretrained_weights_dir, file_name)
                if os.path.exists(file_path):
                    state_dict = torch.load(file_path, weights_only=True, map_location=self.device)
                    submodule = getattr(self, submodule_name)
                    submodule.load_state_dict(state_dict)
                    print(f"Loaded {submodule_name} weights from {file_path} for fine-tuning")
                else:
                    print(f"Warning: {file_name} not found in {pretrained_weights_dir}. Initializing {submodule_name} from scratch.")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sterling Representation Model")
    parser.add_argument("-bag", "-b", type=str, required=True, help="Bag directory with VICReg dataset pickle file inside.")
    parser.add_argument("-batch_size", "-batch", type=int, default=256, help="Batch size for training")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    patches_pkl = load_bag_pkl(args.bag, "vicreg")
    IPT_pkl = load_bag_pkl(args.bag, "_synced")

    # Define the augmentation pipeline
    augment_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        v2.ToTensor(),
    ])

    dataset = TerrainDataset(patches=patches_pkl, synced_data=IPT_pkl, transform=augment_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model with pre-trained weights
    models_dir = os.path.join(args.bag, "models")
    model = SterlingRepresentation(device, pretrained_weights_dir=models_dir).to(device)
    save_path = load_bag_pt_model(args.bag, "terrain_rep", model)

    # Check if weights were loaded
    weights_loaded = False
    if os.path.exists(models_dir):
        weight_files = ["fvis.pt", "fpro.pt"]
        weights_loaded = all(os.path.exists(os.path.join(models_dir, file_name)) for file_name in weight_files)

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
    freeze_epochs = 5 if weights_loaded else 0
    if freeze_epochs > 0:
        for param in model.visual_encoder.parameters():
            param.requires_grad = False
        for param in model.proprioceptive_encoder.parameters():
            param.requires_grad = False

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs - 1 and freeze_epochs > 0:
            for param in model.visual_encoder.parameters():
                param.requires_grad = True
            for param in model.proprioceptive_encoder.parameters():
                param.requires_grad = True
            print("Unfrozen visual and proprioceptive encoders for full fine-tuning.")

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)