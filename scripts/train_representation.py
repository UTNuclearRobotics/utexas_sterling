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

class SterlingPaternRepresentation(nn.Module):
    def __init__(self, device):
        super(SterlingPaternRepresentation, self).__init__()
        self.device = device
        self.latent_size = 128
        self.rep_size = self.latent_size
        self.visual_encoder = VisualEncoderModel(self.latent_size)
        #self.proprioceptive_encoder = ProprioceptionModel(self.latent_size)
        self.projector = nn.Sequential(
            nn.Linear(self.rep_size, self.latent_size),
            nn.PReLU(),
            nn.Linear(self.latent_size, self.latent_size),
        )

        self.vicreg_loss = VICRegLoss()
        self.l1_coeff = 0.5

    def forward(self, patch1, patch2):
        """
        Args:
            patch1 (torch.Tensor): First patch image of shape (3, 128, 128)
            patch2 (torch.Tensor): Second patch image of shape (3, 128, 128)
        """

        # Encode visual patches
        patch1 = patch1.to(self.device)
        patch2 = patch2.to(self.device)
        v_encoded_1 = self.visual_encoder(patch1)
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2)
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        #i_encoded = self.proprioceptive_encoder(inertial_data.float())

        # Project encoded representations to latent space
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        #zi = self.projector(i_encoded)

        return zv1, zv2, v_encoded_1, v_encoded_2
    
    def encode_single_patch(self, patch):
        """
        Encode a single patch and return its representation vector.
        Args:
            patch (torch.Tensor): Single patch image of shape (1, 3, H, W).
        Returns:
            torch.Tensor: Encoded and normalized representation vector.
        """
        # Ensure the input is on the correct device
        patch = patch.to(self.device)

        # Encode the patch
        v_encoded = self.visual_encoder(patch)
        v_encoded = F.normalize(v_encoded, dim=-1)  # Normalize the representation vector
        return v_encoded
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on the batch of data.
        Args:
            batch (tuple): A tuple containing the input data.
            batch_idx (int): The index of the current batch.
        Returns:
            torch.Tensor: The computed loss value.
        """
        patch1, patch2 = batch
        zv1, zv2, _, _ = self.forward(patch1, patch2)

        # Compute VICReg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        #loss_vi = 0.5 * self.vicreg_loss(zv1,zi) + 0.5 * self.vicreg_loss(zv2,zi)

        #loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi

        return loss_vpt_inv

if __name__ == "__main__":
    """
    Train the model using the given dataset.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Sterling Representation Model")
    parser.add_argument("-bag", "-b", type=str, required=True, help="Bag directory with VICReg dataset pickle file inside.")
    parser.add_argument("-batch_size", "-batch", type=int, default=256, help="Batch size for training")
    parser.add_argument("-epochs", type=int, default=50, help="Number of epochs for training")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    patches_pkl = load_bag_pkl(args.bag, "vicreg")
    #IPT_pkl = load_bag_pkl(args.bag, "synced")

    # Define the augmentation pipeline
    augment_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),  
        #v2.RandomVerticalFlip(p=0.1),  
        #v2.RandomRotation(degrees=5),  # Small rotation (too much will distort patterns)
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),  
        #v2.RandomGrayscale(p=0.1),  
        #v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),  # Light blur to prevent over-reliance on high-frequency textures
        v2.ToTensor(),
    ])


    dataset = TerrainDataset(patches=patches_pkl, transform=augment_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = SterlingPaternRepresentation(device).to(device)
    save_path = load_bag_pt_model(args.bag, "terrain_rep", model)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  
        T_mult=2, 
        eta_min=1e-6  # Ensures LR doesn't get too close to zero
    )

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

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
