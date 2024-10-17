#!/usr/bin/env python3
"""
This script trains representation learning models using data from the Spot robot.
It leverages PyTorch Lightning to preprocess data, define datasets, and train models
using a combination of visual and proprioceptive data.

Modules and Classes:
    TerrainDataset: A custom PyTorch Dataset class to handle loading and preprocessing of terrain data from pickle files.
    NATURLDataModule: A PyTorch Lightning DataModule to manage data loading for training and validation.
    NATURLRepresentationsModel: A PyTorch Lightning Module that defines the model architecture and training/validation steps.

Key Functions and Methods:
    TerrainDataset.__init__: Initializes the dataset with paths to pickle files and optional data statistics.
    TerrainDataset.__len__: Returns the number of samples in the dataset.
    TerrainDataset.__getitem__: Loads and preprocesses a single sample from the dataset.

    NATURLDataModule.__init__: Initializes the data module with configuration parameters.
    NATURLDataModule.load: Loads the training and validation datasets, computing data statistics if necessary.
    NATURLDataModule.train_dataloader: Returns a DataLoader for the training dataset.
    NATURLDataModule.val_dataloader: Returns a DataLoader for the validation dataset.

    NATURLRepresentationsModel.__init__: Initializes the model with hyperparameters and architecture components.
    NATURLRepresentationsModel.forward: Defines the forward pass of the model.
    NATURLRepresentationsModel.vicreg_loss: Computes the VICReg loss for representation learning.
    NATURLRepresentationsModel.training_step: Defines the training step for each batch.
    NATURLRepresentationsModel.validation_step: Defines the validation step for each batch.
    NATURLRepresentationsModel.configure_optimizers: Configures the optimizer for training.
    NATURLRepresentationsModel.on_validation_batch_start: Prepares data for validation.
    NATURLRepresentationsModel.validate: Runs the validation process and computes metrics.
    NATURLRepresentationsModel.on_validation_end: Handles actions to be taken at the end of validation.
    NATURLRepresentationsModel.save_models: Saves the trained models and related data.

Constants:
    terrain_label: A dictionary mapping terrain types to numerical preference labels.
    FEET_TOPIC_RATE, LEG_TOPIC_RATE, IMU_TOPIC_RATE: Constants defining the sampling rates for different data types.

Usage:
Pickle files for training.
"""

import torch

torch.multiprocessing.set_sharing_strategy(
    "file_system"
)  # https://github.com/pytorch/pytorch/issues/11201
import argparse
import glob
import os
import pickle
from datetime import datetime

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import tensorboard as tb
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scipy.signal import periodogram
from sklearn import metrics
from termcolor import cprint
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from scripts import cluster_jackal
from scripts.models import (
    ProprioceptionModel,
    VisualEncoderEfficientModel,
    VisualEncoderModel,
)
from scripts.utils import get_transforms, process_feet_data

# terrain_label = {
#     'cement': 0,
#     'pebble_pavement': 1,
#     'grass': 2,
#     'dark_tile': 3,
#     'bush': 4,
#     'asphalt': 5,
#     'marble_rock': 6,
#     'red_brick': 7,
# }

terrain_label = {
    "asphalt": 0,
    "bush": 1,
    "concrete": 2,
    "grass": 3,
    "marble_rock": 4,
    "mulch": 5,
    "pebble_pavement": 6,
    "red_brick": 7,
    "yellow_brick": 8,
}

FEET_TOPIC_RATE = 24.0
LEG_TOPIC_RATE = 24.0
IMU_TOPIC_RATE = 200.0


class TerrainDataset(Dataset):
    def __init__(
        self, pickle_files_root, incl_orientation=False, data_stats=None, train=False
    ):
        # Get all pickle file paths from the specified root directory
        self.pickle_files_paths = glob.glob(pickle_files_root + "/*.pkl")
        self.label = pickle_files_root.split("/")[-2]

        # Whether to include orientation data in the IMU data
        self.incl_orientation = incl_orientation

        # If data statistics are provided, extract min, max, mean, and std values
        self.data_stats = data_stats
        if self.data_stats is not None:
            self.min, self.max = data_stats["min"], data_stats["max"]
            self.mean, self.std = data_stats["mean"], data_stats["std"]

        # If training, apply data augmentation transforms
        if train:
            self.transforms = get_transforms()
        else:
            self.transforms = None

    def __len__(self):
        # Return the number of pickle files
        return len(self.pickle_files_paths)

    def __getitem__(self, idx):
        # Open and load the pickle file at the given index
        with open(self.pickle_files_paths[idx], "rb") as f:
            data = pickle.load(f)

        # Extract IMU, feet, leg, and patches data from the loaded pickle file
        imu, feet, leg = data["imu"], data["feet"], data["leg"]
        patches = data["patches"]

        # Process the feet data to remove mu and std values for non-contacting feet
        feet = process_feet_data(feet)

        # If orientation data is not included, remove the last 4 columns from IMU data
        if not self.incl_orientation:
            imu = imu[:, :-4]

        # Compute the periodogram for IMU, leg, and feet data
        imu = periodogram(imu, fs=IMU_TOPIC_RATE, axis=0)[1]
        leg = periodogram(leg, fs=LEG_TOPIC_RATE, axis=0)[1]
        feet = periodogram(feet, fs=FEET_TOPIC_RATE, axis=0)[1]

        # Normalize the IMU, leg, and feet data if data statistics are provided
        if self.data_stats is not None:
            imu = (imu - self.min["imu"]) / (self.max["imu"] - self.min["imu"] + 1e-7)
            imu = imu.flatten().reshape(1, -1)

            leg = (leg - self.min["leg"]) / (self.max["leg"] - self.min["leg"] + 1e-7)
            leg = leg.flatten().reshape(1, -1)

            feet = (feet - self.min["feet"]) / (
                self.max["feet"] - self.min["feet"] + 1e-7
            )
            feet = feet.flatten().reshape(1, -1)

        # Sample two patch indices: one from the first half and one from the second half of the patches
        patch_1_idx = np.random.randint(0, len(patches) // 2)
        patch_2_idx = np.random.randint(len(patches) // 2, len(patches))

        # Get the patches corresponding to the sampled indices
        patch1, patch2 = patches[patch_1_idx], patches[patch_2_idx]

        # Convert the patches from BGR to RGB color space
        patch1, patch2 = (
            cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB),
        )

        # Apply data augmentation transforms if specified
        if self.transforms is not None:
            patch1 = self.transforms(image=patch1)["image"]
            patch2 = self.transforms(image=patch2)["image"]

        # Normalize the image patches to the range [0, 1]
        patch1 = np.asarray(patch1, dtype=np.float32) / 255.0
        patch2 = np.asarray(patch2, dtype=np.float32) / 255.0

        """
        Transpose the image patches from (H, W, C) to have channels first (C, H, W).
        This is often required by deep learning frameworks like PyTorch, 
        which expect image tensors to have the channels as the first dimension.
        
        C: Channels - This represents the number of color channels in the image (RGB)
        H: Height - Height of the image in pixels.
        W: Width - Width of the image in pixels.
        """
        patch1, patch2 = (
            np.transpose(patch1, (2, 0, 1)),
            np.transpose(patch2, (2, 0, 1)),
        )

        # Return the processed data
        return np.asarray(patch1), np.asarray(patch2), imu, leg, feet, self.label, idx


# Define how data is loaded and preprocessed
class NATURLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_config_path,
        batch_size=64,
        num_workers=2,
        include_orientation_imu=False,
    ):
        super().__init__()

        # Load the yaml file
        cprint("Reading the yaml file at : {}".format(data_config_path), "green")
        self.data_config = yaml.load(
            open(data_config_path, "r"), Loader=yaml.FullLoader
        )
        self.data_config_path = "/".join(data_config_path.split("/")[:-1])

        # Set batch size and number of workers for data loading
        self.batch_size, self.num_workers = batch_size, num_workers

        # Load the training and validation datasets
        self.mean, self.std = {}, {}
        self.min, self.max = {}, {}
        self.load()
        cprint("Train dataset size : {}".format(len(self.train_dataset)), "green")
        cprint("Val dataset size : {}".format(len(self.val_dataset)), "green")

        # Set whether to include orientation data in the IMU data
        self.include_orientation_imu = include_orientation_imu

    def load(self):
        # Check if the data_statistics.pkl file exists in the data configuration path
        if os.path.exists(self.data_config_path + "/data_statistics.pkl"):
            cprint(
                "Loading the mean and std from the data_statistics.pkl file", "green"
            )
            # Load the precomputed data statistics from the pickle file
            data_statistics = pickle.load(
                open(self.data_config_path + "/data_statistics.pkl", "rb")
            )

            # Uncomment the following lines if you need to use the loaded statistics directly
            # self.mean, self.std = data_statistics['mean'], data_statistics['std']
            # self.min, self.max = data_statistics['min'], data_statistics['max']
        else:
            # If the data_statistics.pkl file is not found, compute the statistics from the training dataset
            cprint("data_statistics.pkl file not found!", "yellow")
            cprint("Finding the mean and std of the train dataset", "green")

            # Create a temporary dataset by concatenating all training data
            self.tmp_dataset = ConcatDataset(
                [
                    TerrainDataset(
                        pickle_files_root, incl_orientation=self.include_orientation_imu
                    )
                    for pickle_files_root in self.data_config["train"]
                ]
            )

            # Create a DataLoader for the temporary dataset
            self.tmp_dataloader = DataLoader(
                self.tmp_dataset, batch_size=128, num_workers=2, shuffle=False
            )
            cprint(
                "The length of the tmp_dataloader is : {}".format(
                    len(self.tmp_dataloader)
                ),
                "green",
            )

            # Initialize lists to store IMU, leg, and feet data
            imu_data, leg_data, feet_data = [], [], []

            # Iterate over the DataLoader to collect data
            for _, _, imu, leg, feet, _, _ in tqdm(self.tmp_dataloader):
                imu_data.append(imu.cpu().numpy())
                leg_data.append(leg.cpu().numpy())
                feet_data.append(feet.cpu().numpy())

            # Concatenate the collected data along the first axis
            imu_data = np.concatenate(imu_data, axis=0)
            leg_data = np.concatenate(leg_data, axis=0)
            feet_data = np.concatenate(feet_data, axis=0)
            print("imu_data.shape : ", imu_data.shape)
            print("leg_data.shape : ", leg_data.shape)
            print("feet_data.shape : ", feet_data.shape)
            exit()

            # Reshape the data to 2D arrays for computing statistics
            imu_data = imu_data.reshape(-1, imu_data.shape[-1])
            leg_data = leg_data.reshape(-1, leg_data.shape[-1])
            feet_data = feet_data.reshape(-1, feet_data.shape[-1])

            # Compute mean, standard deviation, minimum, and maximum for IMU data
            self.mean["imu"], self.std["imu"] = (
                np.mean(imu_data, axis=0),
                np.std(imu_data, axis=0),
            )
            self.min["imu"], self.max["imu"] = (
                np.min(imu_data, axis=0),
                np.max(imu_data, axis=0),
            )

            # Compute mean, standard deviation, minimum, and maximum for leg data
            self.mean["leg"], self.std["leg"] = (
                np.mean(leg_data, axis=0),
                np.std(leg_data, axis=0),
            )
            self.min["leg"], self.max["leg"] = (
                np.min(leg_data, axis=0),
                np.max(leg_data, axis=0),
            )

            # Compute mean, standard deviation, minimum, and maximum for feet data
            self.mean["feet"], self.std["feet"] = (
                np.mean(feet_data, axis=0),
                np.std(feet_data, axis=0),
            )
            self.min["feet"], self.max["feet"] = (
                np.min(feet_data, axis=0),
                np.max(feet_data, axis=0),
            )

            # Print the computed statistics
            cprint("Mean : {}".format(self.mean), "green")
            cprint("Std : {}".format(self.std), "green")
            cprint("Min : {}".format(self.min), "green")
            cprint("Max : {}".format(self.max), "green")

            # Save the computed statistics to a pickle file for future use
            cprint(
                "Saving the mean, std, min, max to the data_statistics.pkl file",
                "green",
            )
            data_statistics = {
                "mean": self.mean,
                "std": self.std,
                "min": self.min,
                "max": self.max,
            }
            pickle.dump(
                data_statistics,
                open(self.data_config_path + "/data_statistics.pkl", "wb"),
            )

        # Load the training dataset using the computed or loaded statistics
        self.train_dataset = ConcatDataset(
            [
                TerrainDataset(
                    pickle_files_root,
                    incl_orientation=self.include_orientation_imu,
                    data_stats=data_statistics,
                    train=True,
                )
                for pickle_files_root in self.data_config["train"]
            ]
        )

        # Load the validation dataset using the computed or loaded statistics
        self.val_dataset = ConcatDataset(
            [
                TerrainDataset(
                    pickle_files_root,
                    incl_orientation=self.include_orientation_imu,
                    data_stats=data_statistics,
                )
                for pickle_files_root in self.data_config["val"]
            ]
        )

    # Takes dataset and returns data in batches
    # Handles the batching, shuffling, and loading of the data in parallel using multiple workers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False,
            pin_memory=True,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )


# Define model training, validation, and test steps
class NATURLRepresentationsModel(pl.LightningModule):
    def __init__(
        self,
        lr=3e-4,
        latent_size=64,
        scale_loss=1.0 / 32,
        lambd=3.9e-6,
        weight_decay=1e-6,
        l1_coeff=0.5,
        rep_size=128,
    ):
        super(NATURLRepresentationsModel, self).__init__()

        self.save_hyperparameters(
            "lr",
            "latent_size",
            "scale_loss",
            "lambd",
            "weight_decay",
            "l1_coeff",
            "rep_size",
        )

        self.lr, self.latent_size, self.scale_loss, self.lambd, self.weight_decay = (
            lr,
            latent_size,
            scale_loss,
            lambd,
            weight_decay,
        )
        self.l1_coeff = l1_coeff
        self.rep_size = rep_size

        # Visual encoder architecture
        self.visual_encoder = VisualEncoderModel(latent_size=rep_size)
        # self.visual_encoder = VisualEncoderEfficientModel(latent_size=rep_size)
        self.proprioceptive_encoder = ProprioceptionModel(latent_size=rep_size)

        # Define the projector network to map encoded representations to the latent space
        self.projector = nn.Sequential(
            nn.Linear(rep_size, latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size),
        )

        # Coefficients for vicreg loss
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

        self.max_acc = None

    def forward(self, patch1, patch2, inertial_data, leg, feet):
        # Encode the first visual patch using the visual encoder
        # Normalize the encoded visual representation
        v_encoded_1 = self.visual_encoder(patch1.float())
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2.float())
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        # Encode the proprioceptive data using the proprioceptive encoder
        i_encoded = self.proprioceptive_encoder(
            inertial_data.float(), leg.float(), feet.float()
        )

        # Project the encoded visual/proprioceptive representations to the latent space
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)

        # Return the projected and encoded representations
        return zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded

    def vicreg_loss(self, z1, z2):
        """
        Compute the VICReg (Variance-Invariance-Covariance Regularization) loss between two representations.

        The VICReg loss is composed of three terms:
        1. Representation loss: Mean Squared Error (MSE) between the two representations.
        2. Standard deviation loss: Encourages the standard deviation of the representations to be close to 1.
        3. Covariance loss: Encourages the off-diagonal elements of the covariance matrix to be close to 0.

        Args:
            z1 (torch.Tensor): The first representation tensor of shape (batch_size, feature_dim).
            z2 (torch.Tensor): The second representation tensor of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: The computed VICReg loss.

        Notes:
        - The representation loss is computed using the mean squared error (MSE) between z1 and z2.
        - The standard deviation loss is computed by taking the ReLU of (1 - standard deviation) for both z1 and z2.
        - The covariance loss is computed by summing the squared off-diagonal elements of the covariance matrices of z1 and z2.
        - The final loss is a weighted sum of the three components, with weights defined by self.sim_coeff, self.std_coeff, and self.cov_coeff.
        """
        repr_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div_(
            z1.shape[1]
        ) + self.off_diagonal(cov_y).pow_(2).sum().div_(z2.shape[1])

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

    def off_diagonal(self, x):
        """
        Return a flattened view of the off-diagonal elements of a square matrix.

        Parameters:
            x (torch.Tensor): A square matrix of shape (n, n).

        Returns:
            torch.Tensor: A 1D tensor containing the off-diagonal elements of the input matrix.

        Raises:
            AssertionError: If the input matrix is not square.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def all_reduce(self, c):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

    def training_step(self, batch, batch_idx):
        # Unpack the batch data
        patch1, patch2, inertial, leg, feet, label, _ = batch

        # Forward pass through the model to get encoded representations
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)

        # Compute the viewpoint invariance VICReg loss between the two visual patches
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        
        # Compute the visual-inertial VICReg loss between visual and inertial representations
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)

        # Combine the two losses using the l1_coeff hyperparameter
        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi

        # Log the total training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Log the viewpoint invariance loss
        self.log(
            "train_loss_vpt_inv",
            loss_vpt_inv,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Log the visual-inertial loss
        self.log(
            "train_loss_vi",
            loss_vi,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Return the total loss for backpropagation
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch data
        patch1, patch2, inertial, leg, feet, label, _ = batch

        # Forward pass through the model to get encoded representations
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)

        # Compute the viewpoint invariance VICReg loss between the two visual patches
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        
        # Compute the visual-inertial VICReg loss between visual and inertial representations
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)

        # Combine the two losses using the l1_coeff hyperparameter
        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi

        # Log the total validation loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Log the viewpoint invariance loss
        self.log(
            "val_loss_vpt_inv",
            loss_vpt_inv,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Log the visual-inertial loss
        self.log(
            "val_loss_vi",
            loss_vi,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Return the total loss for logging
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True
        )
        # return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        # return torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        # save the batch data only every other epoch or during the last epoch
        if (
            self.current_epoch % 10 == 0
            or self.current_epoch == self.trainer.max_epochs - 1
        ):
            patch1, patch2, inertial, leg, feet, label, sampleidx = batch
            # combine inertial and leg data
            # inertial = torch.cat((inertial, leg, feet), dim=-1)

            with torch.no_grad():
                _, _, _, zv1, zv2, zi = self.forward(
                    patch1, patch2, inertial, leg, feet
                )
            zv1, zi = zv1.cpu(), zi.cpu()
            patch1 = patch1.cpu()
            label = np.asarray(label)
            sampleidx = sampleidx.cpu()

            if batch_idx == 0:
                self.visual_encoding = [zv1]
                self.inertial_encoding = [zi]
                self.label = label
                self.visual_patch = [patch1]
                self.sampleidx = [sampleidx]
            else:
                self.visual_encoding.append(zv1)
                self.inertial_encoding.append(zi)
                self.label = np.concatenate((self.label, label))
                self.visual_patch.append(patch1)
                self.sampleidx.append(sampleidx)

    # Find random groups of 25 images from each cluster
    def sample_clusters(self, clusters, elbow, vis_patch):
        # Initialize dictionary to store image info for each cluster
        dic = {}
        for a in range(elbow):
            dic[a] = []

        # For each cluster, find indexes of images in that cluster and extract 25 of them
        for i in range(elbow):
            idx = np.where(clusters == i)

            for _ in range(25):
                # Select correct patch
                chosen = np.random.randint(low=0, high=len(idx[0]))
                vp = vis_patch[idx[0][chosen], :, :, :]

                # Formatting for displayable image
                vp = vp.cpu()
                vp = vp.numpy()
                vp = (vp * 255).astype(np.uint8)
                vp = np.moveaxis(vp, 0, -1)

                dic[i].append(vp)

        return dic

    # Create and save 25 image grids for each cluster from dictionary image info
    def img_clusters(self, dic, elbow, path_root="./models/"):
        for i in range(elbow):
            # Initialize grid
            new_im = Image.new("RGB", (64 * 5, 64 * 5))

            for j in range(25):
                vp = dic[i][j]

                # Patch number to grid location
                h = int(j / 5)
                w = j % 5

                # format and paste individual patches to grid
                im = Image.fromarray(vp)
                im = im.convert("RGB")
                im.thumbnail((64, 64))
                new_im.paste(im, (h * 64, w * 64))

            # save grid image
            new_im.save(path_root + "group" + str(i) + ".png")

    def validate(self):
        print("Running validation...")
        dataset = self.trainer.datamodule.val_dataset
        (
            self.visual_encoding,
            self.inertial_encoding,
            self.label,
            self.visual_patch,
            self.sampleidx,
        ) = [], [], [], [], []
        # create dataloader for validation
        dataset = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

        for patch1, patch2, inertial, leg, feet, label, sampleidx in tqdm(dataset):
            # convert to torch tensors
            # patch1, patch2, inertial, leg, feet = torch.from_numpy(patch1), torch.from_numpy(patch2), torch.from_numpy(inertial), torch.from_numpy(leg), torch.from_numpy(feet)
            # move to device
            patch1, patch2, inertial, leg, feet = (
                patch1.to(self.device),
                patch2.to(self.device),
                inertial.to(self.device),
                leg.to(self.device),
                feet.to(self.device),
            )

            with torch.no_grad():
                # _, _, _, zv1, zv2, zi = self.forward(patch1.unsqueeze(0), patch2.unsqueeze(0), inertial.unsqueeze(0), leg.unsqueeze(0), feet.unsqueeze(0))
                _, _, _, zv1, zv2, zi = self.forward(
                    patch1, patch2, inertial, leg, feet
                )
                zv1, zi = zv1.cpu(), zi.cpu()
                patch1 = patch1.cpu()

            self.visual_patch.append(patch1)
            self.visual_encoding.append(zv1)
            self.inertial_encoding.append(zi)
            self.label.append(np.asarray(label))
            self.sampleidx.append(sampleidx)

        self.visual_patch = torch.cat(self.visual_patch, dim=0)
        self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
        self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
        self.sampleidx = torch.cat(self.sampleidx, dim=0)
        self.label = np.concatenate(self.label)

        # print('Visual Encoding Shape: {}'.format(self.visual_encoding.shape))
        # print('Inertial Encoding Shape: {}'.format(self.inertial_encoding.shape))
        # print('Visual Patch Shape: {}'.format(self.visual_patch.shape))
        # print('Sample Index Shape: {}'.format(self.sampleidx.shape))

    def on_validation_end(self):
        if (
            self.current_epoch % 10 == 0
            or self.current_epoch == self.trainer.max_epochs - 1
        ) and torch.cuda.current_device() == 0:
            self.validate()

            # self.visual_patch = torch.cat(self.visual_patch, dim=0)
            # self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            # self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
            # self.sampleidx = torch.cat(self.sampleidx, dim=0)

            # cprint('Visual Encoding Shape: {}'.format(self.visual_encoding.shape), 'white', attrs=['bold'])

            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)

            # limit the number of samples to 2000
            ve = self.visual_encoding  # [idx[:2000]]
            vi = self.inertial_encoding  # [idx[:2000]]
            vis_patch = self.visual_patch  # [idx[:2000]]
            ll = self.label  # [idx[:2000]]

            data = torch.cat((ve, vi), dim=-1)

            # calculate and print accuracy
            cprint("finding accuracy...", "yellow")
            accuracy, kmeanslabels, kmeanselbow, kmeansmodel = (
                cluster_jackal.accuracy_naive(
                    data, ll, label_types=list(terrain_label.keys())
                )
            )
            fms, ari, chs = cluster_jackal.compute_fms_ari(
                data, ll, clusters=kmeanslabels, elbow=kmeanselbow, model=kmeansmodel
            )

            if not self.max_acc or accuracy > self.max_acc:
                self.max_acc = accuracy
                self.kmeanslabels, self.kmeanselbow, self.kmeansmodel = (
                    kmeanslabels,
                    kmeanselbow,
                    kmeansmodel,
                )
                self.vispatchsaved = torch.clone(vis_patch)
                self.sampleidxsaved = torch.clone(self.sampleidx)
                cprint("best model saved", "green")

            # log k-means accurcay and projection for tensorboard visualization
            self.logger.experiment.add_scalar(
                "K-means accuracy", accuracy, self.current_epoch
            )
            self.logger.experiment.add_scalar(
                "Fowlkes-Mallows score", fms, self.current_epoch
            )
            self.logger.experiment.add_scalar(
                "Adjusted Rand Index", ari, self.current_epoch
            )
            self.logger.experiment.add_scalar(
                "Calinski-Harabasz Score", chs, self.current_epoch
            )
            self.logger.experiment.add_scalar(
                "K-means elbow", self.kmeanselbow, self.current_epoch
            )

            # Save the cluster image grids on the final epoch only
            if self.current_epoch == self.trainer.max_epochs - 1:
                path_root = (
                    "./models/acc_"
                    + str(round(self.max_acc, 5))
                    + "_"
                    + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
                    + "_"
                    + "/"
                )
                self.save_models(path_root)

            if self.current_epoch % 10 == 0:
                self.logger.experiment.add_embedding(
                    mat=data[idx[:2500]],
                    label_img=vis_patch[idx[:2500]],
                    global_step=self.current_epoch,
                    metadata=ll[idx[:2500]],
                    tag="visual_encoding",
                )
            del (
                self.visual_patch,
                self.visual_encoding,
                self.inertial_encoding,
                self.label,
            )

    def save_models(self, path_root="./models/"):
        """
        Save the k-means clustering results and visual patches.

        Args:
            path_root (str): The root directory where the models and results will be saved.

        This function performs the following steps:
            1. Creates the directory specified by path_root if it does not exist.
            2. Samples clusters of visual patches and saves them as image grids.
            3. Saves the k-means clustering model.
            4. Saves the k-means clustering labels and sample indices.
            5. Saves the state dictionaries of the visual and proprioceptive encoders.
        """
        cprint("saving models...", "yellow", attrs=["bold"])

        # Create the directory if it does not exist
        if not os.path.exists(path_root):
            cprint("creating directory: " + path_root, "yellow")
            os.makedirs(path_root)
        else:
            cprint("directory already exists: " + path_root, "red")

        # Sample clusters of visual patches and save them as image grids
        dic = self.sample_clusters(
            self.kmeanslabels, self.kmeanselbow, self.vispatchsaved
        )
        self.img_clusters(dic, self.kmeanselbow, path_root=path_root)

        # Save the k-means clustering model
        with open(os.path.join(path_root, "kmeansmodel.pkl"), "wb") as f:
            pickle.dump(self.kmeansmodel, f)
            cprint("kmeans model saved", "green")

        # Save the k-means clustering labels and sample indices
        torch.save(self.kmeanslabels, os.path.join(path_root, "kmeanslabels.pt"))
        torch.save(self.sampleidxsaved, os.path.join(path_root, "sampleidx.pt"))

        # Save the state dictionary of the visual encoder
        torch.save(
            self.visual_encoder.state_dict(),
            os.path.join(path_root, "visual_encoder.pt"),
        )
        cprint("visual encoder saved", "green")

        # Save the state dictionary of the proprioceptive encoder
        torch.save(
            self.proprioceptive_encoder.state_dict(),
            os.path.join(path_root, "proprioceptive_encoder.pt"),
        )
        cprint("proprioceptive encoder saved", "green")

        cprint("All models successfully saved", "green", attrs=["bold"])


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train representations using the NATURL framework"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=512,
        metavar="N",
        help="input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--l1_coeff",
        type=float,
        default=0.5,
        metavar="L1C",
        help="L1 loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--num_gpus",
        "-g",
        type=int,
        default=2,
        metavar="N",
        help="number of GPUs to use (default: 2)",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=128,
        metavar="N",
        help="Size of the common latent space (default: 128)",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        metavar="N",
        help="Whether to save the k-means model and encoders at the end of the run (default: 0)",
    )
    parser.add_argument(
        "--imu_in_rep",
        type=int,
        default=1,
        metavar="N",
        help="Whether to include the inertial data in the representation (default: 1)",
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="spot_data/data_config.yaml",
        help="Path to the data configuration file (default: spot_data/data_config.yaml)",
    )
    args = parser.parse_args()

    # Initialize the model with parsed arguments
    model = NATURLRepresentationsModel(
        lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff
    )

    # Initialize the data module with parsed arguments
    dm = NATURLDataModule(
        data_config_path=args.data_config_path, batch_size=args.batch_size
    )

    # Set up TensorBoard logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="naturl_reptraining_logs/")

    # Initialize the training loop
    print("Training the representation learning model...")
    trainer = pl.Trainer(
        gpus=list(np.arange(args.num_gpus)),
        max_epochs=args.epochs,
        log_every_n_steps=10,
        strategy="ddp",
        num_sanity_val_steps=0,
        logger=tb_logger,
        sync_batchnorm=True,
        gradient_clip_val=100.0,
        gradient_clip_algorithm="norm",
    )

    # Fit the model using the trainer and data module
    trainer.fit(model, dm)
