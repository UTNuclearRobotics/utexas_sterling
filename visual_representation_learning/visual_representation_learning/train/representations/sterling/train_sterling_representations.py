#!/usr/bin/env python3
""" """

import torch

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
from ament_index_python.packages import get_package_share_directory

from visual_representation_learning.train.representations.sterling.cluster import (
    accuracy_naive,
    compute_fms_ari,
)
from visual_representation_learning.train.representations.sterling.models import (
    ProprioceptionModel,
    VisualEncoderModel,
    InertialEncoderModel,
    VisualEncoderEfficientModel,
)
from visual_representation_learning.train.representations.sterling.data_loader import SterlingDataModule

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))

torch.multiprocessing.set_sharing_strategy("file_system")  # https://github.com/pytorch/pytorch/issues/11201

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


class SterlingRepresentationModel(pl.LightningModule):
    def __init__(
        self, lr=3e-4, latent_size=64, scale_loss=1.0 / 32, lambd=3.9e-6, weight_decay=1e-6, l1_coeff=0.5, rep_size=64
    ):
        super(SterlingRepresentationModel, self).__init__()

        self.save_hyperparameters(
            "lr",
            "latent_size",
            "weight_decay",
            "l1_coeff",
            "rep_size",
        )

        self.best_val_loss = 1000000.0

        self.lr = lr
        self.latent_size = latent_size
        self.scale_loss = scale_loss
        self.lambd = lambd
        self.weight_decay = weight_decay
        self.l1_coeff = l1_coeff
        self.rep_size = rep_size

        # Encoder architecture
        self.visual_encoder = VisualEncoderEfficientModel(latent_size=rep_size)
        # self.visual_encoder = VisualEncoderModel(latent_size=rep_size)
        self.inertial_encoder = InertialEncoderModel(latent_size=rep_size)

        self.projector = nn.Sequential(
            nn.Linear(rep_size, latent_size), nn.PReLU(), nn.Linear(latent_size, latent_size)
        )

        # Coefficients for vicreg loss
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

        self.max_acc = None

    def forward(self, patch1, patch2, inertial_data):
        # Encode visual patches
        v_encoded_1 = self.visual_encoder(patch1.float())
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2.float())
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        # Encode inertial data
        i_encoded = self.inertial_encoder(inertial_data.float())

        # Project encoded representations to latent space
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)

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

        # Representation loss
        repr_loss = F.mse_loss(z1, z2)

        # Standard deviation loss
        std_z1 = torch.sqrt(torch.var(z1, dim=0) + 1e-4)
        std_z2 = torch.sqrt(torch.var(z2, dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = (self.off_diagonal(cov_z1).pow(2).sum() + self.off_diagonal(cov_z2).pow(2).sum()) / z1.shape[1]

        # Total loss
        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

    def off_diagonal(self, x):
        # Return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def all_reduce(self, c):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

    def training_step(self, batch, batch_idx):
        patch1, patch2, inertial, label, _ = batch

        # Forward pass
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial)

        # Compute viewpoint invariance VICReg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)

        # Compute visual-inertial VICReg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)

        # Total loss
        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi

        # Log losses
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "train_loss_vpt_inv", loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log("train_loss_vi", loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        patch1, patch2, inertial, label, _ = batch

        # Forward pass
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial)

        # Compute viewpoint invariance VICReg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)

        # Compute visual-inertial VICReg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)

        # Total loss
        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi

        # Log losses
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            "val_loss_vpt_inv", loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log("val_loss_vi", loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def on_validation_batch_start(self, batch, batch_idx):
        # save the batch data only every other epoch or during the last epoch
        if self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            patch1, patch2, inertial, label, sampleidx = batch

            with torch.no_grad():
                _, _, _, zv1, zv2, zi = self.forward(patch1, patch2, inertial)
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

        for patch1, patch2, inertial, label, sampleidx in tqdm(dataset):
            # move to device
            patch1, patch2, inertial = (
                patch1.to(self.device),
                patch2.to(self.device),
                inertial.to(self.device),
            )

            with torch.no_grad():
                # _, _, _, zv1, zv2, zi = self.forward(patch1.unsqueeze(0), patch2.unsqueeze(0), inertial.unsqueeze(0), leg.unsqueeze(0), feet.unsqueeze(0))
                _, _, _, zv1, zv2, zi = self.forward(patch1, patch2, inertial)
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

        # cprint("Visual Encoding Shape: {}".format(self.visual_encoding.shape), "cyan")
        # cprint("Inertial Encoding Shape: {}".format(self.inertial_encoding.shape), "cyan")
        # cprint("Visual Patch Shape: {}".format(self.visual_patch.shape), "cyan")
        # cprint("Sample Index Shape: {}".format(self.sampleidx.shape), "cyan")

    def on_validation_end(self):
        """
        Every 10 epochs or at the very end of training,
        save the model if it has the best validation loss.
        """
        if (
            self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1
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
            cprint("Finding accuracy...", "yellow")
            accuracy, kmeanslabels, kmeanselbow, kmeansmodel = accuracy_naive(
                data, ll, label_types=list(terrain_label.keys())
            )
            fms, ari, chs = compute_fms_ari(data, ll, clusters=kmeanslabels, elbow=kmeanselbow, model=kmeansmodel)

            if not self.max_acc or accuracy > self.max_acc:
                self.max_acc = accuracy
                self.kmeanslabels, self.kmeanselbow, self.kmeansmodel = (
                    kmeanslabels,
                    kmeanselbow,
                    kmeansmodel,
                )
                self.vispatchsaved = torch.clone(vis_patch)
                self.sampleidxsaved = torch.clone(self.sampleidx)
                cprint("Best model saved", "green")

            # log k-means accurcay and projection for tensorboard visualization
            self.logger.experiment.add_scalar("K-means accuracy", accuracy, self.current_epoch)
            self.logger.experiment.add_scalar("Fowlkes-Mallows score", fms, self.current_epoch)
            self.logger.experiment.add_scalar("Adjusted Rand Index", ari, self.current_epoch)
            self.logger.experiment.add_scalar("Calinski-Harabasz Score", chs, self.current_epoch)
            self.logger.experiment.add_scalar("K-means elbow", self.kmeanselbow, self.current_epoch)

            # Save the cluster image grids on the final epoch only
            if self.current_epoch == self.trainer.max_epochs - 1:
                path_root = (
                    "./models/acc_" + str(round(self.max_acc, 5)) + "_" + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
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

    def save_models(self, path_root=os.path.join(ros_ws_dir, "models")):
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
        cprint("Saving models...", "yellow", attrs=["bold"])

        # Create the directory if it does not exist
        if not os.path.exists(path_root):
            cprint("Creating directory: " + path_root, "yellow")
            os.makedirs(path_root)
        else:
            cprint("Directory already exists: " + path_root, "red")

        # Sample clusters of visual patches and save them as image grids
        dic = self.sample_clusters(self.kmeanslabels, self.kmeanselbow, self.vispatchsaved)
        self.img_clusters(dic, self.kmeanselbow, path_root=path_root)

        # Save the k-means clustering model
        with open(os.path.join(path_root, "kmeansmodel.pkl"), "wb") as f:
            pickle.dump(self.kmeansmodel, f)
            cprint("K-means model saved", "green")

        # Save the k-means clustering labels and sample indices
        torch.save(self.kmeanslabels, os.path.join(path_root, "kmeanslabels.pt"))
        torch.save(self.sampleidxsaved, os.path.join(path_root, "sampleidx.pt"))

        # Save the state dictionary of the visual encoder
        torch.save(
            self.visual_encoder.state_dict(),
            os.path.join(path_root, "visual_encoder.pt"),
        )
        cprint("Visual encoder saved", "green")

        # Save the state dictionary of the inertial encoder
        torch.save(
            self.inertial_encoder.state_dict(),
            os.path.join(path_root, "inertial_encoder.pt"),
        )
        cprint("Inertial encoder saved", "green")

        cprint("All models successfully saved", "green", attrs=["bold"])


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train representations using the Sterling framework")
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
        default=1,
        metavar="N",
        help="number of GPUs to use (default: 1)",
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
        default=os.path.join(package_share_directory, "config", "dataset.yaml"),
        help="Path to data config file",
    )
    args = parser.parse_args()

    # Print all arguments in a list
    args_list = [
        f"batch_size: {args.batch_size}",
        f"epochs: {args.epochs}",
        f"lr: {args.lr}",
        f"l1_coeff: {args.l1_coeff}",
        f"num_gpus: {args.num_gpus}",
        f"latent_size: {args.latent_size}",
        f"save: {args.save}",
        f"imu_in_rep: {args.imu_in_rep}",
        f"data_config_path: {args.data_config_path}",
    ]
    for arg in args_list:
        cprint(arg, "cyan")

    return args


def main():
    args = parse_args()

    # Initialize the model with parsed arguments
    model = SterlingRepresentationModel(lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff)

    # Initialize the data module with parsed arguments
    dm = SterlingDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)

    # Set up TensorBoard logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(ros_ws_dir, "log", "sterling_representation_logs"))

    # Initialize the PyTorch Lightning trainer
    cprint("Training the representation learning model...", "yellow", attrs=["bold"])

    trainer = pl.Trainer(
        devices=args.num_gpus,
        max_epochs=args.epochs,
        log_every_n_steps=10,
        strategy="ddp",
        num_sanity_val_steps=0,
        logger=tb_logger,
        sync_batchnorm=True,
        gradient_clip_val=10.0,
        gradient_clip_algorithm="norm",
        # deterministic=True,
    )

    try:
        # Fit the model using the trainer and data module
        trainer.fit(model, dm)
    except Exception as e:
        print(f"An error occurred during training: {e}")
