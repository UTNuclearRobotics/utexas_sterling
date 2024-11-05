"""
This file trains representations using visual and inertial data.
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pytorch_lightning as pl

# import tensorboard as tb
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from clustering_utils import (
    accuracy_naive,
    compute_fms_ari,
)
from data_loader import SterlingDataModule
from models import (
    InertialEncoderModel,
    VisualEncoderEfficientModel,
    VisualEncoderModel,
)

torch.multiprocessing.set_sharing_strategy("file_system")  # https://github.com/pytorch/pytorch/issues/11201

# TODO: Extrapolate to config
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
        self.scale_loss = scale_loss  # Question: Is this being used by LightningModule?
        self.lambd = lambd  # Question: Is this being used by LightningModule?
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
        Args:
            z1 (torch.Tensor): The first representation tensor of shape (batch_size, feature_dim).
            z2 (torch.Tensor): The second representation tensor of shape (batch_size, feature_dim).
        Returns:
            torch.Tensor: The computed VICReg loss.
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
        """
        Return a flattened view of the off-diagonal elements of a square matrix.
        Args:
            x (torch.Tensor): A square matrix of shape (n, n).
        Returns:
            torch.Tensor: A 1D tensor containing the off-diagonal elements of the input matrix.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def all_reduce(self, c):
        """
        Reduces the input tensor across all processes.
        Args:
            c (torch.Tensor): A tensor to be reduced across all processes.
        Returns:
            torch.Tensor: The reduced tensor.
        """
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on the batch of data.
        Args:
            batch (tuple): A tuple containing the input data.
            batch_idx (int): The index of the current batch.
        Returns:
            torch.Tensor: The computed loss value.
        """
        patch1, patch2, inertial, _, _ = batch

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
        """
        Perform a validation step during the training process.
        Args:
            batch (tuple): A tuple containing the input data.
            batch_idx (int): The index of the current batch.
        Returns:
            torch.Tensor: The computed loss value.
        """
        patch1, patch2, inertial, _, _ = batch

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
        """
        Configures the optimizer for training.
        Returns:
            torch.optim.AdamW: The configured AdamW optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def on_validation_batch_start(self, batch, batch_idx):
        """
        Save the batch data for visualization during validation.
        Args:
            batch (tuple): A tuple containing the input data, inertial data, and labels.
            batch_idx (int): The index of the current batch.
        """

        # Save the batch data only every other epoch or during the last epoch
        if self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            patch1, patch2, inertial, label, sample_idx = batch

            with torch.no_grad():
                _, _, _, zv1, zv2, zi = self.forward(patch1, patch2, inertial)
            zv1, zi = zv1.cpu(), zi.cpu()
            patch1 = patch1.cpu()
            label = np.asarray(label)
            sample_idx = sample_idx.cpu()

            if batch_idx == 0:
                self.visual_encoding = [zv1]
                self.inertial_encoding = [zi]
                self.label = label
                self.visual_patch = [patch1]
                self.sample_idx = [sample_idx]
            else:
                self.visual_encoding.append(zv1)
                self.inertial_encoding.append(zi)
                self.label = np.concatenate((self.label, label))
                self.visual_patch.append(patch1)
                self.sample_idx.append(sample_idx)

    def sample_clusters(self, clusters, elbow, vis_patch):
        """
        Sample 25 images from each cluster for visualization.
        Args:
            clusters (np.ndarray): The cluster labels for each image.
            elbow (int): The number of clusters.
            vis_patch (torch.Tensor): The visual patches for each image.
        Returns:
            dict: A dictionary containing the sampled images for each cluster.
        """
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

    def img_clusters(self, dic, elbow, path_root):
        """
        Create and save 25 image grids for each cluster from dictionary image info.
        Args:
            dic (dict): A dictionary containing the sampled images for each cluster.
            elbow (int): The number of clusters.
            path_root (str): The root directory where the models and results will be saved.
        """
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

            # Save grid image
            # TODO: Group name not showing up? Just ''
            new_im.save(os.path.join(path_root, "group" + str(i) + ".png"))

    def validate(self):
        """
        Validates the model using the validation dataset.
        """
        # Initialize empty lists to store visual encodings, inertial encodings, labels, visual patches, and sample indices
        dataset = self.trainer.datamodule.val_dataset
        (
            self.visual_encoding,
            self.inertial_encoding,
            self.label,
            self.visual_patch,
            self.sample_idx,
        ) = [], [], [], [], []

        # Create dataloader for validation
        dataset = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

        for patch1, patch2, inertial, label, sample_idx in tqdm(dataset, desc="Validating"):
            # move to device
            patch1, patch2, inertial = (
                patch1.to(self.device),
                patch2.to(self.device),
                inertial.to(self.device),
            )

            # Performs a forward pass through the model without gradient computation
            with torch.no_grad():
                # _, _, _, zv1, zv2, zi = self.forward(patch1.unsqueeze(0), patch2.unsqueeze(0), inertial.unsqueeze(0), leg.unsqueeze(0), feet.unsqueeze(0))
                _, _, _, zv1, zv2, zi = self.forward(patch1, patch2, inertial)
                zv1, zi = zv1.cpu(), zi.cpu()
                patch1 = patch1.cpu()

            # Stores the outputs
            self.visual_patch.append(patch1)
            self.visual_encoding.append(zv1)
            self.inertial_encoding.append(zi)
            self.label.append(np.asarray(label))
            self.sample_idx.append(sample_idx)

        # Concatenates the collected outputs into tensors or arrays
        self.visual_patch = torch.cat(self.visual_patch, dim=0)
        self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
        self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
        self.sample_idx = torch.cat(self.sample_idx, dim=0)
        self.label = np.concatenate(self.label)

        # cprint("Visual Encoding Shape: {}".format(self.visual_encoding.shape), "cyan")
        # cprint("Inertial Encoding Shape: {}".format(self.inertial_encoding.shape), "cyan")
        # cprint("Visual Patch Shape: {}".format(self.visual_patch.shape), "cyan")
        # cprint("Sample Index Shape: {}".format(self.sample_idx.shape), "cyan")

    def on_validation_end(self):
        """
        Save the model if it has the best validation loss.
        """
        # Every 10 epochs or at the very end of training
        if (
            self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1
        ) and torch.cuda.current_device() == 0:
            self.validate()

            # self.visual_patch = torch.cat(self.visual_patch, dim=0)
            # self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            # self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
            # self.sample_idx = torch.cat(self.sample_idx, dim=0)

            # cprint('Visual Encoding Shape: {}'.format(self.visual_encoding.shape), 'white', attrs=['bold'])

            # Randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)

            # limit the number of samples to 2000
            ve = self.visual_encoding  # [idx[:2000]]
            vi = self.inertial_encoding  # [idx[:2000]]
            vis_patch = self.visual_patch  # [idx[:2000]]
            ll = self.label  # [idx[:2000]]

            data = torch.cat((ve, vi), dim=-1)

            # Calculate and print accuracy
            tqdm.write("Finding accuracy...")

            accuracy, kmeans_labels, kmeanselbow, kmeans_model = accuracy_naive(
                data, ll, label_types=list(terrain_label.keys())
            )
            fms, ari, chs = compute_fms_ari(data, ll, clusters=kmeans_labels, elbow=kmeanselbow, model=kmeans_model)

            if not self.max_acc or accuracy > self.max_acc:
                self.max_acc = accuracy
                self.kmeans_labels, self.kmeans_elbow, self.kmeans_model = (
                    kmeans_labels,
                    kmeanselbow,
                    kmeans_model,
                )
                self.vis_patch_saved = torch.clone(vis_patch)
                self.sample_idx_saved = torch.clone(self.sample_idx)
                cprint("Best model saved", "green")

            # Log k-means accurcay and projection for tensorboard visualization
            self.logger.experiment.add_scalar("K-means accuracy", accuracy, self.current_epoch)
            self.logger.experiment.add_scalar("Fowlkes-Mallows score", fms, self.current_epoch)
            self.logger.experiment.add_scalar("Adjusted Rand Index", ari, self.current_epoch)
            self.logger.experiment.add_scalar("Calinski-Harabasz Score", chs, self.current_epoch)
            self.logger.experiment.add_scalar("K-means elbow", self.kmeans_elbow, self.current_epoch)

            # Save the cluster image grids on the final epoch only
            if self.current_epoch == self.trainer.max_epochs - 1:
                path_root = os.path.normpath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "models",
                        "rep_" + str(round(self.max_acc, 5)) + "_" + str(datetime.now().strftime("%Y%m%d_%H%M%S")),
                    )
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

    def save_models(self, path_root):
        """
        Save the k-means clustering results and visual patches.
        Args:
            path_root (str): The root directory where the models and results will be saved.
        """
        cprint("Saving models...", "yellow", attrs=["bold"])

        # Create the directory if it does not exist
        if not os.path.exists(path_root):
            cprint("Creating directory: " + path_root, "yellow")
            os.makedirs(path_root)
        else:
            cprint("Directory already exists: " + path_root, "red")

        # Sample clusters of visual patches and save them as image grids
        dic = self.sample_clusters(self.kmeans_labels, self.kmeans_elbow, self.vis_patch_saved)
        self.img_clusters(dic, self.kmeans_elbow, path_root=path_root)

        # Save the k-means clustering model
        with open(os.path.join(path_root, "kmeans_model.pkl"), "wb") as f:
            pickle.dump(self.kmeans_model, f)
            cprint("K-means model saved", "green")

        # Save the k-means clustering labels and sample indices
        torch.save(self.kmeans_labels, os.path.join(path_root, "kmeans_labels.pt"))
        torch.save(self.sample_idx_saved, os.path.join(path_root, "sample_idx.pt"))

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
        help="Input batch size for training (default: 512)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=200,
        metavar="N",
        help="Number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Learning rate (default: 3e-4)",
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
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=128,
        metavar="N",
        help="Size of the common latent space (default: 128)",
    )
    # parser.add_argument(
    #     "--save",
    #     type=int,
    #     default=0,
    #     metavar="N",
    #     help="Whether to save the k-means model and encoders at the end of the run (default: 0)",
    # )
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
        default=os.path.join(os.path.dirname(__file__), "config", "data_config.yaml"),
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
        # f"save: {args.save}",
        f"imu_in_rep: {args.imu_in_rep}",
        f"data_config_path: {args.data_config_path}",
    ]
    for arg in args_list:
        cprint(arg, "blue")

    return args


if __name__ == "__main__":
    args = parse_args()

    # Initialize the model with parsed arguments
    model = SterlingRepresentationModel(lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff)

    # Initialize the data module with parsed arguments
    dm = SterlingDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)

    # Set up TensorBoard logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(os.path.dirname(__file__), "..", "..", "logs", "sterling_representations_logs")
    )

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
