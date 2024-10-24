#!/usr/bin/env python3

"""code to train STERLING representations from spot data"""

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
import yaml
from ament_index_python.packages import get_package_share_directory
from PIL import Image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scipy.signal import periodogram
from sklearn import metrics
from termcolor import cprint
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from visual_representation_learning.train.representations.models import VisualEncoderModel

torch.multiprocessing.set_sharing_strategy("file_system")  # https://github.com/pytorch/pytorch/issues/11201

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))


class SterlingModel(pl.LightningModule):
    def __init__(
        self, lr=3e-4, latent_size=64, scale_loss=1.0 / 32, lambd=3.9e-6, weight_decay=1e-6, l1_coeff=0.5, rep_size=64
    ):
        super(SterlingModel, self).__init__()

        self.save_hyperparameters(
            "lr",
            "latent_size",
            "weight_decay",
            "l1_coeff",
            "rep_size",
        )

        self.best_val_loss = 1000000.0

        self.lr, self.latent_size, self.scale_loss, self.lambd, self.weight_decay = (
            lr,
            latent_size,
            scale_loss,
            lambd,
            weight_decay,
        )
        self.l1_coeff = l1_coeff
        self.rep_size = rep_size

        # visual encoder architecture
        self.visual_encoder = VisualEncoderModel(latent_size=rep_size)
        self.ipt_encoder = IPTEncoder(latent_size=rep_size)

        self.projector = nn.Sequential(
            nn.Linear(rep_size, latent_size), nn.PReLU(), nn.Linear(latent_size, latent_size)
        )

        # coefficients for vicreg loss
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

        self.max_acc = None

    def forward(self, patch1, patch2, inertial_data, leg, feet):
        v_encoded_1 = self.visual_encoder(patch1.float())
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2.float())
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        # i_encoded = self.inertial_encoder(inertial_data.float())
        i_encoded = self.ipt_encoder(inertial_data.float(), leg.float(), feet.float())

        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)

        return zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded

    def vicreg_loss(self, z1, z2):
        repr_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) + self.off_diagonal(cov_y).pow_(
            2
        ).sum().div_(z2.shape[1])

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def all_reduce(self, c):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

    def training_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, label, _ = batch

        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)

        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)

        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_vpt_inv", loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_vi", loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, label, _ = batch

        # combine inertial and leg data
        # inertial = torch.cat((inertial, leg, feet), dim=-1)

        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)

        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)

        loss = self.l1_coeff * loss_vpt_inv + (1.0 - self.l1_coeff) * loss_vi

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_vpt_inv", loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_vi", loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def on_validation_batch_start(self, batch, batch_idx):
        # save the batch data only every other epoch or during the last epoch
        if self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            patch1, patch2, inertial, leg, feet, label, sampleidx = batch

            with torch.no_grad():
                _, _, _, zv1, zv2, zi = self.forward(patch1, patch2, inertial, leg, feet)
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

    def on_validation_end(self):
        if (
            self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1
        ) and torch.cuda.current_device() == 0:
            val_loss = self.trainer.callback_metrics["val_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_models()

            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
            self.visual_patch = torch.cat(self.visual_patch, dim=0)
            self.sampleidx = torch.cat(self.sampleidx, dim=0)

            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)

            # limit the number of samples to 2000
            ve = self.visual_encoding  # [idx[:2000]]
            vi = self.inertial_encoding  # [idx[:2000]]
            vis_patch = self.visual_patch  # [idx[:2000]]
            ll = self.label  # [idx[:2000]]

            data = torch.cat((ve, vi), dim=-1)

            if self.current_epoch % 10 == 0:
                self.logger.experiment.add_embedding(
                    mat=data[idx[:2500]],
                    label_img=vis_patch[idx[:2500]],
                    global_step=self.current_epoch,
                    metadata=ll[idx[:2500]],
                    tag="visual_encoding",
                )
            del self.visual_patch, self.visual_encoding, self.inertial_encoding, self.label

    def save_models(self, path_root="./models/"):
        cprint("saving models...", "yellow", attrs=["bold"])
        if not os.path.exists(path_root):
            cprint("creating directory: " + path_root, "yellow")
            os.makedirs(path_root)
        else:
            cprint("directory already exists: " + path_root, "red")

        # save the visual encoder
        torch.save(self.visual_encoder.state_dict(), os.path.join(path_root, "visual_encoder.pt"))
        cprint("visual encoder saved", "green")

        # save the proprioceptive encoder
        torch.save(self.ipt_encoder.state_dict(), os.path.join(path_root, "proprioceptive_encoder.pt"))
        cprint("proprioceptive encoder saved", "green")
        cprint("All models successfully saved", "green", attrs=["bold"])


def main():
    parser = argparse.ArgumentParser(description="Train representations using the STERLING framework")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=128, metavar="N", help="input batch size for training (default: 128)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, metavar="N", help="number of epochs to train (default: 1000)"
    )
    parser.add_argument("--lr", type=float, default=3e-4, metavar="LR", help="learning rate (default: 3e-4)")
    parser.add_argument("--l1_coeff", type=float, default=0.5, metavar="L1C", help="L1 loss coefficient (1)")
    parser.add_argument("--num_gpus", "-g", type=int, default=1, metavar="N", help="number of GPUs to use (default: 1)")
    parser.add_argument(
        "--latent_size", type=int, default=64, metavar="N", help="Size of the common latent space (default: 128)"
    )
    parser.add_argument("--data_config_path", type=str, default="spot_data/data_config.yaml")
    args = parser.parse_args()

    model = SterlingModel(lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff)
    dm = SterlingDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="sterling_logs/")

    print("Training the representation learning model...")

    trainer = pl.Trainer(
        devices=args.num_gpus,
        max_epochs=args.epochs,
        log_every_n_steps=10,
        strategy="ddp",
        #  num_sanity_val_steps=0,
        logger=tb_logger,
        sync_batchnorm=True,
        gradient_clip_val=10.0,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )

    # fit the model
    trainer.fit(model, dm)
