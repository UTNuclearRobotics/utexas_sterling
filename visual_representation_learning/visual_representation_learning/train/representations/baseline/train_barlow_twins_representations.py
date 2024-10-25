import argparse
import os
from datetime import datetime
from typing import List, Tuple, Union

# import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import tensorboard as tb
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
from ament_index_python.packages import get_package_share_directory
# from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from scipy import fftpack, gradient
# from tqdm import tqdm

from visual_representation_learning.train.representations.baseline.data_loader import MyDataModule
from termcolor import cprint

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BarlowTwinsModel(pl.LightningModule):
    def __init__(
        self,
        lr=3e-4,
        latent_size=64,
        inertial_shape=None,
        scale_loss: float = 1.0 / 32,
        lambd: float = 3.9e-6,
        weight_decay=1e-6,
        per_device_batch_size=32,
        num_warmup_steps_or_ratio: Union[int, float] = 0.1,
        inertial_stats=None,
        only_vision=False,
    ):
        super(BarlowTwinsModel, self).__init__()

        self.save_hyperparameters("lr", "latent_size", "inertial_shape", "scale_loss", "lambd", "weight_decay")

        self.scale_loss = scale_loss
        self.only_vision = only_vision
        self.lambd = lambd
        self.weight_decay = weight_decay
        self.lr = lr
        self.per_device_batch_size = per_device_batch_size
        self.num_warmup_steps_or_ratio = num_warmup_steps_or_ratio
        # self.mean_inertial, self.std_inertial = torch.tensor(inertial_stats['mean'], requires_grad=False), \
        # 										torch.tensor(inertial_stats['std'], requires_grad=False)
        self.max_inertial, self.min_inertial = (
            torch.tensor(inertial_stats["max"], requires_grad=False),
            torch.tensor(inertial_stats["min"], requires_grad=False),
        )

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False),  # 63 x 63
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),  # 31 x 31
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False),  # 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False),  # 5 x 5
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 2 x 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Flatten(),  # 256 output
            nn.Linear(256, 64),
        )

        if not only_vision:
            self.inertial_encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm1d(8),
                nn.PReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2),
                nn.Conv1d(8, 16, kernel_size=5, stride=2, bias=False),
                nn.BatchNorm1d(16),
                nn.PReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2),
                nn.Conv1d(16, 32, kernel_size=5, stride=2, bias=False),
                nn.BatchNorm1d(32),
                nn.PReLU(),
                nn.AvgPool1d(kernel_size=3, stride=2),
                nn.Conv1d(32, 64, kernel_size=3, stride=2, bias=False),
                nn.BatchNorm1d(64),
                nn.PReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(256, 64),
            )

        self.projector = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_size, bias=False),
        )

        # Normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(latent_size, affine=False, track_running_stats=False)

        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

    def forward(self, visual_patch, imu_history):
        if not self.only_vision:  # min-max normalize the inertial data
            with torch.no_grad():
                # inertial_mean = self.mean_inertial.to(imu_history.device)
                # inertial_std = self.std_inertial.to(imu_history.device)
                # imu_history = (imu_history - inertial_mean) / inertial_std
                self.min_inertial = self.min_inertial.to(imu_history.device)
                self.max_inertial = self.max_inertial.to(imu_history.device)
                imu_history = (imu_history - self.min_inertial) / (self.max_inertial - self.min_inertial)

        # print('imu history shape : ', imu_history.shape)
        # imu_history = self.normalize_inertial(imu_history)

        z1 = self.visual_encoder(visual_patch)
        if not self.only_vision:
            z2 = self.inertial_encoder(imu_history)
        else:
            z2 = self.visual_encoder(imu_history)

        # L2 normalize along encoding dimension
        # v_encoded = F.normalize(v_encoded, dim=1)
        # i_encoded = F.normalize(i_encoded, dim=1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        return z1, z2

    def barlow_loss(self, z1, z2):
        # z1 = (z1 - torch.mean(z1, dim=0))/(torch.std(z1, dim=0) + 1e-4)
        # z2 = (z2 - torch.mean(z2, dim=0))/(torch.std(z2, dim=0) + 1e-4)

        z1 = self.bn(z1)
        z2 = self.bn(z2)

        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.per_device_batch_size * self.trainer.num_processes)
        self.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale_loss)

        loss = on_diag + self.lambd * off_diag
        return loss

    def vicreg_loss(self, z1, z2):
        repr_loss = F.mse_loss(z1, z2)

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2

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
        visual, visual2, inertial, _ = batch
        if self.only_vision:
            emb1, emb2 = self.forward(visual, visual2)
        else:
            emb1, emb2 = self.forward(visual, inertial)

        # loss = self.barlow_loss(visual_emb, inertial_emb)
        loss = self.vicreg_loss(emb1, emb2)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        visual, visual2, inertial, _ = batch
        if self.only_vision:
            emb1, emb2 = self.forward(visual, visual2)
        else:
            emb1, emb2 = self.forward(visual, inertial)

        # loss = self.barlow_loss(visual_emb, inertial_emb)
        loss = self.vicreg_loss(emb1, emb2)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.current_epoch % 10 == 0:
            visual_patch, visual_patch_2, imu_history, label = batch
            label = np.asarray(label)
            visual_patch = visual_patch.float()
            with torch.no_grad():
                visual_encoding = self.visual_encoder(visual_patch.cuda())
            visual_encoding = visual_encoding.cpu()
            visual_patch = visual_patch.cpu()
            # visual_encoding = F.normalize(visual_encoding, dim=1)

            if batch_idx == 0:
                self.visual_encoding = [visual_encoding[:, :]]
                self.visual_patch = [visual_patch[:, :, :, :]]
                self.label = label[:]
            else:
                self.visual_patch.append(visual_patch[:, :, :, :])
                self.visual_encoding.append(visual_encoding[:, :])
                self.label = np.concatenate((self.label, label[:]))

    def on_validation_end(self) -> None:
        if self.current_epoch % 10 == 0:
            self.visual_patch = torch.cat(self.visual_patch, dim=0)
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            idx = np.arange(self.visual_encoding.shape[0])

            # randomize numpy array
            np.random.shuffle(idx)

            self.logger.experiment.add_embedding(
                mat=self.visual_encoding[idx[:2000], :],
                label_img=self.visual_patch[idx[:2000], :, :, :],
                global_step=self.current_epoch,
                metadata=self.label[idx[:2000]],
            )

    @property
    def total_training_steps(self) -> int:
        dataset_size = len(self.trainer.datamodule.train_dataloader())
        num_devices = self.trainer.tpu_cores if self.trainer.tpu_cores else self.trainer.num_processes
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_training_steps


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Dual Auto Encoder")
    parser.add_argument(
        "--batch_size", type=int, default=256, metavar="N", help="input batch size for training (default: 256)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, metavar="N", help="number of epochs to train (default: 1000)"
    )
    parser.add_argument("--lr", type=float, default=3e-4, metavar="LR", help="learning rate (default: 3e-4)")
    parser.add_argument("--log_dir", type=str, default="logs/", metavar="N", help="log directory (default: logs)")
    parser.add_argument(
        "--model_dir", type=str, default="models/", metavar="N", help="model directory (default: models)"
    )
    parser.add_argument("--num_gpus", type=int, default=1, metavar="N", help="number of GPUs to use (default: 1)")
    parser.add_argument(
        "--latent_size", type=int, default=128, metavar="N", help="Size of the common latent space (default: 128)"
    )
    parser.add_argument("--only_vision", action="store_true", default=False)
    parser.add_argument(
        "--dataset_config_path", type=str, default=os.path.join(package_share_directory, "config", "dataset.yaml")
    )
    args = parser.parse_args()

    cprint("Arguments:", "cyan")
    cprint(f"- Batch size: {args.batch_size}", "cyan")
    cprint(f"- Epochs: {args.epochs}", "cyan")
    cprint(f"- Learning rate: {args.lr}", "cyan")
    cprint(f"- Log directory: {args.log_dir}", "cyan")
    cprint(f"- Model directory: {args.model_dir}", "cyan")
    cprint(f"- Number of GPUs: {args.num_gpus}", "cyan")
    cprint(f"- Latent size: {args.latent_size}", "cyan")
    cprint(f"- Only vision: {args.only_vision}", "cyan")
    cprint(f"- Dataset config path: {args.dataset_config_path}", "cyan")

    # Check if the dataset config yaml file exists
    if not os.path.exists(args.dataset_config_path):
        raise FileNotFoundError(args.dataset_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = MyDataModule(data_config_path=args.dataset_config_path, batch_size=args.batch_size)
    model = BarlowTwinsModel(
        lr=args.lr,
        latent_size=args.latent_size,
        inertial_shape=1200,
        scale_loss=1.0,
        lambd=1.0 / args.latent_size,
        per_device_batch_size=args.batch_size,
        inertial_stats=dm.inertial_stat,
        only_vision=args.only_vision,
    ).to(device)

    early_stopping_cb = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.00, patience=1000)
    model_checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(ros_ws_dir, "models"),
        filename=f'barlow_twins_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        monitor="val_loss",
        verbose=True,
    )

    cprint("Training model...", "green")

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        max_epochs=args.epochs,
        callbacks=[early_stopping_cb, model_checkpoint_cb],
        log_every_n_steps=10,
        strategy="ddp",
        num_sanity_val_steps=0,
        logger=True,
        sync_batchnorm=True,
    )

    trainer.fit(model, dm)
