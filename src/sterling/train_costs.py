#!/usr/bin/env python3
"""
TODO: Add a description of the script here.
"""

import argparse
import os
import pickle

import numpy as np
import pytorch_lightning as pl

# import tensorboard
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from termcolor import cprint

# Sterling imports
from models import CostNet
from data_loader import SterlingDataModule
from models import (
    InertialEncoderModel,
    # VisualEncoderModel,
    VisualEncoderEfficientModel,
)


class CostModel(pl.LightningModule):
    def __init__(self, latent_size=128, model_path=None, temp=1.0):
        super(CostModel, self).__init__()

        # Verify that the model_path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The specified model_path {model_path} does not exist")

        # Verify that the required files exist in the model_path directory
        required_files = [
            "inertial_encoder.pt",
            "visual_encoder.pt",
            "sample_idx.pt",
            "kmeans_labels.pt",
            "kmeans_model.pkl",
        ]
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file {file} not found in {model_path}")

        self.visual_encoder = VisualEncoderEfficientModel(latent_size=latent_size)
        # self.visual_encoder = VisualEncoderModel(latent_size=latent_size)
        self.inertial_encoder = InertialEncoderModel(latent_size=latent_size)

        # Load the weights from the visual encoder
        cprint("Loading the weights from the visual encoder", "green")
        self.visual_encoder.load_state_dict(torch.load(os.path.join(model_path, "visual_encoder.pt")))
        self.visual_encoder.eval()

        # Load the weights from the inertial encoder
        cprint("Loading the weights from the inertial encoder", "green")
        self.inertial_encoder.load_state_dict(torch.load(os.path.join(model_path, "inertial_encoder.pt")))
        self.inertial_encoder.eval()

        # Load the k-means model from the pickle file
        cprint("Loading the k-means model from the pickle file", "green")
        with open(os.path.join(model_path, "kmeans_model.pkl"), "rb") as f:
            self.kmeans_model = pickle.load(f)

        self.cost_net = CostNet(latent_size=latent_size)
        self.temp = temp

        # Load the kmeans_labels and sample_idx
        self.kmeans_labels = torch.load(os.path.join(model_path, "kmeans_labels.pt"))
        self.sample_idx = torch.load(os.path.join(model_path, "sample_idx.pt"))

        # Sort the sample_idx and get the idx
        _, idx = torch.sort(self.sample_idx)
        self.kmeans_labels = self.kmeans_labels[idx]

        cprint("The kmeans_labels are : {}".format(np.unique(self.kmeans_labels)), "green")
        cprint("Number of kmeans_labels : {}".format(self.kmeans_labels.shape[0]), "green")
        cprint("Number of vals in self.sample_idx : {}".format(self.sample_idx.shape[0]), "green")
        cprint("Number of vals in self.kmeans_labels : {}".format(self.kmeans_labels.shape[0]), "green")

        # Define preferences for each cluster
        # TODO: Extrapolate to config
        self.preferences = {
            0: 0,  # yellow_bricks
            1: 4,  # marble_rocks
            # 2: 0,  # red_bricks
            # 3: 3,  # grass
            # 4: 0,  # cement_sidewalk
            # 5: 5,  # bush
            # 6: 0,  # asphalt
            # 7: 0,  # pebble_sidewalk
            # 8: 0,  # mulch
        }

        self.best_val_loss = 1000000.0
        self.cost_model_save_path = os.path.join(model_path, "cost_model.pt")

        assert len(self.preferences) == len(
            np.unique(self.kmeans_labels)
        ), "The number of preferences must be equal to the number of clusters"

    def forward(self, visual, inertial):
        """
        Forward pass for the cost network.
        Args:
            visual (torch.Tensor): The visual input tensor.
            inertial (torch.Tensor): The inertial input tensor.
        Returns:
            torch.Tensor: The cost encoding tensor.
            torch.Tensor: The combined features tensor.
        """
        with torch.no_grad():
            visual_encoding = self.visual_encoder(visual.float())
            inertial_encoding = self.inertial_encoder(inertial.float())
        cost_encoding = self.cost_net(visual_encoding)
        combined_features = torch.cat((visual_encoding, inertial_encoding), dim=1)
        return cost_encoding, combined_features

    def softmax_with_temp(self, x, y, temp=1.0):
        """
        Compute the softmax of the two costs with a temperature scaling factor.
        Args:
            x (torch.Tensor): The first cost tensor.
            y (torch.Tensor): The second cost tensor.
            temp (float): The temperature scaling factor.
        Returns:
            torch.Tensor: The softmax of the two costs.
        """
        x = torch.exp(x / temp)
        y = torch.exp(y / temp)
        return x / (x + y)

    def compute_preference_loss(self, cost, preference_labels, temp=1.0):
        """
        Compute the preference loss for a batch of costs.
        Args:
            cost (torch.Tensor): The cost tensor.
            preference_labels (List[int]): The list of preference labels.
            temp (float): The temperature scaling factor.
        Returns:
            torch.Tensor: The preference loss.
        """
        # Shuffle the batch and compute the cost per sample
        loss = 0.0
        for i in range(cost.shape[0]):
            # Randomly select a sample from the batch
            j = torch.randint(0, cost.shape[0], (1,))[0]
            if preference_labels[i] < preference_labels[j]:
                loss += self.softmax_with_temp(cost[i], cost[j], temp=temp)
            elif preference_labels[i] > preference_labels[j]:
                loss += self.softmax_with_temp(cost[j], cost[i], temp=temp)
            else:
                loss += (cost[i] - cost[j]) ** 2
        return loss / cost.shape[0]

    def compute_ranking_loss(self, cost, preference_labels):
        # convert list of preferences to a tensor
        preference_labels = torch.tensor(preference_labels).float().to(cost.device)
        return torch.nn.SmoothL1Loss()(cost.flatten(), preference_labels)
        # loss = torch.nn.MSELoss()(cost, torch.tensor(preference_labels).float().to(cost.device))

    def training_step(self, batch, batch_idx):
        patch1, patch2, inertial, _, _ = batch
        # sample_idx = sample_idx.cpu()

        # kmeans_labels = self.kmeans_labels[sample_idx]
        # preference_labels = [self.preferences[i] for i in kmeans_labels]

        cost1, rep1 = self.forward(patch1, inertial)
        cost2, rep2 = self.forward(patch2, inertial)

        rep1, rep2 = (
            rep1.detach().cpu().detach().numpy(),
            rep2.detach().cpu().detach().numpy(),
        )

        labels1, labels2 = (
            self.kmeans_model.predict(rep1),
            self.kmeans_model.predict(rep2),
        )
        preference_labels1 = [self.preferences[i] for i in labels1]
        preference_labels2 = [self.preferences[i] for i in labels2]

        # compute the preference loss
        # pref_loss = 0.5*self.compute_preference_loss(cost1, preference_labels1, temp=self.temp) + \
        #     0.5*self.compute_preference_loss(cost2, preference_labels2, temp=self.temp)
        pref_loss = 0.5 * self.compute_ranking_loss(cost1, preference_labels1) + 0.5 * self.compute_ranking_loss(
            cost2, preference_labels2
        )

        # cost must be invariant to the viewpoint of the patch
        vpt_inv_loss = torch.mean((cost1 - cost2) ** 2)
        # penalty for the cost crossing 25.0
        penalty_loss = torch.mean(torch.relu(cost1 - 25.0)) + torch.mean(torch.relu(cost2 - 25.0))

        loss = pref_loss + 0.1 * vpt_inv_loss + penalty_loss

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_pref_loss", pref_loss, prog_bar=True, on_epoch=True)
        self.log("train_vpt_inv_loss", vpt_inv_loss, prog_bar=True, on_epoch=True)
        self.log("train_penalty_loss", penalty_loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch1, patch2, inertial, _, _ = batch
        # sample_idx = sample_idx.cpu()

        # kmeans_labels = self.kmeans_labels[sample_idx]
        # preference_labels = [self.preferences[i] for i in kmeans_labels]

        cost1, rep1 = self.forward(patch1, inertial)
        cost2, rep2 = self.forward(patch2, inertial)

        rep1, rep2 = (
            rep1.detach().cpu().detach().numpy(),
            rep2.detach().cpu().detach().numpy(),
        )

        labels1, labels2 = (
            self.kmeans_model.predict(rep1),
            self.kmeans_model.predict(rep2),
        )
        preference_labels1 = [self.preferences[i] for i in labels1]
        preference_labels2 = [self.preferences[i] for i in labels2]

        # compute the preference loss
        # pref_loss = 0.5*self.compute_preference_loss(cost1, preference_labels1, temp=self.temp) + \
        #     0.5*self.compute_preference_loss(cost2, preference_labels2, temp=self.temp)
        pref_loss = 0.5 * self.compute_ranking_loss(cost1, preference_labels1) + 0.5 * self.compute_ranking_loss(
            cost2, preference_labels2
        )

        # cost must be invariant to the viewpoint of the patch
        vpt_inv_loss = torch.mean((cost1 - cost2) ** 2)
        # penalty for the cost crossing 25.0
        penalty_loss = torch.mean(torch.relu(cost1 - 25.0)) + torch.mean(torch.relu(cost2 - 25.0))

        loss = pref_loss + 0.1 * vpt_inv_loss + penalty_loss

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_pref_loss", pref_loss, prog_bar=True, on_epoch=True)
        self.log("val_vpt_inv_loss", vpt_inv_loss, prog_bar=True, on_epoch=True)
        self.log("val_penalty_loss", penalty_loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_end(self):
        """
        Save the batch data for visualization during validation.
        Args:
            batch (tuple): A tuple containing the input data, inertial data, and labels.
            batch_idx (int): The index of the current batch.
        """

        # Get the validation loss from the trainer's callback metrics
        val_loss = self.trainer.callback_metrics["val_loss"]

        # If running in a distributed setting, aggregate the validation loss across all GPUs
        if self.trainer.world_size > 1:
            # Convert the validation loss to a tensor and move it to the current GPU
            val_loss = torch.tensor(val_loss).cuda()

            # Perform an all-reduce operation to sum the validation losses from all GPUs
            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)

            # Average the validation loss by dividing by the number of GPUs
            val_loss = val_loss / self.trainer.world_size

            # Move the validation loss back to the CPU and convert it to a numpy array
            val_loss = val_loss.cpu().numpy()

        # Check if the current validation loss is the best so far and if we are on GPU 0
        if val_loss < self.best_val_loss and torch.cuda.current_device() == 0:
            self.best_val_loss = val_loss

            # Wrap the visual encoder and the cost network in a single module
            model = nn.Sequential(self.visual_encoder, self.cost_net)

            # Save the state dictionary of the combined model to the specified path
            torch.save(model.state_dict(), self.cost_model_save_path)
            cprint("Saved the model with the best validation loss", "green")

            # If this is the last epoch, display the model save path
            if self.trainer.current_epoch == self.trainer.max_epochs - 1:
                cprint(
                    "The model was saved at {}".format(self.cost_model_save_path),
                    "green",
                    attrs=["bold"],
                )

        cprint("The validation loss is {}".format(val_loss), "green")

    def configure_optimizers(self):
        # use only costnet parameters
        return torch.optim.AdamW(self.cost_net.parameters(), lr=3e-4, weight_decay=1e-5, amsgrad=True)


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=128,
        metavar="N",
        help="Input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        metavar="N",
        help="Number of epochs to train (default: 100)",
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
        "-l",
        type=int,
        default=64,
        metavar="N",
        help="Size of the common latent space (default: 64)",
    )
    # parser.add_argument(
    #     "--save",
    #     type=int,
    #     default=0,
    #     metavar="N",
    #     help="Whether to save the k means model and encoders at the end of the run",
    # )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        default="",
        help="Path to the saved representations models (required)",
    )
    parser.add_argument(
        "--data_config_path",
        "-c",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "data_config.yaml"),
        help="Path to the data configuration file.",
    )
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature scaling factor (default: 1.0)")
    args = parser.parse_args()

    # Print all arguments in a list
    args_list = [
        f"batch_size: {args.batch_size}",
        f"epochs: {args.epochs}",
        f"num_gpus: {args.num_gpus}",
        f"latent_size: {args.latent_size}",
        # f"save: {args.save}",
        f"model_path: {args.model_path}",
        f"data_config_path: {args.data_config_path}",
        f"temp: {args.temp}",
    ]
    for arg in args_list:
        cprint(arg, "cyan")

    return args


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Initialize the CostModel with the parsed arguments
    model = CostModel(
        latent_size=args.latent_size,
        model_path=args.model_path,
        temp=args.temp,
    )

    # Initialize the data module with the parsed arguments
    dm = SterlingDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)

    # Initialize the TensorBoard logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(os.path.dirname(__file__), "..", "..", "logs", "sterling_costs_logs")
    )

    cprint("Training the cost function model...", "green", attrs=["bold"])

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        devices=args.num_gpus,
        max_epochs=args.epochs,  # Maximum number of epochs
        log_every_n_steps=10,  # Log every n steps
        strategy="ddp_find_unused_parameters_true",
        # strategy="ddp",  # Distributed data parallel strategy
        num_sanity_val_steps=0,  # Number of sanity validation steps
        sync_batchnorm=True,  # Synchronize batch normalization
        logger=tb_logger,  # Logger for TensorBoard
    )

    # Fit the model using the trainer and data module
    trainer.fit(model, dm)
