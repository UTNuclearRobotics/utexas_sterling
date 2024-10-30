#!/usr/bin/env python3
""" 
train_sterling_costs.py

TODO: Add a description of the script here.
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import tensorboard
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from termcolor import cprint

from visual_representation_learning.train.costs.models import CostNet
from visual_representation_learning.train.representations.data_loader import SterlingDataModule
from visual_representation_learning.train.representations.models import InertialEncoderModel, VisualEncoderModel


class CostModel(pl.LightningModule):
    def __init__(self, latent_size=128, visual_encoder_weights=None, temp=1.0):
        super(CostModel, self).__init__()
        assert visual_encoder_weights is not None, "visual_encoder_weights cannot be None"

        self.visual_encoder = VisualEncoderModel(latent_size=latent_size)
        self.inertial_encoder = InertialEncoderModel(latent_size=latent_size)

        # load the weights from the visual encoder
        cprint("Loading the weights from the visual encoder", "green")
        self.visual_encoder.load_state_dict(torch.load(visual_encoder_weights))
        self.visual_encoder.eval()
        cprint("Loaded the weights from the visual encoder", "green")
        cprint("Loading the weights from the proprioceptive encoder", "green")
        self.inertial_encoder.load_state_dict(
            torch.load(visual_encoder_weights.replace("visual_encoder", "inertial_encoder"))
        )
        self.inertial_encoder.eval()
        cprint("Loaded the weights from the proprioceptive encoder", "green")

        cprint("Loading the k-means model from the pickle file", "green")
        self.kmeansmodel = pickle.load(
            open(
                visual_encoder_weights.replace("visual_encoder.pt", "kmeansmodel.pkl"),
                "rb",
            )
        )
        cprint("Loaded the k-means model from the pickle file", "green")

        self.cost_net = CostNet(latent_size=latent_size)
        self.temp = temp

        # load the kmeanslabels
        self.kmeanslabels = torch.load(visual_encoder_weights.replace("visual_encoder", "kmeanslabels"))
        self.sampleidx = torch.load(visual_encoder_weights.replace("visual_encoder", "sampleidx"))
        # sort the sampleidx and get the idx
        _, idx = torch.sort(self.sampleidx)
        self.kmeanslabels = self.kmeanslabels[idx]
        cprint("The kmeanslabels are : {}".format(np.unique(self.kmeanslabels)), "green")
        cprint("Number of kmeanslabels : {}".format(self.kmeanslabels.shape[0]), "green")
        # now the kmeanslabels are sorted according to the sampleidx

        cprint("The kmeanslabels are : {}".format(np.unique(self.kmeanslabels)), "green")
        cprint(
            "Number of vals in self.sampleidx : {}".format(self.sampleidx.shape[0]),
            "green",
        )
        cprint(
            "Number of vals in self.kmeanslabels : {}".format(self.kmeanslabels.shape[0]),
            "green",
        )

        # TODO: Extrapolate to config
        self.preferences = {
            0: 0,  # yellow_bricks
            1: 4,  # marble_rocks
            2: 0,  # red_bricks
            3: 3,  # grass
            4: 0,  # cement_sidewalk
            5: 5,  # bush
            6: 0,  # asphalt
            7: 0,  # pebble_sidewalk
            8: 0,  # mulch
        }

        self.best_val_loss = 1000000.0
        self.cost_model_save_path = visual_encoder_weights.replace("visual_encoder", "cost_model_grass_eq")

        assert len(self.preferences) == len(
            np.unique(self.kmeanslabels)
        ), "The number of preferences must be equal to the number of clusters"

    def forward(self, visual, inertial, leg, feet):
        with torch.no_grad():
            visual_encoding = self.visual_encoder(visual.float())
            proprioceptive_encoding = self.inertial_encoder(inertial.float(), leg.float(), feet.float())

        return self.cost_net(visual_encoding), torch.cat((visual_encoding, proprioceptive_encoding), dim=-1)

    def softmax_with_temp(self, x, y, temp=1.0):
        x = torch.exp(x / temp)
        y = torch.exp(y / temp)
        return x / (x + y)

    def compute_preference_loss(self, cost, preference_labels, temp=1.0):
        # shuffle the batch and compute the cost per sample
        loss = 0.0
        for i in range(cost.shape[0]):
            # randomly select a sample from the batch
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
        patch1, patch2, inertial, leg, feet, _, _ = batch
        # sampleidx = sampleidx.cpu()

        # kmeanslabels = self.kmeanslabels[sampleidx]
        # preference_labels = [self.preferences[i] for i in kmeanslabels]

        cost1, rep1 = self.forward(patch1, inertial, leg, feet)
        cost2, rep2 = self.forward(patch2, inertial, leg, feet)

        rep1, rep2 = (
            rep1.detach().cpu().detach().numpy(),
            rep2.detach().cpu().detach().numpy(),
        )

        labels1, labels2 = (
            self.kmeansmodel.predict(rep1),
            self.kmeansmodel.predict(rep2),
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
        patch1, patch2, inertial, leg, feet, _, _ = batch
        # sampleidx = sampleidx.cpu()

        # kmeanslabels = self.kmeanslabels[sampleidx]
        # preference_labels = [self.preferences[i] for i in kmeanslabels]

        cost1, rep1 = self.forward(patch1, inertial, leg, feet)
        cost2, rep2 = self.forward(patch2, inertial, leg, feet)

        rep1, rep2 = (
            rep1.detach().cpu().detach().numpy(),
            rep2.detach().cpu().detach().numpy(),
        )

        labels1, labels2 = (
            self.kmeansmodel.predict(rep1),
            self.kmeansmodel.predict(rep2),
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
        This method is called at the end of the validation phase.
        It checks if the current validation loss is the best so far and saves the model if it is.

        - Retrieves the validation loss from the trainer's callback metrics.
        - If running in a distributed setting, aggregates the validation loss across all GPUs.
        - Checks if the current validation loss is the best so far and if we are on GPU 0.
        - If the current validation loss is the best, updates the best validation loss and saves the model.

        Note:
        The PyTorch neural network model being saved is a combination of the visual encoder and the cost network.
        This combined model is saved as a state dictionary, which includes the parameters (weights and biases) of the neural network.
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

    # Argument for batch size
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 512)",
    )

    # Argument for number of epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 1000)",
    )

    # Argument for number of GPUs to use
    parser.add_argument(
        "--num_gpus",
        "-g",
        type=int,
        default=8,
        metavar="N",
        help="number of GPUs to use (default: 8)",
    )

    # Argument for latent size
    parser.add_argument(
        "--latent_size",
        type=int,
        default=512,
        metavar="N",
        help="Size of the common latent space (default: 128)",
    )

    # Argument for whether to save the k-means model and encoders at the end of the run
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        metavar="N",
        help="Whether to save the k means model and encoders at the end of the run",
    )

    # Argument for experiment save path
    parser.add_argument(
        "--expt_save_path",
        "-e",
        type=str,
        default="/robodata/haresh92/spot-vrl/models/acc_0.98154_22-01-2023-05-13-46_",
    )

    # Argument for data configuration path
    parser.add_argument("--data_config_path", type=str, default="spot_data/data_config.yaml")

    # Argument for temperature scaling
    parser.add_argument("--temp", type=float, default=1.0)

    # Parse the arguments
    args = parser.parse_args()
    
    # Print all arguments in a list
    args_list = [
        f"batch_size: {args.batch_size}",
        f"epochs: {args.epochs}",
        f"num_gpus: {args.num_gpus}",
        f"latent_size: {args.latent_size}",
        f"save: {args.save}",
        f"expt_save_path: {args.expt_save_path}",
        f"data_config_path: {args.data_config_path}",
        f"temp: {args.temp}",
    ]
    for arg in args_list:
        cprint(arg, "cyan")
    
    return args

def main():
    args = parse_args()

    # Initialize the CostModel with the parsed arguments
    model = CostModel(
        latent_size=128,
        visual_encoder_weights=os.path.join(args.expt_save_path, "visual_encoder.pt"),
        temp=args.temp,
    )

    # Initialize the data module with the parsed arguments
    dm = SterlingDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)

    # Initialize the TensorBoard logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="cost_training_logs/")

    print("Training the cost function model...")

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        gpus=list(np.arange(args.num_gpus)),  # List of GPUs to use
        max_epochs=args.epochs,  # Maximum number of epochs
        log_every_n_steps=10,  # Log every n steps
        strategy="ddp",  # Distributed data parallel strategy
        num_sanity_val_steps=0,  # Number of sanity validation steps
        sync_batchnorm=True,  # Synchronize batch normalization
        logger=tb_logger,  # Logger for TensorBoard
    )

    # Fit the model using the trainer and data module
    trainer.fit(model, dm)
