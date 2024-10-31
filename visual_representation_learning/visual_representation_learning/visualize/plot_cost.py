""" """

# TODO: Untested
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from visual_representation_learning.train.costs.models import CostNet
from visual_representation_learning.train.costs.train_costs import CostModel
from visual_representation_learning.train.representations.models import VisualEncoderEfficientModel, VisualEncoderModel
from visual_representation_learning.train.representations.data_loader import SterlingDataModule


def main():
    parser = argparse.ArgumentParser(description="Plot cost for different labels.")
    parser.add_argument("--data_config", type=str, required=True, help="Path to the data configuration file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the folder of models.")
    args = parser.parse_args()
    MODEL_PATH = args.model_path

    # Get validation dataset
    dm = SterlingDataModule(data_config_path=args.data_config)
    dm.setup()
    dataset = dm.val_dataset

    visual_encoder = VisualEncoderEfficientModel(latent_size=64)
    cost_net = CostNet(latent_size=64)

    # Load weights of model
    model = nn.Sequential(visual_encoder, cost_net)
    model_state_dict = torch.load(os.path.join(MODEL_PATH, "cost_model.pt"))
    model.load_state_dict(model_state_dict)
    model.eval()
    model.cuda()

    # Load the clusters
    kmeans_labels = torch.load(os.path.join(MODEL_PATH, "kmeans_labels.pt"))
    unique_labels = np.unique(kmeans_labels)

    # Initialize the data dictionary
    data = {label: [] for label in unique_labels}

    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)
    for batch in tqdm(dataloader, desc="Processing batches"):
        patch1, patch2, imu, label, idx = batch

        # Move to device
        patch1 = patch1.cuda()
        patch2 = patch2.cuda()
        imu = imu.cuda()

        # Forward pass
        with torch.no_grad():
            cost = model(patch1.float())
            cost = cost.detach().cpu().numpy().flatten()

        for i in range(len(idx)):
            kmeans_label = kmeans_labels[idx[i]]
            data[kmeans_label].append(cost[i])

    # Plot the labels on the x-axis and the mean cost with standard deviation on the y-axis
    labels = list(data.keys())
    mean_costs = [np.mean(data[label]) for label in labels]
    std_costs = [np.std(data[label]) for label in labels]

    # Plot vertical bars, labels text on x-axis is rotated
    plt.figure()
    plt.bar(labels, mean_costs, yerr=std_costs, align="center", alpha=0.5, ecolor="black", capsize=10)
    plt.xticks(rotation=45)
    plt.ylabel("Cost")
    plt.title("Costs for different labels")
    plt.tight_layout()  # Prevent the labels from being cut off

    # Save the plot
    plt.savefig(os.path.join(MODEL_PATH, "costs_bar.png"))
    cprint("Saved costs_bar.png", "green")

    # Draw a boxplot
    plt.figure()
    plt.boxplot(data.values())
    plt.xticks(range(1, len(data) + 1), labels, rotation=45)
    plt.ylabel("Cost")
    plt.title("Costs for different labels")
    plt.tight_layout()  # Prevent the labels from being cut off

    # Save the plot
    plt.savefig(os.path.join(MODEL_PATH, "costs_boxplot.png"))
    cprint("Saved costs_boxplot.png", "green")


if __name__ == "__main__":
    main()
