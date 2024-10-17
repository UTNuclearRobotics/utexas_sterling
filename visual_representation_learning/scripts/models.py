#!/usr/bin/env python3

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from termcolor import cprint


# create a pytorch model for the proprioception data
class ProprioceptionModel(nn.Module):
    def __init__(self, latent_size=64, p=0.05):
        super(ProprioceptionModel, self).__init__()

        self.inertial_encoder = nn.Sequential(  # input shape : (batch_size, 1, 1407)
            nn.Flatten(),
            nn.Linear(201 * 6, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.leg_encoder = nn.Sequential(  # input shape : (batch_size, 1, 900)
            nn.Flatten(),
            nn.Linear(900, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.feet_encoder = nn.Sequential(  # input shape : (batch_size, 1, 500)
            nn.Flatten(),
            nn.Linear(500, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 + 32 + 32, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, inertial, leg, feet):
        inertial = self.inertial_encoder(inertial)
        leg = self.leg_encoder(leg)
        feet = self.feet_encoder(feet)

        features = self.fc(torch.cat([inertial, leg, feet], dim=1))

        # normalize the features
        features = F.normalize(features, dim=-1)

        return features


class RCAModel(nn.Module):
    def __init__(self, n_classes=6):
        super(RCAModel, self).__init__()
        self.model = nn.Sequential(  # input shape : (batch_size, 3, 64, 64)
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8 x 32 x 32
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x 16 x 16
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 8 x 8
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 4 x 4
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 x 2 x 2
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, n_classes),
        )

    def forward(self, x):
        return self.model(x)


class RCAModelWrapped(nn.Module):
    def __init__(self, model, rca_costs_pkl_path=None):
        super(RCAModelWrapped, self).__init__()
        self.model = model
        assert rca_costs_pkl_path is not None, "rca_costs_pkl_path is None"
        cprint("Loading WrappedModel for RCA", "green")
        self.rca_costs = pickle.load(open(rca_costs_pkl_path, "rb"))
        self.terrain_classes = list(self.rca_costs.keys())

    def forward(self, x):
        x = self.model(x)
        # get the class
        class_ = torch.argmax(x, dim=1)
        # get the cost
        return torch.tensor([self.rca_costs[self.terrain_classes[tc]] for tc in class_])


class CostNet(nn.Module):
    def __init__(self, latent_size=64):
        super(CostNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.BatchNorm1d(latent_size // 2),
            nn.ReLU(),
            nn.Linear(latent_size // 2, 1),
            nn.Sigmoid(),  # nn.ReLU(), #nn.Softplus(),
        )

    def forward(self, x):
        return self.fc(x)


class VisualEncoderEfficientModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderEfficientModel, self).__init__()

        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        del self.model._fc
        self.model._fc = nn.Linear(1280, latent_size)

    def forward(self, x):
        # image is between 0 and 1, normalize it to -1 and 1
        # x = x * 2 - 1
        x = self.model(x)
        x = F.normalize(x, dim=-1)
        return x


class VisualEncoderModel(nn.Module):
    """
    Convolutional Neural Network for encoding images into a lower-dimensional latent space.
    This model uses a series of convolutional layers followed by batch normalization and PReLU activation functions
    to progressively reduce the spatial dimensions of the input image while increasing the depth. The final output
    is a latent vector of a specified size.

    Attributes:
        model (nn.Sequential): A sequential container of the layers that make up the neural network.

    Args:
        latent_size (int): The size of the output latent vector. Default is 64.

    Layers:
        - Conv2d: Applies a 2D convolution over an input signal composed of several input planes.
        - BatchNorm2d: Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension).
        - PReLU: Applies the Parametric Rectified Linear Unit function element-wise.
        - AvgPool2d: Applies a 2D average pooling over an input signal composed of several input planes.
        - Flatten: Flattens the input tensor to a 1D tensor.
        - Linear: Applies a linear transformation to the incoming data.
        - ReLU: Applies the Rectified Linear Unit function element-wise.

    Methods:
        forward(x):
            Defines the computation performed at every call. Takes an input tensor `x` and returns the encoded latent vector.

            Args:
                x (torch.Tensor): The input tensor representing a batch of images.

            Returns:
                torch.Tensor: The output latent vector after passing through the network.
    """

    def __init__(self, latent_size=64):
        super(VisualEncoderModel, self).__init__()
        self.model = nn.Sequential(
            # First convolution layer: input channels = 3 (RGB), output channels = 8, kernel size = 3x3, stride = 2
            # This layer reduces the spatial dimensions and increases the depth to capture more complex features.
            nn.Conv2d(3, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.PReLU(),  # Output shape : (batch_size, 8, 31, 31)
            # Second convolution layer: input channels = 8, output channels = 16, kernel size = 3x3, stride = 2
            # Further reduces the spatial dimensions and increases the depth to capture more detailed features.
            nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),  # Output shape : (batch_size, 16, 15, 15)
            # Third convolution layer: input channels = 16, output channels = 32, kernel size = 3x3, stride = 2
            # Continues to reduce the spatial dimensions and increases the depth to capture even more detailed features.
            nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # Output shape : (batch_size, 32, 7, 7)
            # Fourth convolution layer: input channels = 32, output channels = 64, kernel size = 3x3, stride = 2
            # Further reduces the spatial dimensions to a very small size and increases the depth to capture high-level features.
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.PReLU(),  # Output shape : (batch_size, 64, 3, 3)
            # Average pooling layer: kernel size = 3x3
            # Reduces each feature map to a single value, effectively summarizing the feature map.
            nn.AvgPool2d(kernel_size=3),  # Output shape : (batch_size, 64, 1, 1)
            # Flatten the tensor to a 1D tensor
            nn.Flatten(),
            # Fully connected layer: input features = 64, output features = latent_size
            # Reduces the feature map to the desired latent size.
            nn.Linear(64, latent_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # return F.normalize(self.model(x), dim=-1)
        return self.model(x)
