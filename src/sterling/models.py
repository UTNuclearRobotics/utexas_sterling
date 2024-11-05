#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import timm
from torchinfo import summary


class IPTEncoderModel(nn.Module):
    def __init__(self, latent_size=64, p=0.2, l2_normalize=True):
        super(IPTEncoderModel, self).__init__()

        self.inertial_encoder = nn.Sequential(  # input shape : (batch_size, 1, 603)
            nn.Flatten(),
            nn.Linear(201 * 3, 128),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size // 2),
        )

        self.leg_encoder = nn.Sequential(  # input shape : (batch_size, 1, 900)
            nn.Flatten(),
            nn.Linear(25 * 36, 128),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size // 2),
        )

        self.feet_encoder = nn.Sequential(  # input shape : (batch_size, 1, 500)
            nn.Flatten(),
            nn.Linear(25 * 20, 128),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size // 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(3 * latent_size // 2, latent_size), nn.Mish(), nn.Linear(latent_size, latent_size)
        )

        self.l2_normalize = l2_normalize

    def forward(self, inertial, leg, feet):
        inertial_features = self.inertial_encoder(inertial)
        leg_features = self.leg_encoder(leg)
        feet_features = self.feet_encoder(feet)

        combined_features = torch.cat([inertial_features, leg_features, feet_features], dim=1)
        nonvis_features = self.fc(combined_features)

        # Normalize
        if self.l2_normalize:
            nonvis_features = F.normalize(nonvis_features, dim=-1)

        return nonvis_features


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


class InertialEncoderModel(nn.Module):
    def __init__(self, latent_size=64, p=0.2, l2_normalize=True):
        super(InertialEncoderModel, self).__init__()

        self.inertial_encoder = nn.Sequential(  # input shape : (batch_size, 1, 1200)
            nn.Flatten(),
            nn.Linear(1200, 128),
            nn.Mish(),
            nn.Dropout(p),
            nn.Linear(128, latent_size),
        )

        self.l2_normalize = l2_normalize

    def forward(self, inertial):
        inertial_features = self.inertial_encoder(inertial)

        # Normalize
        if self.l2_normalize:
            inertial_features = F.normalize(inertial_features, dim=-1)

        return inertial_features


class VisualEncoderEfficientModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderEfficientModel, self).__init__()

        # Load a pre-trained EfficientNet model (efficientnet-b0)
        self.model = EfficientNet.from_pretrained("efficientnet-b0")

        # Remove the final fully connected layer of the EfficientNet model
        del self.model._fc

        # Replace the final fully connected layer with a new one that has 'latent_size' output features
        self.model._fc = nn.Linear(1280, latent_size)

    def forward(self, x):
        # image is between 0 and 1, normalize it to -1 and 1
        # x = x * 2 - 1
        x = self.model(x)
        x = F.normalize(x, dim=-1)
        return x


class VisualEncoderModel(nn.Module):
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


# class VisualEncoderModel(nn.Module):
#     def __init__(self, latent_size=64, replace_bn_w_gn=False, l2_normalize=True):
#         super(VisualEncoderModel, self).__init__()
#         self.encoder = timm.create_model("efficientnet_b0", pretrained=True)

#         # Replace batchnorms with groupnorms if needed
#         if replace_bn_w_gn:
#             self.encoder = self.convert_bn_to_gn(self.encoder, features_per_group=16)

#         # Remove the final classification layer
#         self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

#         # Fully connected layers
#         self.fc = nn.Sequential(nn.Linear(1280, latent_size), nn.Mish(), nn.Linear(latent_size, latent_size))

#         self.l2_normalize = l2_normalize

#     def forward(self, x):
#         vis_features = self.encoder(x)
#         vis_features = self.fc(vis_features)
#         if self.l2_normalize:
#             vis_features = F.normalize(vis_features, dim=-1)
#         return vis_features

#     def convert_bn_to_gn(self, module, features_per_group=16):
#         """Replace all BatchNorm layers with GroupNorm layers."""
#         if isinstance(module, nn.BatchNorm2d):
#             num_groups = max(1, module.num_features // features_per_group)
#             return nn.GroupNorm(num_groups, module.num_features, eps=module.eps, affine=module.affine)

#         for name, child_module in module.named_children():
#             module.add_module(name, self.convert_bn_to_gn(child_module, features_per_group=features_per_group))

#         return module


class VisualEncoderTinyModel(nn.Module):
    def __init__(self, latent_size=64, l2_normalize=True):
        super(VisualEncoderTinyModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(8),  # output shape : (batch_size, 8, 64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output shape : (batch_size, 8, 32, 32),
        )

        self.skipblock = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(8),  # output shape : (batch_size, 8, 32, 32),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(8),  # output shape : (batch_size, 8, 32, 32),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(16),  # output shape : (batch_size, 16, 32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output shape : (batch_size, 16, 16, 16),
        )

        self.skipblock2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(16),  # output shape : (batch_size, 16, 16, 16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(16),  # output shape : (batch_size, 16, 16, 16),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(32),  # output shape : (batch_size, 32, 16, 16),
            nn.AvgPool2d(kernel_size=2, stride=2),  # output shape : (batch_size, 32, 8, 8),
        )

        self.skipblock3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(32),  # output shape : (batch_size, 32, 8, 8),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(32),  # output shape : (batch_size, 32, 8, 8),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(64),  # output shape : (batch_size, 64, 2, 2),
        )

        self.skipblock4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(64),  # output shape : (batch_size, 64, 2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.BatchNorm2d(64),  # output shape : (batch_size, 64, 2, 2),
        )

        self.fc = nn.Linear(256, latent_size)

        self.l2_normalize = l2_normalize

    def forward(self, x):
        x = self.block1(x)
        x = self.skipblock(x) + x
        x = self.block2(x)
        x = self.skipblock2(x) + x
        x = self.block3(x)
        x = self.skipblock3(x) + x
        x = self.block4(x)
        x = self.skipblock4(x) + x
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)

        x = self.fc(x)

        # Normalize
        if self.l2_normalize:
            x = F.normalize(x, dim=-1)

        return x


def visualize_models():
    # IPT Encoder
    ipt_encoder = IPTEncoderModel()
    inertial, leg, feet = torch.randn(1, 1, 603), torch.randn(1, 1, 900), torch.randn(1, 1, 500)
    out = ipt_encoder(inertial, leg, feet)
    print(out.shape)
    summary(ipt_encoder, [(1, 1, 603), (1, 1, 900), (1, 1, 500)])

    # Inertial Encoder
    inertial_encoder = InertialEncoderModel()
    inertial = torch.randn(1, 1, 603)
    out = inertial_encoder(inertial)
    print(out.shape)
    summary(inertial_encoder, (1, 1, 603))

    # Visual Encoder
    vision_encoder = VisualEncoderModel()
    x = torch.randn(1, 3, 64, 64)
    out = vision_encoder(x)
    print(out.shape)
    summary(vision_encoder, (1, 3, 64, 64))

    # Visual Encoder Tiny
    vision_encoder_tiny = VisualEncoderTinyModel()
    x = torch.randn(1, 3, 64, 64)
    out = vision_encoder_tiny(x)
    print(out.shape)
    summary(vision_encoder_tiny, (1, 3, 64, 64))
