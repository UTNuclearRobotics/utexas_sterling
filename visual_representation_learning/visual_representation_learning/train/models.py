import torch
import torch.nn as nn
import torch.nn.functional as F


class InertialEncoder(nn.Module):
    """
    Neural Network for encoding inertial data into a lower-dimensional latent space.
    This model uses a series of linear layers followed by activation functions and dropout to process the input data.
    The final output is a latent vector of a specified size.

    Layers:
        - Flatten: Flattens the input tensor to a 1D tensor.
        - Linear: Applies a linear transformation to the incoming data.
        - Mish: Applies the Mish activation function element-wise.
        - Dropout: Randomly zeroes some of the elements of the input tensor with probability p.
    """

    def __init__(self, latent_size=64, p=0.2, l2_normalize=True):
        super(InertialEncoder, self).__init__()

        self.inertial_encoder = nn.Sequential(
            nn.Flatten(),  # Flattens the input tensor to a 1D tensor
            nn.Linear(201 * 3, 128),  # Linear transformation from input size to 128
            nn.Mish(),  # Mish activation function
            nn.Dropout(p),  # Dropout with probability p
            nn.Linear(128, latent_size),  # Linear transformation from 128 to latent_size
        )

        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size),  # Linear transformation within latent_size
            nn.Mish(),  # Mish activation function
            nn.Linear(latent_size, latent_size)  # Linear transformation within latent_size
        )

        self.l2_normalize = l2_normalize

    def forward(self, inertial):
        inertial = self.inertial_encoder(inertial)  # Encode inertial data

        nonvis_features = self.fc(inertial)  # Further process the encoded data

        # Normalize the features if l2_normalize is set to True
        if self.l2_normalize:
            nonvis_features = F.normalize(nonvis_features, dim=-1)

        return nonvis_features


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


class VisualEncoderModel(nn.Module):
    """
    Convolutional Neural Network for encoding images into a lower-dimensional latent space.
    This model uses a series of convolutional layers followed by batch normalization and PReLU activation functions
    to progressively reduce the spatial dimensions of the input image while increasing the depth. The final output
    is a latent vector of a specified size.

    Layers:
        - Conv2d: Applies a 2D convolution over an input signal composed of several input planes.
        - BatchNorm2d: Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension).
        - PReLU: Applies the Parametric Rectified Linear Unit function element-wise.
        - AvgPool2d: Applies a 2D average pooling over an input signal composed of several input planes.
        - Flatten: Flattens the input tensor to a 1D tensor.
        - Linear: Applies a linear transformation to the incoming data.
        - ReLU: Applies the Rectified Linear Unit function element-wise.
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


# class VisualEncoder(nn.Module):
#     def __init__(self, latent_size=64, replace_bn_w_gn=False, l2_normalize=True):
#         super(VisualEncoder, self).__init__()
#         self.encoder = timm.create_model('efficientnet_b0', pretrained=True)
#         # replace batchnorms with groupnorms if needed
#         if replace_bn_w_gn:
#             self.encoder = self.convert_bn_to_gn(self.encoder, features_per_group=16)
#         self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

#         self.fc = nn.Sequential(
#             nn.Linear(1280, latent_size), nn.Mish(),
#             nn.Linear(latent_size, latent_size)
#         )

#         self.l2_normalize = l2_normalize

#     def forward(self, x):
#         vis_features = self.encoder(x)
#         vis_features = self.fc(vis_features)
#         if self.l2_normalize:
#             vis_features = F.normalize(vis_features, dim=-1)
#         return vis_features


#     # replace all batchnorms with groupnorms
#     def convert_bn_to_gn(self, module, features_per_group=16):
#         if isinstance(module, nn.BatchNorm2d):
#             num_groups = max(1, module.num_features // features_per_group)  # Calculate num_groups
#             return nn.GroupNorm(num_groups, module.num_features, eps=module.eps, affine=module.affine)
#         for name, child_module in module.named_children():
#             module.add_module(name, self.convert_bn_to_gn(child_module, features_per_group=features_per_group))
#         return module
