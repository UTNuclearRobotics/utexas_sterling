import torch
import torch.nn as nn

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial Attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_out, avg_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        return x


class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=256):
        super(VisualEncoderModel, self).__init__()
        self.rep_size = 256  # Increased latent space for better feature representation
        self.latent_size = latent_size

        self.model = nn.Sequential(
            # Input shape: (batch_size, 3, H, W), where 3 = RGB channels
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # Output shape: (batch_size, 32, H/2, W/2)
            CBAM(32),  # Add CBAM for better attention

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),  # Output shape: (batch_size, 64, H/4, W/4)
            CBAM(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),  # Output shape: (batch_size, 128, H/8, W/8)
            CBAM(128),

            nn.Conv2d(128, self.rep_size, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),  # Output shape: (batch_size, 256, H/16, W/16)
            CBAM(self.rep_size),

            # Squeeze-and-Excitation (SE) Block
            nn.AdaptiveAvgPool2d(1),  # Global average pooling (batch_size, channels, 1, 1)
            nn.Conv2d(self.rep_size, self.rep_size // 16, kernel_size=1),  # Squeeze
            nn.ReLU(),
            nn.Conv2d(self.rep_size // 16, self.rep_size, kernel_size=1),  # Excitation
            nn.Sigmoid(),  # Output shape: (batch_size, channels, 1, 1)

            nn.AdaptiveAvgPool2d((1, 1)),  # Output shape: (batch_size, 256, 1, 1)
            nn.Flatten(),  # Flattens to (batch_size, 256)
            nn.Linear(self.rep_size, latent_size),  # Reduces to (batch_size, latent_size)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class CostNet(nn.Module):
    def __init__(self, latent_size=256):  # Updated latent size to match VisualEncoderModel
        super(CostNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2), nn.BatchNorm1d(latent_size // 2), nn.ReLU(),
            nn.Linear(latent_size // 2, 1), nn.Sigmoid(),  # Output a probability
        )

    def forward(self, x):
        return self.fc(x)