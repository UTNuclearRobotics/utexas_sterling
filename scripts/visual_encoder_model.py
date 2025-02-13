import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=128):
        super(VisualEncoderModel, self).__init__()
        self.rep_size = 128  # Keeping representation size fixed
        self.latent_size = latent_size

        self.model = nn.Sequential(
            # Input shape: (batch_size, 3, H, W), where 3 = RGB channels

            # Larger kernel to capture broad textures (e.g., grass vs. pavement)
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),  # Output shape: (batch_size, 16, H/2, W/2)

            # Medium kernel for mid-level features
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # Output shape: (batch_size, 32, H/4, W/4)

            # Standard 3x3 kernel for fine details
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),  # Output shape: (batch_size, 64, H/8, W/8)

            # Final feature extraction before embedding
            nn.Conv2d(64, self.rep_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.rep_size),
            nn.PReLU(),  # Output shape: (batch_size, 128, H/16, W/16)

            SEBlock(self.rep_size),  # Channel Attention
            MultiScaleSpatialAttention(),      # Spatial Attention

            # Feature Fusion: GAP + GMP with Learnable Weight
            FeatureFusion(self.rep_size),

            nn.Flatten(),
            nn.Linear(self.rep_size, latent_size),  # Reduce to latent space
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1) * self.alpha + F.adaptive_max_pool2d(x, 1) * (1 - self.alpha)
        return pooled  # Only storing one output tensor, not two


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        """ Optimized SE Block with 1x1 Convolutions instead of Linear layers """
        super(SEBlock, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
        
        # Replacing nn.Linear with 1x1 Convolution to reduce parameters
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        squeeze_avg = self.global_avg(x)
        squeeze_max = self.global_max(x)
        squeeze = squeeze_avg + squeeze_max  # Combine both pathways

        excitation = self.conv2(self.relu(self.conv1(squeeze)))  # 1x1 conv replaces Linear layers
        return x * self.sigmoid(excitation)
    
class MultiScaleSpatialAttention(nn.Module):
    """ Multi-Scale Spatial Attention for better texture differentiation """
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)

        attn1 = self.conv1(attention)
        attn2 = self.conv2(attention)
        attn3 = self.conv3(attention)

        attention = self.sigmoid(attn1 + attn2 + attn3)  # Sum multi-scale responses
        return x * attention


class CostNet(nn.Module):
    def __init__(self, latent_size=128):  # Updated latent size to match VisualEncoderModel
        super(CostNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2), nn.BatchNorm1d(latent_size // 2), nn.ReLU(),
            nn.Linear(latent_size // 2, 1), nn.Sigmoid(),  # Output a probability
        )

    def forward(self, x):
        return self.fc(x)