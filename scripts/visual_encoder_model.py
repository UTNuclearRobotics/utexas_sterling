import torch.nn as nn


class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=128):
        super(VisualEncoderModel, self).__init__()
        self.rep_size = 128
        self.latent_size = latent_size

        self.model = nn.Sequential(
            # Input shape: (batch_size, 3, H, W), where 3 = RGB channels
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),  # Output shape: (batch_size, 16, H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # Output shape: (batch_size, 32, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),  # Output shape: (batch_size, 64, H/8, W/8)
            nn.Conv2d(64, self.rep_size, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),  # Output shape: (batch_size, 64, H/16, W/16)
            nn.AdaptiveAvgPool2d((1, 1)), # Output shape: (batch_size, 64, 1, 1)
            nn.Flatten(), # Flattens to (batch_size, 64)
            nn.Linear(self.rep_size, latent_size), # Reduces to (batch_size, latent_size)
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

class CostNet(nn.Module):
    def __init__(self, latent_size=64):
        super(CostNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size//2), nn.BatchNorm1d(latent_size//2), nn.ReLU(),
            nn.Linear(latent_size//2, 1), nn.Sigmoid(), #nn.ReLU(), #nn.Softplus(), 
        )
        
    def forward(self, x):
        return self.fc(x)