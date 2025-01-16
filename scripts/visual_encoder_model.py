import torch.nn as nn


class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderModel, self).__init__()
        self.rep_size = 64
        self.latent_size = latent_size

        self.model = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.PReLU(),  # torch.Size([batch_size, 8, 64, 64])
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),  # torch.Size([batch_size, 16, 32, 32])
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # torch.Size([batch_size, 32, 16, 16])
            nn.Conv2d(32, self.rep_size, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),  # torch.Size([batch_size, rep_size, 8, 8])
            nn.AdaptiveAvgPool2d((1, 1)), # output shape : (batch_size, 64, 1, 1),  # torch.Size([batch_size, rep_size, 1, 1])
            nn.Flatten(),
            nn.Linear(self.rep_size, latent_size),
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