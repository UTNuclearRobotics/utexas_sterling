import torch.nn as nn

class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderModel, self).__init__()
        self.rep_size = 64
        self.latent_size = latent_size

        self.model = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(3, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.PReLU(),  # torch.Size([batch_size, 8, 31, 31])
            nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),  # torch.Size([batch_size, 16, 15, 15])
            nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # torch.Size([batch_size, 32, 7, 7])
            nn.Conv2d(32, self.rep_size, kernel_size=3, stride=2),
            nn.PReLU(),  # torch.Size([batch_size, rep_size, 3, 3])
            nn.AvgPool2d(kernel_size=3),  # torch.Size([batch_size, rep_size, 1, 1])
            nn.Flatten(),
            nn.Linear(self.rep_size, latent_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)