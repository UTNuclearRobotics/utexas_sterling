from torch import nn


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
