import torch.nn as nn
import torch
import torch.nn.functional as F

# create a pytorch model for the proprioception data
class ProprioceptionModel(nn.Module):
    def __init__(self, latent_size=64, p=0.05, input_dim=132):  # Default to 606
        super(ProprioceptionModel, self).__init__()
        
        self.inertial_encoder = nn.Sequential(
            nn.Flatten(),  # (batch_size, 1, 606) -> (batch_size, 606)
            nn.Linear(input_dim, 128, bias=False),  # 606 input features
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 32), nn.ReLU(),
        )
        
        #self.leg_encoder = nn.Sequential( # input shape : (batch_size, 1, 900)
        #    nn.Flatten(),
        #    nn.Linear(900, 128, bias=False), nn.ReLU(),
        #    nn.Dropout(p),
        #    nn.Linear(128, 32), nn.ReLU(),
        #)
        
        #self.feet_encoder = nn.Sequential( # input shape : (batch_size, 1, 500)
        #    nn.Flatten(),
        #    nn.Linear(500, 128, bias=False), nn.ReLU(),
        #    nn.Dropout(p),
        #    nn.Linear(128, 32), nn.ReLU(),
        #)
        
        self.fc = nn.Sequential(
            nn.Linear(32, latent_size), nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )
        
    def forward(self, inertial):
        inertial = self.inertial_encoder(inertial)
        #leg = self.leg_encoder(leg)
        #feet = self.feet_encoder(feet)
        
        # Pass through fully connected layer
        #features = self.fc(torch.cat([inertial, leg, feet], dim=1))

        features = self.fc(inertial)
        
        # normalize the features
        features = F.normalize(features, dim=-1)
        
        return features