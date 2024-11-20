import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import pickle
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

'''
1 0 1
0 1 0
1 0 1

0.5 1 2

0.5 0   0.5
0   0.5 0
0.5 0   0.5

    1   0   1
    0   1   0
    1   0   1

        2   0   2
        0   2   0
        2   0   2

0.5 1   2.5 
0   0.5 1
0.5 1   2.5

'''

matplotlib.use('TkAgg')

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, '../datasets/')
dataset_pkl = "nrg_ahg_courtyard.pkl"
dataset_file = dataset_dir + dataset_pkl

with open(dataset_file, 'rb') as file:
    data_pkl = pickle.load(file)

patches = data_pkl['patches']
print(len(patches[0]))

class TerrainDataset(Dataset):
    def __init__(self, patches):
        #Convert to tensor
        #Convert to BLUE 64x64, R, G <-- Order not important
        # self.patches = patches
        #Shape here is N_SAMPLES, N_PATCHES, 3, 64, 64
        #DROP FIRST 10 FRAMES DURING PRE-PROCESS
        #Specify float16 or float32 in the constructor QOL
        print('')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        #Return a tensor of shape [2, 3, 64, 64] First dimension is the 2 patches
        return 0
        #return 2 random patches from the same sample first half latter half


class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderModel, self).__init__()
        self.rep_size = 64
        self.latent_size = latent_size
        self.model = nn.Sequential(
           
            # torch.nn.Conv2d(
            #   in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(3, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.PReLU(),  # Output shape : (batch_size, 8, 31, 31)

            nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(),  # Output shape : (batch_size, 16, 15, 15)

            nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),  # Output shape : (batch_size, 32, 7, 7)

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.PReLU(),  # Output shape : (batch_size, 64, 3, 3)

            nn.AvgPool2d(kernel_size=3),  # Output shape : (batch_size, 64, 1, 1)   <-- self.rep_size
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

class SterlingRepresentation(nn.Module):
    def __init__(self):
        self.latent_size = 64
        self.visual_encoder = VisualEncoderModel(self.latent_size)
        self.projector = nn.Sequential(
            nn.Linear(self.visual_encoder.rep_size, self.latent_size),
            nn.PReLU(), nn.Linear(self.latent_size, self.latent_size)
        )

    def forward(self, x):
        #Shape should be [batch size, 2, 3, 64, 64]
        patch1 = x[:, 0:1, :, :, :]
        patch2 = x[:, 1:2, :, :, :]
        # Encode visual patches
        v_encoded_1 = self.visual_encoder(patch1)
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2)
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        # Project encoded representations to latent space
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)

        return zv1, zv2, v_encoded_1, v_encoded_2


'''
# for index, patch in enumerate(patches):
#     plt.imshow(patch[0])
#     plt.axis('off')  # Turn off the axes for a cleaner image
#     print(index)
#     plt.pause(0.001)   # Pause for 0.2 seconds

# print(data_pkl.keys())
'''