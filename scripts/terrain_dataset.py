import numpy as np
import torch
from torch.utils.data import Dataset


class TerrainDataset(Dataset):
    def __init__(self, patches, dtype=torch.float32):
        # Convert patches to tensor
        patches_array = np.array(patches)
        self.patches = torch.tensor(patches_array, dtype=dtype)  # torch.Size([N_SAMPLES, N_PATCHES, 128, 128, 3])

        # Convert to RGB, 128x128
        self.patches = self.patches.permute(0, 1, 4, 2, 3)  # torch.Size([N_SAMPLES, N_PATCHES, 3, 128, 128])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sample = self.patches[idx]
        num_patches = sample.shape[0]

        # Pick a patch from first half and second half
        patch1_idx = torch.randint(0, num_patches // 2, (1,)).item()
        patch2_idx = torch.randint(num_patches // 2, num_patches, (1,)).item()
        patch1 = sample[patch1_idx]  # torch.Size([3, 128, 128])
        patch2 = sample[patch2_idx]  # torch.Size([3, 128, 128])

        return patch1, patch2
