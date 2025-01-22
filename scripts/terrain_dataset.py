import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TerrainDataset(Dataset):
    def __init__(self, patches, transform = None, dtype=torch.float32):
        # Convert each patch to a tensor and ensure they are resized or padded if necessary
        self.patches = [
            torch.tensor(np.array(patch), dtype=dtype).permute(0, 3, 1, 2)  # (N_PATCHES, 3, 128, 128)
            for patch in patches
        ]
        self.transform = transform


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sample = self.patches[idx]
        num_patches = sample.shape[0]

        # Pick a patch from first half and second half
        #patch1_idx = torch.randint(0, num_patches // 2, (1,)).item()
        #patch2_idx = torch.randint(num_patches // 2, num_patches, (1,)).item()

        # Ensure there are enough patches to sample from
        if num_patches < 2:
            raise ValueError(f"Sample {idx} has fewer than 2 patches.")

        patch1_idx = 0
        patch2_idx = torch.randint(1, num_patches, (1,)).item()
        patch1 = sample[patch1_idx]  # torch.Size([3, 128, 128])
        patch2 = sample[patch2_idx]  # torch.Size([3, 128, 128])

        # Apply transforms if available
        if self.transform:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)

        return patch1, patch2
