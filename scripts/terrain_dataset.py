import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TerrainDataset(Dataset):
    def __init__(self, patches, transform = None, dtype=torch.float32, incl_orientation = False):
        # Convert each patch to a tensor and ensure they are resized or padded if necessary
        self.patches = [
            torch.tensor(np.array(patch), dtype=dtype).permute(0, 3, 1, 2)  # (N_PATCHES, 3, 128, 128)
            for patch in patches]
        #self.robot_data = synced_data
        
        # Extract IMU and Odom data
        #self.imu_data = self.robot_data["imu"]
        #self.odom_data = self.robot_data["odom"]  # If needed for further use
        self.incl_orientation = incl_orientation  # If False, remove last 4 IMU columns

        # Ensure number of patches and IMU data match
        #if len(self.patches) != len(self.imu_data):
        #    raise ValueError(f"Mismatch: {len(self.patches)} patches vs {len(self.imu_data)} IMU samples")

        self.transform = transform
        # Compute normalization statistics (global min, max, mean, std)
        #imu_array = np.concatenate(self.imu_data, axis=0)  # Flatten all IMU data for global statistics
        #self.imu_min = np.min(imu_array, axis=0)
        #self.imu_max = np.max(imu_array, axis=0)
        #self.imu_mean = np.mean(imu_array, axis=0)
        #self.imu_std = np.std(imu_array, axis=0) + 1e-7  # Avoid division by zero

    def normalize_imu(self, imu_sample):
        """Apply normalization to a given IMU sample."""
        return (imu_sample - self.imu_min) / (self.imu_max - self.imu_min + 1e-7)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sample = self.patches[idx]
        #imu_sample = self.imu_data[idx]
        num_patches = sample.shape[0]

        # Ensure there are enough patches to sample from
        if num_patches < 2:
            raise ValueError(f"Sample {idx} has fewer than 2 patches.")

        # Pick a patch from first half and second half
        patch1_idx = torch.randint(0, num_patches // 2, (1,)).item()
        patch2_idx = torch.randint(num_patches // 2, num_patches, (1,)).item()
        patch1 = sample[patch1_idx]  # torch.Size([3, 128, 128])
        patch2 = sample[patch2_idx]  # torch.Size([3, 128, 128])

        # Apply transforms if available
        if self.transform:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)

        # Remove last 4 columns from IMU data if orientation is excluded
        #if not self.incl_orientation:
        #    imu_sample = imu_sample[:, :-4]

        # Normalize IMU data
        #imu_sample = self.normalize_imu(imu_sample)

        # Convert IMU data to a PyTorch tensor
        #imu_sample = torch.tensor(imu_sample, dtype=torch.float32)

        return patch1, patch2
