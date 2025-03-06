import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import argparse, os, pickle
from scipy.signal import periodogram, butter, filtfilt
from scipy.spatial.transform import Rotation

IMU_TOPIC_RATE = 100

class TerrainDataset(Dataset):
    def __init__(self, patches, synced_data, transform=None, dtype=torch.float32, incl_orientation=False):
        self.raw_patches = patches  # List of 26,955 patch batches, each with up to 10 patches
        self.dtype = dtype
        self.robot_data = synced_data
        self.imu_data = self.robot_data["imu"]
        self.incl_orientation = incl_orientation
        self.transform = transform
        
        # Convert IMU data to 2D NumPy array with gravity removal, vectorized for speed
        ang_vels = np.array([sample["angular_velocity"] for sample in self.imu_data])
        lin_accs = np.array([sample["linear_acceleration"] for sample in self.imu_data])
        
        if self.incl_orientation:
            orientations = np.array([sample["orientation"] for sample in self.imu_data])
            lin_accs = np.apply_along_axis(lambda x: self.remove_gravity(x, orientations[i]), axis=1, arr=lin_accs)
            self.imu_data = np.concatenate([ang_vels, lin_accs, orientations], axis=1)
        else:
            lin_accs = np.apply_along_axis(lambda x: self.remove_gravity(x, None), axis=1, arr=lin_accs)
            self.imu_data = np.concatenate([ang_vels, lin_accs], axis=1)

        # Precompute PSD min/max with 201-sample windows (2-second at 100 Hz), using ang_vel_x, ang_vel_y, lin_acc_z
        samples_per_window = 201  # 2 seconds at 100 Hz
        # Use sliding windows with stride 1 for overlapping windows, or adjust for non-overlapping
        num_windows = (len(self.imu_data) - samples_per_window + 1) // 1  # Sliding window count, stride 1
        if num_windows <= 0:
            num_windows = 1  # Ensure at least one window if data is short
        
        psd_arrays = []
        for i in range(num_windows):
            start = i
            end = start + samples_per_window
            window = self.imu_data[start:end]
            if len(window) < samples_per_window:
                window = np.pad(window, ((0, samples_per_window - len(window)), (0, 0)), mode='constant')
            imu_subset = window[:, [0, 1, 5]]  # ang_vel_x, ang_vel_y, lin_acc_z
            if not self.incl_orientation:
                imu_subset[:, 2] = self.high_pass_filter(imu_subset[:, 2], fs=IMU_TOPIC_RATE)
            
            psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)[1].flatten()  # (303,)
            psd_arrays.append(psd)
        
        if psd_arrays:  # Ensure psd_arrays is not empty
            self.imu_min = np.min(psd_arrays, axis=0)
            self.imu_max = np.max(psd_arrays, axis=0)
        else:
            # Default to zeros if no windows (unlikely with your data size)
            self.imu_min = np.zeros(303)
            self.imu_max = np.ones(303)

    def remove_gravity(self, linear_acceleration, orientation):
        if orientation is None:
            return linear_acceleration  # No gravity removal if no orientation
        gravity_world = np.array([0, 0, -9.81])
        rot = Rotation.from_quat(orientation)
        gravity_imu = rot.apply(gravity_world)
        return linear_acceleration - gravity_imu

    def high_pass_filter(self, data, fs, cutoff=0.1):
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    def normalize_imu(self, imu_sample):
        if np.allclose(self.imu_max - self.imu_min, 0, atol=1e-8):  # If range is effectively zero
            return imu_sample  # Return unnormalized if range is zero
        return (imu_sample - self.imu_min) / (self.imu_max - self.imu_min + 1e-7)

    def __len__(self):
        return len(self.raw_patches)  # 26,955 batches

    def __getitem__(self, idx):
        # Map patch batch idx to IMU timestep and 2-second window
        imu_timestep_start = idx // 5
        if imu_timestep_start >= len(self.imu_data) - 200:
            raise ValueError(f"Patch batch index {idx} exceeds IMU data length {len(self.imu_data)}")
        
        # Get IMU data for a 2-second history
        start_idx = imu_timestep_start
        end_idx = min(imu_timestep_start + 200, len(self.imu_data))
        imu_segment = self.imu_data[start_idx:end_idx]
        
        if imu_segment.shape[0] < 201:
            imu_segment = np.pad(imu_segment, ((0, 201 - imu_segment.shape[0]), (0, 0)), mode='constant')

        # Extract patches
        patch_batch = self.raw_patches[idx]
        patch_array = np.array(patch_batch)  # Shape: (N, 128, 128, 3) where N <= 10
        
        if len(patch_array.shape) == 0 or patch_array.size == 0:
            patch_array = np.zeros((0, 128, 128, 3), dtype=np.float32)
        if len(patch_array.shape) != 4:
            raise ValueError(f"Unexpected patch array shape for batch {idx}: {patch_array.shape}. Expected (N, 128, 128, 3)")
        
        sample = torch.tensor(patch_array, dtype=self.dtype).permute(0, 3, 1, 2)  # (N, 3, 128, 128)
        num_patches = sample.shape[0]
        
        if num_patches < 2 and num_patches > 0:
            raise ValueError(f"Patch batch {idx} has fewer than 2 patches: {num_patches}")
        
        if num_patches > 0:
            patch1_idx = torch.randint(0, num_patches // 2, (1,)).item()
            patch2_idx = torch.randint(num_patches // 2, num_patches, (1,)).item()
            patch1 = sample[patch1_idx]
            patch2 = sample[patch2_idx]
        else:
            patch1 = torch.zeros((3, 128, 128), dtype=self.dtype)
            patch2 = torch.zeros((3, 128, 128), dtype=self.dtype)
        
        if self.transform and num_patches > 0:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)

        # Extract ang_vel_x (0), ang_vel_y (1), lin_acc_z (5)
        imu_subset = imu_segment[:, [0, 1, 5]]
        if not self.incl_orientation:
            imu_subset[:, 2] = self.high_pass_filter(imu_subset[:, 2], fs=IMU_TOPIC_RATE)
        
        # Compute statistical measures and PSD
        imu_mean = np.mean(imu_subset, axis=0)  # (3,)
        imu_std = np.std(imu_subset, axis=0)    # (3,)
        
        freqs, psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)
        psd_flat = psd.flatten()  # (303,)
        normalized_psd = self.normalize_imu(psd_flat)

        # Combine into imu_sample
        imu_features = np.concatenate([imu_mean, imu_std, normalized_psd])  # (3 + 3 + 303 = 309,)
        imu_sample = torch.tensor(imu_features, dtype=torch.float32).reshape(1, -1)

        return patch1, patch2, imu_sample

def visualize_psd(dataset, idx):
    print(f"Visualizing PSD and patches for idx={idx}")
    print(f"IMU data length: {len(dataset.imu_data)}")
    print(f"Number of patch batches: {len(dataset.raw_patches)}")
    print(f"Number of patch timesteps: {len(dataset.raw_patches) // 5}")  # 5,391 timesteps (26,955 / 5)
    
    # Get patches and IMU sample from dataset (specific batch)
    patch1, patch2, imu_sample = dataset[idx]
    
    # Recompute PSD for visualization
    num_imu_samples = len(dataset.imu_data)  # 8,901
    patch_timestep_start = idx // 5  # Map patch batch idx to patch timestep start (0 to 5,390)
    if patch_timestep_start >= num_imu_samples - 200:  # Ensure we don’t exceed IMU data
        raise ValueError(f"Patch batch index {idx} exceeds IMU data length {num_imu_samples}")
    
    start_idx = patch_timestep_start  # Start at the patch timestep
    end_idx = min(patch_timestep_start + 200, num_imu_samples)  # Extend 2 seconds forward (201 samples if possible)
    imu_segment = dataset.imu_data[start_idx:end_idx]
    
    print(f"Start idx: {start_idx}, End idx: {end_idx}")
    print(f"IMU segment length before padding: {imu_segment.shape[0]}")
    
    if imu_segment.shape[0] < 201:  # Expect 201 samples for 2 seconds at 100 Hz
        padding = np.zeros((201 - imu_segment.shape[0], 
                            6 if not dataset.incl_orientation else 10))
        imu_segment = np.vstack((imu_segment, padding))
        print(f"Padded to: {imu_segment.shape}")
    
    # Extract ang_vel_x (0), ang_vel_y (1), ang_vel_z (2), lin_acc_z (5)
    imu_subset = imu_segment[:, [0, 1, 5]]  # Shape: (201, 4) or less if padded
    
    freqs, psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)
    
    # Normalize the PSD using normalize_imu, matching __getitem__
    psd_flat = psd.flatten()  # (404,) for 4 channels
    normalized_psd = dataset.normalize_imu(psd_flat)  # Use dataset's normalize_imu method
    normalized_psd = normalized_psd.reshape(psd.shape)  # Reshape back to (303, 4) for plotting
    
    # Debug IMU data for all 5 batches of the corresponding patch timestep to verify consistency
    base_psd = normalized_psd  # Store the normalized PSD for comparison
    for batch_offset in range(5):
        patch_batch_idx = patch_timestep_start * 5 + batch_offset  # Map patch timestep to patch batches (0-4 for timestep 0, 5-9 for timestep 1, etc.)
        if patch_batch_idx < len(dataset.raw_patches):
            # Recalculate IMU data for the same timestep to verify
            imu_segment_check = dataset.imu_data[start_idx:end_idx]
            if imu_segment_check.shape[0] < 201:
                padding = np.zeros((201 - imu_segment_check.shape[0], 
                                    6 if not dataset.incl_orientation else 10))
                imu_segment_check = np.vstack((imu_segment_check, padding))
            
            imu_subset_check = imu_segment_check[:, [0, 1, 5]]  # Shape: (201, 4)
            if not dataset.incl_orientation:
                imu_subset_check[:, 2] = dataset.high_pass_filter(imu_subset_check[:, 2], fs=IMU_TOPIC_RATE)
            
            freqs_check, psd_check = periodogram(imu_subset_check, fs=IMU_TOPIC_RATE, axis=0)
            psd_check_flat = psd_check.flatten()  # (404,)
            normalized_psd_check = dataset.normalize_imu(psd_check_flat)  # Normalize
            normalized_psd_check = normalized_psd_check.reshape(psd_check.shape)  # Reshape back to (303, 4)
            
            # Verify normalized PSD is identical (within numerical precision)
            psd_diff = np.max(np.abs(normalized_psd_check - base_psd))
            print(f"Normalized PSD difference for batch {patch_batch_idx} (vs base at idx={patch_timestep_start}): {psd_diff}")
            if psd_diff > 1e-10:  # Threshold for floating-point comparison
                print(f"Warning: Normalized PSD mismatch detected for batch {patch_batch_idx} at idx={patch_timestep_start}")
            
            # Debug patch data
            patch_batch = dataset.raw_patches[patch_batch_idx]
            patch_array = np.array(patch_batch)
            sample = torch.tensor(patch_array, dtype=dataset.dtype).permute(0, 3, 1, 2)
            num_patches = sample.shape[0]
            
            print(f"Patch array shape (batch {patch_batch_idx}): {patch_array.shape}")
            print(f"Number of patches in batch {patch_batch_idx}: {num_patches}")
            print(f"Patch array mean (all patches, batch {patch_batch_idx}): {patch_array.mean(axis=0).mean():.4f}")
            print(f"Patch array std (all patches, batch {patch_batch_idx}): {patch_array.std(axis=0).mean():.4f}")
    
    # Define labels for the 4 channels
    channel_labels = [
        'Angular Velocity X', 'Angular Velocity Y', 'Linear Acceleration Z'
    ]
    
    # Create figure with subplots: 1 for PSD, 2 for patches from the current batch
    fig = plt.figure(figsize=(18, 6))
    
    # Plot normalized PSD
    ax1 = fig.add_subplot(1, 3, 1)
    for i in range(normalized_psd.shape[1]):
        ax1.plot(freqs, normalized_psd[:, i], label=channel_labels[i])
    ax1.set_title(f'Normalized Power Spectral Density at Index {idx} (Patch Timestep {patch_timestep_start})')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Normalized Power/Frequency')
    ax1.legend()
    ax1.grid(True)
    
    # Prepare patches for display (convert from torch tensor to numpy, move channels to last dim)
    patch1_np = patch1.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    patch2_np = patch2.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    
    # Normalize patches to [0, 1] if they aren't already (assuming range is typical for images)
    # Handle empty patches (zeros) gracefully
    if patch1_np.size > 0 and (patch1_np.max() > 1.0 or patch1_np.min() < 0.0):
        patch1_np = (patch1_np - patch1_np.min()) / (patch1_np.max() - patch1_np.min() + 1e-7)
    if patch2_np.size > 0 and (patch2_np.max() > 1.0 or patch2_np.min() < 0.0):
        patch2_np = (patch2_np - patch2_np.min()) / (patch2_np.max() - patch2_np.min() + 1e-7)
    
    # Plot Patch 1 (from the current batch)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(patch1_np if patch1_np.size > 0 else np.zeros((128, 128, 3)), cmap='gray')  # Use gray for empty patches
    ax2.set_title(f'Patch 1 (idx {patch_timestep_start}, batch {idx})')
    ax2.axis('off')
    
    # Plot Patch 2 (from the current batch)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(patch2_np if patch2_np.size > 0 else np.zeros((128, 128, 3)), cmap='gray')  # Use gray for empty patches
    ax3.set_title(f'Patch 2 (idx {patch_timestep_start}, batch {idx})')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sterling Representation Model")
    parser.add_argument("-bag", "-b", type=str, required=True, help="Bag directory with VICReg dataset pickle file inside.")
    args = parser.parse_args()

    bag_path = args.bag
    vicreg_path = os.path.join(bag_path, [f for f in os.listdir(bag_path) if f.endswith("vicreg.pkl")][0])
    synced_path = os.path.join(bag_path, [f for f in os.listdir(bag_path) if f.endswith("_synced.pkl")][0])

    with open(vicreg_path, "rb") as file:
        vicreg_pkl = pickle.load(file)

    with open(synced_path, "rb") as file:
        synced_pkl = pickle.load(file)

    dataset = TerrainDataset(vicreg_pkl, synced_pkl, incl_orientation=False)
    for idx in range(15000, len(dataset), 10):
        visualize_psd(dataset, idx)
