import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy.signal import periodogram, butter, filtfilt
from scipy.spatial.transform import Rotation
import cv2
import os
import tempfile

IMU_TOPIC_RATE = 20

class TerrainDataset(Dataset):
    def __init__(self, synced_h5_path=None, vicreg_h5_path=None, labeled_dataset=None, transform=None, dtype=torch.float32, incl_orientation=False, train=False):
        self.dtype = dtype
        self.transform = transform
        self.incl_orientation = incl_orientation
        self.train = train

        if labeled_dataset is not None:
            if isinstance(labeled_dataset, str):  # Handle HDF5 file path
                print("Labeled data mode activated (HDF5 file path detected)")
                self.is_labeled = True
                self.hdf5_path = labeled_dataset
                
                # Load the full dataset into memory
                with h5py.File(self.hdf5_path, 'r') as h5f:
                    if not all(key in h5f for key in ['patches', 'terrain_labels']):
                        raise ValueError("HDF5 file missing required datasets: 'patches' or 'terrain_labels'")
                    
                    # Load all data into memory
                    self.patches = np.array(h5f['patches'])  # Shape: (N, H, W, C) or similar
                    self.terrain_labels = [label.decode('utf-8') for label in h5f['terrain_labels']]
                    self.inertial = np.array(h5f['inertial']) if 'inertial' in h5f else None
                    self.preferences = np.array(h5f['preferences'], dtype=np.float32) if 'preferences' in h5f else np.zeros(len(self.patches), dtype=np.float32)
                    self.length = len(self.patches)

                    # Always compute pref_min and pref_max
                    self.pref_min = self.preferences.min()
                    self.pref_max = self.preferences.max()
                    print(f"Global preferences range: {self.pref_min} to {self.pref_max}")
                    if self.pref_max <= self.pref_min:
                        print("Warning: Preferences max <= min; setting default range 0-1")
                        self.pref_min, self.pref_max = 0.0, 1.0  # Fallback range
                
                # Convert patches to torch tensor and adjust dimensions if needed
                self.patches = torch.from_numpy(self.patches).to(dtype=self.dtype)
                if self.patches.shape[-3:] != (3, 128, 128):
                    self.patches = self.patches.permute(0, 3, 1, 2)  # Adjust to (N, C, H, W)
                
                # Convert inertial to tensor if it exists
                if self.inertial is not None:
                    self.inertial = torch.from_numpy(self.inertial).to(dtype=self.dtype)
                
                # Convert preferences to tensor
                self.preferences = torch.from_numpy(self.preferences).to(dtype=self.dtype)
                
                print(f"Loaded full dataset into memory: {self.length} samples")
                
            elif isinstance(labeled_dataset, Dataset):
                print("Detected Dataset subclass, attempting to load as dataset object")
                self.patches = getattr(labeled_dataset, 'patches', None)
                self.terrain_labels = getattr(labeled_dataset, 'terrain_labels', None)
                self.inertial = getattr(labeled_dataset, 'inertial', None)
                self.preferences = getattr(labeled_dataset, 'preferences', None)
                if self.patches is None or self.terrain_labels is None:
                    raise ValueError("Loaded dataset object missing required 'patches' or 'terrain_labels'")
                self.is_labeled = True
                print("Labeled data mode activated (dataset object detected)")
                self.length = len(self.patches)
                # Precompute min/max for preferences if available
                if self.preferences is not None:
                    self.pref_min = self.preferences.min()
                    self.pref_max = self.preferences.max()
                    print(f"Global preferences range: {self.pref_min} to {self.pref_max}")
                    if self.pref_max <= self.pref_min:
                        raise ValueError("Preferences max <= min; cannot scale data meaningfully.")
                    
            elif isinstance(labeled_dataset, (list, tuple)) and all(isinstance(sample, dict) for sample in labeled_dataset):
                print("Labeled data mode activated (list of dicts detected)")
                self.patches = [sample['patch'] for sample in labeled_dataset]
                self.inertial = [sample['inertial'] for sample in labeled_dataset]
                self.terrain_labels = [sample['terrain_label'] for sample in labeled_dataset]
                self.preferences = [sample['preference'] for sample in labeled_dataset]
                self.is_labeled = True
                self.length = len(self.patches)
                # Convert preferences to tensor and precompute min/max
                self.preferences = torch.tensor(self.preferences, dtype=self.dtype)
                self.pref_min = self.preferences.min()
                self.pref_max = self.preferences.max()
                print(f"Global preferences range: {self.pref_min} to {self.pref_max}")
                if self.pref_max <= self.pref_min:
                    raise ValueError("Preferences max <= min; cannot scale data meaningfully.")
            else:
                raise ValueError(f"Unsupported labeled_dataset type or structure: {type(labeled_dataset)}")
        else:
            print("Unlabeled data mode activated")
            if synced_h5_path is None or vicreg_h5_path is None:
                raise ValueError("Must provide synced_h5_path and vicreg_h5_path")
            if not os.path.exists(synced_h5_path) or not os.path.exists(vicreg_h5_path):
                raise FileNotFoundError(f"Files not found: {synced_h5_path}, {vicreg_h5_path}")

            self.synced_h5_path = synced_h5_path
            self.vicreg_h5_path = vicreg_h5_path
            self.is_labeled = False

            with h5py.File(synced_h5_path, 'r') as synced_f:
                imu_group = synced_f['imu']
                self.imu_length = len(imu_group)
                print("Loading IMU data...")
                imu_data = np.array([
                    np.concatenate([
                        imu_group[str(i)]['angular_velocity'][:],
                        imu_group[str(i)]['linear_acceleration'][:],
                        imu_group[str(i)]['orientation'][:] if self.incl_orientation else np.zeros(4)
                    ])
                    for i in range(self.imu_length)
                ])

            with h5py.File(vicreg_h5_path, 'r') as vicreg_f:
                self.vicreg_length = len(vicreg_f)

            if self.train:
                print("Precomputing patch pairs to disk cache...")
                # Create temporary files for memory-mapped arrays
                self.patch1_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                self.patch2_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                
                # Initialize memory-mapped arrays
                self.patch1_data = np.memmap(
                    self.patch1_file.name, dtype=np.float32, mode='w+', 
                    shape=(self.vicreg_length, 128, 128, 3)
                )
                self.patch2_data = np.memmap(
                    self.patch2_file.name, dtype=np.float32, mode='w+', 
                    shape=(self.vicreg_length, 128, 128, 3)
                )
                
                # Populate memory-mapped arrays
                with h5py.File(vicreg_h5_path, 'r') as vicreg_f:
                    for i in range(self.vicreg_length):
                        timestep_group = vicreg_f[f'timestep_{i}']
                        num_patches = len(timestep_group)
                        if num_patches > 1:
                            patch1_idx = i % (num_patches // 2)
                            patch2_idx = num_patches // 2 + (i % (num_patches - num_patches // 2))
                            self.patch1_data[i] = timestep_group[f'patch_{patch1_idx}'][:]
                            self.patch2_data[i] = timestep_group[f'patch_{patch2_idx}'][:]
                        else:
                            self.patch1_data[i] = np.zeros((128, 128, 3), dtype=np.float32)
                            self.patch2_data[i] = np.zeros((128, 128, 3), dtype=np.float32)
                
                # Flush to disk
                self.patch1_data.flush()
                self.patch2_data.flush()
                print(f"Patch pairs cached to disk. Shape: {(self.vicreg_length, 128, 128, 3)}")
            else:
                self.patch1_data = None
                self.patch2_data = None
                self.patch1_file = None
                self.patch2_file = None

            samples_per_window = IMU_TOPIC_RATE * 2
            self.num_timesteps = self.vicreg_length // 5
            all_psd_features = []
            for timestep in range(self.num_timesteps):
                start_idx = timestep
                end_idx = min(start_idx + samples_per_window, self.imu_length)
                imu_window = imu_data[start_idx:end_idx]
                if len(imu_window) < samples_per_window:
                    imu_window = np.pad(imu_window, ((0, samples_per_window - len(imu_window)), (0, 0)), 'constant')
                ang_vels = imu_window[:, :3]
                lin_accs = imu_window[:, 3:6]
                if self.incl_orientation:
                    orientations = imu_window[:, 6:]
                    lin_accs = np.array([self.remove_gravity(lin_accs[j], orientations[j]) for j in range(len(lin_accs))])
                else:
                    lin_accs = np.apply_along_axis(lambda x: self.remove_gravity(x, None), 1, arr=lin_accs)
                imu_subset = np.hstack([ang_vels, lin_accs])[:, [0, 1, 2, 3, 4, 5]]
                if not self.incl_orientation:
                    for j in range(3, 6):
                        imu_subset[:, j] = self.high_pass_filter(imu_subset[:, j], fs=IMU_TOPIC_RATE)
                psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)[1].flatten()
                std = np.std(imu_subset, axis=0)
                features = np.concatenate([std, psd])
                all_psd_features.append(features)

            psd_features = torch.from_numpy(np.array(all_psd_features)).to(dtype=self.dtype)
            self.imu_min = torch.min(psd_features, dim=0)[0]
            self.imu_max = torch.max(psd_features, dim=0)[0]
            self.psd_features = self.normalize_imu(psd_features).contiguous()
            self.length = self.vicreg_length
            print(f"Dataset initialized with {self.length} samples")

    def __del__(self):
        """Clean up memory-mapped files when dataset is destroyed."""
        if self.train and hasattr(self, 'patch1_file') and self.patch1_file is not None:
            try:
                self.patch1_data.flush()
                self.patch1_file.close()
                os.unlink(self.patch1_file.name)
            except Exception as e:
                print(f"Warning: Failed to clean up patch1 cache: {e}")
        if self.train and hasattr(self, 'patch2_file') and self.patch2_file is not None:
            try:
                self.patch2_data.flush()
                self.patch2_file.close()
                os.unlink(self.patch2_file.name)
            except Exception as e:
                print(f"Warning: Failed to clean up patch2 cache: {e}")

    def _compute_imu_normalization_params(self):
        samples_per_window = IMU_TOPIC_RATE * 2
        num_windows = max(1, (self.imu_length - samples_per_window + 1) // 1)
        chunk_size = 1000

        with h5py.File(self.synced_h5_path, 'r') as f:
            imu_group = f['imu']
            all_psd_features = []

            for start in range(0, num_windows, chunk_size):
                end = min(start + chunk_size, num_windows)
                chunk_features = []

                for i in range(start, end):
                    window_start = i
                    window_end = min(window_start + samples_per_window, self.imu_length)
                    pad_size = samples_per_window - (window_end - window_start) if window_end < window_start + samples_per_window else 0

                    # Load and concatenate IMU data for this window
                    imu_window = np.array([
                        np.concatenate([
                            imu_group[str(j)]['angular_velocity'][:],
                            imu_group[str(j)]['linear_acceleration'][:],
                            imu_group[str(j)]['orientation'][:] if self.incl_orientation else np.zeros(4)
                        ])
                        for j in range(window_start, window_end)
                    ])  # Shape: (window_size, 10) or (window_size, 6) if not incl_orientation

                    if pad_size > 0:
                        imu_window = np.pad(imu_window, ((0, pad_size), (0, 0)), mode='constant')

                    # Process IMU data
                    ang_vels = imu_window[:, :3]
                    lin_accs = imu_window[:, 3:6]
                    if self.incl_orientation:
                        orientations = imu_window[:, 6:]
                        lin_accs = np.array([self.remove_gravity(lin_accs[j], orientations[j]) 
                                           for j in range(len(lin_accs))])
                    else:
                        lin_accs = np.apply_along_axis(lambda x: self.remove_gravity(x, None), axis=1, arr=lin_accs)

                    imu_subset = np.hstack([ang_vels, lin_accs])[:, [0, 1, 2, 3, 4, 5]]
                    if not self.incl_orientation:
                        for j in range(3, 6):
                            imu_subset[:, j] = self.high_pass_filter(imu_subset[:, j], fs=IMU_TOPIC_RATE)

                    psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)[1].flatten()
                    std = np.std(imu_subset, axis=0)
                    features = np.concatenate([std, psd])
                    chunk_features.append(features)

                all_psd_features.extend(chunk_features)

            self.psd_features = np.array(all_psd_features)
            self.imu_min = np.min(self.psd_features, axis=0)
            self.imu_max = np.max(self.psd_features, axis=0)
    
    def get_scaled_preferences(self, preferences):
        """Helper method to scale preferences using precomputed min/max."""
        return ((preferences - self.pref_min) / (self.pref_max - self.pref_min)) * 255.0

    def remove_gravity(self, linear_acceleration, orientation):
        if orientation is None or not self.incl_orientation:
            return linear_acceleration
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
        """Normalize IMU features using precomputed min/max."""
        if torch.allclose(self.imu_max - self.imu_min, torch.tensor(0.0, dtype=self.dtype), atol=1e-8):
            return imu_sample  # Avoid division by zero
        return (imu_sample - self.imu_min) / (self.imu_max - self.imu_min + 1e-7)  # Add epsilon for stability

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_labeled:
            patch = self.patches[idx].clone().detach()
            inertial = self.inertial[idx].clone().detach() if self.inertial is not None else None
            if inertial is not None and inertial.dim() == 1:  # If inertial is (48,)
                inertial = inertial.unsqueeze(0)  # Add channel dim: (1, 48)
            terrain_label = self.terrain_labels[idx]
            preference = self.preferences[idx].clone().detach().unsqueeze(0)

            if self.transform:
                patch = self.transform(patch)

            return patch, inertial, terrain_label, preference
        else:
            if self.train and self.patch1_data is not None:
                patch1 = torch.from_numpy(self.patch1_data[idx]).permute(2, 0, 1).to(dtype=self.dtype)
                patch2 = torch.from_numpy(self.patch2_data[idx]).permute(2, 0, 1).to(dtype=self.dtype)
            else:
                with h5py.File(self.vicreg_h5_path, 'r') as vicreg_f:
                    timestep_group = vicreg_f[f'timestep_{idx}']
                    num_patches = len(timestep_group)
                    if num_patches > 1:
                        patch1_idx = idx % (num_patches // 2)
                        patch2_idx = num_patches // 2 + (idx % (num_patches - num_patches // 2))
                        patch1 = torch.from_numpy(timestep_group[f'patch_{patch1_idx}'][:]).permute(2, 0, 1).to(self.dtype)
                        patch2 = torch.from_numpy(timestep_group[f'patch_{patch2_idx}'][:]).permute(2, 0, 1).to(self.dtype)
                    else:
                        patch1 = torch.zeros((3, 128, 128), dtype=self.dtype)
                        patch2 = torch.zeros((3, 128, 128), dtype=self.dtype)
            imu_sample = self.psd_features[min(idx // 5, len(self.psd_features) - 1)]
            return patch1, patch2, imu_sample
""" 
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
    if patch_timestep_start >= num_imu_samples - IMU_TOPIC_RATE*2:  # Ensure we don’t exceed IMU data
        raise ValueError(f"Patch batch index {idx} exceeds IMU data length {num_imu_samples}")
    
    start_idx = patch_timestep_start  # Start at the patch timestep
    end_idx = min(patch_timestep_start + 1+IMU_TOPIC_RATE*2, num_imu_samples)  # Extend 2 seconds forward (201 samples if possible)
    imu_segment = dataset.imu_data[start_idx:end_idx]
    
    print(f"Start idx: {start_idx}, End idx: {end_idx}")
    print(f"IMU segment length before padding: {imu_segment.shape[0]}")
    
    if imu_segment.shape[0] < (1+IMU_TOPIC_RATE*2):  # Expect 201 samples for 2 seconds at 100 Hz
        padding = np.zeros(((1+IMU_TOPIC_RATE*2) - imu_segment.shape[0], 
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
            if imu_segment_check.shape[0] < (1+IMU_TOPIC_RATE*2):
                padding = np.zeros(((1+IMU_TOPIC_RATE*2) - imu_segment_check.shape[0], 
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

def visualize_pca_terrain(dataset, n_clusters=3, n_components=2, save_dir=None):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(dataset.psd_features)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result)

    # Updated feature names for 612 features (std + all PSDs)
    feature_names = ['std_ang_vel_x', 'std_ang_vel_y', 'std_ang_vel_z',
                     'std_lin_acc_x', 'std_lin_acc_y', 'std_lin_acc_z'] + \
                    [f'psd_ang_vel_x_{i}' for i in range(101)] + \
                    [f'psd_ang_vel_y_{i}' for i in range(101)] + \
                    [f'psd_ang_vel_z_{i}' for i in range(101)] + \
                    [f'psd_lin_acc_x_{i}' for i in range(101)] + \
                    [f'psd_lin_acc_y_{i}' for i in range(101)] + \
                    [f'psd_lin_acc_z_{i}' for i in range(101)]

    loadings = pca.components_.T  # Shape: (612, n_components)

    # 1. Variance Ratio Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, color='skyblue', 
            tick_label=[f'PC{i+1}' for i in range(n_components)])
    plt.title('Explained Variance Ratio by Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.ylim(0, max(pca.explained_variance_ratio_) * 1.2)
    for i, v in enumerate(pca.explained_variance_ratio_):
        plt.text(i + 1, v + 0.005, f'{v:.2%}', ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'variance_ratio.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Top Features per Component
    for i, pc in enumerate([f'PC{j+1}' for j in range(n_components)]):
        top_indices = np.argsort(np.abs(loadings[:, i]))[-10:][::-1]
        top_features = [feature_names[idx] for idx in top_indices]
        top_loadings = [loadings[idx, i] for idx in top_indices]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_features, top_loadings, color='salmon' if i == 0 else 'lightgreen')
        plt.title(f'Top 10 Features for {pc} ({pca.explained_variance_ratio_[i]:.2%} Variance)')
        plt.xlabel('Loading Value')
        plt.gca().invert_yaxis()
        for bar, val in zip(bars, top_loadings):
            plt.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', 
                     ha='left' if val < 0 else 'right')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'top_features_{pc}.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nTop features for {pc} ({pca.explained_variance_ratio_[i]:.2%} variance):")
        for idx in top_indices:
            print(f"{feature_names[idx]}: {loadings[idx, i]:.4f}")

    # 3. Top 10 Most Descriptive Features Overall
    weighted_loadings = loadings * pca.explained_variance_ratio_
    total_contribution = np.sum(np.abs(weighted_loadings), axis=1)
    top_indices_total = np.argsort(total_contribution)[-10:][::-1]
    top_features_total = [feature_names[idx] for idx in top_indices_total]
    top_contributions = [total_contribution[idx] for idx in top_indices_total]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_features_total, top_contributions, color='lightcoral')
    plt.title('Top 10 Most Descriptive IMU Features Overall')
    plt.xlabel('Total Weighted Contribution')
    plt.gca().invert_yaxis()
    for bar, val in zip(bars, top_contributions):
        plt.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', ha='right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'top_features_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("\nTop 10 most descriptive IMU features overall:")
    for idx in top_indices_total:
        print(f"{feature_names[idx]}: {total_contribution[idx]:.4f}")

    # 4. 2D Scatter Plot (if applicable)
    if n_components >= 2:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"PCA of IMU Features ({n_components} Components)")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'pca_scatter.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return cluster_labels

def inspect_cluster_samples(dataset, cluster_labels, num_samples=3):
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        print(f"\nInspecting Cluster {cluster}:")
        cluster_indices = np.where(cluster_labels == cluster)[0]
        sample_indices = np.random.choice(cluster_indices, min(num_samples, len(cluster_indices)), replace=False)
        
        for idx in sample_indices:
            patch_idx = idx * 5
            if patch_idx >= len(dataset):
                patch_idx = len(dataset) - 1
            
            patch1, patch2, imu_sample = dataset[patch_idx]
            print(f"Patch batch {patch_idx}:")
            print(f"IMU std (ang_vel_x, y, lin_acc_y, z): {imu_sample[:, :4].numpy()}")
            
            patch1_np = patch1.permute(1, 2, 0).numpy()
            patch2_np = patch2.permute(1, 2, 0).numpy()
            if patch1_np.max() > 0 and (patch1_np.dtype == np.float32 or patch1_np.dtype == np.float64):
                if patch1_np.max() > 1.0 or patch1_np.min() < 0.0:
                    patch1_np = (patch1_np - patch1_np.min()) / (patch1_np.max() - patch1_np.min() + 1e-7)
            if patch2_np.max() > 0 and (patch2_np.dtype == np.float32 or patch2_np.dtype == np.float64):
                if patch2_np.max() > 1.0 or patch2_np.min() < 0.0:
                    patch2_np = (patch2_np - patch2_np.min()) / (patch2_np.max() - patch2_np.min() + 1e-7)

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(patch1_np if patch1_np.max() > 0 else np.zeros((128, 128, 3)))
            plt.title(f"Patch1 (Cluster {cluster})")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(patch2_np if patch2_np.max() > 0 else np.zeros((128, 128, 3)))
            plt.title(f"Patch2 (Cluster {cluster})")
            plt.axis('off')
            plt.show()


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
    save_dir = "scripts/IMU_PCA"
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cluster_labels = visualize_pca_terrain(dataset, n_clusters=5, n_components=2, save_dir=save_dir)

    # Inspect samples from each cluster
    inspect_cluster_samples(dataset, cluster_labels, num_samples=10)

    #for idx in range(15000, len(dataset), 10):
    #    visualize_psd(dataset, idx)

"""
