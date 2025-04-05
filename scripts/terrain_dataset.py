import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy.signal import periodogram, butter, filtfilt
from scipy.spatial.transform import Rotation
import cv2
import os
import tempfile
from tqdm import tqdm
from multiprocessing import Pool
import gc

IMU_TOPIC_RATE = 20

class TerrainDataset(Dataset):
    def __init__(self, synced_h5_path=None, vicreg_h5_path=None, labeled_dataset=None, transform=None, dtype=torch.float32, incl_orientation=False, train=False, debug=False):
        self.dtype = dtype
        self.transform = transform
        self.incl_orientation = incl_orientation
        self.train = train
        self.debug = debug
        self.rng = np.random.default_rng()  # Cached RNG for faster random choices

        # Map PyTorch dtype to NumPy dtype
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            # Add more mappings if needed
        }
        self.np_dtype = dtype_map.get(self.dtype, np.float32)  # Default to np.float32 if dtype not found

        if labeled_dataset is not None:
            # Labeled mode unchanged from new implementation
            if isinstance(labeled_dataset, str):
                print("Labeled data mode activated (HDF5 file path detected)")
                self.is_labeled = True
                self.hdf5_path = labeled_dataset
                
                with h5py.File(self.hdf5_path, 'r') as h5f:
                    if not all(key in h5f for key in ['patches', 'terrain_labels']):
                        raise ValueError("HDF5 file missing required datasets: 'patches' or 'terrain_labels'")
                    
                    self.patches = np.array(h5f['patches'])
                    self.terrain_labels = [label.decode('utf-8') for label in h5f['terrain_labels']]
                    self.inertial = np.array(h5f['inertial']) if 'inertial' in h5f else None
                    self.preferences = np.array(h5f['preferences'], dtype=np.float32) if 'preferences' in h5f else np.zeros(len(self.patches), dtype=np.float32)
                    self.length = len(self.patches)

                    self.pref_min = self.preferences.min()
                    self.pref_max = self.preferences.max()
                    print(f"Global preferences range: {self.pref_min} to {self.pref_max}")
                    if self.pref_max <= self.pref_min:
                        print("Warning: Preferences max <= min; setting default range 0-1")
                        self.pref_min, self.pref_max = 0.0, 1.0
                
                self.patches = torch.from_numpy(self.patches).to(dtype=self.dtype)
                if self.patches.shape[-3:] != (3, 128, 128):
                    self.patches = self.patches.permute(0, 3, 1, 2)
                
                if self.inertial is not None:
                    self.inertial = torch.from_numpy(self.inertial).to(dtype=self.dtype)
                
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
                self.length = len(self.patches)
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

            # [Previous HDF5 loading and metadata setup unchanged...]
            with h5py.File(synced_h5_path, 'r') as synced_f:
                self.imu_length = len(synced_f['imu'])
                self.odom_length = len(synced_f['odom']) if 'odom' in synced_f else self.imu_length
                if 'odom' in synced_f:
                    odom_group = synced_f['odom']
                    self.odom_positions = np.array([odom_group[str(i)]['pose'][:2] for i in range(self.odom_length)])
                else:
                    self.odom_positions = np.zeros((self.odom_length, 2))

            with h5py.File(vicreg_h5_path, 'r') as vicreg_f:
                self.num_timesteps = len([k for k in vicreg_f.keys() if k.startswith('timestep_')])
                self.global_positions = np.array([vicreg_f[f'timestep_{i}']['global_position'][:] 
                                                for i in range(self.num_timesteps)])
                self.patch_counts = []
                self.timestep_offsets = [0]
                self.shift_keys = []
                self.patches_per_shift = []
                self.shift_idx_map = []
                total_patches = 0
                for t in range(self.num_timesteps):
                    timestep_group = vicreg_f[f'timestep_{t}']
                    shift_keys_t = [k for k in timestep_group.keys() if k.startswith('shift_')]
                    self.shift_keys.append(shift_keys_t)
                    patches_per_shift_t = [len([pk for pk in timestep_group[sk].keys() if pk.startswith('patch_')])
                                        for sk in shift_keys_t]
                    self.patches_per_shift.append(patches_per_shift_t)
                    num_patches = sum(patches_per_shift_t)
                    self.patch_counts.append(num_patches)
                    total_patches += num_patches
                    self.timestep_offsets.append(total_patches)
                    shift_idx_map_t = []
                    for i, count in enumerate(patches_per_shift_t):
                        shift_idx_map_t.extend([i] * count)
                    self.shift_idx_map.append(shift_idx_map_t)

                self.idx_to_timestep = []
                for t in range(self.num_timesteps):
                    self.idx_to_timestep.extend([t] * self.patch_counts[t])

            self.imu_min = None
            self.imu_max = None
            self.precompute_features()

            self.zero_patch = torch.zeros((3, 128, 128), dtype=self.dtype)
            self.patch1_file = None
            self.patch2_file = None
            self.patch1_path = None
            self.patch2_path = None
            self.vicreg_h5 = None

            # Common parameters for both modes
            pairs_per_shift = 5  # Number of pairs to precompute per shift for variability

            if self.train:
                print("Precomputing multiple patch pairs per shift for training mode...")
                self.vicreg_h5 = h5py.File(self.vicreg_h5_path, 'r')
                
                shifts_per_timestep = 5  # Fixed number of shifts for training
                total_shifts = self.num_timesteps * shifts_per_timestep
                self.length = total_shifts  # One index per shift, pairs selected dynamically
                
                # Temporary files
                self.patch1_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                self.patch2_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                self.patch1_path = self.patch1_file.name
                self.patch2_path = self.patch2_file.name
                
                # Memory-mapped arrays for all pairs
                total_pairs = total_shifts * pairs_per_shift
                self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, 128, 128))
                self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, 128, 128))
                
                self.pair_offsets = []  # Start index of pairs for each shift
                self.idx_to_timestep = []
                patch_idx = 0
                
                for t in tqdm(range(self.num_timesteps), desc="Caching patches (training)"):
                    timestep_group = self.vicreg_h5[f'timestep_{t}']
                    shift_keys = self.shift_keys[t]
                    num_shifts = len(shift_keys)
                    selected_shifts = shift_keys[:min(shifts_per_timestep, num_shifts)]
                    
                    if num_shifts < shifts_per_timestep:
                        print(f"Warning: Timestep {t} has only {num_shifts} shifts, less than {shifts_per_timestep}")
                    
                    self.pair_offsets.extend([patch_idx + i * pairs_per_shift for i in range(len(selected_shifts))])
                    
                    for shift_key in selected_shifts:
                        shift_group = timestep_group[shift_key]
                        all_patches = [shift_group[patch_key][:] for patch_key in shift_group.keys() if patch_key.startswith('patch_')]
                        num_patches = len(all_patches)
                        
                        for _ in range(pairs_per_shift):
                            if num_patches > 1:
                                half_point = num_patches // 2
                                patch1_idx = self.rng.choice(half_point)
                                patch2_idx = self.rng.choice(num_patches - half_point) + half_point
                                patch1 = np.transpose(all_patches[patch1_idx], (2, 0, 1))
                                patch2 = np.transpose(all_patches[patch2_idx], (2, 0, 1))
                            elif num_patches == 1:
                                patch1 = np.transpose(all_patches[0], (2, 0, 1))
                                patch2 = np.zeros((3, 128, 128), dtype=self.np_dtype)
                            else:
                                patch1 = np.zeros((3, 128, 128), dtype=self.np_dtype)
                                patch2 = np.zeros((3, 128, 128), dtype=self.np_dtype)
                            
                            self.patch1_data[patch_idx] = patch1
                            self.patch2_data[patch_idx] = patch2
                            patch_idx += 1
                        self.idx_to_timestep.append(t)
                    
                    # Pad with zero pairs if fewer shifts
                    if num_shifts < shifts_per_timestep:
                        remaining_shifts = shifts_per_timestep - num_shifts
                        self.pair_offsets.extend([patch_idx + i * pairs_per_shift for i in range(remaining_shifts)])
                        for _ in range(remaining_shifts * pairs_per_shift):
                            self.patch1_data[patch_idx] = np.zeros((3, 128, 128), dtype=self.np_dtype)
                            self.patch2_data[patch_idx] = np.zeros((3, 128, 128), dtype=self.np_dtype)
                            patch_idx += 1
                        self.idx_to_timestep.extend([t] * remaining_shifts)
                
                self.patch1_data.flush()
                self.patch2_data.flush()
                del self.patch1_data
                del self.patch2_data
                self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, 128, 128))
                self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, 128, 128))
                
                self.patch1_tensor = torch.from_numpy(self.patch1_data).to(dtype=self.dtype)
                self.patch2_tensor = torch.from_numpy(self.patch2_data).to(dtype=self.dtype)
                self.pairs_per_shift = pairs_per_shift
                self.vicreg_h5.close()
                self.vicreg_h5 = None
                print(f"Training mode: Cached {total_pairs} patch pairs ({pairs_per_shift} per shift). Dataset length: {self.length} shifts")
            else:
                print("Precomputing multiple patch pairs per shift for non-training mode...")
                self.vicreg_h5 = h5py.File(self.vicreg_h5_path, 'r')
                
                total_shifts = sum(len(shift_keys) for shift_keys in self.shift_keys)
                self.length = total_shifts  # One index per shift, pairs selected dynamically
                
                # Temporary files
                self.patch1_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                self.patch2_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                self.patch1_path = self.patch1_file.name
                self.patch2_path = self.patch2_file.name
                
                # Memory-mapped arrays
                total_pairs = total_shifts * pairs_per_shift
                self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, 128, 128))
                self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, 128, 128))
                
                self.pair_offsets = []
                self.idx_to_timestep = []
                patch_idx = 0
                
                for t in tqdm(range(self.num_timesteps), desc="Caching patches (non-training)"):
                    timestep_group = self.vicreg_h5[f'timestep_{t}']
                    shift_keys = self.shift_keys[t]
                    
                    self.pair_offsets.extend([patch_idx + i * pairs_per_shift for i in range(len(shift_keys))])
                    
                    for shift_key in shift_keys:
                        shift_group = timestep_group[shift_key]
                        all_patches = [shift_group[patch_key][:] for patch_key in shift_group.keys() if patch_key.startswith('patch_')]
                        num_patches = len(all_patches)
                        
                        for _ in range(pairs_per_shift):
                            if num_patches > 1:
                                half_point = num_patches // 2
                                patch1_idx = self.rng.choice(half_point)
                                patch2_idx = self.rng.choice(num_patches - half_point) + half_point
                                patch1 = np.transpose(all_patches[patch1_idx], (2, 0, 1))
                                patch2 = np.transpose(all_patches[patch2_idx], (2, 0, 1))
                            elif num_patches == 1:
                                patch1 = np.transpose(all_patches[0], (2, 0, 1))
                                patch2 = np.zeros((3, 128, 128), dtype=self.np_dtype)
                            else:
                                patch1 = np.zeros((3, 128, 128), dtype=self.np_dtype)
                                patch2 = np.zeros((3, 128, 128), dtype=self.np_dtype)
                            
                            self.patch1_data[patch_idx] = patch1
                            self.patch2_data[patch_idx] = patch2
                            patch_idx += 1
                        self.idx_to_timestep.append(t)
                
                self.patch1_data.flush()
                self.patch2_data.flush()
                del self.patch1_data
                del self.patch2_data
                self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, 128, 128))
                self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, 128, 128))
                
                self.patch1_tensor = torch.from_numpy(self.patch1_data).to(dtype=self.dtype)
                self.patch2_tensor = torch.from_numpy(self.patch2_data).to(dtype=self.dtype)
                self.pairs_per_shift = pairs_per_shift
                self.vicreg_h5.close()
                self.vicreg_h5 = None
                if self.length == 0:
                    raise ValueError("No valid shifts found in the dataset.")
                print(f"Non-training mode: Cached {total_pairs} patch pairs ({pairs_per_shift} per shift). Dataset length: {self.length} shifts")

    def process_chunk(self, chunk_args):
        chunk_idx, start_idx, end_idx = chunk_args
        chunk_size_local = end_idx - start_idx
        chunk_indices = np.arange(start_idx, end_idx)
        
        # Preallocate results
        features = np.zeros((chunk_size_local, 138), dtype=np.float32)
        matching_odom_chunk = [None] * chunk_size_local

        samples_per_window = IMU_TOPIC_RATE * 2  # 40 samples

        # Open HDF5 file within the worker process
        with h5py.File(self.synced_h5_path, 'r') as synced_f:
            imu_group = synced_f['imu']
            odom_group = synced_f['odom']

            # Define IMU/odom range for the chunk
            min_odom_idx = max(0, start_idx)
            max_odom_idx = min(end_idx + samples_per_window, self.odom_length)
            
            # Load IMU and odom data for the chunk's window
            imu_data = np.array([
                np.concatenate([
                    imu_group[str(i)]['angular_velocity'][:],
                    imu_group[str(i)]['linear_acceleration'][:],
                    imu_group[str(i)]['orientation'][:] if self.incl_orientation else np.zeros(4)
                ]) for i in range(min_odom_idx, max_odom_idx)
            ])
            odom_poses = np.array([odom_group[str(i)]['pose'][:3] for i in range(min_odom_idx, max_odom_idx)])
            odom_positions = np.array([odom_group[str(i)]['pose'][:2] for i in range(self.odom_length)])

            # Vectorized processing for the chunk
            global_pos_xy = self.global_positions[chunk_indices, :2]  # Shape: (chunk_size, 2)
            odom_start_indices = np.maximum(0, chunk_indices[:, None])  # Shape: (chunk_size, 1)
            odom_end_indices = np.minimum(odom_start_indices + samples_per_window, len(odom_positions))  # Shape: (chunk_size, 1)
            
            # Extract odom subsets for all timesteps in chunk
            odom_indices = np.arange(len(odom_positions))
            odom_mask = (odom_indices >= odom_start_indices) & (odom_indices < odom_end_indices)  # Shape: (chunk_size, odom_length)
            odom_subset = odom_positions[None, :, :] * odom_mask[:, :, None]  # Shape: (chunk_size, odom_length, 2)
            distances = np.linalg.norm(odom_subset - global_pos_xy[:, None, :], axis=2)  # Shape: (chunk_size, odom_length)
            valid_mask = distances <= 1.0  # Shape: (chunk_size, odom_length)

            # Precompute filter coefficients if needed
            if not self.incl_orientation:
                nyquist = IMU_TOPIC_RATE / 2
                cutoff = 0.1
                normal_cutoff = cutoff / nyquist
                b, a = butter(1, normal_cutoff, btype='high', analog=False)

            for i, idx in enumerate(chunk_indices):
                start = max(0, idx)
                end = min(start + samples_per_window, len(odom_positions))
                valid_indices = odom_indices[start:end][valid_mask[i, start:end]]
                valid_count = len(valid_indices)

                if valid_count > 0:
                    imu_subset = np.zeros((valid_count, 6))
                    matching_odom = []
                    
                    # Vectorized IMU data extraction and processing
                    imu_valid = imu_data[valid_indices - min_odom_idx]  # Adjust indices relative to loaded data
                    ang_vels = imu_valid[:, :3]  # Shape: (valid_count, 3)
                    lin_accs = imu_valid[:, 3:6]  # Shape: (valid_count, 3)
                    
                    if self.incl_orientation:
                        orientations = imu_valid[:, 6:]  # Shape: (valid_count, 4)
                        rot = Rotation.from_quat(orientations)
                        gravity_world = np.array([0, 0, -9.81])
                        gravity_imu = rot.apply(gravity_world)  # Shape: (valid_count, 3)
                        lin_accs = lin_accs - gravity_imu
                    else:
                        # Simplified gravity removal without orientation
                        lin_accs = lin_accs  # Adjust if a default gravity vector is needed

                    imu_subset = np.hstack([ang_vels, lin_accs])  # Shape: (valid_count, 6)
                    
                    if not self.incl_orientation:
                        # Batch high-pass filtering
                        for k in range(3, 6):
                            imu_subset[:, k] = filtfilt(b, a, imu_subset[:, k])

                    # Pad or truncate to fixed window size
                    imu_data_padded = np.zeros((samples_per_window, 6))
                    if valid_count < samples_per_window:
                        summary_sample = np.mean(imu_subset, axis=0)
                        imu_data_padded[:valid_count] = imu_subset
                        imu_data_padded[valid_count:] = summary_sample
                        matching_odom = [(j, odom_poses[j - min_odom_idx][:3], distances[i, j - start]) 
                                        for j in valid_indices] + \
                                       [(start, np.zeros(3), 0.0)] * (samples_per_window - valid_count)
                    else:
                        imu_data_padded = imu_subset[:samples_per_window]
                        matching_odom = [(j, odom_poses[j - min_odom_idx][:3], distances[i, j - start]) 
                                        for j in valid_indices[:samples_per_window]]
                else:
                    imu_data_padded = np.zeros((samples_per_window, 6))
                    matching_odom = [(start, np.zeros(3), 0.0)] * samples_per_window

                # Compute features
                mean = np.mean(imu_data_padded, axis=0)  # Shape: (6,)
                std = np.std(imu_data_padded, axis=0)  # Shape: (6,)
                freqs, psd = periodogram(imu_data_padded, fs=IMU_TOPIC_RATE, axis=0)  # psd: (freq_bins, 6)
                psd_flat = psd.flatten()  # Shape: (126,)
                features[i] = np.concatenate([mean, std, psd_flat])  # Shape: (138,)
                features[i] = np.nan_to_num(features[i], nan=0.0, posinf=0.0, neginf=0.0)
                matching_odom_chunk[i] = matching_odom

        return list(zip(chunk_indices, features, matching_odom_chunk))

    def precompute_features(self):
        print("Precomputing IMU features and odometry data...")
        self.imu_features = np.zeros((self.num_timesteps, 138), dtype=np.float32)
        self.matching_odom_data = [None] * self.num_timesteps

        chunk_size = 1000
        num_chunks = (self.num_timesteps + chunk_size - 1) // chunk_size

        # Parallel processing with minimal data transfer
        with Pool(processes=os.cpu_count()) as pool:
            chunk_args = [(i, i * chunk_size, min((i + 1) * chunk_size, self.num_timesteps)) 
                          for i in range(num_chunks)]
            results = list(tqdm(pool.imap_unordered(self.process_chunk, chunk_args), 
                                total=num_chunks, desc="Processing chunks"))

        # Aggregate results
        for chunk_result in results:
            for idx, feat, odom in chunk_result:
                self.imu_features[idx] = feat
                self.matching_odom_data[idx] = odom

        # Final normalization
        self.imu_features = torch.from_numpy(self.imu_features).to(dtype=self.dtype)
        self.imu_min = torch.min(self.imu_features, dim=0)[0]
        self.imu_max = torch.max(self.imu_features, dim=0)[0]
        self.imu_features = self.normalize_imu(self.imu_features)
        print(f"IMU features and odometry data precomputed. Features shape: {self.imu_features.shape}")

    def __del__(self):
        if self.train and hasattr(self, 'patch1_file') and self.patch1_file is not None:
            try:
                if self.patch1_data is not None:
                    self.patch1_data.flush()
                self.patch1_file.close()
                os.unlink(self.patch1_path)
            except Exception as e:
                print(f"Warning: Failed to clean up patch1 cache: {e}")
        if self.train and hasattr(self, 'patch2_file') and self.patch2_file is not None:
            try:
                if self.patch2_data is not None:
                    self.patch2_data.flush()
                self.patch2_file.close()
                os.unlink(self.patch2_path)
            except Exception as e:
                print(f"Warning: Failed to clean up patch2 cache: {e}")
        # Add cleanup for persistent HDF5 handle in non-training mode
        if hasattr(self, 'vicreg_h5') and self.vicreg_h5 is not None:
            self.vicreg_h5.close()

    def get_scaled_preferences(self, preferences):
        return ((preferences - self.pref_min) / (self.pref_max - self.pref_min)) * 100.0

    def remove_gravity(self, linear_acceleration, orientation):
        if orientation is None or not self.incl_orientation:
            return linear_acceleration
        gravity_world = np.array([0, 0, -9.81])
        rot = Rotation.from_quat(orientation)
        gravity_imu = rot.apply(gravity_world)
        return linear_acceleration - gravity_imu

    def high_pass_filter(self, data, fs, cutoff=0.1):
        if np.isscalar(data) or data.size == 1:
            return data
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    def normalize_imu(self, imu_sample):
        if self.imu_min is None or self.imu_max is None or torch.allclose(self.imu_max - self.imu_min, torch.tensor(0.0, dtype=self.dtype), atol=1e-8):
            return imu_sample
        return (imu_sample - self.imu_min) / (self.imu_max - self.imu_min + 1e-7)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_labeled:
            patch = self.patches[idx]
            inertial = self.inertial[idx] if self.inertial is not None else None
            terrain_label = self.terrain_labels[idx]
            preference = self.preferences[idx]
            return patch, inertial, terrain_label, preference
        else:
            if idx >= self.length or idx < 0:
                raise IndexError(f"Index {idx} out of range for dataset length {self.length}")
            
            timestep_idx = self.idx_to_timestep[idx]
            imu_sample = self.imu_features[timestep_idx]
            
            # Select a random pair from the precomputed set for this shift
            pair_start = self.pair_offsets[idx]
            pair_idx = pair_start + self.rng.integers(0, self.pairs_per_shift)  # Randomly pick one of the pairs
            patch1 = self.patch1_tensor[pair_idx]
            patch2 = self.patch2_tensor[pair_idx]
            
            if self.debug:
                assert 0 <= timestep_idx < self.num_timesteps, f"Invalid timestep_idx: {timestep_idx}"
                global_pos = self.global_positions[timestep_idx]
                matching_odom = self.matching_odom_data[timestep_idx]
                return patch1, patch2, imu_sample, global_pos, matching_odom
            
            return patch1, patch2, imu_sample

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        if not hasattr(dataset, 'patch1_tensor'):
            dataset.patch1_tensor = torch.from_numpy(np.memmap(dataset.patch1_path, dtype=dataset.np_dtype, mode='r', shape=(dataset.length, 3, 128, 128))).to(dtype=dataset.dtype)
            dataset.patch2_tensor = torch.from_numpy(np.memmap(dataset.patch2_path, dtype=dataset.np_dtype, mode='r', shape=(dataset.length, 3, 128, 128))).to(dtype=dataset.dtype)

def process_timestep_batch_wrapper(args):
    dataset, sub_chunk, imu_data_dict, odom_data_dict, global_positions, odom_positions = args  # Unpack 6 values
    return dataset.process_timestep_batch((sub_chunk, imu_data_dict, odom_data_dict, global_positions, odom_positions))

"""
if __name__ == "__main__":
    synced_h5_path = "bags/agh_courtyard_2/agh_courtyard_2_synced.h5"
    vicreg_h5_path = "bags/agh_courtyard_2/agh_courtyard_2_vicreg.h5"

    dataset = TerrainDataset(
        synced_h5_path=synced_h5_path,
        vicreg_h5_path=vicreg_h5_path,
        dtype=torch.float32,
        incl_orientation=True,
        train=False,
        debug=True
    )

    # Define a range of patch indices to test
    patch_start = 1500
    patch_end = 1520  # Inclusive, so this tests 1500 to 1520
    print(f"Testing patch indices from {patch_start} to {patch_end} (total: {patch_end - patch_start + 1} patches)")
    
    for idx in range(patch_start, patch_end + 1):
        if idx >= len(dataset):
            print(f"Index {idx} exceeds dataset length ({len(dataset)}), stopping.")
            break
        
        patch1, patch2, imu_sample, global_pos, matching_odom = dataset[idx]
        
        print(f"\nPatch Sample {idx}:")
        print(f"Global Position of Patch: {global_pos} (x, y, z in meters)")
        print(f"Patch1 Shape: {patch1.shape}, Patch2 Shape: {patch2.shape}")
        print(f"IMU Sample Shape: {imu_sample.shape}")
        
        if matching_odom and matching_odom[0][1].any():
            print(f"Matching Odometry Segment (within 1 foot in x and y, {len(matching_odom)} samples):")
            start_idx = matching_odom[0][0]
            end_idx = matching_odom[-1][0]
            print(f"  Segment Range: Index {start_idx} to {end_idx}")
            for odom_idx, odom_pos, xy_distance in matching_odom:
                print(f"    Index {odom_idx}: Position {odom_pos} (meters), XY Distance: {xy_distance:.4f} meters")
        else:
            print("No matching odometry segment found within 1 foot in x and y.")


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
