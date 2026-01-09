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
import glob
# import gc
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

IMU_TOPIC_RATE = 20

class TerrainDataset(Dataset):
    def __init__(self, synced_h5_path=None, vicreg_h5_path=None, labeled_dataset=None, transform=None, 
                dtype=torch.float32, incl_orientation=False, train=False, debug=False, patch_size=128):
        self.transform = transform
        self.incl_orientation = incl_orientation
        self.train = train
        self.debug = debug
        self.patch_size = patch_size
        self.rng = np.random.default_rng()
        self.imu_stats_path = None
        self.imu_mean = None
        self.imu_std = None
        self.imu_min   = None
        self.imu_max   = None
        self.is_labeled = None
        self.dtype = dtype

        # Map PyTorch dtype to NumPy dtype
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }
        self.np_dtype = dtype_map.get(self.dtype, np.float32)
        try:
            if labeled_dataset is not None:
                print("Entering labeled data mode")
                self.is_labeled = True
                if isinstance(labeled_dataset, str):
                    print("Labeled data mode activated (HDF5 file path)")
                    self.hdf5_path = labeled_dataset
                    # Derive imu_stats_dir one directory above the bag folder
                    imu_stats_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(labeled_dataset))))
                    imu_stats_files = glob.glob(os.path.join(imu_stats_dir, "*_imu_stats.pt"))
                    if not imu_stats_files:
                        raise FileNotFoundError(f"No *_imu_stats.pt file found in {imu_stats_dir}")
                    if len(imu_stats_files) > 1:
                        print(f"Warning: Multiple IMU stats files found in {imu_stats_dir}: {imu_stats_files}. Using the first one.")
                    self.imu_stats_path = imu_stats_files[0]
                    print(f"IMU stats path for reference: {self.imu_stats_path}")

                    with h5py.File(self.hdf5_path, 'r') as h5f:
                        if not all(key in h5f for key in ['patches', 'terrain_labels']):
                            raise ValueError("HDF5 file missing required datasets: 'patches' or 'terrain_labels'")
                        
                        self.patches = np.array(h5f['patches'])
                        self.terrain_labels = [label.decode('utf-8') for label in h5f['terrain_labels']]
                        self.inertial = np.array(h5f['inertial'], dtype=self.np_dtype) if 'inertial' in h5f else np.zeros((len(self.patches), 115), dtype=self.np_dtype)
                        self.preferences = np.array(h5f['preferences'], dtype=np.float32) if 'preferences' in h5f else np.zeros(len(self.patches), dtype=np.float32)
                        self.length = len(self.patches)

                        self.pref_min = self.preferences.min()
                        self.pref_max = self.preferences.max()
                        print(f"Global preferences range: {self.pref_min} to {self.pref_max}")
                        if self.pref_max <= self.pref_min:
                            print("Warning: Preferences max <= min; setting default range 0-1")
                            self.pref_min, self.pref_max = 0.0, 1.0
                    
                    self.patches = torch.from_numpy(self.patches).to(dtype=self.dtype)
                    if self.patches.shape[-3:] != (3, self.patch_size, self.patch_size):
                        self.patches = self.patches.permute(0, 3, 1, 2)
                    
                    if self.inertial is not None:
                        self.inertial = torch.from_numpy(self.inertial).to(dtype=self.dtype)
                        # Load IMU stats for reference but do not normalize
                        if self.imu_stats_path and os.path.exists(self.imu_stats_path):
                            print(f"Loading IMU statistics from {self.imu_stats_path} for reference")
                            try:
                                stats = torch.load(self.imu_stats_path)
                                self.imu_mean = stats['mean'].to(dtype=self.dtype)
                                self.imu_std = stats['std'].to(dtype=self.dtype)
                                if self.imu_mean.shape != (115,) or self.imu_std.shape != (115,):
                                    raise ValueError(f"Loaded IMU stats have incorrect shape: mean {self.imu_mean.shape}, std {self.imu_std.shape}")
                                print(f"Successfully loaded IMU statistics (not applied, as inertial data is pre-normalized)")
                            except Exception as e:
                                print(f"Error loading IMU stats: {e}. Proceeding without IMU stats.")
                                self.imu_mean = None
                                self.imu_std = None
                    
                    self.preferences = torch.from_numpy(self.preferences).to(dtype=self.dtype)
                    print(f"Loaded HDF5 dataset: {self.length} samples")

                elif isinstance(labeled_dataset, list):
                    print("Labeled data mode activated (list of dictionaries)")
                    if not labeled_dataset:
                        raise ValueError("labeled_dataset list is empty")
                    
                    self.patches = []
                    self.inertial = []
                    self.terrain_labels = []
                    self.preferences = []
                    
                    for item in labeled_dataset:
                        if not isinstance(item, dict):
                            raise ValueError(f"Expected dict in labeled_dataset, got {type(item)}")
                        if not all(key in item for key in ["patch", "inertial", "terrain_label", "preference"]):
                            raise ValueError(f"Item missing required keys: {item.keys()}")
                        
                        self.patches.append(item["patch"])
                        self.inertial.append(item["inertial"])
                        self.terrain_labels.append(item["terrain_label"])
                        self.preferences.append(item["preference"])
                    
                    self.patches = torch.stack([p.to(dtype=self.dtype) for p in self.patches])
                    self.inertial = torch.stack([i.to(dtype=self.dtype) for i in self.inertial]) if self.inertial else None
                    self.preferences = torch.tensor(self.preferences, dtype=self.dtype)
                    self.length = len(self.patches)

                    if self.patches.shape[1:] != (3, self.patch_size, self.patch_size):
                        raise ValueError(f"Invalid patch shape: {self.patches.shape[1:]}; expected (3, {self.patch_size}, {self.patch_size})")
                    
                    self.pref_min = self.preferences.min().item()
                    self.pref_max = self.preferences.max().item()
                    print(f"Global preferences range: {self.pref_min} to {self.pref_max}")
                    if self.pref_max <= self.pref_min:
                        print("Warning: Preferences max <= min; setting default range 0-1")
                        self.pref_min, self.pref_max = 0.0, 1.0
                    
                    if self.inertial is not None and self.imu_stats_path is None:
                        print("Warning: No IMU stats path provided for labeled dataset (list mode). Assuming inertial data is pre-normalized.")
                    elif self.inertial is not None and self.imu_stats_path and os.path.exists(self.imu_stats_path):
                        print(f"Loading IMU statistics from {self.imu_stats_path} for reference")
                        try:
                            stats = torch.load(self.imu_stats_path)
                            self.imu_mean = stats['mean'].to(dtype=self.dtype)
                            self.imu_std = stats['std'].to(dtype=self.dtype)
                            if self.imu_mean.shape != (115,) or self.imu_std.shape != (115,):
                                raise ValueError(f"Loaded IMU stats have incorrect shape: mean {self.imu_mean.shape}, std {self.imu_std.shape}")
                            print(f"Successfully loaded IMU statistics (not applied, as inertial data is pre-normalized)")
                        except Exception as e:
                            print(f"Error loading IMU stats: {e}. Proceeding without IMU stats.")
                            self.imu_mean = None
                            self.imu_std = None
                    elif self.inertial is not None:
                        print(f"Warning: IMU stats file {self.imu_stats_path} not found. Assuming inertial data is pre-normalized.")
                    
                    print(f"Loaded list dataset: {self.length} samples")
                print(f"Labeled dataset initialized with length: {self.length}")
            else:
                print("Entering unlabeled data mode")
                if synced_h5_path is None or vicreg_h5_path is None:
                    raise ValueError(f"Must provide synced_h5_path and vicreg_h5_path, got synced_h5_path={synced_h5_path}, vicreg_h5_path={vicreg_h5_path}")
                
                # Convert to absolute paths
                synced_h5_path = os.path.abspath(synced_h5_path)
                vicreg_h5_path = os.path.abspath(vicreg_h5_path)

                if not os.path.exists(synced_h5_path):
                    raise FileNotFoundError(f"Synced HDF5 file not found at: {synced_h5_path}")
                if not os.path.exists(vicreg_h5_path):
                    raise FileNotFoundError(f"VICReg HDF5 file not found at: {vicreg_h5_path}")

                self.synced_h5_path = synced_h5_path
                self.vicreg_h5_path = vicreg_h5_path

                # Derive imu_stats_dir
                imu_stats_dir = os.path.dirname(os.path.dirname(os.path.abspath(synced_h5_path)))
                imu_stats_files = glob.glob(os.path.join(imu_stats_dir, "*_imu_stats.pt"))
                if imu_stats_files:
                    self.imu_stats_path = imu_stats_files[0]
                    print(f"IMU stats path: {self.imu_stats_path}")
                else:
                    bag_name = os.path.basename(os.path.dirname(os.path.abspath(synced_h5_path)))
                    self.imu_stats_path = os.path.join(imu_stats_dir, f"{bag_name}_imu_stats.pt")
                    print(f"IMU stats will be saved to: {self.imu_stats_path}")
                self.is_labeled = False

                # Load synced_h5_path
                try:
                    print("Loading synced_h5_path")
                    with h5py.File(synced_h5_path, 'r') as synced_f:
                        self.imu_length = len(synced_f['imu'])
                        self.odom_length = len(synced_f['odom']) if 'odom' in synced_f else self.imu_length
                        print(f"IMU data: {self.imu_length} samples, Odometry data: {self.odom_length} samples")
                        if 'odom' in synced_f:
                            odom_group = synced_f['odom']
                            self.odom_positions = np.array([odom_group[str(i)]['pose'][:2] for i in range(self.odom_length)])
                            odom_valid = np.sum(np.all(self.odom_positions != 0, axis=1) & ~np.isnan(self.odom_positions).any(axis=1))
                            print(f"Odometry validity: {odom_valid}/{self.odom_length} positions are non-zero and valid "
                                f"({100 * odom_valid / self.odom_length:.2f}%)")
                        else:
                            self.odom_positions = np.zeros((self.odom_length, 2))
                            print("Warning: No odometry data found. All odometry positions set to zero.")
                    print("Synced HDF5 loaded successfully")
                except Exception as e:
                    print(f"Error loading synced_h5_path: {e}")
                    raise

                # Load vicreg_h5_path
                try:
                    print("Loading vicreg_h5_path")
                    with h5py.File(vicreg_h5_path, 'r') as vicreg_f:
                        self.num_timesteps = len([k for k in vicreg_f.keys() if k.startswith('timestep_')])
                        print(f"Number of timesteps: {self.num_timesteps}")
                        if self.num_timesteps == 0:
                            raise ValueError(f"No timesteps found in {vicreg_h5_path}")
                        self.global_positions = np.array([vicreg_f[f'timestep_{i}']['global_position'][:] 
                                                        for i in range(self.num_timesteps)])
                        global_valid = np.sum(np.all(self.global_positions != 0, axis=1) & ~np.isnan(self.global_positions).any(axis=1))
                        print(f"Global positions validity: {global_valid}/{self.num_timesteps} positions are non-zero and valid "
                            f"({100 * global_valid / self.num_timesteps:.2f}%)")
                        if global_valid < 0.5 * self.num_timesteps:
                            print("Warning: Sparse or invalid global positions detected.")
                        global_min = np.min(self.global_positions, axis=0) if global_valid > 0 else [0, 0]
                        global_max = np.max(self.global_positions, axis=0) if global_valid > 0 else [0, 0]
                        print(f"Global position range: X [{global_min[0]:.2f}, {global_max[0]:.2f}], Y [{global_min[1]:.2f}, {global_max[1]:.2f}]")

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
                        print(f"Total patches: {total_patches}")
                    print("VICReg HDF5 loaded successfully")
                except Exception as e:
                    print(f"Error loading vicreg_h5_path: {e}")
                    raise

                self.imu_min = None
                self.imu_max = None
                try:
                    self.precompute_features()
                except Exception as e:
                    print(f"Error in precompute_features: {e}")
                    raise

                # Patch caching
                self.zero_patch = torch.zeros((3, self.patch_size, self.patch_size), dtype=self.dtype)
                self.patch1_file = None
                self.patch2_file = None
                self.patch1_path = None
                self.patch2_path = None
                self.vicreg_h5 = None
                pairs_per_shift = 10

                if self.train:
                    print("Precomputing patches for training mode")
                    try:
                        self.vicreg_h5 = h5py.File(self.vicreg_h5_path, 'r')
                        
                        shifts_per_timestep = 5
                        total_shifts = self.num_timesteps * shifts_per_timestep
                        self.length = total_shifts
                        
                        self.patch1_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                        self.patch2_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                        self.patch1_path = self.patch1_file.name
                        self.patch2_path = self.patch2_file.name
                        
                        total_pairs = total_shifts * pairs_per_shift
                        self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        
                        self.pair_offsets = []
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
                                        patch2 = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                    else:
                                        patch1 = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                        patch2 = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                    
                                    self.patch1_data[patch_idx] = patch1
                                    self.patch2_data[patch_idx] = patch2
                                    patch_idx += 1
                                self.idx_to_timestep.append(t)
                            
                            if num_shifts < shifts_per_timestep:
                                remaining_shifts = shifts_per_timestep - num_shifts
                                self.pair_offsets.extend([patch_idx + i * pairs_per_shift for i in range(remaining_shifts)])
                                for _ in range(remaining_shifts * pairs_per_shift):
                                    self.patch1_data[patch_idx] = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                    self.patch2_data[patch_idx] = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                    patch_idx += 1
                                self.idx_to_timestep.extend([t] * remaining_shifts)
                        
                        self.patch1_data.flush()
                        self.patch2_data.flush()
                        del self.patch1_data
                        del self.patch2_data
                        self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        
                        self.patch1_tensor = torch.from_numpy(self.patch1_data).to(dtype=self.dtype)
                        self.patch2_tensor = torch.from_numpy(self.patch2_data).to(dtype=self.dtype)
                        self.pairs_per_shift = pairs_per_shift
                        self.vicreg_h5.close()
                        self.vicreg_h5 = None
                        print(f"Training mode: Cached {total_pairs} patch pairs ({pairs_per_shift} per shift). Dataset length: {self.length} shifts")
                    except Exception as e:
                        print(f"Error in training patch caching: {e}")
                        raise
                    print(f"Training mode completed with length: {self.length}")
                else:
                    print("Precomputing patches for non-training mode")
                    try:
                        self.vicreg_h5 = h5py.File(self.vicreg_h5_path, 'r')
                        
                        total_shifts = sum(len(shift_keys) for shift_keys in self.shift_keys)
                        self.length = total_shifts
                        
                        self.patch1_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                        self.patch2_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                        self.patch1_path = self.patch1_file.name
                        self.patch2_path = self.patch2_file.name
                        
                        total_pairs = total_shifts * pairs_per_shift
                        self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='w+', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        
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
                                        patch2 = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                    else:
                                        patch1 = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                        patch2 = np.zeros((3, self.patch_size, self.patch_size), dtype=self.np_dtype)
                                    
                                    self.patch1_data[patch_idx] = patch1
                                    self.patch2_data[patch_idx] = patch2
                                    patch_idx += 1
                                self.idx_to_timestep.append(t)
                        
                        self.patch1_data.flush()
                        self.patch2_data.flush()
                        del self.patch1_data
                        del self.patch2_data
                        self.patch1_data = np.memmap(self.patch1_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        self.patch2_data = np.memmap(self.patch2_path, dtype=self.np_dtype, mode='r', shape=(total_pairs, 3, self.patch_size, self.patch_size))
                        
                        self.patch1_tensor = torch.from_numpy(self.patch1_data).to(dtype=self.dtype)
                        self.patch2_tensor = torch.from_numpy(self.patch2_data).to(dtype=self.dtype)
                        self.pairs_per_shift = pairs_per_shift
                        self.vicreg_h5.close()
                        self.vicreg_h5 = None
                        if self.length == 0:
                            raise ValueError(f"No valid shifts found in {vicreg_h5_path}")
                    except Exception as e:
                        print(f"Error in non-training patch caching: {e}")
                        raise

        except Exception as e:
            print(f"Error in TerrainDataset.__init__: {e}")
            raise

        print(f"TerrainDataset initialized with length: {self.length}")

    def process_chunk(self, chunk_args):
        chunk_idx, start_idx, end_idx = chunk_args
        chunk_size_local = end_idx - start_idx
        chunk_indices = np.arange(start_idx, end_idx)
        
        features = np.zeros((chunk_size_local, 115), dtype=np.float32)
        matching_odom_chunk = [None] * chunk_size_local
        samples_per_window = IMU_TOPIC_RATE * 2
        match_count = 0

        with h5py.File(self.synced_h5_path, 'r') as synced_f:
            imu_group = synced_f['imu']
            odom_group = synced_f['odom']
            
            min_odom_idx = max(0, start_idx)
            max_odom_idx = min(end_idx + samples_per_window, self.odom_length)
            
            imu_data = np.array([
                np.concatenate([
                    imu_group[str(i)]['angular_velocity'][:2],
                    imu_group[str(i)]['linear_acceleration'][:],
                    imu_group[str(i)]['orientation'][:] if self.incl_orientation else np.zeros(4)
                ]) for i in range(min_odom_idx, max_odom_idx)
            ])
            odom_positions = np.array([odom_group[str(i)]['pose'][:2] for i in range(self.odom_length)])

            global_pos_xy = self.global_positions[chunk_indices, :2]
            
            distances = np.full((chunk_size_local, self.odom_length), np.inf)
            for i, idx in enumerate(chunk_indices):
                start = max(0, idx)
                end = min(start + samples_per_window, self.odom_length)
                distances[i, start:end] = np.linalg.norm(
                    odom_positions[start:end] - global_pos_xy[i], axis=1
                )
            valid_mask = distances <= 2.0

            if not self.incl_orientation:
                nyquist = IMU_TOPIC_RATE / 2
                cutoff = 0.1
                normal_cutoff = cutoff / nyquist
                b, a = butter(1, normal_cutoff, btype='high', analog=False)

            for i, idx in enumerate(chunk_indices):
                start = max(0, idx)
                end = min(start + samples_per_window, len(odom_positions))
                valid_indices = np.where(valid_mask[i, start:end])[0] + start
                valid_count = len(valid_indices)

                if valid_count > 0:
                    match_count += 1
                    imu_subset = np.zeros((valid_count, 5))
                    matching_odom = []
                    
                    imu_valid = imu_data[valid_indices - min_odom_idx]
                    ang_vels = imu_valid[:, :2]
                    lin_accs = imu_valid[:, 2:5]
                    
                    if self.incl_orientation:
                        orientations = imu_valid[:, 5:]
                        rot = Rotation.from_quat(orientations)
                        gravity_world = np.array([0, 0, -9.81])
                        gravity_imu = rot.apply(gravity_world)
                        lin_accs = lin_accs - gravity_imu
                    else:
                        gravity = np.array([0, 0, -9.81])
                        lin_accs = lin_accs - gravity

                    imu_subset = np.hstack([ang_vels, lin_accs])
                    
                    if not self.incl_orientation:
                        min_length = 7
                        for k in range(2, 5):
                            if valid_count < min_length:
                                imu_subset[:, k] = imu_subset[:, k]
                            else:
                                imu_subset[:, k] = filtfilt(b, a, imu_subset[:, k])

                    imu_data_padded = np.zeros((samples_per_window, 5))
                    if valid_count < samples_per_window:
                        imu_data_padded[:valid_count] = imu_subset
                        imu_data_padded[valid_count:] = imu_subset[-1] if valid_count > 0 else 0
                    else:
                        imu_data_padded = imu_subset[:samples_per_window]
                else:
                    imu_data_padded = np.zeros((samples_per_window, 5))
                    matching_odom = [(start, np.zeros(3), 0.0)] * samples_per_window

                mean = np.mean(imu_data_padded, axis=0)
                std = np.std(imu_data_padded, axis=0)
                freqs, psd = periodogram(imu_data_padded, fs=IMU_TOPIC_RATE, axis=0)
                psd_flat = psd.flatten()
                features[i] = np.concatenate([mean, std, psd_flat])
                features[i] = np.nan_to_num(features[i], nan=0.0, posinf=0.0, neginf=0.0)
                matching_odom_chunk[i] = matching_odom

            print(f"Chunk {chunk_idx}: {match_count}/{chunk_size_local} timesteps have valid odometry matches")
            return list(zip(chunk_indices, features, matching_odom_chunk)), match_count

    def precompute_features(self):
        print("Precomputing IMU features and odometry data...")
        self.imu_features = np.zeros((self.num_timesteps, 115), dtype=np.float32)
        self.matching_odom_data = [None] * self.num_timesteps

        chunk_size = 1000
        num_chunks = (self.num_timesteps + chunk_size - 1) // chunk_size
        total_matches = 0

        with Pool(processes=os.cpu_count()) as pool:
            chunk_args = [(i, i * chunk_size, min((i + 1) * chunk_size, self.num_timesteps)) 
                          for i in range(num_chunks)]
            results = list(tqdm(pool.imap_unordered(self.process_chunk, chunk_args), 
                                total=num_chunks, desc="Processing chunks"))

        for chunk_result, match_count in results:
            for idx, feat, odom in chunk_result:
                self.imu_features[idx] = feat
                self.matching_odom_data[idx] = odom
            total_matches += match_count

        self.imu_features = torch.from_numpy(self.imu_features).to(dtype=self.dtype)
        self.imu_min   = None
        self.imu_max   = None
        self.imu_mean  = None
        self.imu_std   = None
        if self.imu_stats_path and os.path.exists(self.imu_stats_path):
            print(f"Loading IMU statistics from {self.imu_stats_path}")
            try:
                stats = torch.load(self.imu_stats_path, map_location='cpu')

                # Backward compatibility
                self.imu_mean = stats.get('mean', stats.get('imu_mean'))
                self.imu_std  = stats.get('std',  stats.get('imu_std'))
                self.imu_min  = stats.get('min',  stats.get('imu_min'))
                self.imu_max  = stats.get('max',  stats.get('imu_max'))

                expected = self.imu_features.shape[1]
                for name, val in [('mean', self.imu_mean), ('std', self.imu_std),
                                  ('min', self.imu_min),   ('max', self.imu_max)]:
                    if val is not None and val.shape != (expected,):
                        raise ValueError(f"Loaded {name} has wrong shape {val.shape}")

                print(f"Loaded stats – min-max={'present' if self.imu_min is not None else 'missing'}, "
                      f"mean-std={'present' if self.imu_mean is not None else 'missing'}")
            except Exception as e:
                print(f"Error loading IMU stats: {e}. Will compute fresh stats.")
                self.imu_mean = self.imu_std = self.imu_min = self.imu_max = None
        else:
            print(f"No IMU stats file at {self.imu_stats_path}. Computing from scratch.")

        if self.imu_mean is None or self.imu_std is None:
            print("Computing mean/std statistics...")
            self.imu_mean = torch.mean(self.imu_features, dim=0)
            self.imu_std  = torch.std(self.imu_features, dim=0)

        if self.imu_min is None or self.imu_max is None:
            print("Computing min/max statistics...")
            self.imu_min = torch.min(self.imu_features, dim=0)[0]
            self.imu_max = torch.max(self.imu_features, dim=0)[0]

        if self.train and self.imu_stats_path and not os.path.exists(self.imu_stats_path):
            try:
                os.makedirs(os.path.dirname(self.imu_stats_path), exist_ok=True)
                torch.save({
                    'min':  self.imu_min,
                    'max':  self.imu_max,
                    'mean': self.imu_mean,
                    'std':  self.imu_std
                }, self.imu_stats_path)
                print(f"Saved BOTH min-max & mean-std stats to {self.imu_stats_path}")
            except Exception as e:
                print(f"Warning: Failed to save IMU stats: {e}")

        # Normalize using min-max (preferred)
        self.imu_features = self.normalize_imu(self.imu_features)

        print(f"Dataset summary: {total_matches}/{self.num_timesteps} timesteps have valid odometry matches "
              f"({100 * total_matches / self.num_timesteps:.2f}%)")
        if total_matches < 0.1 * self.num_timesteps:
            print("Warning: Less than 10% of timesteps have valid odometry matches. Consider adjusting spatial_threshold "
                  "or checking coordinate frame alignment.")
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
        """
        Normalizes using min-max by default.
        Falls back to z-score only if min-max is missing or degenerate.
        """
        # Prefer min-max
        if self.imu_min is not None and self.imu_max is not None:
            denom = self.imu_max - self.imu_min
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            return (imu_sample - self.imu_min) / (denom + 1e-7)

        # Fallback to z-score
        if self.imu_mean is not None and self.imu_std is not None:
            std = torch.where(self.imu_std == 0, torch.ones_like(self.imu_std), self.imu_std)
            return (imu_sample - self.imu_mean) / (std + 1e-7)

        # No stats → return raw
        return imu_sample
    
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
            
            pair_start = self.pair_offsets[idx]
            pair_idx = pair_start + self.rng.integers(0, self.pairs_per_shift)
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
            dataset.patch1_tensor = torch.from_numpy(np.memmap(dataset.patch1_path, dtype=dataset.np_dtype, mode='r', 
                                                              shape=(dataset.length * dataset.pairs_per_shift, 3, dataset.patch_size, dataset.patch_size))).to(dtype=dataset.dtype)
            dataset.patch2_tensor = torch.from_numpy(np.memmap(dataset.patch2_path, dtype=dataset.np_dtype, mode='r', 
                                                              shape=(dataset.length * dataset.pairs_per_shift, 3, dataset.patch_size, dataset.patch_size))).to(dtype=dataset.dtype)

def process_timestep_batch_wrapper(args):
    dataset, sub_chunk, imu_data_dict, odom_data_dict, global_positions, odom_positions = args
    return dataset.process_timestep_batch((sub_chunk, imu_data_dict, odom_data_dict, global_positions, odom_positions))

"""

def visualize_psd(dataset, idx):
    print(f"Visualizing PSD and patches for idx={idx}")
    
    # Unpack five values from dataset[idx] in debug mode
    patch1, patch2, imu_sample, global_pos, matching_odom = dataset[idx]
    
    # Print debugging info
    print(f"Global Position: {global_pos}")
    print(f"Matching Odometry Samples: {len(matching_odom)}")
    
    # Check patch tensor availability
    if not hasattr(dataset, 'patch1_tensor') or not hasattr(dataset, 'patch2_tensor'):
        raise AttributeError("Dataset missing patch1_tensor or patch2_tensor attributes")
    
    # Print patch batch info
    print(f"Number of patch batches: {len(dataset.patch1_tensor)}")
    print(f"Number of patch timesteps: {len(dataset.patch1_tensor) // dataset.pairs_per_shift}")
    
    # Open HDF5 file to load IMU data
    with h5py.File(dataset.synced_h5_path, 'r') as synced_f:
        imu_group = synced_f['imu']
        num_imu_samples = len(imu_group)
        print(f"IMU data length: {num_imu_samples}")
        
        # Map patch batch idx to IMU timestep
        patch_timestep_start = dataset.idx_to_timestep[idx]
        if patch_timestep_start >= num_imu_samples - IMU_TOPIC_RATE * 2:
            raise ValueError(f"Patch batch index {idx} (timestep {patch_timestep_start}) exceeds IMU data length {num_imu_samples}")
        
        start_idx = patch_timestep_start
        end_idx = min(patch_timestep_start + 1 + IMU_TOPIC_RATE * 2, num_imu_samples)
        
        # Load IMU segment
        imu_segment = np.array([
            np.concatenate([
                imu_group[str(i)]['angular_velocity'][:],
                imu_group[str(i)]['linear_acceleration'][:],
                imu_group[str(i)]['orientation'][:] if dataset.incl_orientation else np.zeros(4)
            ]) for i in range(start_idx, end_idx) if str(i) in imu_group
        ])
        
        print(f"Start idx: {start_idx}, End idx: {end_idx}")
        print(f"IMU segment length before padding: {imu_segment.shape[0]}")
        
        # Pad if necessary
        expected_samples = 1 + IMU_TOPIC_RATE * 2  # 41 for IMU_TOPIC_RATE=20
        if imu_segment.shape[0] < expected_samples:
            padding = np.zeros((expected_samples - imu_segment.shape[0], 
                               6 if not dataset.incl_orientation else 10))
            imu_segment = np.vstack((imu_segment, padding))
            print(f"Padded to: {imu_segment.shape}")
        
        # Extract all 6 IMU channels (ang_vel_x,y,z; lin_acc_x,y,z)
        imu_subset = imu_segment[:, :6]
        
        # Apply gravity correction if incl_orientation=True
        if dataset.incl_orientation:
            orientations = imu_segment[:, 6:]  # Shape: (41, 4)
            rot = Rotation.from_quat(orientations)
            gravity_world = np.array([0, 0, -9.81])
            gravity_imu = rot.apply(gravity_world)  # Shape: (41, 3)
            imu_subset[:, 3:6] = imu_subset[:, 3:6] - gravity_imu
        else:
            if hasattr(dataset, 'high_pass_filter'):
                for k in range(3, 6):  # lin_acc_x,y,z
                    imu_subset[:, k] = dataset.high_pass_filter(imu_subset[:, k], fs=IMU_TOPIC_RATE)
            else:
                print("Warning: high_pass_filter not defined, skipping filtering")
        
        # Compute PSD for all 6 channels
        freqs, psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)  # psd: (21, 6)
        
        # Compute full feature vector (mean, std, PSD)
        mean = np.mean(imu_subset, axis=0)  # Shape: (6,)
        std = np.std(imu_subset, axis=0)  # Shape: (6,)
        psd_flat = psd.flatten()  # Shape: (126,)
        feature_vector = np.concatenate([mean, std, psd_flat])  # Shape: (138,)
        
        # Normalize feature vector
        if hasattr(dataset, 'normalize_imu'):
            feature_tensor = torch.tensor(feature_vector, dtype=dataset.dtype)
            normalized_features = dataset.normalize_imu(feature_tensor)
            normalized_features = normalized_features.numpy() if isinstance(normalized_features, torch.Tensor) else normalized_features
        else:
            print("Warning: normalize_imu not defined, using raw features")
            normalized_features = feature_vector
        
        # Extract PSD for plotting (ang_vel_x, ang_vel_y, lin_acc_z)
        psd_indices = np.concatenate([
            np.arange(0, 21),           # ang_vel_x
            np.arange(21, 42),          # ang_vel_y
            np.arange(5*21, 6*21)       # lin_acc_z
        ])
        normalized_psd = normalized_features[12:][psd_indices].reshape(21, 3)
        
        # Debug consistency across patch batches
        base_psd = normalized_psd
        for batch_offset in range(dataset.pairs_per_shift):
            patch_batch_idx = patch_timestep_start * dataset.pairs_per_shift + batch_offset
            if patch_batch_idx < len(dataset.patch1_tensor):
                # Reload IMU segment for consistency
                imu_segment_check = np.array([
                    np.concatenate([
                        imu_group[str(i)]['angular_velocity'][:],
                        imu_group[str(i)]['linear_acceleration'][:],
                        imu_group[str(i)]['orientation'][:] if dataset.incl_orientation else np.zeros(4)
                    ]) for i in range(start_idx, end_idx) if str(i) in imu_group
                ])
                if imu_segment_check.shape[0] < expected_samples:
                    padding = np.zeros((expected_samples - imu_segment_check.shape[0], 
                                       6 if not dataset.incl_orientation else 10))
                    imu_segment_check = np.vstack((imu_segment_check, padding))
                
                imu_subset_check = imu_segment_check[:, :6]
                if dataset.incl_orientation:
                    orientations_check = imu_segment_check[:, 6:]
                    rot_check = Rotation.from_quat(orientations_check)
                    gravity_imu_check = rot_check.apply(gravity_world)
                    imu_subset_check[:, 3:6] = imu_subset_check[:, 3:6] - gravity_imu_check
                elif hasattr(dataset, 'high_pass_filter'):
                    for k in range(3, 6):
                        imu_subset_check[:, k] = dataset.high_pass_filter(imu_subset_check[:, k], fs=IMU_TOPIC_RATE)
                
                freqs_check, psd_check = periodogram(imu_subset_check, fs=IMU_TOPIC_RATE, axis=0)
                mean_check = np.mean(imu_subset_check, axis=0)
                std_check = np.std(imu_subset_check, axis=0)
                psd_check_flat = psd_check.flatten()
                feature_check = np.concatenate([mean_check, std_check, psd_check_flat])
                
                if hasattr(dataset, 'normalize_imu'):
                    feature_check_tensor = torch.tensor(feature_check, dtype=dataset.dtype)
                    normalized_features_check = dataset.normalize_imu(feature_check_tensor)
                    normalized_features_check = normalized_features_check.numpy() if isinstance(normalized_features_check, torch.Tensor) else normalized_features_check
                else:
                    normalized_features_check = feature_check
                
                normalized_psd_check = normalized_features_check[12:][psd_indices].reshape(21, 3)
                psd_diff = np.max(np.abs(normalized_psd_check - base_psd))
                print(f"Normalized PSD difference for batch {patch_batch_idx}: {psd_diff}")
                if psd_diff > 1e-10:
                    print(f"Warning: PSD mismatch for batch {patch_batch_idx}")
                
                # Debug patch data
                patch1_batch = dataset.patch1_tensor[patch_batch_idx].numpy()
                patch2_batch = dataset.patch2_tensor[patch_batch_idx].numpy()
                patch_array = np.stack([patch1_batch, patch2_batch], axis=0)  # Shape: (2, 3, self.patch_size, self.patch_size)
                sample = torch.tensor(patch_array, dtype=dataset.dtype).permute(0, 3, 1, 2)
                num_patches = sample.shape[0]
    
    # Plotting
    channel_labels = ['Angular Velocity X', 'Angular Velocity Y', 'Linear Acceleration Z']
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(1, 3, 1)
    for i in range(normalized_psd.shape[1]):
        ax1.plot(freqs, normalized_psd[:, i], label=channel_labels[i])
    ax1.set_title(f'Normalized Power Spectral Density at Index {idx} (Timestep {patch_timestep_start})')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Normalized Power/Frequency')
    ax1.legend()
    ax1.grid(True)
    
    patch1_np = patch1.permute(1, 2, 0).numpy()
    patch2_np = patch2.permute(1, 2, 0).numpy()
    if patch1_np.size > 0 and (patch1_np.max() > 1.0 or patch1_np.min() < 0.0):
        patch1_np = (patch1_np - patch1_np.min()) / (patch1_np.max() - patch1_np.min() + 1e-7)
    if patch2_np.size > 0 and (patch2_np.max() > 1.0 or patch2_np.min() < 0.0):
        patch2_np = (patch2_np - patch2_np.min()) / (patch2_np.max() - patch2_np.min() + 1e-7)
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(patch1_np if patch1_np.size > 0 else np.zeros((self.patch_size, self.patch_size, 3)), cmap='gray')
    ax2.set_title(f'Patch 1 (Timestep {patch_timestep_start}, Batch {idx})')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(patch2_np if patch2_np.size > 0 else np.zeros((self.patch_size, self.patch_size, 3)), cmap='gray')
    ax3.set_title(f'Patch 2 (Timestep {patch_timestep_start}, Batch {idx})')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_pca_terrain(dataset, n_clusters=2, n_components=2, save_dir=None):
    # Ensure imu_features exists
    if not hasattr(dataset, 'imu_features'):
        raise AttributeError("Dataset missing imu_features attribute")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(dataset.imu_features.numpy())
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result)
    
    # Define feature names (138 features: 6 mean, 6 std, 126 PSD)
    feature_names = (
        ['mean_ang_vel_x', 'mean_ang_vel_y', 'mean_ang_vel_z',
         'mean_lin_acc_x', 'mean_lin_acc_y', 'mean_lin_acc_z'] +
        ['std_ang_vel_x', 'std_ang_vel_y', 'std_ang_vel_z',
         'std_lin_acc_x', 'std_lin_acc_y', 'std_lin_acc_z'] +
        [f'psd_{ch}_{i}' for ch in ['ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                                    'lin_acc_x', 'lin_acc_y', 'lin_acc_z']
                         for i in range(21)]  # 126 / 6 = 21 PSD bins per channel
    )
    
    loadings = pca.components_.T  # Shape: (138, n_components)
    
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
        os.makedirs(save_dir, exist_ok=True)
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
    
    # 4. 2D Scatter Plot (if n_components >= 2)
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
        sample_timesteps = np.random.choice(cluster_indices, min(num_samples, len(cluster_indices)), replace=False)
        
        for timestep_idx in sample_timesteps:
            # Map timestep to dataset index (pick first batch for simplicity)
            idx = dataset.idx_to_timestep.index(timestep_idx)
            if idx >= len(dataset):
                idx = len(dataset) - 1
                print(f"Warning: Adjusted idx to {idx} to stay within dataset length")
            
            patch1, patch2, imu_sample, global_pos, matching_odom = dataset[idx]
            print(f"Timestep {timestep_idx} (batch {idx}):")
            print(f"IMU features (mean ang_vel_x,y,z; lin_acc_x,y,z; std; first few PSD): {imu_sample[:12].numpy()}")
            print(f"Global Position: {global_pos}")
            
            patch1_np = patch1.permute(1, 2, 0).numpy()
            patch2_np = patch2.permute(1, 2, 0).numpy()
            if patch1_np.size > 0 and (patch1_np.max() > 1.0 or patch1_np.min() < 0.0):
                patch1_np = (patch1_np - patch1_np.min()) / (patch1_np.max() - patch1_np.min() + 1e-7)
            if patch2_np.size > 0 and (patch2_np.max() > 1.0 or patch2_np.min() < 0.0):
                patch2_np = (patch2_np - patch2_np.min()) / (patch2_np.max() - patch2_np.min() + 1e-7)
            
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(patch1_np if patch1_np.size > 0 else np.zeros((self.patch_size, self.patch_size, 3)), cmap='gray')
            plt.title(f"Patch 1 (Cluster {cluster}, Timestep {timestep_idx})")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(patch2_np if patch2_np.size > 0 else np.zeros((self.patch_size, self.patch_size, 3)), cmap='gray')
            plt.title(f"Patch 2 (Cluster {cluster}, Timestep {timestep_idx})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Initialize dataset
    synced_h5_path = "bags/dirt_grass/dirt_grass_synced.h5"
    vicreg_h5_path = "bags/dirt_grass/dirt_grass_vicreg.h5"

    dataset = TerrainDataset(
        synced_h5_path=synced_h5_path,
        vicreg_h5_path=vicreg_h5_path,
        dtype=torch.float32,
        incl_orientation=True,
        train=False,
        debug=True
    )

    # Verify attributes
    #print("Attributes:", [attr for attr in dir(dataset) if not attr.startswith('__')])
    #print("Patch1 tensor shape:", dataset.patch1_tensor.shape)
    print("IMU features shape:", dataset.imu_features.shape)
    print("Odom length:", dataset.odom_length)
    print("IMU_TOPIC_RATE:", IMU_TOPIC_RATE)

    # Test visualize_psd
    patch_start = 17000
    patch_end = patch_start + 5
    for idx in range(patch_start, patch_end + 1):
        if idx >= len(dataset):
            print(f"Index {idx} exceeds dataset length ({len(dataset)})")
            break
        visualize_psd(dataset, idx)
        plt.savefig(f"psd_idx_{idx}.png")
        plt.close()

    # Test visualize_pca_terrain and inspect_cluster_samples
    os.makedirs("pca_plots", exist_ok=True)
    cluster_labels = visualize_pca_terrain(dataset, n_clusters=2, n_components=2, save_dir="pca_plots")
    inspect_cluster_samples(dataset, cluster_labels, num_samples=3)

"""