import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import argparse, os, pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import periodogram, butter, filtfilt
from scipy.spatial.transform import Rotation

IMU_TOPIC_RATE = 20

class TerrainDataset(Dataset):
    def __init__(self, patches=None, synced_data=None, labeled_data=None, transform=None, dtype=torch.float32, incl_orientation=False):
        print("Loading TerrainDataset version with is_labeled")
        self.dtype = dtype
        self.transform = transform
        self.incl_orientation = incl_orientation

        if labeled_data is not None:
            # Labeled data mode
            self.patches = [sample['patch'] for sample in labeled_data]
            self.inertial = [sample['inertial'] for sample in labeled_data]
            self.terrain_labels = [sample['terrain_label'] for sample in labeled_data]
            self.preferences = [sample['preference'] for sample in labeled_data]
            self.is_labeled = True
        else:
            # Unlabeled data mode (self-supervised)
            if patches is None or synced_data is None:
                raise ValueError("Must provide patches and synced_data for unlabeled mode")
            self.raw_patches = patches
            self.robot_data = synced_data
            self.imu_data = self.robot_data["imu"]
            self.is_labeled = False

            ang_vels = np.array([sample["angular_velocity"] for sample in self.imu_data])
            lin_accs = np.array([sample["linear_acceleration"] for sample in self.imu_data])

            if self.incl_orientation:
                orientations = np.array([sample["orientation"] for sample in self.imu_data])
                lin_accs = np.array([self.remove_gravity(lin_accs[i], orientations[i]) 
                                    for i in range(len(lin_accs))])
                self.imu_data = np.concatenate([ang_vels, lin_accs, orientations], axis=1)
            else:
                lin_accs = np.apply_along_axis(lambda x: self.remove_gravity(x, None), axis=1, arr=lin_accs)
                self.imu_data = np.concatenate([ang_vels, lin_accs], axis=1)

            samples_per_window = IMU_TOPIC_RATE * 2
            num_windows = (len(self.imu_data) - samples_per_window + 1) // 1
            if num_windows <= 0:
                num_windows = 1

            self.psd_features = []
            for i in range(num_windows):
                start = i
                end = start + samples_per_window
                window = self.imu_data[start:end]
                if len(window) < samples_per_window:
                    window = np.pad(window, ((0, samples_per_window - len(window)), (0, 0)), mode='constant')
                imu_subset = window[:, [0, 1, 2, 3, 4, 5]]  # All 6 channels
                if not self.incl_orientation:
                    for j in range(3, 6):
                        imu_subset[:, j] = self.high_pass_filter(imu_subset[:, j], fs=IMU_TOPIC_RATE)
                
                psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)[1].flatten()  # 606 values
                std = np.std(imu_subset, axis=0)  # 6 values
                features = np.concatenate([std, psd])  # 6 std + 606 PSD = 612
                self.psd_features.append(features)

            self.psd_features = np.array(self.psd_features)  # Shape: (num_windows, 612)
            self.imu_min = np.min(self.psd_features, axis=0)
            self.imu_max = np.max(self.psd_features, axis=0)

    def remove_gravity(self, linear_acceleration, orientation):
        if orientation is None:
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
        if np.allclose(self.imu_max - self.imu_min, 0, atol=1e-8):
            return imu_sample
        return (imu_sample - self.imu_min) / (self.imu_max - self.imu_min + 1e-7)

    def __len__(self):
        return len(self.patches if self.is_labeled else self.raw_patches)

    def __getitem__(self, idx):
        if self.is_labeled:
            # Labeled data mode
            patch = torch.tensor(self.patches[idx], dtype=self.dtype).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            inertial = torch.tensor(self.inertial[idx], dtype=self.dtype)
            terrain_label = self.terrain_labels[idx]
            preference = torch.tensor(self.preferences[idx], dtype=self.dtype)

            if self.transform:
                patch = self.transform(patch)

            return {
                'patch': patch,
                'inertial': inertial,
                'terrain_label': terrain_label,
                'preference': preference
            }
        else:
            # Unlabeled data mode (self-supervised)
            imu_timestep_start = idx // 5
            start_idx = imu_timestep_start
            end_idx = min(imu_timestep_start + IMU_TOPIC_RATE*2, len(self.imu_data))
            imu_segment = self.imu_data[start_idx:end_idx]

            if imu_segment.shape[0] < IMU_TOPIC_RATE*2:
                imu_segment = np.pad(imu_segment, ((0, IMU_TOPIC_RATE*2 - imu_segment.shape[0]), (0, 0)), mode='constant')

            patch_batch = self.raw_patches[idx]
            patch_array = np.array(patch_batch)
            if len(patch_array.shape) != 4:
                raise ValueError(f"Unexpected patch array shape: {patch_array.shape}")

            sample = torch.tensor(patch_array, dtype=self.dtype).permute(0, 3, 1, 2)
            num_patches = sample.shape[0]

            patch1_idx = torch.randint(0, num_patches // 2, (1,)).item() if num_patches > 0 else 0
            patch2_idx = torch.randint(num_patches // 2, num_patches, (1,)).item() if num_patches > 0 else 0
            patch1 = sample[patch1_idx] if num_patches > 0 else torch.zeros((3, 128, 128), dtype=self.dtype)
            patch2 = sample[patch2_idx] if num_patches > 0 else torch.zeros((3, 128, 128), dtype=self.dtype)

            if self.transform and num_patches > 0:
                patch1 = self.transform(patch1)
                patch2 = self.transform(patch2)

            imu_subset = imu_segment[:, [0, 1, 2, 3, 4, 5]]
            if not self.incl_orientation:
                for j in range(3, 6):
                    imu_subset[:, j] = self.high_pass_filter(imu_subset[:, j], fs=IMU_TOPIC_RATE)

            psd = periodogram(imu_subset, fs=IMU_TOPIC_RATE, axis=0)[1].flatten()  # 606 values
            std = np.std(imu_subset, axis=0)  # 6 values
            imu_features = np.concatenate([std, psd])  # 612 features
            normalized_features = self.normalize_imu(imu_features)
            imu_sample = torch.tensor(normalized_features, dtype=self.dtype).reshape(1, -1)

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
