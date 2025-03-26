import argparse
import math
import os
import gc
import h5py

import cv2
import numpy as np
from homography_params import get_homography_params
from homography_utils import *
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils import *

class RobotDataAtTimestep:
    def __init__(self, file_path):
        # Load the .h5 file
        with h5py.File(file_path, "r") as h5f:
            # Ensure the file contains the expected groups
            required_keys = {"image", "imu", "odom"}
            if not all(key in h5f.keys() for key in required_keys):
                raise ValueError(f"The .h5 file must contain the groups: {required_keys}")

            # Load data into memory
            self.data = {
                "image": [{"timestamp": d["timestamp"][()], "data": d["data"][:]} 
                         for d in [h5f["image"][str(i)] for i in range(len(h5f["image"]))]],
                "imu": [{"timestamp": d["timestamp"][()], 
                        "orientation": d["orientation"][:], 
                        "angular_velocity": d["angular_velocity"][:], 
                        "linear_acceleration": d["linear_acceleration"][:]} 
                       for d in [h5f["imu"][str(i)] for i in range(len(h5f["imu"]))]],
                "odom": [{"timestamp": d["timestamp"][()], 
                         "pose": d["pose"][:], 
                         "twist": d["twist"][:]} 
                        for d in [h5f["odom"][str(i)] for i in range(len(h5f["odom"]))]]
            }

        # Determine the number of timesteps from one of the keys
        self.nTimesteps = len(self.data["image"])

    def getNTimesteps(self):
        """Return the number of timesteps."""
        return self.nTimesteps

    def getImageAtTimestep(self, idx):
        """Return the image at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            img_data = self.data["image"][idx]["data"]
            return cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        else:
            raise IndexError("Index out of range for timesteps.")

    def getIMUAtTimestep(self, idx):
        """Return the IMU data as a numpy array at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            imu_data = self.data["imu"][idx]
            # Stack orientation, angular_velocity, and linear_acceleration
            imu = np.hstack([
                imu_data["orientation"],
                imu_data["angular_velocity"],
                imu_data["linear_acceleration"]
            ])
            return imu
        else:
            raise IndexError("Index out of range for timesteps.")

    def getOdomAtTimestep(self, idx):
        """Return the odometry data as a 4x4 matrix at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            odom_data = self.data["odom"][idx]

            # Extract position and quaternion from the pose
            position = np.array(odom_data["pose"][:3], dtype=np.float32)  # x, y, z position
            quaternion = np.array(odom_data["pose"][3:], dtype=np.float32)  # quaternion (qx, qy, qz, qw)

            # Construct the 4x4 transformation matrix
            transformation_matrix = np.eye(4, dtype=np.float32)  # 4x4 identity matrix

            # Set the translation part (position)
            transformation_matrix[:3, 3] = position

            # Convert quaternion to rotation matrix and set it
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            transformation_matrix[:3, :3] = rotation_matrix

            return transformation_matrix
        else:
            raise IndexError("Index out of range for timesteps.")

def ComputeVicRegData(H, K, plane_normal, plane_distance, robot_data, history_size=10, patch_size=(128, 128), batch_size=1000, output_dir=""):
    n_timesteps = robot_data.getNTimesteps()

    num_patches = 2
    shift_step = 128
    shifts = np.arange(-(num_patches), num_patches + 1) * shift_step
    n_shifts = len(shifts)

    T_shifts = np.tile(np.eye(3), (n_shifts, 1, 1))
    T_shifts[:, 0, 2] = shifts
    H_shifted_all = np.matmul(T_shifts, H)

    camera_offset = np.array([0.2286, 0, 0.5715])

    patch_corners = np.array([
        [0, 0], [patch_size[0], 0],
        [patch_size[0], patch_size[1]], [0, patch_size[1]]
    ], dtype=np.float32).reshape(-1, 1, 2)

    batch_files_created = []

    for batch_start in range(history_size, n_timesteps, batch_size):
        batch_end = min(batch_start + batch_size, n_timesteps)
        batch_patches = []

        batch_start_with_history = max(0, batch_start - history_size)
        images = {t: robot_data.getImageAtTimestep(t) for t in range(batch_start_with_history, batch_end)}
        odometry = {t: robot_data.getOdomAtTimestep(t) for t in range(batch_start_with_history, batch_end)}

        for timestep in tqdm(range(batch_start, batch_end), desc=f"Processing batch {batch_start}-{batch_end}"):
            cur_image = images[timestep]
            cur_rt = odometry[timestep]
            R_cur, T_cur = cur_rt[:3, :3], cur_rt[:3, 3] + camera_offset

            cur_patches = np.array([
                cv2.warpPerspective(cur_image, H_shifted, dsize=patch_size)
                for H_shifted in H_shifted_all
            ])

            for shift_idx, H_shifted in enumerate(H_shifted_all):
                timestep_patches = [cur_patches[shift_idx]]
                past_patches = []

                valid_past_timesteps = range(max(0, timestep - history_size + 1), timestep)
                past_images = [images[t] for t in valid_past_timesteps]
                past_rts = [odometry[t] for t in valid_past_timesteps]

                for past_image, past_rt in zip(past_images, past_rts):
                    R_past, T_past = past_rt[:3, :3], past_rt[:3, 3] + camera_offset
                    R_rel = R_cur.T @ R_past
                    T_rel = R_cur.T @ (T_past - T_cur)

                    H_past2cur = compute_homography_from_rt(K, R_rel, T_rel, plane_normal, plane_distance)
                    H_past2patch = H_shifted @ H_past2cur

                    transformed_corners = cv2.perspectiveTransform(patch_corners, H_past2cur)
                    bbox_coords = transformed_corners[:, 0, :].astype(int)
                    x_min, y_min = np.maximum(np.min(bbox_coords, axis=0), 0)
                    x_max, y_max = np.minimum(np.max(bbox_coords, axis=0), past_image.shape[1::-1])
                    img_height, img_width = past_image.shape[:2]
                    
                    past_bbox = [x_min, y_min, x_max, y_max]

                    if does_overlap([0, 0, patch_size[0], patch_size[1]], past_bbox, img_width, img_height):
                        past_patch = cv2.warpPerspective(past_image, H_past2patch, dsize=patch_size)
                        past_patches.append(past_patch)

                timestep_patches.extend(past_patches)
                batch_patches.append(timestep_patches)

        batch_file = os.path.join(output_dir, f"batch_{batch_start}_{batch_end}.h5")
        with h5py.File(batch_file, "w") as h5f:
            for i, timestep_patches in enumerate(batch_patches):
                group = h5f.create_group(f"timestep_{i}")
                for j, patch in enumerate(timestep_patches):
                    group.create_dataset(f"patch_{j}", data=patch)

        if not os.path.exists(batch_file) or os.path.getsize(batch_file) == 0:
            raise RuntimeError(f"Failed to save batch file: {batch_file}")
        
        print(f"Saved batch to {batch_file}")
        batch_files_created.append(batch_file)

        batch_patches = None
        images = None
        odometry = None
        gc.collect()

    return batch_files_created

def does_overlap(cur_bbox, past_bbox, img_width, img_height):
    x_min_cur, y_min_cur, x_max_cur, y_max_cur = cur_bbox
    x_min_past, y_min_past, x_max_past, y_max_past = past_bbox

    if (x_min_cur < 0 or x_max_cur > img_width or y_min_cur < 0 or y_max_cur > img_height or
        x_min_past < 0 or x_max_past > img_width or y_min_past < 0 or y_max_past > img_height):
        return False

    return not (
        x_max_cur < x_min_past or x_max_past < x_min_cur or
        y_max_cur < y_min_past or y_max_past < y_min_cur
    )

def stitch_patches_in_grid(patches, grid_size=None, gap_size=10, gap_color=(255, 255, 255)):
    if not patches:
        return np.full((128, 128, 3), gap_color, dtype=np.uint8)

    if grid_size is None:
        num_patches = len(patches) - 1
        grid_cols = max(math.ceil(math.sqrt(num_patches)), 1)
        grid_rows = math.ceil(num_patches / grid_cols) if num_patches > 0 else 1
    else:
        grid_rows, grid_cols = grid_size

    patch_height, patch_width, _ = patches[0].shape
    grid_height = (grid_rows + 1) * patch_height + grid_rows * gap_size
    grid_width = max(grid_cols * patch_width + (grid_cols - 1) * gap_size, patch_width)
    canvas = np.full((int(grid_height), int(grid_width), 3), gap_color, dtype=np.uint8)

    canvas[:patch_height, :patch_width] = patches[0]

    for idx, patch in enumerate(patches[1:], start=1):
        row = (idx - 1) // grid_cols + 1
        col = (idx - 1) % grid_cols
        start_y = row * (patch_height + gap_size)
        start_x = col * (patch_width + gap_size)
        canvas[start_y : start_y + patch_height, start_x : start_x + patch_width] = patch

    return canvas

def validate_vicreg_data(vicreg_data_path):
    with h5py.File(vicreg_data_path, "r") as h5f:
        vicreg_data = []
        for timestep_key in sorted(h5f.keys(), key=lambda x: int(x.split('_')[1])):
            timestep_patches = [h5f[timestep_key][patch_key][:] for patch_key in sorted(h5f[timestep_key].keys(), key=lambda x: int(x.split('_')[1]))]
            vicreg_data.append(timestep_patches)

    print("Number of patches: ", len(vicreg_data))
    print("Number of patches per timestep: ", len(vicreg_data[0]))

    counter = 0
    cv2.namedWindow("VICReg Data")
    while counter < len(vicreg_data):
        patch_images = stitch_patches_in_grid(vicreg_data[counter])
        cv2.imshow("VICReg Data", patch_images)

        key = cv2.waitKey(0)
        if key == 113:  # 'q'
            counter = len(vicreg_data)
        elif key == 82:  # Up arrow
            counter += len(vicreg_data[0])
        else:
            counter += 1
    cv2.destroyAllWindows()

def combine_batches_to_single_h5(batch_files, output_h5_path, chunk_size=500):
    batch_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    total_timesteps = 0

    with h5py.File(output_h5_path, "w") as h5f:
        for batch_file in batch_files:
            with h5py.File(batch_file, "r") as batch_h5f:
                batch_timesteps = len(batch_h5f.keys())
                print(f"Processing {batch_file} with {batch_timesteps} patches")

                for timestep_key in sorted(batch_h5f.keys(), key=lambda x: int(x.split('_')[1])):
                    timestep_patches = [batch_h5f[timestep_key][patch_key][:] for patch_key in sorted(batch_h5f[timestep_key].keys(), key=lambda x: int(x.split('_')[1]))]
                    group = h5f.create_group(f"timestep_{total_timesteps}")
                    for j, patch in enumerate(timestep_patches):
                        group.create_dataset(f"patch_{j}", data=patch)
                    total_timesteps += 1

                gc.collect()

    print(f"Combined {total_timesteps} timesteps into {output_h5_path}")

def cleanup_batch_files(batch_files):
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            os.remove(batch_file)
            print(f"Deleted batch file: {batch_file}")
    print("All batch files cleaned up")

if __name__ == "__main__":
    H = get_homography_params().homography_matrix()
    RT = get_homography_params().rigid_transform()
    plane_normal = get_homography_params().plane_norm()
    plane_distance = get_homography_params().plane_dist()
    K, _ = get_homography_params().camera_intrinsics()

    parser = argparse.ArgumentParser(description="Preprocess data for VICReg.")
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronized HDF5 file inside.")
    args = parser.parse_args()

    bag_path = args.b
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    synced_h5 = [file for file in os.listdir(bag_path) if file.endswith("_synced.h5")]
    if len(synced_h5) != 1:
        raise FileNotFoundError(f"Synced HDF5 file not found in: {bag_path}")
    synced_h5_path = os.path.join(bag_path, synced_h5[0])

    robot_data = RobotDataAtTimestep(synced_h5_path)
    save_path = "/".join(synced_h5_path.split("/")[:-1])
    vicreg_data_path = os.path.join(save_path, save_path.split("/")[-1] + "_vicreg.h5")

    if os.path.exists(vicreg_data_path):
        print(f"Using existing HDF5 file: {vicreg_data_path}")
        validate_vicreg_data(vicreg_data_path)
    else:
        history_size = 10
        batch_size = 1000

        expected_batches = []
        for batch_start in range(history_size, robot_data.getNTimesteps(), batch_size):
            batch_end = min(batch_start + batch_size, robot_data.getNTimesteps())
            expected_batches.append(os.path.join(bag_path, f"batch_{batch_start}_{batch_end}.h5"))

        batch_files = expected_batches
        all_batches_exist = all(os.path.exists(f) for f in batch_files)

        if not all_batches_exist:
            batch_files = ComputeVicRegData(
                H, K, plane_normal, plane_distance, robot_data, history_size,
                patch_size=(128, 128), batch_size=batch_size, output_dir=bag_path
            )
            print(f"Saved VICReg data batches to {bag_path}")

            missing_files = [f for f in expected_batches if f not in batch_files]
            if missing_files:
                raise RuntimeError(f"Missing batch files: {missing_files}")
            print("All batch files verified successfully")
        else:
            print(f"All batch files already exist: {len(batch_files)} files found")

        gc.collect()
        print("Memory cleared before combining batches")
        H, K, RT, plane_normal, plane_distance, robot_data = None, None, None, None, None, None

        combine_batches_to_single_h5(batch_files, vicreg_data_path)

        if not os.path.exists(vicreg_data_path) or os.path.getsize(vicreg_data_path) == 0:
            raise RuntimeError(f"Failed to save combined HDF5 file: {vicreg_data_path}")
        print(f"Combined .h5 file verified: {vicreg_data_path}")

        gc.collect()

        cleanup_batch_files(batch_files)
        validate_vicreg_data(vicreg_data_path)