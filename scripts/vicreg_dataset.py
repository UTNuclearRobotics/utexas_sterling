import argparse
import math
import os
import pickle
import gc
import h5py

import cv2
import numpy as np
from homography_params import get_homography_params
from homography_utils import *
from robot_data_at_timestep import RobotDataAtTimestep
from tqdm import tqdm
from utils import *

def ComputeVicRegData(H, K, plane_normal, plane_distance, robot_data, history_size=10, patch_size=(128, 128), batch_size=1000, output_dir=""):
    """
    Creates and saves batch .pkl files with memory clearing after each batch.
    Returns list of batch file paths for later processing.
    """
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

        batch_file = os.path.join(output_dir, f"batch_{batch_start}_{batch_end}.pkl")
        with open(batch_file, "wb") as f:
            pickle.dump(batch_patches, f)
        
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

    # Determine the grid size if not provided
    if grid_size is None:
        num_patches = len(patches) - 1  # Exclude the first patch for the grid
        grid_cols = max(math.ceil(math.sqrt(num_patches)), 1)  # Ensure at least 1 column
        grid_rows = math.ceil(num_patches / grid_cols) if num_patches > 0 else 1
    else:
        grid_rows, grid_cols = grid_size

    # Get the dimensions of the patches (assuming all patches are the same size)
    patch_height, patch_width, _ = patches[0].shape

    # Create a blank canvas to hold the grid with gaps
    grid_height = (grid_rows + 1) * patch_height + grid_rows * gap_size  # +1 for the first patch row
    grid_width = max(grid_cols * patch_width + (grid_cols - 1) * gap_size, patch_width)
    canvas = np.full((int(grid_height), int(grid_width), 3), gap_color, dtype=np.uint8)

    # Place the first patch on its own row
    canvas[:patch_height, :patch_width] = patches[0]

    # Place the remaining patches in the grid
    for idx, patch in enumerate(patches[1:], start=1):
        row = (idx - 1) // grid_cols + 1  # +1 to account for the first patch row
        col = (idx - 1) % grid_cols
        start_y = row * (patch_height + gap_size)
        start_x = col * (patch_width + gap_size)
        canvas[start_y : start_y + patch_height, start_x : start_x + patch_width] = patch

    return canvas

def validate_vicreg_data(vicreg_data_path):
    
    # Load the pickle file
    with open(vicreg_data_path, "rb") as f:  # "rb" means read binary mode
        vicreg_data = pickle.load(f)

    print("Number of patches: ", len(vicreg_data))
    print("Number of patches per timestep: ", len(vicreg_data[0]))

    counter = 0
    cv2.namedWindow("VICReg Data")
    while counter < len(vicreg_data):
        patch_images = stitch_patches_in_grid(vicreg_data[counter])
        cv2.imshow("VICReg Data", patch_images)

        key = cv2.waitKey(0)
        if key == 113:  # Hitting 'q' quits the program
            counter = len(vicreg_data)
        elif key == 82:  # Up arrow key
            counter += len(vicreg_data[0])
        else:
            counter += 1
    exit(0)

def combine_batches_to_single_pkl(batch_files, output_pkl_path, chunk_size=500):
    batch_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    total_patches = 0
    temp_h5_path = output_pkl_path.replace(".pkl", ".h5")

    # Step 1: Combine batches into an HDF5 file incrementally
    with h5py.File(temp_h5_path, "w") as h5f:
        dset = h5f.create_dataset("patches", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.uint8))
        timestep_counts = h5f.create_dataset("timestep_counts", (0,), maxshape=(None,), dtype=np.int32)

        while batch_files:
            batch_file = batch_files.pop(0)
            with open(batch_file, "rb") as in_f:
                batch_data = pickle.load(in_f)
                batch_size = len(batch_data)
                print(f"Processing {batch_file} with {batch_size} timesteps")

                # Flatten patches and track counts
                all_patches_in_batch = [patch for timestep_patches in batch_data for patch in timestep_patches]
                timestep_patch_counts = [len(timestep_patches) for timestep_patches in batch_data]
                num_patches = len(all_patches_in_batch)

                for start in range(0, num_patches, chunk_size):
                    chunk = all_patches_in_batch[start:start + chunk_size]
                    # Convert patches to flattened uint8 arrays (not bytes objects)
                    chunk_flat = [p.flatten() for p in chunk]  # Keep as NumPy arrays
                    # Convert to a NumPy array with object dtype containing uint8 sequences
                    chunk_flat_array = np.array(chunk_flat, dtype=object)
                    curr_size = dset.shape[0]
                    dset.resize((curr_size + len(chunk_flat_array),))
                    dset[curr_size:] = chunk_flat_array  # Assign to vlen dataset
                    del chunk_flat, chunk_flat_array, chunk
                    gc.collect()

                # Append timestep counts
                curr_count_size = timestep_counts.shape[0]
                timestep_counts.resize((curr_count_size + batch_size,))
                timestep_counts[curr_count_size:] = timestep_patch_counts

                total_patches += num_patches
                del batch_data, all_patches_in_batch, timestep_patch_counts
                gc.collect()

    print(f"Combined {total_patches} patches into temporary HDF5 file: {temp_h5_path}")

    # Step 2: Convert HDF5 back to pickle format with nested structure
    with h5py.File(temp_h5_path, "r") as h5f:
        all_patches_raw = h5f["patches"][:]
        timestep_counts = h5f["timestep_counts"][:]
        all_patches_flat = [np.frombuffer(patch, dtype=np.uint8).reshape(128, 128, 3) for patch in all_patches_raw]

        # Reconstruct nested structure
        all_patches = []
        start_idx = 0
        for count in timestep_counts:
            end_idx = start_idx + count
            all_patches.append(all_patches_flat[start_idx:end_idx])
            start_idx = end_idx

    with open(output_pkl_path, "wb") as out_f:
        pickle.dump(all_patches, out_f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Converted and saved {total_patches} patches into {output_pkl_path} with {len(all_patches)} timesteps")
    os.remove(temp_h5_path)
    del all_patches_raw, all_patches_flat, all_patches, timestep_counts
    gc.collect()

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
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronized pickle file inside.")
    args = parser.parse_args()

    bag_path = args.b
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    synced_pkl = [file for file in os.listdir(bag_path) if file.endswith("_synced.pkl")]
    if len(synced_pkl) != 1:
        raise FileNotFoundError(f"Synced pickle file not found in: {bag_path}")
    synced_pkl_path = os.path.join(bag_path, synced_pkl[0])

    robot_data = RobotDataAtTimestep(synced_pkl_path)
    save_path = "/".join(synced_pkl_path.split("/")[:-1])
    vicreg_data_path = os.path.join(save_path, save_path.split("/")[-1] + "_vicreg.pkl")

    if os.path.exists(vicreg_data_path):
        print(f"Using existing pickle file: {vicreg_data_path}")
        validate_vicreg_data(vicreg_data_path)
    else:
        history_size = 10
        batch_size = 1000

        # Generate expected batch file paths
        expected_batches = []
        for batch_start in range(history_size, robot_data.getNTimesteps(), batch_size):
            batch_end = min(batch_start + batch_size, robot_data.getNTimesteps())
            expected_batches.append(os.path.join(bag_path, f"batch_{batch_start}_{batch_end}.pkl"))

        # Check if all batch files exist
        batch_files = expected_batches  # Start with expected list
        all_batches_exist = all(os.path.exists(f) for f in batch_files)

        if not all_batches_exist:
            # If any batch files are missing, compute them
            batch_files = ComputeVicRegData(
                H, K, plane_normal, plane_distance, robot_data, history_size, 
                patch_size=(128, 128), batch_size=batch_size, output_dir=bag_path
            )
            print(f"Saved VICReg data batches to {bag_path}")

            # Verify all expected batches were created
            missing_files = [f for f in expected_batches if f not in batch_files]
            if missing_files:
                raise RuntimeError(f"Missing batch files: {missing_files}")
            print("All batch files verified successfully")
        else:
            print(f"All batch files already exist: {len(batch_files)} files found")

        # Clear memory before combining
        gc.collect()
        print("Memory cleared before combining batches")
        H, K, RT, plane_normal, plane_distance, robot_data = None, None, None, None, None, None

        combine_batches_to_single_pkl(batch_files, vicreg_data_path)

        if not os.path.exists(vicreg_data_path) or os.path.getsize(vicreg_data_path) == 0:
            raise RuntimeError(f"Failed to save combined pickle file: {vicreg_data_path}")
        print(f"Combined .pkl file verified: {vicreg_data_path}")

        # Clear memory before cleanup and validation
        gc.collect()

        cleanup_batch_files(batch_files)
        validate_vicreg_data(vicreg_data_path)