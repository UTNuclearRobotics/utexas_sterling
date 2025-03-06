import argparse
import math
import os
import pickle

import cv2
import numpy as np
from camera_intrinsics import CameraIntrinsics
from homography_matrix import HomographyMatrix
from homography_utils import *
from robot_data_at_timestep import RobotDataAtTimestep
from tqdm import tqdm
from utils import *

def ComputeVicRegData(H, K, plane_normal, plane_distance, robot_data, history_size=10, patch_size=(128, 128)):
    """
    Highly optimized version for high-speed processing with numpy vectorization and batch operations.
    """
    n_timesteps = robot_data.getNTimesteps() #- 3500 subtract for agh2 to improve terrain training
    patches = []

    # Define shifts
    num_patches = 2  
    shift_step = 128  
    shifts = np.arange(-(num_patches), num_patches + 1) * shift_step  
    n_shifts = len(shifts)

    # Vectorized shift matrices
    T_shifts = np.tile(np.eye(3), (n_shifts, 1, 1))
    T_shifts[:, 0, 2] = shifts  
    H_shifted_all = np.matmul(T_shifts, H)  

    # Static camera offset
    camera_offset = np.array([0.2286, 0, 0.5715])  

    # Precompute patch bounding box corners
    patch_corners = np.array([
        [0, 0], [patch_size[0], 0], 
        [patch_size[0], patch_size[1]], [0, patch_size[1]]
    ], dtype=np.float32).reshape(-1, 1, 2)  

    # Preload all images and odometry data
    images = [robot_data.getImageAtTimestep(t) for t in range(n_timesteps)]
    odometry = [robot_data.getOdomAtTimestep(t) for t in range(n_timesteps)]

    for timestep in tqdm(range(history_size, n_timesteps), desc="Processing patches"):
        cur_image = images[timestep]
        cur_rt = odometry[timestep]
        R_cur, T_cur = cur_rt[:3, :3], cur_rt[:3, 3] + camera_offset  

        # Warp current image for all shifts in batch
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

                # Transform past patch bounding box
                transformed_corners = cv2.perspectiveTransform(patch_corners, H_past2cur)

                # Compute bounding box limits
                bbox_coords = transformed_corners[:, 0, :].astype(int)
                x_min, y_min = np.maximum(np.min(bbox_coords, axis=0), 0)
                x_max, y_max = np.minimum(np.max(bbox_coords, axis=0), past_image.shape[1::-1])
                img_height, img_width = past_image.shape[:2]
                
                past_bbox = [x_min, y_min, x_max, y_max]

                if does_overlap([0, 0, patch_size[0], patch_size[1]], past_bbox, img_width, img_height):
                    past_patch = cv2.warpPerspective(past_image, H_past2patch, dsize=patch_size)
                    past_patches.append(past_patch)

            timestep_patches.extend(past_patches)
            patches.append(timestep_patches)
    
    return patches

def does_overlap(cur_bbox, past_bbox, img_width, img_height):
    """
    Checks if two bounding boxes overlap and ensures they are within image bounds.
    """
    x_min_cur, y_min_cur, x_max_cur, y_max_cur = cur_bbox
    x_min_past, y_min_past, x_max_past, y_max_past = past_bbox

    # Check if bounding boxes are within image bounds
    if (x_min_cur < 0 or x_max_cur > img_width or y_min_cur < 0 or y_max_cur > img_height or
        x_min_past < 0 or x_max_past > img_width or y_min_past < 0 or y_max_past > img_height):
        return False  # Bounding box is out of image bounds

    # Check if bounding boxes overlap
    return not (
        x_max_cur < x_min_past or x_max_past < x_min_cur or
        y_max_cur < y_min_past or y_max_past < y_min_cur
    )


def stitch_patches_in_grid(patches, grid_size=None, gap_size=10, gap_color=(255, 255, 255)):
    # Determine the grid size if not provided
    if grid_size is None:
        num_patches = len(patches) - 1  # Exclude the first patch for the grid
        grid_cols = math.ceil(math.sqrt(num_patches))
        grid_rows = math.ceil(num_patches / grid_cols)
    else:
        grid_rows, grid_cols = grid_size

    # Get the dimensions of the patches (assuming all patches are the same size)
    patch_height, patch_width, _ = patches[0].shape  # Extract first patch image


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


def validate_vicreg_data(robot_data, vicreg_data):
    history_size = robot_data.getNTimesteps() - len(vicreg_data)

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

if __name__ == "__main__":
    # Load parameters for compute vicreg data from camera intrinsics yaml and homography yaml
    H = HomographyMatrix().get_homography_matrix()
    RT = HomographyMatrix().get_rigid_transform()
    plane_normal = HomographyMatrix().get_plane_normal()
    plane_distance = HomographyMatrix().get_plane_distance()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    parser = argparse.ArgumentParser(description="Preprocess data for VICReg.")
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronzied pickle file inside.")
    args = parser.parse_args()

    # Check if the bag file exists
    bag_path = args.b
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    # Validate the sycned pickle file
    synced_pkl = [file for file in os.listdir(bag_path) if file.endswith("_synced.pkl")]
    if len(synced_pkl) != 1:
        raise FileNotFoundError(f"Synced pickle file not found in: {bag_path}")
    synced_pkl_path = os.path.join(bag_path, synced_pkl[0])

    robot_data = RobotDataAtTimestep(synced_pkl_path)

    save_path = "/".join(synced_pkl_path.split("/")[:-1])
    vicreg_data_path = os.path.join(save_path, save_path.split("/")[-1] + "_vicreg.pkl")

    # Load or compute vicreg data
    if os.path.exists(vicreg_data_path):
        # --- DELETE THE .PKL IF YOU WANT TO RECALCULATE VICREG DATA ---
        with open(vicreg_data_path, "rb") as f:
            vicreg_data = pickle.load(f)
    else:
        history_size = 10
        vicreg_data = ComputeVicRegData(
            H, K, plane_normal, plane_distance, robot_data, history_size, patch_size=(128, 128)
        )
        with open(vicreg_data_path, "wb") as f:
            pickle.dump(vicreg_data, f)

    # Visualize vicreg data
    validate_vicreg_data(robot_data, vicreg_data)
