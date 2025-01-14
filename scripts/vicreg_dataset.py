import argparse
import math
import os

import cv2
import numpy as np
from camera_intrinsics import CameraIntrinsics
from homography_from_chessboard import HomographyFromChessboardImage
from homography_utils import *
from robot_data_at_timestep import RobotDataAtTimestep
from tqdm import tqdm
from utils import *
import pickle


def ComputeVicRegData(H, K, plane_normal, plane_distance, robot_data, history_size=10, patch_size=(128, 128)):
    n_timesteps = robot_data.getNTimesteps()
    patches = []

    for timestep in tqdm(range(history_size, n_timesteps), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)
        timestep_patches = []

        # Adjust the current translation for the camera offset
        R_cur, T_cur = cur_rt[:3, :3], cur_rt[:3, 3]

        # Get current patch
        cur_patch = cv2.warpPerspective(cur_image, H, dsize=patch_size)
        if cur_patch.shape != (128,128):
            cur_patch = cv2.resize(cur_patch,(128,128))
            timestep_patches.append(cur_patch)
        else:
            timestep_patches.append(cur_patch)


        # --- Draw Past Patches ---
        for past_hist in range(1, history_size):
            past_timestep = timestep - past_hist
            if past_timestep < 0:
                continue

            # Get past image and past odometry data
            past_image = robot_data.getImageAtTimestep(past_timestep)
            past_rt = robot_data.getOdomAtTimestep(past_timestep)
            R_past, T_past = past_rt[:3, :3], past_rt[:3, 3]

            R_rel = R_cur.T @ R_past  # Past to current rotation
            T_rel = R_cur.T @ (T_past - T_cur) / plane_distance  # Past to current translation

            # Compute homography for past -> current -> patch
            H_past2cur = compute_homography_from_rt(K, R_rel, T_rel, plane_normal, plane_distance)
            H_past2patch = H @ H_past2cur

            past_patch = cv2.warpPerspective(past_image, H_past2patch, dsize=patch_size)
            if past_patch.shape != (128,128):
                past_patch = cv2.resize(past_patch,(128,128))
                timestep_patches.append(past_patch)
            else:
                timestep_patches.append(past_patch)

        patches.append(timestep_patches)

    return patches


def stitch_patches_in_grid(patches, grid_size=None, gap_size=10, gap_color=(255, 255, 255)):
    # Determine the grid size if not provided
    if grid_size is None:
        num_patches = len(patches) - 1  # Exclude the first patch for the grid
        grid_cols = math.ceil(math.sqrt(num_patches))
        grid_rows = math.ceil(num_patches / grid_cols)
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


def validate_vicreg_data(robot_data, vicreg_data):
    history_size = robot_data.getNTimesteps() - len(vicreg_data)
    
    counter = 0
    cv2.namedWindow("VICReg Data")
    while counter < len(vicreg_data):
        patch_images = stitch_patches_in_grid(vicreg_data[counter])
        cv2.imshow("VICReg Data", patch_images)

        key = cv2.waitKey(0)
        if key == 113:  # Hitting 'q' quits the program
            counter = len(vicreg_data)
        elif key == 82:  # Up arrow key
            counter += history_size
        else:
            counter += 1
    exit(0)


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    # Parameters for compute vicreg data
    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)
    H = np.linalg.inv(chessboard_homography.H)  # get_homography_image_to_model()
    #H, dsize = chessboard_homography.plot_BEV_full(plot_BEV_full=False)
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()
    plane_normal = chessboard_homography.get_plane_norm()
    plane_distance = chessboard_homography.get_plane_dist()

    bag_name = "panther_ahg_courtyard"
    robot_data = RobotDataAtTimestep(os.path.join(script_dir, f"../bags/{bag_name}/{bag_name}.pkl"))

    # Load or compute vicreg data
    vicreg_data_path = os.path.join(script_dir, f"../bags/{bag_name}/vicreg_data.pkl")
    if os.path.exists(vicreg_data_path):
        # --- DELETE THE .PKL IF YOU WANT TO RECALCULATE VICREG DATA ---
        with open(vicreg_data_path, "rb") as f:
            vicreg_data = pickle.load(f)
    else:
        history_size = 10
        vicreg_data = ComputeVicRegData(
            H, K, plane_normal, plane_distance, robot_data, history_size, patch_size=(128,128)
        )
        with open(vicreg_data_path, "wb") as f:
            pickle.dump(vicreg_data, f)

    # Visualize vicreg data
    validate_vicreg_data(robot_data, vicreg_data)
