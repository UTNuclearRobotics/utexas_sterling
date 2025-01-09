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

    # Static transform from base_link to camera_link
    x_offset = 0.2413
    y_offset = 0.0
    z_offset = 0.1
    T_camera_base = np.array([-x_offset, -y_offset, -z_offset])

    for timestep in tqdm(range(history_size, n_timesteps), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        # past_image_test = robot_data.getImageAtTimestep(start - history_size)
        cur_rt = robot_data.getOdomAtTimestep(timestep)
        timestep_patches = []

        # Adjust the current translation for the camera offset
        R_cur = cur_rt[:3, :3]
        T_cur = cur_rt[:3, 3] - T_camera_base

        # Get current patch
        cur_patch = cv2.warpPerspective(cur_image, H, dsize=patch_size)
        timestep_patches.append(cur_patch)

        # # Define patch corners in homogeneous coordinates
        # patch_corners = np.array(
        #     [
        #         [0, patch_size[0], patch_size[0], 0],  # x-coordinates
        #         [0, 0, patch_size[1], patch_size[1]],  # y-coordinates
        #         [1, 1, 1, 1],  # homogeneous coordinates
        #     ]
        # )

        # # Copy current image for visualization
        # cur_image_with_patches = past_image_test.copy()

        # # --- Draw the Current Patch ---
        # H_inv = np.linalg.inv(H)  # Inverse homography for current patch
        # current_corners = H_inv @ patch_corners
        # current_corners /= current_corners[2]  # Normalize to (x, y) coordinates
        # pts_current = current_corners[:2].T.astype(np.int32)  # Convert to integer pixel coordinates
        # cv2.polylines(
        #     cur_image_with_patches, [pts_current], isClosed=True, color=(0, 255, 0), thickness=3
        # )  # Green for current patch

        # --- Draw Past Patches ---
        for past_hist in range(1, history_size):
            past_timestep = timestep - past_hist
            if past_timestep < 0:
                continue

            # Get past image and past odometry data
            past_image = robot_data.getImageAtTimestep(past_timestep)
            past_rt = robot_data.getOdomAtTimestep(past_timestep)

            # Compute relative rotation and translation
            R_past = past_rt[:3, :3]
            T_past = past_rt[:3, 3] - T_camera_base

            # R_rel = R_past.T @ R_cur  # Current to past rotation
            # T_rel = R_past.T @ (T_cur - T_past)  # Current to past translation

            R_rel = R_cur.T @ R_past  # Past to current rotation
            T_rel = R_cur.T @ (T_past - T_cur)  # Past to current translation

            # Scale translation using plane distance
            T_test = T_rel / plane_distance

            # Compute homography for past -> current -> patch
            H_past2cur = compute_homography_from_rt(K, R_rel, T_test, plane_normal, plane_distance)
            H_past2patch = H @ H_past2cur

            past_patch = cv2.warpPerspective(past_image, H_past2patch, dsize=patch_size)
            timestep_patches.append(past_patch)

            # # Compute patch corners for the past patch in the current image
            # H_patch2cur = np.linalg.inv(H_past2patch)  # Inverse homography
            # past_corners = H_patch2cur @ patch_corners
            # past_corners /= past_corners[2]  # Normalize to (x, y) coordinates

            # # Draw the past patch on the current image
            # pts_past = past_corners[:2].T.astype(np.int32)  # Convert to integer pixel coordinates
            # cv2.polylines(
            #     cur_image_with_patches, [pts_past], isClosed=True, color=(255, 0, 0), thickness=2
            # )  # Blue for past patches

        # # Show the current image with all patches
        # cv2.imshow("All Patches on Current Image", cur_image_with_patches)
        # cv2.waitKey(1)

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
    canvas = np.full((grid_height, grid_width, 3), gap_color, dtype=np.uint8)

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
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()
    plane_normal = chessboard_homography.get_plane_norm()
    plane_distance = chessboard_homography.get_plane_dist()

    bag_name = "panther_ahg_courtyard_0"
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
            H, K, plane_normal, plane_distance, robot_data, history_size, patch_size=(128, 128)
        )
        with open(vicreg_data_path, "wb") as f:
            pickle.dump(vicreg_data, f)

    # Visualize vicreg data
    validate_vicreg_data(robot_data, vicreg_data)
