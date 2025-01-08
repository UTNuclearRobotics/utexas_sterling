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


class Homography:
    def __init__(self, homography_tensor):
        self.homography_tensor = homography_tensor

'''
Find homography to chessboard
Get rigid transform to chessboard
Transform a 3D version of the model chessboard such that its projection lies
    on top of that of the imaged chessboard

    1) Find rigid transform, as per decompose_homography...
    2) Make a 3D version of your model chessboard (x, y, 0, 1)
    3) Transform the 3D chessboard as per the rigid transform from the decomposition.
    4) Project the 3D chessboard to 2D as per K [Identity Projection]
    5) If correct, the 2D projected points will lie on top of each other
        (image, as per homography, as per 3D transform)
    6) Transform a second model chessboard such that it takes up as much of the image as possible
        Rotation and translation relative to the RT from the homography decomposition.
        Keep it co-planar, you want to SCALE, ROTATE, and TRANSLATE IN 2D
    7) What's the best way to do this? Use a numerical optimizer to find those terms. Argmax such that..
'''
class FiddlyBEVHomography:
    def __init__(self, in_cb_image, cb_rows, cb_cols):
        self.in_cb_image = in_cb_image
        self.cb_rows = cb_rows
        self.cb_cols = cb_cols

    def extract_chessboard_points(self):
        # Get image chessboard corners, cartesian NX2
        gray = cv2.cvtColor(self.in_cb_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.cb_cols, self.cb_rows), None)
        corners = corners.reshape(-1, 2)


def draw_patch_corners_on_image(image, homography, patch_size=(128, 128)):
    """
    Draws the corners of a patch on the original image using the given homography.

    Args:
        image: Original image (numpy array).
        homography: Homography matrix mapping patch space to image space.
        patch_size: Size of the patch (width, height).

    Returns:
        Image with patch corners drawn.
    """

    # Copy current image for visualization
    image_with_patches = image.copy()
    
    # Define patch corners in homogeneous coordinates
    patch_corners = np.array([
        [0, patch_size[0], patch_size[0], 0],  # x-coordinates
        [0, 0, patch_size[1], patch_size[1]],  # y-coordinates
        [1, 1, 1, 1]                           # homogeneous coordinates
    ])

    # --- Draw the Current Patch ---
    H_inv = np.linalg.inv(homography)  # Inverse homography for current patch
    corners = H_inv @ patch_corners
    corners /= corners[2]  # Normalize to (x, y) coordinates
    pts = corners[:2].T.astype(np.int32)  # Convert to integer pixel coordinates

    cv2.polylines(image_with_patches, [pts], isClosed=True, color=(0, 255, 0), thickness=3)  # Green for current patch

    return image_with_patches


def ComputeVicRegData(H, K, RT, plane_normal, plane_distance, robot_data, history_size=10, patch_size=(128, 128), start=0):
    """
    Preprocesses the robot data to compute multiple viewpoints 
    of the same patch for each timestep.
    Args:
        H: Homography matrix.
        K: Camera intrinsic matrix.
        RT: Rotation and translation matrix.
        robot_data: Instance of RobotDataAtTimestep.
        history_size: Number of timesteps to consider in the past.
        patch_size: Size of the patch (width, height).
    Returns:
        patches: List of patches for each timestep.
    """

    n_timesteps = robot_data.getNTimesteps()
    if start >= n_timesteps:
        raise ValueError(f"Invalid cur_timestep: {start}. Must be less than {n_timesteps}.")

    patches = []
    x_offset = 0.2413
    y_offset = 0.0
    z_offset = 0.1

    T_camera_base = np.array([-x_offset, -y_offset, -z_offset])

    for timestep in tqdm(range(start, start + history_size), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        past_image_test = robot_data.getImageAtTimestep(start - history_size)
        cur_rt = robot_data.getOdomAtTimestep(timestep)
        timestep_patches = []

        # Adjust the current translation for the camera offset
        R_cur = cur_rt[:3, :3]
        T_cur = cur_rt[:3, 3] - T_camera_base

        # Get current patch
        cur_patch = cv2.warpPerspective(cur_image, H, dsize=patch_size)
        timestep_patches.append(cur_patch)

        # Define patch corners in homogeneous coordinates
        patch_corners = np.array([
            [0, patch_size[0], patch_size[0], 0],  # x-coordinates
            [0, 0, patch_size[1], patch_size[1]],  # y-coordinates
            [1, 1, 1, 1]                           # homogeneous coordinates
        ])

        # Copy current image for visualization
        cur_image_with_patches = past_image_test.copy()

        # --- Draw the Current Patch ---
        H_inv = np.linalg.inv(H)  # Inverse homography for current patch
        current_corners = H_inv @ patch_corners
        current_corners /= current_corners[2]  # Normalize to (x, y) coordinates
        pts_current = current_corners[:2].T.astype(np.int32)  # Convert to integer pixel coordinates
        cv2.polylines(cur_image_with_patches, [pts_current], isClosed=True, color=(0, 255, 0), thickness=3)  # Green for current patch

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

            #R_rel = R_past.T @ R_cur  # Current to past rotation
            #T_rel = R_past.T @ (T_cur - T_past)  # Current to past translation

            R_rel = R_cur.T @ R_past  # Past to current rotation
            T_rel = R_cur.T @ (T_past - T_cur) # Past to current translation

            # Scale translation using plane distance
            T_test = T_rel / plane_distance

            # Compute homography for past -> current -> patch
            H_past2cur = compute_homography_from_rt(K, R_rel, T_test, plane_normal, plane_distance)
            H_past2patch = H @ H_past2cur

            past_patch = cv2.warpPerspective(past_image, H_past2patch, dsize=patch_size)
            timestep_patches.append(past_patch)

            # Compute patch corners for the past patch in the current image
            H_patch2cur = np.linalg.inv(H_past2patch)  # Inverse homography
            past_corners = H_patch2cur @ patch_corners
            past_corners /= past_corners[2]  # Normalize to (x, y) coordinates

            # Draw the past patch on the current image
            pts_past = past_corners[:2].T.astype(np.int32)  # Convert to integer pixel coordinates
            cv2.polylines(cur_image_with_patches, [pts_past], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue for past patches

        # Show the current image with all patches
        cv2.imshow("All Patches on Current Image", cur_image_with_patches)
        cv2.waitKey(1)

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


def visualize_pkl(robot_data, H, patch_size=(128, 128)):
    """
    Playsback the pickle file and shows the camera image and
    the patch image at each timestep.
    Args:
        robot_data: Instance of RobotDataAtTimestep.
        H: Homography matrix.
    """
    for idx in range(0, robot_data.getNTimesteps()):
        camera_image = robot_data.getImageAtTimestep(idx)
        cur_patch = cv2.warpPerspective(camera_image, H, dsize=patch_size)
        cv2.imshow("camera_image", camera_image)
        cv2.imshow("cur_patch", cur_patch)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Homography")
    parser.add_argument("-val", action="store_true", help="Show plots to validate homography")
    parser.add_argument("-bev", action="store_true", help="Show plot of BEV image of the chessboard region")
    parser.add_argument("-bev_full", action="store_true", help="Show plot of the BEV of the whole image")
    parser.add_argument(
        "-vis_pkl", action="store_true", help="Show video feed and extracted terrain patch from pickle file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    args = parse_args()

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)
    H = np.linalg.inv(chessboard_homography.H)  #get_homography_image_to_model()
    RT = chessboard_homography.get_rigid_transform()
    plane_normal = chessboard_homography.get_plane_norm()
    plane_distance = chessboard_homography.get_plane_dist()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    robot_data = RobotDataAtTimestep(
        os.path.join(script_dir, "../bags/panther_ahg_courtyard_0/panther_ahg_courtyard_0.pkl")
    )

    output_dimensions = (
        int(chessboard_homography.cb_tile_width * (chessboard_homography.cb_cols - 1)),
        int(chessboard_homography.cb_tile_width * (chessboard_homography.cb_rows - 1)),
    )

    match args:
        case _ if args.val:
            chessboard_homography.validate_homography()
        case _ if args.bev:
            chessboard_homography.plot_BEV_chessboard()
        case _ if args.bev_full:
            chessboard_homography.BEVEditor()
            # chessboard_homography.plot_BEV_full()
        case _ if args.vis_pkl:
            visualize_pkl(robot_data, H)

    index = 100
    history_size = 10
    vicreg_data = ComputeVicRegData(H, K, RT, plane_normal, plane_distance, robot_data, history_size, patch_size=(128, 128), start = index)

    # Get the current image from robot_data
    current_image = robot_data.getImageAtTimestep(index + history_size)
    patch_images = stitch_patches_in_grid(vicreg_data[0])

    cv2.imshow("Original Image", current_image)
    cv2.imshow("Patches Grid", patch_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Show the current image side by side with the vicreg_data
# show_images_separately(current_image, vicreg_data[5])

# Access the frames (history) for the current timestep
# print(f"Image history for timestep 15:")
# for i, img in enumerate(frame_history.frames):
#   print(f"Image {frame_history.start_frame-i}: {img}")

# if ret:
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#     corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
#         criteria)
