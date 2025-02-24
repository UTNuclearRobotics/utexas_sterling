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


"""
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
"""


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



def ComputeVicRegData(
    H, K, RT, plane_normal, plane_distance, robot_data, history_size=10, patch_size=(128, 128), start=0, visualize = False
):
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
    
    # Define horizontal shifts: 5 left, original, 5 right
    num_patches = 3
    shift_step = 128
    shifts = np.arange(-(num_patches), num_patches + 2) * shift_step

    # Each batch stores patches of the same shift across timesteps
    patches = [[] for _ in range(len(shifts))]

    # For visualization, we'll draw patches on the image from the first timestep
    annotated_image = None
    if visualize:
        annotated_image = robot_data.getImageAtTimestep(start).copy()

    for timestep in tqdm(range(start, start + history_size), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)

        R_cur, T_cur = cur_rt[:3, :3], cur_rt[:3, 3]

        for shift_idx, shift in enumerate(shifts):
            # Compute the translated homography
            T_shift = np.array([[1, 0, shift], [0, 1, 0], [0, 0, 1]])
            H_shifted = T_shift @ H

            # Warp and resize the patch
            cur_patch = cv2.warpPerspective(cur_image, H_shifted, dsize=patch_size)
            if cur_patch.shape != (128, 128):
                cur_patch = cv2.resize(cur_patch, (128, 128))

            # Visualize the patch on the first timestep's image
            if visualize and timestep == start:
                annotated_image = draw_points(annotated_image, H_shifted, patch_size, color=(0, 255, 0), thickness=2)

            # Store the current patch
            batch_patches = [cur_patch]

            # --- Draw Past Patches for Each Shifted Homography ---
            for past_hist in range(1, history_size):
                past_timestep = timestep - past_hist
                if past_timestep < 0:
                    continue

                # Get past image and past odometry data
                past_image = robot_data.getImageAtTimestep(past_timestep)
                past_rt = robot_data.getOdomAtTimestep(past_timestep)
                R_past, T_past = past_rt[:3, :3], past_rt[:3, 3]

                R_rel = R_cur.T @ R_past  # Past to current rotation
                T_rel = R_cur.T @ (T_past - T_cur)  # Past to current translation

                # Compute homography for past -> current -> patch
                H_past2cur = compute_homography_from_rt(K, R_rel, T_rel, plane_normal, plane_distance)
                
                # Apply past transformation to the *shifted* homography
                H_past2patch = H_shifted @ H_past2cur  

                past_patch = cv2.warpPerspective(past_image, H_past2patch, dsize=patch_size)
                if past_patch.shape != (128, 128):
                    past_patch = cv2.resize(past_patch, (128, 128))
                
                batch_patches.append(past_patch)

            # Store batch (all patches of the same shift)
            patches[shift_idx].append(batch_patches)

    if visualize:
        return patches, annotated_image
    
    return patches

def crop_bottom_to_content(img, threshold=1):
    """
    Crops the bottom of the image so that the last row containing
    any pixel value above the threshold becomes the new bottom.
    
    Parameters:
      img: A color image (NumPy array) in BGR or RGB.
      threshold: Pixel intensity threshold (default 1); 
                 rows with all pixel values <= threshold are considered black.
    
    Returns:
      Cropped image.
    """
    # Convert to grayscale for simplicity.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    h, w = gray.shape
    # Initialize the crop index to h (no crop if no black bottom is found).
    crop_row = h  
    # Iterate from the bottom row upward.
    for row in range(h - 1, -1, -1):
        # If at least one pixel in this row exceeds the threshold,
        # then this row is part of the actual image.
        if np.any(gray[row, :] > threshold):
            crop_row = row + 1  # +1 so that this row is included
            break
    return img[:crop_row, :]

def plot_BEV_full(
    H, patch_size=(128, 128), timestep=0, visualize = False
):
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
    
    # Define horizontal shifts: 5 left, original, 5 right
    num_patches_x = 6
    num_patches_y = 10
    shift_step = 128
    shift_x = np.arange(-(num_patches_x), num_patches_x + 2) * shift_step
    shift_y = np.arange(-2, num_patches_y) * shift_step

    # For visualization, if desired.
    annotated_image = None
    if visualize:
        annotated_image = robot_data.getImageAtTimestep(timestep).copy()

    # Process a single timestep. (You could loop over history_size and stitch each one.)
    # Here we'll stitch for each timestep and append to a list.
    cur_image = robot_data.getImageAtTimestep(timestep)
    # List to hold each row of patches.
    row_images = []
    # Loop over y-shifts (vertical order; top-to-bottom)
    for sy in sorted(shift_y, reverse=True):
        col_patches = []
        # Loop over x-shifts (horizontal order; left-to-right)
        for sx in sorted(shift_x, reverse=True):
            # Create a translation matrix for the given shift
            T_shift = np.array([[1, 0, sx],
                                [0, 1, sy],
                                [0, 0, 1]])
            # Shift the homography
            H_shifted = T_shift @ H

            # Warp the current image patch using the shifted homography
            cur_patch = cv2.warpPerspective(cur_image, H_shifted, dsize=patch_size)
            if cur_patch.shape[:2] != patch_size:
                cur_patch = cv2.resize(cur_patch, patch_size)

            col_patches.append(cur_patch)

            if visualize and timestep == timestep:
                annotated_image = draw_points(
                    annotated_image, H_shifted, patch_size, color=(0, 255, 0), thickness=2
                )
        # Concatenate patches in the current row horizontally.
        row_image = cv2.hconcat(col_patches)
        row_images.append(row_image)
    # Now vertically concatenate all rows to form the full birds-eye view.
    stitched_image = cv2.vconcat(row_images)
    stitched_image = crop_bottom_to_content(stitched_image)

    if visualize:
        return stitched_image, annotated_image
    return stitched_image

def stitch_patches_in_grid(patches, grid_size=None, gap_size=10, gap_color=(255, 255, 255)):
    # Determine the grid size if not provided
    if not patches or not patches[0]:  
        raise ValueError("Patches list is empty or improperly structured.")

    # Extract first actual image from the batch
    patch_height, patch_width, _ = patches[0][0].shape  # Fix: Use first patch inside batch

    # Determine grid size if not provided
    if grid_size is None:
        num_patches = len(patches)  # Include all batches
        grid_cols = math.ceil(math.sqrt(num_patches))
        grid_rows = math.ceil(num_patches / grid_cols)
    else:
        grid_rows, grid_cols = grid_size

    # Create a blank canvas to hold the grid with gaps
    grid_height = grid_rows * patch_height + (grid_rows - 1) * gap_size
    grid_width = grid_cols * patch_width + (grid_cols - 1) * gap_size
    canvas = np.full((int(grid_height), int(grid_width), 3), gap_color, dtype=np.uint8)

    # Place patches in the grid
    for idx, batch in enumerate(patches):
        patch = batch[0]  # Fix: Extract first image from batch
        row = idx // grid_cols
        col = idx % grid_cols
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
    H = np.linalg.inv(chessboard_homography.H)  # get_homography_image_to_model()
    #H, dsize,_ = chessboard_homography.plot_BEV_full(image,plot_BEV_full=False)
    RT = chessboard_homography.get_rigid_transform()
    plane_normal = chessboard_homography.get_plane_norm()
    plane_distance = chessboard_homography.get_plane_dist()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    robot_data = RobotDataAtTimestep(
        os.path.join(script_dir, "../bags/ahg_courtyard_1/ahg_courtyard_1_synced.pkl")
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
            #chessboard_homography.BEVEditor()
            chessboard_homography.plot_BEV_full(image,plot_BEV_full=True)
        case _ if args.vis_pkl:
            visualize_pkl(robot_data, H)

    index = 10
    history_size = 10

    #vicreg_data, imagewithpatches = ComputeVicRegData(
    #    H, K, RT, plane_normal, plane_distance, robot_data, history_size, patch_size=(128,128), start=index, visualize=True
    #)
    all_patches, imagewithpatches = BEV_full(
    H, patch_size=(128,128), timestep=index, visualize=True
    )
    print(all_patches)
    # Get the current image from robot_data
    current_image = robot_data.getImageAtTimestep(index + history_size)
    cv2.imshow("Stitched View", all_patches)

    cv2.imshow("Current Image with patches", imagewithpatches)
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

"""Next Steps: 
- Stiching map together from ROS bag
- Improve pattern recognition
- Making sure patches remain in the image ie when robots making a hard turn
- Classification: Segmenting patches for more accurate
"""

#Base link 
#Rigid Transform

#Calculate rigid transfrom between odom and camera using non linear optimizer