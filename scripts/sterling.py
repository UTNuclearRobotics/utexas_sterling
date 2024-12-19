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
    # Define the corners of the patch in the patch coordinate space
    patch_corners = np.array(
        [[0, 0, 1], [patch_size[0], 0, 1], [0, patch_size[0], 1], [patch_size[0], patch_size[0], 1]]
    ).T  # Shape (3, 4)

    # Transform the patch corners to the original image coordinate space
    transformed_corners = homography @ patch_corners
    transformed_corners /= transformed_corners[2, :]  # Normalize by the third row
    transformed_corners = transformed_corners[:2, :].T.astype(int)  # Convert to (x, y) integers

    # Draw the corners on the image
    image_with_corners = image.copy()
    for corner in transformed_corners:
        cv2.circle(image_with_corners, tuple(corner), radius=5, color=(0, 0, 255), thickness=-1)

    # Optionally, connect the corners to form a polygon
    cv2.polylines(image_with_corners, [transformed_corners], isClosed=True, color=(0, 255, 0), thickness=2)

    return image_with_corners


def ComputeVicRegData(H, K, RT, robot_data, history_size=10, patch_size=(128, 128)):
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
    patches = []

    # Loops through entire dataset
    for timestep in tqdm(range(history_size, history_size * 2), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)

        timestep_patches = []

        # Get current patch
        cur_homography = RT[:3, [0, 1, 3]]
        cur_patch = cv2.warpPerspective(cur_image, H, dsize=patch_size)

        # Draw the patch corners in the original image
        # cur_image_with_corners = draw_patch_corners_on_image(cur_image, np.linalg.inv(cur_homography))
        # cv2.imshow("Current Patch Corners", cur_patch)
        # cv2.waitKey(1)

        timestep_patches.append(cur_patch)

        # Return the image at the given timestep along with images from previous `history_size` timesteps
        for past_hist in range(1, history_size):
            past_timestep = timestep - past_hist

            # Get past image
            past_image = robot_data.getImageAtTimestep(past_timestep)
            # cv2.imshow("Past image{past_hist}", past_image)
            # cv2.waitKey(5)

            # Get homography from past image
            past_rt = robot_data.getOdomAtTimestep(past_timestep)
            cur_to_past_rt = np.linalg.inv(past_rt) @ cur_rt
            cool_transform = cur_to_past_rt * RT
            calibrated_hom_past = cool_transform[:3, [0, 1, 3]]
            # print("Calibrated homography past matrix:   ", calibrated_hom_past)

            past_patch = cv2.warpPerspective(past_image, K * calibrated_hom_past, dsize=patch_size)
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
    H = chessboard_homography.get_homography_image_to_model()
    RT = chessboard_homography.get_rotation_translation_matrix()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    robot_data = RobotDataAtTimestep(
        os.path.join(script_dir, "../bags/panther_ahg_courtyard_1/panther_ahg_courtyard_1.pkl")
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
            # TODO: Implement this case
            chessboard_homography.plot_BEV_full()
        case _ if args.vis_pkl:
            visualize_pkl(robot_data, H)

    history_size = 10
    vicreg_data = ComputeVicRegData(H, K, RT, robot_data, history_size, patch_size=(128, 128))

    # Get the current image from robot_data
    index = 0
    current_image = robot_data.getImageAtTimestep(index + history_size)
    patch_images = stitch_patches_in_grid(vicreg_data[index])

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
