import argparse
import math
import os
from joblib import Parallel, delayed

import cv2
import numpy as np
from homography_params import get_homography_params
from homography_utils import *
from utils import *
from homography_params import get_homography_params

def crop_bottom_to_content(img):
    nonzero_rows = np.any(img > 0, axis=(1, 2))  # Check for non-black rows
    last_nonzero = np.where(nonzero_rows)[0][-1]  # Find last non-black row
    return img[:last_nonzero + 1]

def plot_BEV_full(
    H, patch_size, image, visualize = False
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

    cols, rows = [14,20]
    patch_width, patch_height = patch_size
    shift_step = patch_size[0]

    # Precompute shift arrays
    shift_x = np.linspace(-cols * shift_step, (cols + 1) * shift_step, 2 * cols + 2, dtype=np.float32)
    shift_y = np.linspace(-shift_step, (rows - 1) * shift_step, rows + 1, dtype=np.float32)

    # Precompute sorted shifts
    sorted_sy = np.sort(shift_y)[::-1]  # Reverse to match sorted(reverse=True)
    sorted_sx = np.sort(shift_x)[::-1]
    
    # Create shift combinations
    sy, sx = np.meshgrid(sorted_sy, sorted_sx, indexing='ij')
    shift_combinations = np.stack([sy.ravel(), sx.ravel()], axis=1)

    # Compute homography shifts
    T_shifts = np.eye(3, dtype=np.float32)[None, :, :].repeat(len(shift_combinations), axis=0)
    T_shifts[:, 0, 2] = shift_combinations[:, 1]  # sx
    T_shifts[:, 1, 2] = shift_combinations[:, 0]  # sy
    H_shifted = T_shifts @ H.astype(np.float32)

    def warp_patch(i):
        return cv2.warpPerspective(
            image,
            H_shifted[i],
            dsize=patch_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # Parallelize patch warping
    patches = Parallel(n_jobs=-1, backend='loky', prefer='processes')(
        delayed(warp_patch)(i) for i in range(len(shift_combinations))
    )

    # Reshape patches into stitched image
    num_rows = len(sorted_sy)  # 16
    num_cols = len(sorted_sx)  # 30
    stitched_height = num_rows * patch_height
    stitched_width = num_cols * patch_width
    patches_array = np.array(patches, dtype=image.dtype).reshape(
        num_rows, num_cols, patch_height, patch_width, 3
    )
    stitched_image = patches_array.transpose(0, 2, 1, 3, 4).reshape(
        stitched_height, stitched_width, 3
    )

    # Crop black rows from bottom
    stitched_image = crop_bottom_to_content(stitched_image)

    # Visualization (only if enabled)
    if visualize:
        annotated_image = image.copy()
        annotated_image = draw_points_old(annotated_image, H_shifted, patch_size, color=(0, 255, 0), thickness=2)
        cv2.namedWindow("Current Image with patches", cv2.WINDOW_NORMAL)
        cv2.imshow("Current Image with patches", annotated_image)

        # Display the image
        cv2.namedWindow("Stitched BEV Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Stitched BEV Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

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
    image_file = "sim_pgl_calibration.png"
    image = cv2.imread(os.path.join(image_dir, image_file))

    #chessboard_homography = HomographyFromChessboardImage(image, 9, 7)
    H = get_homography_params().homography_matrix()
    RT = get_homography_params().rigid_transform()
    plane_normal = get_homography_params().plane_norm()
    plane_distance = get_homography_params().plane_dist()
    K, _ = get_homography_params().camera_intrinsics()
    px_meter = get_homography_params().px_meter()

    all_patches = plot_BEV_full(
    H, patch_size=(128,128), image=image, visualize=True
    )
    #print(all_patches)
    # Get the current image from robot_data