import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor

def rotation_matrix_to_euler_angles_scipy(R_matrix, order='xyz', degrees=True):
    """
    Convert a rotation matrix to Euler angles using SciPy.

    Args:
        R_matrix (numpy.ndarray): A 3x3 rotation matrix.
        order (str): The order of axes for Euler angles (default 'xyz').
        degrees (bool): Whether to return angles in degrees (default True).

    Returns:
        numpy.ndarray: Euler angles.
    """
    rotation = R.from_matrix(R_matrix)
    euler_angles = rotation.as_euler(order, degrees=degrees)
    return euler_angles

def decompose_homography(H, K):
    """
    Decomposes a homography matrix H into a 4x4 transformation matrix RT
    using OpenCV's decomposeHomographyMat and selecting the valid decomposition.

    Args:
        H (np.ndarray): 3x3 homography matrix.
        K (np.ndarray): 3x3 intrinsic camera matrix.

    Returns:
        np.ndarray: 4x4 transformation matrix RT combining rotation and translation.
    """

    # Normalize homography by intrinsic matrix
    K_inv = np.linalg.inv(K)
    H_normalized = K_inv @ H

    # Extract column vectors
    h1 = H_normalized[:, 0]
    h2 = H_normalized[:, 1]
    h3 = H_normalized[:, 2]

    # Compute scale factor (magnitude of h1)
    scale = np.linalg.norm(h1)

    # Normalize to get rotation and translation
    r1 = h1 / scale
    r2 = h2 / scale
    r3 = np.cross(r1, r2)  # Ensure orthonormality

    R = np.column_stack((r1, r2, r3))
    T = h3 / scale

    # Compute plane normal and distance
    plane_normal = np.cross(r1, r2)
    plane_distance = 1 / scale

    # Combine R and T into a single RT matrix
    RT = np.column_stack((R, T))
    RT = np.vstack([RT, np.array([0, 0, 0, 1])])
    # print("RT:  ", RT)
    # print("plane_normal:  ", plane_normal)
    # print("scale:  ", scale)

    return RT, plane_normal, plane_distance

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

def plot_BEV_full(img, H, patch_size=(128, 128)):
    """
    Preprocesses the robot data to compute multiple viewpoints
    of the same patch for each timestep.
    Args:
        H: Homography matrix.
        patch_size: Size of the patch (width, height).
    Returns:
        stitched_image: Reconstructed bird's-eye view image.
    """
    # Define horizontal and vertical shifts
    num_patches_x = 6
    num_patches_y = 10
    shift_step = 128

    # Compute all shifts using vectorized NumPy operations
    shift_x = np.arange(-(num_patches_x), num_patches_x + 2) * shift_step  # -6 to 7 (14 steps)
    shift_y = np.arange(-2, num_patches_y) * shift_step                   # -2 to 9 (12 steps)

    # Sort shifts in descending order to match sorted(..., reverse=True)
    shift_x = sorted(shift_x, reverse=True)  # Largest negative to largest positive (left to right)
    shift_y = sorted(shift_y, reverse=True)  # Largest negative to largest positive (top to bottom)

    # Generate all possible (sx, sy) shift pairs with the correct order
    # Use explicit indexing to match the original reverse-sorted order
    shift_pairs = []
    for sy in shift_y:  # Top to bottom (largest negative to largest positive)
        for sx in shift_x:  # Left to right (largest negative to largest positive)
            shift_pairs.append([sx, sy])
    shift_pairs = np.array(shift_pairs)  # Shape: (168, 2)

    def process_patch(shift):
        """Applies homography and warps a patch."""
        sx, sy = shift

        # Create transformation matrix
        T_shift = np.array([[1, 0, sx],
                            [0, 1, sy],
                            [0, 0, 1]])
        H_shifted = T_shift @ H  # Matrix multiplication

        # Warp image using shifted homography
        cur_patch = cv2.warpPerspective(img, H_shifted, dsize=patch_size, flags=cv2.INTER_LINEAR)

        # Ensure patch size matches exactly
        if cur_patch.shape[:2] != patch_size:
            cur_patch = cv2.resize(cur_patch, patch_size, interpolation=cv2.INTER_LINEAR)

        return cur_patch

    # Use multi-threading to process patches in parallel
    with ThreadPoolExecutor(max_workers=min(8, len(shift_pairs))) as executor:
        patches = list(executor.map(process_patch, shift_pairs))

    # Reconstruct the grid (rows x cols)
    rows = len(shift_y)  # 12
    cols = len(shift_x)  # 14

    # Reshape patches into row-wise groups and concatenate efficiently
    patches_array = np.array(patches, dtype=np.uint8)  # Convert to numpy array for efficiency
    row_images = [cv2.hconcat(patches_array[i * cols:(i + 1) * cols]) for i in range(rows)]

    # No reversal needed since shift_y is already sorted reverse=True
    # Concatenate all rows to form the final stitched image
    stitched_image = cv2.vconcat(row_images)

    # Crop the bottom part if necessary
    stitched_image = crop_bottom_to_content(stitched_image)

    return stitched_image