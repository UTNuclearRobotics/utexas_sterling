import cv2
import numpy as np
import os
import pickle

import torch
from termcolor import cprint

script_dir = os.path.dirname(os.path.abspath(__file__))

def compute_model_chessboard_2d(rows, cols, scalar_factor=20, center_at_zero=False):
    model_chessboard = np.zeros((rows * cols, 2), dtype=np.float32)
    midpoint_row = rows / 2
    midpoint_col = cols / 2
    for row in range(0, rows):
        for col in range(0, cols):
            if center_at_zero:
                model_chessboard[row * cols + col, 0] = (col + 0.5) - midpoint_col
                model_chessboard[row * cols + col, 1] = (row + 0.5) - midpoint_row
            else:
                model_chessboard[row * cols + col, 0] = col
                model_chessboard[row * cols + col, 1] = row
    model_chessboard = model_chessboard * scalar_factor
    return model_chessboard


def compute_model_chessboard_3d(rows, cols, scalar_factor=20, center_at_zero=False):
    """
    Generate 3D coordinates of the chessboard corners.
    Since chessboard lies on the plane z=0, augment the 2D points with 0 z-coordinate.
    """
    model_chessboard = compute_model_chessboard_2d(rows, cols, scalar_factor, center_at_zero)
    # Convert to 3D points by adding a z-coordinate of 0
    model_chessboard_3D = np.hstack((model_chessboard, np.zeros((model_chessboard.shape[0], 1))))
    # Add homogeneous coordinate
    model_chessboard_3D_hom = np.hstack((model_chessboard_3D, np.ones((model_chessboard_3D.shape[0], 1))))
    return model_chessboard_3D_hom

def compute_homography_from_rt(K, R, T, plane_normal, plane_distance):
    """
    Compute homography matrix from camera parameters.

    Args:
        K: Intrinsic matrix of the current camera.
        K_prime: Intrinsic matrix of the past camera (assume identical if same camera).
        R: Rotation matrix between current and past camera frames.
        T: Translation vector between current and past camera frames.
        plane_normal: Normal vector of the plane in world coordinates.
        plane_distance: Distance of the plane from the camera origin.

    Returns:
        Homography matrix H.
    """
    # Compute the plane-induced term: T * plane_normal^T / plane_distance

    plane_term = np.outer(T, plane_normal) / plane_distance

    # Compute the full homography matrix
    H = K @ (R - plane_term) @ np.linalg.inv(K)
    return H

def load_bag_h5(bag_dir, suffix):
    """Utility function to load HDF5 file paths from a bag directory."""
    file_path = os.path.join(bag_dir, f"{os.path.basename(bag_dir)}_{suffix}.h5")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No {suffix}.h5 file found at {file_path}")
    return file_path


def load_bag_pt_model(bag_path, suffix, model=None):
    model_path = os.path.join(bag_path, "models")
    # Create models directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        cprint("Created models directory", "yellow")

    # Validate the PyTorch model file exists
    pt_model = [file for file in os.listdir(model_path) if file.endswith(f"{suffix}.pt")]
    if len(pt_model) != 1:
        cprint("Existing model weights not found", "yellow")
        
    else:
        pt_model_path = os.path.join(model_path, pt_model[0])
        model.load_state_dict(torch.load(pt_model_path, weights_only=True))
        cprint("Existing model weights loaded successfully", "green")
    
    return os.path.join(model_path, f"{bag_path.rstrip('/').split('/')[-1]}_{suffix}.pt")


def cart_to_hom(points):
    """Convert Cartesian coordinates to homogeneous coordinates."""
    ones = np.ones((1, points.shape[1]))
    return_value = np.vstack((points, ones))
    return return_value

def hom_to_cart(points):
    """Convert homogeneous coordinates to Cartesian coordinates."""
    points /= points[-1, :]
    points = points[:-1, :]
    return points


def draw_points_old(image, H_shifted, patch_size, color=(0, 255, 0), thickness=2):
    """
    Draw patch boundaries on the original image using shifted homographies.

    Args:
        image (numpy.ndarray): The input image (BGR format).
        H_shifted (numpy.ndarray): Array of homography matrices, shape (N, 3, 3).
        patch_size (tuple): Patch dimensions (width, height).
        color (tuple): Color of the boundaries in BGR format (default: green).
        thickness (int): Thickness of the boundary lines (default: 2).

    Returns:
        numpy.ndarray: The image with patch boundaries drawn.
    """
    output_image = image.copy()
    patch_width, patch_height = patch_size

    # Define patch corners in BEV space (homogeneous coordinates)
    patch_corners = np.array([
        [0, 0, 1],
        [patch_width, 0, 1],
        [patch_width, patch_height, 1],
        [0, patch_height, 1]
    ], dtype=np.float32)

    for H in H_shifted:
        # Compute inverse homography to map from BEV back to original image
        H_inv = np.linalg.inv(H)
        # Transform patch corners to original image space
        original_corners = (H_inv @ patch_corners.T).T
        # Normalize homogeneous coordinates
        original_corners = original_corners[:, :2] / original_corners[:, [2]]
        # Convert to integer points for drawing
        points = original_corners.astype(int)
        # Draw the patch boundary as a closed polygon
        cv2.polylines(output_image, [points], isClosed=True, color=color, thickness=thickness)

    return output_image

def draw_points(image, corners, color=(0, 255, 0), radius=3, thickness=-1):
    """
    Draw specified corners as points on the original image.

    Args:
        image (numpy.ndarray): The input image (BGR format).
        corners (numpy.ndarray): Array of corner points, shape (N, 2) or (2, N), where each point is (x, y).
        H (numpy.ndarray, optional): Homography matrix (not used in this version but kept for compatibility).
        color (tuple): Color of the points in BGR format (default: green).
        radius (int): Radius of the drawn circles (default: 3 pixels).
        thickness (int): Thickness of the circle (-1 for filled circles, default: -1).

    Returns:
        numpy.ndarray: The image with corners drawn as points.
    """
    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()

    # Ensure corners are in the correct shape (N, 2)
    corners = np.array(corners)
    if corners.shape[0] == 2 and corners.shape[1] != 2:
        corners = corners.T  # Transpose if shape is (2, N)
    elif len(corners.shape) == 3 and corners.shape[0] == 1:
        corners = corners.reshape(-1, 2)  # Handle cases like (1, N, 2)

    # Convert corners to integer coordinates for drawing
    corners = corners.astype(int)

    # Draw each corner as a circle
    for point in corners:
        pt = tuple(point)
        cv2.circle(output_image, pt, radius, color, thickness)

    return output_image
