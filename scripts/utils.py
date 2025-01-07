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

def compute_model_rectangle_3d_hom(theta=0, x1=1.0, y1=1.0, x2=1.0, y2=1.0, center_at_zero=True):
    """
    Generate 3D coordinates of the rectangle corners with separate scaling for x and y coordinates.
    """
    # Create 2D rectangle centered at 0
    if x1 > 0:
        x1 = -x1
    else:
        x1 = x1

    if y1 > 0:
        y1 = -y1
    else:
        y1 = y1

    if x2 < 0:
        x2 = -x2
    else:
        x2 = x2

    if y2 < 0:
        y2 = -y2
    else:
        y2 = y2

    model_rectangle_2d = np.array([[x1, y1],
                                   [x1, y2],
                                   [x2, y2],
                                   [x2, y1]], dtype=np.float32)

    # if center_at_zero:
    #     model_rectangle_2d[:, 0] -= np.mean(model_rectangle_2d[:, 0])
    #     model_rectangle_2d[:, 1] -= np.mean(model_rectangle_2d[:, 1])

    # Apply rotation around Z by theta
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    model_rectangle_2d = model_rectangle_2d @ rot.T

    # # Scale the rectangle with separate factors for x and y
    # model_rectangle_2d[[0,3], 0] *= x1 #left points of rectangle
    # model_rectangle_2d[[0,1], 1] *= y1 #bottom points of rectangle

    # model_rectangle_2d[[1,2], 0] *= x2 #right points of rectangle
    # model_rectangle_2d[[2,3], 1] *= y2 #top points of rectangle

    # Convert to 3D points by adding a z-coordinate of 0
    model_rectangle_3d = np.hstack((model_rectangle_2d, np.zeros((model_rectangle_2d.shape[0], 1))))

    # Add homogeneous coordinate
    model_rectangle_3d_hom = np.hstack((model_rectangle_3d, np.ones((model_rectangle_3d.shape[0], 1))))

    return model_rectangle_3d_hom


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


def load_dataset():
    dataset_dir = os.path.join(script_dir, "../datasets/")
    dataset_file = "nrg_ahg_courtyard.pkl"
    dataset_path = dataset_dir + dataset_file

    with open(dataset_path, "rb") as file:
        data_pkl = pickle.load(file)

    return data_pkl


def load_model(model):
    model_dir = os.path.join(script_dir, "../models/")
    model_file = "vis_rep.pt"
    model_path = model_dir + model_file

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        cprint("Existing model weights loaded successfully", "green")
    else:
        cprint("Existing model weights not found", "yellow")
    
    return model_path

def fixedWarpPerspective(H, image):
    image_height, image_width, channels = image.shape
    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())], axis=1)
    transformed_coords = (H @ coords.T).T
    
    w_prime = transformed_coords[:, 2]  # Extract z-coordinate
    depth_matrix = w_prime.reshape(image_height, image_width)  # Reshape to h x w
    print("image.shape: ", image.shape)
    print("depth_matrix.shape: ", depth_matrix.shape)
    thresholded = np.where(depth_matrix > 0, 255, 0).astype(np.uint8)
    # normalized_depth = cv2.normalize(depth_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b, g, r = cv2.split(image)
    b_out = cv2.bitwise_and(b, thresholded)
    g_out = cv2.bitwise_and(g, thresholded)
    r_out = cv2.bitwise_and(r, thresholded)
    # new_map_image = map_image.copy()
    # new_map_image[:, :, 2] = depth_indicator
    merged_image = cv2.merge((b_out, g_out, r_out))
    return cv2.warpPerspective(merged_image, H, (image_width * 2, image_height * 2), \
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

def cart_to_hom(points):
    """Convert Cartesian coordinates to homogeneous coordinates."""
    ones = np.ones((1, points.shape[1]))
    return_value = np.vstack((points, ones))
    return return_value

# def cart_to_hom_pt(points):
#     """Convert Cartesian coordinates to homogeneous coordinates."""
#     ones = np.ones((1, points.shape[1]))
#     return_value = np.vstack((points, ones))
#     return return_value

def hom_to_cart(points):
    """Convert homogeneous coordinates to Cartesian coordinates."""
    points /= points[-1, :]
    points = points[:-1, :]
    return points

def draw_points(image, points, color=(0, 255, 0), radius=5, thickness=-1):
    """
    Draw a list of points as circles on an image.
    
    Args:
        image (numpy.ndarray): The input image (BGR format).
        points (list of tuples): List of (x, y) coordinates to draw as circles.
        color (tuple): Color of the circles in BGR format (default: green).
        radius (int): Radius of the circles (default: 5 pixels).
        thickness (int): Thickness of the circles (-1 for filled, >0 for border thickness).
    
    Returns:255
        numpy.ndarray: The image with the points drawn.
    """
    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()
    
    # Iterate over the list of points and draw each as a circle
    for point in points:
        cv2.circle(output_image, tuple(map(int, tuple(point))), radius, color, thickness)
    
    return output_image