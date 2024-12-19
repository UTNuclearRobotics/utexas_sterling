import cv2
import numpy as np
import os
import pickle

import torch
from termcolor import cprint

script_dir = os.path.dirname(os.path.abspath(__file__))


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

# Function to compute intersection of two lines
def compute_intersection(line1, line2):
    vx1, vy1, x01, y01 = line1
    vx2, vy2, x02, y02 = line2

    a1 = vy1
    b1 = -vx1
    c1 = vy1 * x01 - vx1 * y01

    a2 = vy2
    b2 = -vx2
    c2 = vy2 * x02 - vx2 * y02

    D = a1 * b2 - a2 * b1
    if D == 0:
        return None  # Lines are parallel
    x = (b1 * c2 - b2 * c1) / D
    y = (c1 * a2 - c2 * a1) / D
    return np.array([x, y])