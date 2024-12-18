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