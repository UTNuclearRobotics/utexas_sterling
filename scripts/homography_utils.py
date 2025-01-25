import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    print("RT:  ", RT)

    return RT, plane_normal, plane_distance