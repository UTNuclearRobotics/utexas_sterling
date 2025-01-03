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

    H = H.T
    K_inv = np.linalg.inv(K)

    # Normalize the first column of H to extract the scaling factor
    L = 1 / np.linalg.norm(np.dot(K_inv, H[0]))

    h1 = H[0]
    h2 = H[1]
    h3 = H[2]

    # Compute the rotation vectors
    r1 = L * np.dot(K_inv, H[0])
    r2 = L * np.dot(K_inv, H[1])
    r3 = np.cross(r1, r2)

    # Compute the translation vector
    T = L * np.dot(K_inv, H[2]).reshape(3, 1)

    # Combine rotation and translation into a single transformation matrix
    R = np.stack((r1, r2, r3), axis=1)
    RT = np.hstack((R, T))

    return RT