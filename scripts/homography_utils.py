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
    # Normalize the homography using the intrinsic matrix
    K_inv = np.linalg.inv(K)
    normalized_H = K_inv @ H

    # Decompose the homography matrix
    num_decompositions, rotations, translations, normals = cv2.decomposeHomographyMat(normalized_H, K)

    # Logic to select the correct decomposition
    best_index = -1
    max_z_translation = -np.inf  # Example criterion: largest positive translation in Z-axis
    for i in range(num_decompositions):
        # Ensure the plane normal points towards the camera (positive Z-axis)
        normal_z = normals[i][2]
        translation_z = translations[i][2]
        # print("normal_z:    ", normal_z)
        # print("translation_z:    ", translation_z)

        if normal_z > 0 and translation_z > max_z_translation:
            max_z_translation = translation_z
            best_index = i

    if best_index == -1:
        raise ValueError("No valid decomposition found.")

    # Use the selected decomposition
    R = rotations[best_index]
    t = translations[best_index].flatten()

    angs = rotation_matrix_to_euler_angles_scipy(R)
    print("angs:    ", angs)

    # Create the 4x4 transformation matrix
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R
    RT[:3, 3] = t

    return np.linalg.inv(RT)