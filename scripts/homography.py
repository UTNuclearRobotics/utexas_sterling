import argparse
import math
import os
import pickle

import cv2
import numpy as np
import torch
from cam_calibration import CameraIntrinsics
from homography_util import *
from tqdm import tqdm
from utils import *


def compute_model_chessboard(rows, cols, scalar_factor=20, center_at_zero=False):
    model_chessboard = np.zeros((rows * cols, 2), dtype=np.float32)
    midpoint_row = rows / 2
    midpoint_col = cols / 2
    for row in range(0, rows):
        for col in range(0, cols):
            if center_at_zero:
                model_chessboard[row * cols + col, 0] = (col + 0.5) - midpoint_col
                model_chessboard[row * cols + col, 1] = (row + 0.5) - midpoint_row
            else:
                model_chessboard[row * cols + col, 0] = col + 0.5
                model_chessboard[row * cols + col, 1] = row + 0.5
    model_chessboard = model_chessboard * scalar_factor
    return model_chessboard


class Homography:
    def __init__(self, homography_tensor):
        self.homography_tensor = homography_tensor


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


class HomographyFromChessboardImage(Homography):
    def __init__(self, image, cb_rows, cb_cols):
        super().__init__(torch.eye(3))
        self.image = image
        self.cb_rows = cb_rows
        self.cb_cols = cb_cols
        self.chessboard_size = (cb_rows, cb_cols)

        # Get image chessboard corners, cartesian NX2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
        self.corners = corners.reshape(-1, 2)
        self.pixel_width = self.corner_pixel_width()

        # Get model chessboard corners, cartesian NX2
        self.model_chessboard = compute_model_chessboard(cb_rows, cb_cols, self.pixel_width)

        self.H, mask = cv2.findHomography(self.corners, self.model_chessboard, cv2.RANSAC)
        self.K, K_inv = CameraIntrinsics().get_camera_calibration_matrix()
        self.RT = self.decompose_homography(self.H, self.K)

        # Transform model chessboard points to image points
        self.transformed_model_corners = self.transform_points(
            self.model_chessboard.T, self.get_homography_model_to_image()
        )
        self.transformed_model_corners = self.transformed_model_corners.T.reshape(-1, 2).astype(np.float32)
        # print("transformed_model_corners:   ", transformed_model_corners)
        # print("corners:   ", corners)
        # print("Diff:    ", transformed_model_corners - corners)

    def get_homography_image_to_model(self):
        return self.H

    def get_homography_model_to_image(self):
        return np.linalg.inv(self.H)

    def get_rotation_translation_matrix(self):
        return self.RT

    def corner_pixel_width(self):
        """Calculate the maximum distance between two consecutive corners in each row of the chessboard."""
        # Sort corners by y value to group them by rows
        sorted_corners = sorted(self.corners, key=lambda x: x[1])

        # Split sorted_corners into rows
        interval = self.cb_cols
        rows = [sorted_corners[i * interval : (i + 1) * interval] for i in range(len(sorted_corners) // interval)]

        # Calculate distances between consecutive points in each row
        max_distance = 0
        for row in rows:
            row = sorted(row, key=lambda x: x[0])
            for i in range(len(row) - 1):
                distance = np.linalg.norm(np.array(row[i]) - np.array(row[i + 1]))
                max_distance = max(max_distance, distance)

        return max_distance

    def validate_homography(self):
        keepRunning = True
        renderTransformed = False
        while keepRunning:
            if renderTransformed:
                rend_image = draw_points(self.image, self.transformed_model_corners, color=(0, 255, 255))
            else:
                rend_image = draw_points(self.image, self.corners, color=(255, 0, 0))
            cv2.imshow("Chessboard", rend_image)
            key = cv2.waitKey(0)
            renderTransformed = not renderTransformed
            if key == 113:  # Hitting 'q' quits the program
                keepRunning = False
        exit(0)

    def transform_points(self, points, H):
        """Transform points using the homography matrix."""
        hom_points = cart_to_hom(points)
        transformed_points = H @ hom_points
        return hom_to_cart(transformed_points)

    def decompose_homography(self, H, K):
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

            if normal_z > 0 and translation_z > max_z_translation:
                max_z_translation = translation_z
                best_index = i

        if best_index == -1:
            raise ValueError("No valid decomposition found.")

        # Use the selected decomposition
        R = rotations[best_index]
        t = translations[best_index].flatten()

        # Create the 4x4 transformation matrix
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = R
        RT[:3, 3] = t

        return RT

    def get_BEV_chessboard_region(self):
        image = self.image.copy()
        H = self.get_homography_image_to_model()
        dimensions = (int(self.pixel_width * self.cb_cols), int(self.pixel_width * self.cb_rows))
        warped_image = cv2.warpPerspective(image, H, dsize=dimensions)

        cv2.imshow("BEV", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)

    def plot_BEV_entire_image(self, image, K, H):
        # TODO: Get BEV of entire image
        image = image.copy()
        height, width = image.shape[:2]

        # Define the corners of the original image
        img_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        # print("Image corners:   ", img_corners)

        # Compute the transformed corners
        # K_inv * RRT
        inv_H = np.linalg.inv(H)
        transformation_matrix = K @ inv_H
        transformed_corners = cv2.perspectiveTransform(np.array([img_corners]), transformation_matrix)[0]
        # print("Transformed image corners:   ", transformed_corners)

        # Calculate the bounding box of the transformed corners
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        # print("Translated image corners:   ", translated_corners)

        # Compute new dimensions
        new_width = int(np.ceil(max_x - min_x))
        new_height = int(np.ceil(max_y - min_y))

        # Adjust the transformation matrix to account for translation
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

        # Scale the translated corners to the desired width
        translated_corners = cv2.perspectiveTransform(np.array([transformed_corners]), translation_matrix)[0]
        scale_factor = width / new_width
        scaled_corners = translated_corners * scale_factor

        # Warp the image to get BEV
        combined_matrix = translation_matrix @ transformation_matrix
        warped_image = cv2.warpPerspective(image, transformation_matrix, dsize=(new_width, new_height))
        # Print the warped image dimensions
        warped_image = cv2.resize(warped_image, dsize=(width, int(new_height * (width / new_width))))
        warped_image = cv2.resize(warped_image, (width, height))
        cv2.polylines(warped_image, [scaled_corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the result
        cv2.imshow("Translated Corners", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)


class RobotDataAtTimestep:
    def __init__(self, file_path):
        # Load the .pkl file
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)

        # Ensure the file contains the expected keys
        required_keys = {"image", "imu", "odom"}
        if not required_keys.issubset(self.data.keys()):
            raise ValueError(f"The .pkl file must contain the keys: {required_keys}")

        # Determine the number of timesteps from one of the keys
        self.nTimesteps = len(self.data["image"])

    def getNTimesteps(self):
        """Return the number of timesteps."""
        return self.nTimesteps

    def getImageAtTimestep(self, idx):
        """Return the image at the given timestep index."""
        img_data = self.data["image"][idx]["data"]
        return cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if 0 <= idx < self.nTimesteps:
            image_data = self.data["image"][idx]
            if isinstance(image_data, dict):
                # Handle the dictionary (e.g., extract the 'data' field)
                image_data = image_data.get("data", None)  # Adjust based on actual structure
                if image_data is None:
                    raise TypeError("No 'data' field found in the image dictionary.")
            return torch.tensor(image_data, dtype=torch.float32)
        else:
            raise IndexError("Index out of range for timesteps.")

    def getIMUAtTimestep(self, idx):
        """Return the IMU data as a 4x4 matrix at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            imu_data = self.data["imu"][idx]

            # Extract relevant data from the dictionary
            orientation = imu_data["orientation"]  # Should be a 4-element vector
            angular_velocity = imu_data["angular_velocity"]  # Should be a 3-element vector
            linear_acceleration = imu_data["linear_acceleration"]  # Should be a 3-element vector

            # Convert to tensors
            orientation_tensor = torch.tensor(orientation, dtype=torch.float32)  # 4 elements
            angular_velocity_tensor = torch.tensor(angular_velocity, dtype=torch.float32)  # 3 elements
            linear_acceleration_tensor = torch.tensor(linear_acceleration, dtype=torch.float32)  # 3 elements

            # Pad the angular velocity and linear acceleration tensors with zeros to make them 4-element tensors
            angular_velocity_tensor = torch.cat([angular_velocity_tensor, torch.zeros(1, dtype=torch.float32)])
            linear_acceleration_tensor = torch.cat([linear_acceleration_tensor, torch.zeros(1, dtype=torch.float32)])

            # Combine the tensors into a 4x4 matrix (by stacking them row-wise)
            imu_matrix = torch.stack(
                [
                    orientation_tensor,
                    angular_velocity_tensor,
                    linear_acceleration_tensor,
                    torch.zeros(4, dtype=torch.float32),
                ],
                dim=0,
            )

            return imu_matrix

        else:
            raise IndexError("Index out of range for timesteps.")

    def getOdomAtTimestep(self, idx):
        """Return the IMU data as a 4x4 matrix at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            odom_data = self.data["odom"][idx]

            # Extract position and quaternion from the pose
            position = np.array(odom_data["pose"][:3], dtype=np.float32)  # x, y, z position
            quaternion = np.array(odom_data["pose"][3:], dtype=np.float32)  # quaternion (qx, qy, qz, qw)

            # Construct the 4x4 transformation matrix
            transformation_matrix = np.eye(4, dtype=np.float32)  # 4x4 identity matrix

            # Set the translation part (position)
            transformation_matrix[:3, 3] = position

            # Convert quaternion to rotation matrix and set it
            rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
            transformation_matrix[:3, :3] = rotation_matrix

            return transformation_matrix

    def quaternion_to_rotation_matrix(self, quaternion):
        """Convert a quaternion to a 3x3 rotation matrix using PyTorch."""
        qx, qy, qz, qw = quaternion

        # Compute the rotation matrix using the quaternion
        R = torch.tensor(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ],
            dtype=torch.float32,
        )

        return R


class FramePlusHistory:
    def __init__(self, robot_data, start_frame, history_size=10):
        self.robot_data = robot_data  # Instance of RobotDataAtTimestep
        self.start_frame = start_frame  # The frame at the current timestep
        self.history_size = history_size  # The size of the history
        self.frames = self.getImagesHistory(start_frame)

    def getImagesHistory(self, idx):
        """Return the image at the given timestep along with images from previous `history_size` timesteps."""
        # Ensure the history does not go out of bounds (e.g., at the start of the dataset)
        start_idx = max(0, idx - self.history_size)
        end_idx = idx

        # Collect the images from the history
        history_images = []
        for i in range(end_idx - 1, start_idx - 1, -1):
            image = self.robot_data.getImageAtTimestep(i)
            history_images.append(image)

        return history_images


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


def ComputeVicRegData(K, rt_to_calibrated_homography, robot_data, history_size=10):
    """
    Args:
        K: Camera intrinsic matrix.
        rt_to_calibrated_homography: Homography from the chessboard image.
        robot_data: Instance of RobotDataAtTimestep.
        history_size: Number of timesteps to consider in the past.
    Returns:
        patches: List of patches for each timestep.
    """
    n_timesteps = 500
    # n_timesteps = robot_data.getNTimesteps()
    patches = []

    # Loops through entire dataset
    for timestep in tqdm(range(490, n_timesteps), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)

        # Get past patches from current frame
        # frame_history = FramePlusHistory(robot_data, start_frame=timestep, history_size=history_size).frames

        timestep_patches = []

        # Get current patch
        cur_homography = rt_to_calibrated_homography[:3, [0, 1, 3]]
        cur_patch = cv2.warpPerspective(cur_image, cur_homography, dsize=(128, 128))

        # Draw the patch corners in the original image
        # cur_image_with_corners = draw_patch_corners_on_image(cur_image, np.linalg.inv(cur_homography))
        cv2.imshow("Current Patch Corners", cur_patch)
        cv2.waitKey(1)

        timestep_patches.append(cur_patch)

        for past_hist in range(1, history_size):
            past_timestep = timestep - past_hist

            # Get past image
            past_image = robot_data.getImageAtTimestep(past_timestep)
            cv2.imshow("Past image{past_hist}", past_image)
            cv2.waitKey(5)

            # Get homography from past image
            past_rt = robot_data.getOdomAtTimestep(past_timestep)
            cur_to_past_rt = np.linalg.inv(past_rt) @ cur_rt
            cool_transform = cur_to_past_rt @ rt_to_calibrated_homography
            calibrated_hom_past = cool_transform[:3, [0, 1, 3]]
            # print("Calibrated homography past matrix:   ", calibrated_hom_past)

            past_patch = cv2.warpPerspective(past_image, K @ calibrated_hom_past, dsize=(128, 128))
            timestep_patches.append(past_patch)

        patches.append(timestep_patches)

    return patches


def stitch_patches_in_grid(patches, grid_size=None, gap_size=10, gap_color=(255, 255, 255)):
    # Determine the grid size if not provided
    if grid_size is None:
        num_patches = len(patches)
        grid_cols = math.ceil(math.sqrt(num_patches))
        grid_rows = math.ceil(num_patches / grid_cols)
    else:
        grid_rows, grid_cols = grid_size

    # Get the dimensions of the patches (assuming all patches are the same size)
    patch_height, patch_width, _ = patches[0].shape

    # Create a blank canvas to hold the grid with gaps
    grid_height = grid_rows * patch_height + (grid_rows - 1) * gap_size
    grid_width = grid_cols * patch_width + (grid_cols - 1) * gap_size
    canvas = np.full((grid_height, grid_width, 3), gap_color, dtype=np.uint8)

    # Place each patch in the appropriate position on the canvas
    for idx, patch in enumerate(patches):
        row = idx // grid_cols
        col = idx % grid_cols
        start_y = row * (patch_height + gap_size)
        start_x = col * (patch_width + gap_size)
        canvas[start_y : start_y + patch_height, start_x : start_x + patch_width] = patch

    return canvas


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    def parse_args():
        parser = argparse.ArgumentParser(description="Homography")
        parser.add_argument("--val", "-v", action="store_true", help="Run the validate homography")
        return parser.parse_args()

    args = parse_args()

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)

    if args.val:
        chessboard_homography.validate_homography()
    chessboard_homography.get_BEV_chessboard_region()

    H = chessboard_homography.get_homography_image_to_model()

    # RT = chessboard_homography.get_rotation_translation_matrix()
    # K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    # robot_data = RobotDataAtTimestep(
    #     os.path.join(script_dir, "../bags/panther_ahg_courtyard_1/panther_ahg_courtyard_1.pkl")
    # )

    # for idx in range(0, robot_data.getNTimesteps()):
    #     camera_image = robot_data.getImageAtTimestep(idx)
    #     cur_patch = cv2.warpPerspective(camera_image, H, dsize=(128, 128))
    #     cv2.imshow("camera_image", camera_image)
    #     cv2.imshow("cur_patch", cur_patch)
    #     cv2.waitKey(1)

    # vicreg_data = ComputeVicRegData(K, H_calibrated, robot_data, 10)

    # Get the current image from robot_data
"""
    current_image = robot_data.getImageAtTimestep(5)
    patch_images = stitch_patches_in_grid(vicreg_data[5])

    #cv2.imshow("Original Image", current_image)
    cv2.imshow("Patches Grid", patch_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

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
