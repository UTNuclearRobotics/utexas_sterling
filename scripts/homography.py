"""
TR-98 Zhang "A flexible technique for.."
H - Homography
Hx = x'
Where x is in frame 1, and x' is the point in frame 2, or the homography point.
Often x is expressed as a "model point" for a frontal-parallel chessboard
x' is as imaged in a real image

How do I compute a homography?
Direct Linear Transformation -> DLT

[
H1 H2 H3
H4 H5 H6
H7 H8 H9
] *
[
X
Y
W
] =
[
H1 * X + H2 * X + H3 * X,
..Y
..W
]

cv.getPerspectiveTransform does all of this for us

Explicit representation of a calibrated homography

Camera Calibration
Intrinsic Parameters <-- Unique to the camera
Extrinsic Parameters <-- Position and orientation of the camera (actually, transformation about the camera)

Camera Intrinsic Matrix
K = [
    fx gamma u0
    0   fy   v0
    0   0   1
]

You can assume:
fx = fy
(u0, v0) is the center of the image
gamma = 0

SO, you really only need focal length

Camera Extrinsic Matrix <-- Projective, taking a 3D point down to 2D homogeneous coordinates
[
R1 R2 R3 Tx
R1 R2 R3 Ty
R1 R2 R3 Tz
]           ^- T

Rigid Transform about the camera
[
R R R Tx
R R R Ty
R R R Tz
0 0 0 1
]

Let's suppose that I'm projecting a 3D point into 2D

X = [X, Y, Z, W] = (X/W, Y/W, Z/W)
x = [X, Y, W] = (X/W, Y/W)

<X, Y, W> ~= 2 * <X, Y, W> = We don't know W, but Z is a valid W
<X, Y, 1> ~= <2X, 2Y, 2>
<X/Z, Y/Z>

Ideal Projection <-- Not subject to camera intrinsics
R R R Tx
R R R Ty
R R R Tz
]
We took Z from 3D we made it W for 2D (and dropped the 3D W)

Ideal Projection from a Rigid Transformation <-- Is the camera extrinsic matrix

Calibrated Homography
H_calibrated = K * [R R T]

H <-- Computed from a chessboard
H^-1 * H = [R1 R2 T]
Rigid Transform =
[R1 R2 R1xR2 T]

So, if we want to move the chessboard around with the vehicle.

Get

Rigid transform in frame i
Rigid transform from IMU data for i+1..n (up to 10, but not if the homography drifts off of the camera)

Formula becomes
RT_i <-- Homography to first frame, which just stays as computed from the initial homography chessboard
BEVi = from chessboard
BEVvideo_frame,0 <-- from chessboard
BEVvideo_frame,i..(n) <-- from IMU

RT_i^-1 * RT_imu ->> Turn this into R R T - H_ideal
H = K * h_ideal


class HomographyFromChessboard

class HomographyTransformed
    Use IMURTsByTimestamp to tranform homography into other video frames

class ImageDataForTraining
    BEV_ts,0 CUR_IMAGE * HomographyFromChessboard
    BEV_ts - 1..n IMAGE_ts-1..n * HomographyTransformed_ts - 1..n

    "Good" image for timestamp + n vicreg into past images (transformed by homography)
    for each timestamp

Training the representation input vector requires IMURTsByTimestamp, ImagesByTimestamp, HomographyFromChessboard
    HomographyTransformed, and puts it into ImageDataForTraining

How do we build a ground image for a bag? A BEV picture of the ground?

Images -> BEV (with Rigid Transform)
Use Rigid Transform to compute relative homographies onto a larger plane.
    The resultant homography is not constrained to the model (0,0),(64,64) model, but lives in coordinates
    on this larger plane

Costmap is just that computation after applying the scoring function neural net

Metric calibration
5 images of chessboard from different views
Cannot be coparallel


"""

import cv2
import numpy as np
import os
import torch
import pickle
import math

from camera_intrinsics import CameraIntrinsics
from tqdm import tqdm


class Homography:
    def __init__(self, homography_tensor):
        self.homography_tensor = homography_tensor


class HomographyFromChessboardImage(Homography):
    def __init__(self, image, cb_rows, cb_cols):
        super().__init__(torch.eye(3))

        # Get image chessboard corners, cartesian NX2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
        corners = corners.reshape(-1, 2)

        # Get model chessboard corners, cartesian NX2
        model_chessboard = self.compute_model_chessboard(cb_rows, cb_cols)

        self.H, mask = cv2.findHomography(model_chessboard, corners, cv2.RANSAC)
        K, K_inv = CameraIntrinsics().get_camera_calibration_matrix()
        self.H_calibrated = self.decompose_homography(self.H, K_inv)

        ### These 2 images should be the same
        # points_out = H @ self.cart_to_hom(model_chessboard.T)
        # cart_pts_out = self.hom_to_cart(points_out)
        # validate_pts = cart_pts_out.T.reshape(-1, 1, 2).astype(np.float32)
        # self.draw_corner_image(image, (cb_rows, cb_cols), validate_pts, ret)
        # self.draw_corner_image(image, (cb_rows, cb_cols), corners, ret)

    def get_homography(self):
        """
        Return the homography matrix from the chessboard image.
        """
        return self.H

    def get_calibrated_homography(self):
        """
        Return the calibrated homography matrix from the chessboard image.
        """
        return self.H_calibrated

    def draw_corner_image(self, image, chessboard_size, corners, ret):
        if ret:
            print("Chessboard corners found!")
            # Draw the corners on the image
            cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
            cv2.imshow("Chessboard Corners", image)
        else:
            cv2.imshow("Loaded Image", image)
            print("Chessboard corners not found.")

        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

    def compute_model_chessboard(self, rows, cols):
        model_chessboard = np.zeros((rows * cols, 2), dtype=np.float32)
        midpoint_row = rows / 2
        midpoint_col = cols / 2
        for row in range(0, rows):
            for col in range(0, cols):
                model_chessboard[row * cols + col, 0] = (col + 0.5) - midpoint_col
                model_chessboard[row * cols + col, 1] = (row + 0.5) - midpoint_row
        return model_chessboard

    def cart_to_hom(self, points):
        row_of_ones = np.ones((1, points.shape[1]))
        return np.vstack((points, row_of_ones))

    def hom_to_cart(self, points):
        w = points[-1]
        cart_pts = points / w
        return cart_pts[:-1]

    def decompose_homography(self, H, K_inv):
        """
        Returns:
            RT: 4x4 transformation matrix.
        """
        H = np.transpose(H)
        h1 = H[0]
        h2 = H[1]
        h3 = H[2]

        L = 1 / np.linalg.norm(np.dot(K_inv, h1))

        r1 = L * np.dot(K_inv, h1)
        r2 = L * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)

        T = L * np.dot(K_inv, h3)

        R = np.array([[r1], [r2], [r3]])
        R = np.reshape(R, (3, 3))
        U, S, V = np.linalg.svd(R, full_matrices=True)

        U = np.matrix(U)
        V = np.matrix(V)
        R = U * V

        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = R
        RT[:3, 3] = T.ravel()
        return RT


"""class RobotDataAtTimestep:
    def __init__(self, nTimesteps):
        print("FILLER")
        self.nTimesteps = nTimesteps

    def getNTimesteps(self):
        return self.nTimesteps
    
    def getImageAtTimestep(self, idx):
        print("FILLER")
        #return the image
    
    def getIMUAtTimestep(self, idx):
        print("FILLER")
        #return the IMU as a 4x4 matrix
    
    def getOdomAtTimestep(self, idx):
        print("FILLER")
        #return the Odom as a 4x4 matrix"""


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


"""
Inertial frame "the tape x in the drone cage"
Base link cur where the robot is
Base link past where the robot was
Base link to camera

Inertial frame * base link * base link to camera -- Do some magic -- BEV image
Inertial frame * base link past * base link to camera -- Do some magic -- BEV image in the past

Transform from cur to past

(Inertial frame * base link)^-1 <-- inverse * (Inertial frame * base link past) = cur_to_past
Assume the inertial frame is identity
base link^-1 * base link past = cur_to_past

Assume that base link to camera is always the same..
Meaning it was the same in the inertial frame
And in the current frame
And in the past frame

Determining how the camera moved is now just cur_to_past
The past orientation of the camera with respect to the homography is just
cur_to_past * rt_to_calibrated_homography = cool_tranform
Turn cool_tranform into a calibrated homography
[   R1 R2 R3   T
    0           1
]

[R1 R2 T] = calibrated_hom_past

Turn it into something you can use to get the same BEV image patch

At the current frame it is cv2.warpImage(cur_image, H)
In the past frame it is cv2.warpImage(past_image, K * calibrated_hom_past)
"""


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
    n_timesteps = 20
    # n_timesteps = robot_data.getNTimesteps()
    patches = []

    # Loops through entire dataset
    for timestep in tqdm(range(history_size, n_timesteps), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)

        # Get past patches from current frame
        # frame_history = FramePlusHistory(robot_data, start_frame=timestep, history_size=history_size).frames

        timestep_patches = []

        # Get current patch
        cur_patch = cv2.warpPerspective(cur_image, rt_to_calibrated_homography[:3, [0, 1, 3]], dsize=(64, 64))
        timestep_patches.append(cur_patch)

        for past_hist in range(1, history_size):
            past_timestep = timestep - past_hist

            # Get past image
            past_image = robot_data.getImageAtTimestep(past_timestep)

            # Get homography from past image
            past_rt = robot_data.getOdomAtTimestep(past_timestep)
            cur_to_past_rt = past_rt @ np.linalg.inv(cur_rt)
            cool_transform = cur_to_past_rt @ rt_to_calibrated_homography
            calibrated_hom_past = cool_transform[:3, [0, 1, 3]]
            # print("Calibrated homography past matrix:   ", calibrated_hom_past)

            past_patch = cv2.warpPerspective(past_image, K @ calibrated_hom_past, dsize=(64, 64))
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
    image_dir = script_dir + "/homography/"
    path_to_image = image_dir + "raw_image.jpg"

    image = cv2.imread(path_to_image)
    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)

    H = chessboard_homography.get_homography()
    H_calibrated = chessboard_homography.get_calibrated_homography()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    robot_data = RobotDataAtTimestep(
        os.path.join(script_dir, "../bags/panther_ahg_courtyard_1/panther_ahg_courtyard_1.pkl")
    )
    vicreg_data = ComputeVicRegData(K, H_calibrated, robot_data, 10)

    # Get the current image from robot_data
    current_image = robot_data.getImageAtTimestep(5)
    patch_images = stitch_patches_in_grid(vicreg_data[5])
    cv2.imshow("Original Image", current_image)
    cv2.imshow("Patches Grid", patch_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
