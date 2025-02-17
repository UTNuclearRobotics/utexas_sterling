import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


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

    def getIMUAtTimestep(self, idx):
        """Return the IMU data as a 4x4 matrix at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            imu_data = self.data["imu"][idx]
            imu = imu_data[:, :-4]
            return imu
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
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            transformation_matrix[:3, :3] = rotation_matrix

            return transformation_matrix
