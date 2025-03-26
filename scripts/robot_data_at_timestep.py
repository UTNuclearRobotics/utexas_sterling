import h5py
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class RobotDataAtTimestep:
    def __init__(self, file_path):
        # Load the .h5 file
        self.file_path = file_path
        with h5py.File(file_path, "r") as h5f:
            # Ensure the file contains the expected groups
            required_keys = {"image", "imu", "odom"}
            if not required_keys.issubset(h5f.keys()):
                raise ValueError(f"The .h5 file must contain the groups: {required_keys}")

            # Determine the number of timesteps from one of the groups
            self.nTimesteps = len(h5f["image"])

            # We'll keep the file open in read mode to access data on demand
            self.h5f = h5py.File(file_path, "r")

    def __del__(self):
        # Ensure the file is closed when the object is destroyed
        if hasattr(self, 'h5f'):
            self.h5f.close()

    def getNTimesteps(self):
        """Return the number of timesteps."""
        return self.nTimesteps

    def getImageAtTimestep(self, idx):
        """Return the image at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            img_data = self.h5f["image"][str(idx)]["data"][:]
            return cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        else:
            raise IndexError("Index out of range for timesteps.")

    def getIMUAtTimestep(self, idx):
        """Return the IMU data as a 4x4 matrix at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            imu_data = self.h5f["imu"][str(idx)]
            # Concatenate orientation, angular_velocity, and linear_acceleration
            orientation = imu_data["orientation"][:]
            angular_velocity = imu_data["angular_velocity"][:]
            linear_acceleration = imu_data["linear_acceleration"][:]
            imu = np.concatenate([orientation, angular_velocity, linear_acceleration])
            return imu
        else:
            raise IndexError("Index out of range for timesteps.")

    def getOdomAtTimestep(self, idx):
        """Return the odometry data as a 4x4 matrix at the given timestep index."""
        if 0 <= idx < self.nTimesteps:
            odom_data = self.h5f["odom"][str(idx)]

            # Extract position and quaternion from the pose
            pose = odom_data["pose"][:]
            position = pose[:3]  # x, y, z position
            quaternion = pose[3:]  # quaternion (qx, qy, qz, qw)

            # Construct the 4x4 transformation matrix
            transformation_matrix = np.eye(4, dtype=np.float32)

            # Set the translation part (position)
            transformation_matrix[:3, 3] = position

            # Convert quaternion to rotation matrix and set it
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            transformation_matrix[:3, :3] = rotation_matrix

            return transformation_matrix
        else:
            raise IndexError("Index out of range for timesteps.")
