import os

import torch
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))


class CameraIntrinsics:
    def __init__(self, config_path=os.path.join(script_dir, "homography", "camera_config.yaml")):
        # Load the configuration from the YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            self.CAMERA_INTRINSICS = config["camera_intrinsics"]
            self.CAMERA_IMU_TRANSFORM = config["camera_imu_transform"]

    def get_camera_intrinsics(self):
        """
        Get camera intrinsics and its inverse as a tensors.
        Returns:
            C_i: Camera intrinsic matrix.
            C_i_inv: Inverse of the camera intrinsic matrix.
        """
        fx = self.CAMERA_INTRINSICS["fx"]
        fy = self.CAMERA_INTRINSICS["fy"]
        cx = self.CAMERA_INTRINSICS["cx"]
        cy = self.CAMERA_INTRINSICS["cy"]

        C_i = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        C_i_inv = torch.inverse(C_i)
        return C_i, C_i_inv
    
    
