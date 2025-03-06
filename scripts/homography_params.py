import numpy as np
import os
import yaml

class get_homography_params:
    def __init__(self, config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "homography", "config.yaml")):
        """
        Initialize the class by loading configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file (default: 'homography/config.yaml').
        """
        self.config_path = config_path
        self._load_config()  # Load initial configuration

    def _load_config(self):
        """
        Load or reload the configuration from the YAML file.
        """
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file) or {}

        # Always load camera_intrinsics (required)
        if "camera_intrinsics" not in config:
            raise ValueError("YAML file must contain a 'camera_intrinsics' key")
        self.CAMERA_INTRINSICS = config["camera_intrinsics"]

        # Load homography (optional initially, required after calculation)
        if "homography" in config:
            homography_list = config["homography"]
            if len(homography_list) != 3 or any(len(row) != 3 for row in homography_list):
                raise ValueError("Homography data in YAML is not a 3x3 matrix")
            self.H = np.array(homography_list, dtype=np.float32)
        else:
            self.H = None  # Will be calculated or set later

        # Load rigid transform (optional)
        if "rigid_transform" in config:
            rt_list = config["rigid_transform"]
            if len(rt_list) not in [3, 4] or any(len(row) != 4 for row in rt_list):
                raise ValueError("Rigid transform data in YAML is not a 3x4 or 4x4 matrix")
            self.RT = np.array(rt_list, dtype=np.float32)
        else:
            self.RT = None

        # Load plane normal (optional)
        if "plane_normal" in config:
            plane_normal_list = config["plane_normal"]
            if len(plane_normal_list) != 3:
                raise ValueError("Plane normal data in YAML is not a 3-element vector")
            self.plane_normal = np.array(plane_normal_list, dtype=np.float32)
        else:
            self.plane_normal = None

        # Load plane distance (optional)
        if "plane_distance" in config:
            self.plane_distance = float(config["plane_distance"])
        else:
            self.plane_distance = None

    def reload_config(self):
        """
        Reload the configuration from the YAML file to reflect any updates.
        """
        self._load_config()
        print(f"Configuration reloaded from {self.config_path}")

    def camera_intrinsics(self):
        """
        Get camera intrinsics and its inverse as tensors.

        Returns:
            K: Camera intrinsic matrix.
            K_inv: Inverse of the camera intrinsic matrix.
        """
        if not hasattr(self, 'CAMERA_INTRINSICS'):
            raise ValueError("Camera intrinsics not loaded. Reload configuration.")
        fx = self.CAMERA_INTRINSICS["fx"]
        fy = self.CAMERA_INTRINSICS["fy"]
        cx = self.CAMERA_INTRINSICS["cx"]
        cy = self.CAMERA_INTRINSICS["cy"]

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        K_inv = np.linalg.inv(K)
        return K, K_inv
    
    def homography_matrix(self):
        """
        Get the homography matrix as a 3x3 NumPy array. Calculates it if not present.

        Returns:
            H (np.ndarray): The 3x3 homography matrix loaded from the YAML file or calculated.
        """
        if self.H is None:
            # Placeholder: Implement calculation using camera_intrinsics if possible
            # Example: You might need additional data (e.g., image points, world points)
            raise NotImplementedError("Homography calculation not implemented. Update config.yaml with 'homography' or provide calculation logic.")
        return self.H

    def rigid_transform(self):
        """
        Get the rigid transform matrix as a NumPy array (e.g., 3x4 or 4x4).

        Returns:
            RT (np.ndarray or None): The rigid transform matrix loaded from the YAML file, or None if not present.
        """
        return self.RT

    def plane_norm(self):
        """
        Get the plane normal vector as a 3x1 NumPy array.

        Returns:
            plane_normal (np.ndarray or None): The plane normal vector loaded from the YAML file, or None if not present.
        """
        return self.plane_normal

    def plane_dist(self):
        """
        Get the plane distance as a float.

        Returns:
            plane_distance (float or None): The plane distance loaded from the YAML file, or None if not present.
        """
        return self.plane_distance