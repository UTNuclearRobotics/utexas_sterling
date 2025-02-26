import numpy as np
import os
import yaml

class HomographyMatrix:
    def __init__(self, 
                 config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "homography", "homography.yaml")):
        """
        Initialize the HomographyMatrix class by loading the homography, rigid transform, plane normal,
        and plane distance from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file containing the homography data.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find YAML file at {config_path}")
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            
            # Load homography (required)
            if "homography" not in config:
                raise ValueError("YAML file does not contain a 'homography' key")
            homography_list = config["homography"]
            if len(homography_list) != 3 or any(len(row) != 3 for row in homography_list):
                raise ValueError("Homography data in YAML is not a 3x3 matrix")
            self.H = np.array(homography_list, dtype=np.float32)

            # Load rigid transform (optional, with validation)
            if "rigid_transform" in config:
                rt_list = config["rigid_transform"]
                # Assuming RT is 3x4 or 4x4; adjust validation as needed
                if len(rt_list) not in [3, 4] or any(len(row) != 4 for row in rt_list):
                    raise ValueError("Rigid transform data in YAML is not a 3x4 or 4x4 matrix")
                self.RT = np.array(rt_list, dtype=np.float32)
            else:
                self.RT = None  # Set to None if not present

            # Load plane normal (optional, with validation)
            if "plane_normal" in config:
                plane_normal_list = config["plane_normal"]
                if len(plane_normal_list) != 3:
                    raise ValueError("Plane normal data in YAML is not a 3-element vector")
                self.plane_normal = np.array(plane_normal_list, dtype=np.float32)
            else:
                self.plane_normal = None  # Set to None if not present

            # Load plane distance (optional, with validation)
            if "plane_distance" in config:
                self.plane_distance = float(config["plane_distance"])
            else:
                self.plane_distance = None  # Set to None if not present

    def get_homography_matrix(self):
        """
        Get the homography matrix as a 3x3 NumPy array.

        Returns:
            H (np.ndarray): The 3x3 homography matrix loaded from the YAML file.
        """
        return self.H

    def get_rigid_transform(self):
        """
        Get the rigid transform matrix as a NumPy array (e.g., 3x4 or 4x4).

        Returns:
            RT (np.ndarray or None): The rigid transform matrix loaded from the YAML file, or None if not present.
        """
        return self.RT

    def get_plane_normal(self):
        """
        Get the plane normal vector as a 3x1 NumPy array.

        Returns:
            plane_normal (np.ndarray or None): The plane normal vector loaded from the YAML file, or None if not present.
        """
        return self.plane_normal

    def get_plane_distance(self):
        """
        Get the plane distance as a float.

        Returns:
            plane_distance (float or None): The plane distance loaded from the YAML file, or None if not present.
        """
        return self.plane_distance