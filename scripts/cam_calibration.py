import cv2 as cv
import glob
import numpy as np
import os
import torch
import yaml

#   Squares on Justin's big chessboard are 100mm
def prepare_object_points(grid_size, square_width = 100):
    """
    Prepare object points like (0,0,0), (1,0,0), ..., (grid_width-1, grid_height-1, 0)
    """
    grid_width, grid_height = grid_size
    objp = torch.zeros((grid_width * grid_height, 3), dtype=torch.float32).numpy()
    objp[:, :2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1, 2)
    objp = objp * square_width
    return objp

def find_image_points(grid_size, objp, image_path_pattern):
    """
    Detect corners in chessboard images and refine the points.
    """
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

    print("find_image_points()")
    print(" Loading Images: ", image_path_pattern)
    images = glob.glob(image_path_pattern)

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, grid_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(img, grid_size, corners2, ret)
            cv.imshow('Detected Corners', img)
            cv.waitKey(1)
            print("GOOD")
        else:
            print("BAD")

    return objpoints, imgpoints

            # Draw and display corners

    #cv.destroyAllWindows()

class CameraIntrinsics:
    def __init__(self, \
        config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "homography", "camera_config.yaml")):
        # Load the configuration from the YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            self.CAMERA_INTRINSICS = config["camera_intrinsics"]
            # self.CAMERA_IMU_TRANSFORM = config["camera_imu_transform"]

    def get_camera_calibration_matrix(self):
        """
        Get camera intrinsics and its inverse as a tensors.
        Returns:
            K: Camera intrinsic matrix.
            K_inv: Inverse of the camera intrinsic matrix.
        """

        fx = self.CAMERA_INTRINSICS["fx"]
        fy = self.CAMERA_INTRINSICS["fy"]
        cx = self.CAMERA_INTRINSICS["cx"]
        cy = self.CAMERA_INTRINSICS["cy"]

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        K_inv = np.linalg.inv(K)
        return K, K_inv

class CameraCalibration:
    def __init__(self, image_path_pattern, grid_size=(8, 6)):
        objp = prepare_object_points(grid_size)
        self.objpoints, self.imgpoints = \
            find_image_points(grid_size, objp, image_path_pattern)

    def calibrate_camera(self, image_path_pattern):
        """
        Perform camera calibration using detected points.
        """
        if not self.objpoints or not self.imgpoints:
            raise ValueError("Object points or image points are empty. Run find_image_points() first.")

        # Use the shape of the last processed image for calibration
        h, w = cv.imread(glob.glob(image_path_pattern)[-1]).shape[:2]

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, (w, h), None, None
        )

        mean_error = self.calculate_reprojection_error(rvecs, tvecs, mtx, dist)

        return {
            "ret": ret,
            "camera_matrix": mtx,
            "distortion_coefficients": dist,
            "rotation_vectors": rvecs,
            "translation_vectors": tvecs,
            "mean_error": mean_error,
        }
    
    def calculate_reprojection_error(self, rvecs, tvecs, mtx, dist):
        """
        Calculate the mean reprojection error for the calibration.
        """
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        return mean_error / len(self.objpoints)
    
class MetricCalibration:
    def __init__(self, camera_matrix, image_path_pattern):
        grid_size=(8, 6)
        objp = prepare_object_points(grid_size)
        objpoints, imgpoints_list = \
            find_image_points(grid_size, objp, image_path_pattern)
        camera_matrix = torch.tensor(camera_matrix)
        objpoints = torch.tensor(objpoints)
        imgpoints = torch.tensor(imgpoints_list)
        imgpoints = imgpoints.squeeze(-2)

        # for image_cb in imgpoints_list:self.
        #     H, mask = cv2.findHomography(model_chessboard, corners, cv2.RANSAC)
        #     homography = HomographyFromChessboardImage(image_cb, 8, 6)
        #     print(homography)

        print(camera_matrix.shape, " ", objpoints.shape, " ", imgpoints.shape)

def render3DPoints(in_pts):
        


# Example usage

if __name__ == "__main__":
    image_path_pattern='./homography/calibration_images/*.jpg'
    calibration = CameraCalibration(image_path_pattern=image_path_pattern)
    calibration_values = calibration.calibrate_camera(image_path_pattern)

    print("Camera matrix:\n", calibration_values["camera_matrix"])
    #print("Distortion coefficients:\n", results["distortion_coefficients"])
    #print("Rotation vectors:\n", results["rotation_vectors"])
    #print("Translation vectors:\n", results["translation_vectors"])
    print("Mean error:\n", calibration_values["mean_error"])

    metric_calibration = MetricCalibration(calibration_values["camera_matrix"], image_path_pattern)