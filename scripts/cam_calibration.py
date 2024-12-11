import torch
import cv2 as cv
import glob
import numpy as np

class CameraCalibration:
    def __init__(self, image_path_pattern, grid_size=(8, 6), termination_criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)):
        self.image_path_pattern = image_path_pattern
        self.grid_size = grid_size
        self.criteria = termination_criteria
        self.objpoints = []  # 3D points in real-world space
        self.imgpoints = []  # 2D points in image plane
        self.objp = self.prepare_object_points()

    def prepare_object_points(self):
        """
        Prepare object points like (0,0,0), (1,0,0), ..., (grid_width-1, grid_height-1, 0)
        """
        grid_width, grid_height = self.grid_size
        objp = torch.zeros((grid_width * grid_height, 3), dtype=torch.float32).numpy()
        objp[:, :2] = np.mgrid[0:grid_width, 0:grid_height].T.reshape(-1, 2)
        return objp

    def find_image_points(self):
        """
        Detect corners in chessboard images and refine the points.
        """
        images = glob.glob(self.image_path_pattern)

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, self.grid_size, None)

            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)

                # Draw and display corners
                #cv.drawChessboardCorners(img, self.grid_size, corners2, ret)
                #cv.imshow('Detected Corners', img)
                #cv.waitKey(500)

        #cv.destroyAllWindows()

    def calibrate_camera(self):
        """
        Perform camera calibration using detected points.
        """
        if not self.objpoints or not self.imgpoints:
            raise ValueError("Object points or image points are empty. Run find_image_points() first.")

        # Use the shape of the last processed image for calibration
        h, w = cv.imread(glob.glob(self.image_path_pattern)[-1]).shape[:2]

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

# Example usage

if __name__ == "__main__":
    calibration = CameraCalibration(image_path_pattern='./scripts/homography/calibration_images/*.jpg')
    calibration.find_image_points()
    results = calibration.calibrate_camera()

    print("Camera matrix:\n", results["camera_matrix"])
    #print("Distortion coefficients:\n", results["distortion_coefficients"])
    #print("Rotation vectors:\n", results["rotation_vectors"])
    #print("Translation vectors:\n", results["translation_vectors"])
    print("Mean error:\n", results["mean_error"])