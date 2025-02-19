import cv2 as cv
import glob
import numpy as np
from pathlib import Path
import pickle
from scipy.optimize import least_squares
import sys
import torch

from chess_board_renderer import *
from homography_utils import *
from model_chessboard import *
from utils import *

'''
 [[759.85478155   0.         633.28655077]
 [  0.         760.95738925 357.70398778]
 [  0.           0.           1.        ]]
'''

def find_image_points(grid_size,image_path_pattern):
    """
    Detect corners in chessboard images and refine the points.
    """
    imgpoints = []  # 2D points in image plane
    criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

    print("find_image_points()")
    print(" Loading Images: ", image_path_pattern)
    images = glob.glob(image_path_pattern)
    w = 0
    h = 0

    for fname in images:
        img = cv.imread(fname)
        # print(img.shape)
        h = img.shape[0]
        w = img.shape[1]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, grid_size, None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # cv.drawChessboardCorners(img, grid_size, corners2, ret)
            # cv.imshow('Detected Corners', img)
            # cv.waitKey(1)
            # print("GOOD")
        # else:
        #     print("BAD")

    return imgpoints, h, w

            # Draw and display corners

    #cv.destroyAllWindows()

class CVBasedCalibrator:
    def __init__(self, image_path_pattern, rows=6, cols=8, square_size=100):
        self.rows = rows
        self.cols = cols
        self.square_size = square_size
        grid_size=(self.cols, self.rows)
        self.imgpoints, self.img_height, self.img_width = find_image_points(grid_size,image_path_pattern)
        # Ensure self.imgpoints is a NumPy array of shape [N, 2]
        self.imgpoints = np.array(self.imgpoints, dtype=np.float32)  # Convert list to NumPy array

        # Ensure the shape is [N, 2] before converting to a tensor
        if len(self.imgpoints.shape) == 3:  # [N, 1, 2] -> [N, 2]
            self.imgpoints = self.imgpoints.squeeze(1)

        # Convert to PyTorch tensor correctly
        self.img_pts_tensor = torch.tensor(self.imgpoints)  # Remove squeeze(2)
        

        # Use the shape of the last processed image for calibration
        self.h, self.w = cv.imread(glob.glob(image_path_pattern)[-1]).shape[:2]

        self.model_chessboard = ModelChessboard(rows, cols, square_size)
        # print("self.model_chessboard.cb_pts_3D: ", self.model_chessboard.cb_pts_3D)
        # print("self.model_chessboard.cb_pts_3D_cart: ", self.model_chessboard.cb_pts_3D_cart)
        # objpoints = []
        objpoints = []
        for i in range(0, len(self.imgpoints)):
            objpoints.append(self.model_chessboard.cb_pts_3D_cart.T.numpy())
            # objpoints.append(objp)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            objpoints, self.imgpoints, (self.w, self.h), None, None
        )

        mean_error = 0
        for i in range(len(self.imgpoints)):
            imgpoints2, _ = cv.projectPoints(
                self.model_chessboard.cb_pts_3D_cart.T.numpy(), self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        self.mean_error = mean_error / len(self.imgpoints)

class CameraCalibration:
    def __init__(self, alpha, beta, u0, v0, distortion = [0.0, 0.0, 0.0, 0.0, 0.0], scale = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.u0 = u0
        self.v0 = v0
        self.distortion = distortion
        self.scale = scale

        self.intrinsic_matrix = torch.tensor([
            [self.alpha,    0.0, self.u0],
            [0.0,    self.beta, self.v0],
            [0.0,      0.0,     1.0]
        ], dtype=torch.float32)
    
    def adjust_focal_length(self, f):
        self.alpha = f
        self.beta = f
        self.intrinsic_matrix[0, 0] = f
        self.intrinsic_matrix[1, 1] = f

    def adjust_alpha_beta(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.intrinsic_matrix[0, 0] = alpha
        self.intrinsic_matrix[1, 1] = beta

    def adjust_alpha_beta_u0_v0(self, alpha, beta, u0, v0):
        self.alpha = alpha
        self.beta = beta
        self.u0 = u0
        self.v0 = v0
        self.intrinsic_matrix[0, 0] = alpha
        self.intrinsic_matrix[1, 1] = beta
        self.intrinsic_matrix[0, 2] = u0
        self.intrinsic_matrix[1, 2] = v0

    def adjust_distortion(self, distortion):
        self.distortion = distortion

    def undistort(self, src_img):
        return cv.undistort(
            src_img, self.intrinsic_matrix.numpy(), self.distortion)

def projection(k_mat_torch, r,t):
    r, _ = cv.Rodrigues(r)
    r_torch = torch.from_numpy(r).float()
    t_torch = torch.from_numpy(t).float().view(3, 1)
    RT = torch.cat([r_torch, t_torch], dim=1)
    # RT = torch.cat([RT, torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)], dim=0)
    return k_mat_torch @ RT
        
class ZhangNonlinear:
    def __init__(self, calibration_values):
        super().__init__()
        print("ZhangNonlinear()")

        self.calibration_values = calibration_values
        self.model_chessboard = ModelChessboard(
            calibration_values.rows, calibration_values.cols, calibration_values.square_size)
        self.num_images = len(calibration_values.imgpoints)
        self.num_pts = calibration_values.rows * calibration_values.cols
        
        #self.calibration_values.dist
        distortion = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        self.K = CameraCalibration(
            calibration_values.mtx[0,0],
            calibration_values.mtx[1,1],
            calibration_values.mtx[0,2],
            calibration_values.mtx[1,2],
            distortion)
        
        print("num_images: ", self.num_images)
        print("Image Dimensions:    ", self.calibration_values.img_height, "   ", self.calibration_values.img_width)
        print("DISTORTION COEFFICIENTS: ", self.calibration_values.dist)

        rvecs = [calibration.rvecs[i].flatten() for i in range(len(calibration.rvecs))]
        tvecs = [calibration.tvecs[i].flatten() for i in range(len(calibration.tvecs))]

        result = least_squares(
            self.residuals_alpha_beta_u0_v0_distortion_2_radial,
            self.init_params_alpha_beta_u0_v0_distortion_2_radial(rvecs, tvecs),
            method='trf', verbose=2, xtol=1e-15)
        rvecs, tvecs = self.parse_result_alpha_beta_u0_v0_distortion_2_radial(result)

        result = least_squares(
            self.residuals_f_u0_v0_keep_distortion,
            self.init_params_f_u0_v0_keep_distortion(rvecs, tvecs),
            method='trf', verbose=2, xtol=1e-15)
        rvecs, tvecs = self.parse_result_f_u0_v0_keep_distortion(result)

        #Move u0, v0 to middle of image and retry
        init_params =  self.init_params_alpha_beta_u0_v0_distortion_2_radial(rvecs, tvecs)
        init_params[3] = self.calibration_values.img_width / 2.0
        init_params[4] = self.calibration_values.img_height / 2.0
        result = least_squares(
            self.residuals_alpha_beta_u0_v0_distortion_2_radial,
            init_params,
            method='trf', verbose=2, xtol=1e-15)
        rvecs, tvecs = self.parse_result_alpha_beta_u0_v0_distortion_2_radial(result)
        

    def parse_result_alpha_beta_u0_v0_distortion_2_radial(self, result):
        print("result:  ", result)

        alpha, beta, u0, v0, k1, k2 = result.x[:6]
        distortion = np.array([[k1, k2, 0.0, 0.0, 0.0]])
        self.K.adjust_alpha_beta_u0_v0(alpha, beta, u0, v0)
        self.K.adjust_distortion(distortion)
        print("alpha, beta, u0, v0: ", alpha, beta, u0, v0)
        print("distortion: ", self.K.distortion)

        rvecs = result.x[6:6 + 3*self.num_images].reshape(self.num_images, 3)
        tvecs = result.x[6 + 3*self.num_images:].reshape(self.num_images, 3)
        return rvecs, tvecs

    def init_params_alpha_beta_u0_v0_distortion_2_radial(self, rvecs, tvecs):
        return np.hstack([
            self.K.intrinsic_matrix[0,0], self.K.intrinsic_matrix[1,1],
            self.K.intrinsic_matrix[0,2], self.K.intrinsic_matrix[1,2],
            self.K.distortion[0,0], self.K.distortion[0,1],
            *rvecs, *tvecs])

    def residuals_alpha_beta_u0_v0_distortion_2_radial(self, params):
        alpha, beta, u0, v0, k1, k2 = params[:6]
        distortion = np.array([k1, k2, 0.0, 0.0, 0.0])
        self.K.adjust_alpha_beta_u0_v0(alpha, beta, u0, v0)
        rvecs = params[6:6 + 3*self.num_images].reshape(self.num_images, 3)
        tvecs = params[6 + 3*self.num_images:].reshape(self.num_images, 3)
        return self.compute_pts_diff(rvecs, tvecs, distortion)

    def parse_result_f_u0_v0_keep_distortion(self, result):
        print("result:  ", result)
        f, u0, v0 = result.x[:3]
        self.K.adjust_alpha_beta_u0_v0(f, f, u0, v0)
        print("f, u0, v0: ", f, f, u0, v0)
        rvecs = result.x[3:3 + 3*self.num_images].reshape(self.num_images, 3)
        tvecs = result.x[3 + 3*self.num_images:].reshape(self.num_images, 3)
        return rvecs, tvecs

    def init_params_f_u0_v0_keep_distortion(self, rvecs, tvecs):
        return np.hstack([
            (self.K.intrinsic_matrix[0,0] + self.K.intrinsic_matrix[1,1]) / 2.0,
            self.K.intrinsic_matrix[0,2], self.K.intrinsic_matrix[1,2],
            *rvecs, *tvecs])

    def residuals_f_u0_v0_keep_distortion(self, params):
        f, u0, v0 = params[:3]
        distortion = self.K.distortion
        self.K.adjust_alpha_beta_u0_v0(f, f, u0, v0)
        rvecs = params[3:3 + 3*self.num_images].reshape(self.num_images, 3)
        tvecs = params[3 + 3*self.num_images:].reshape(self.num_images, 3)
        return self.compute_pts_diff(rvecs, tvecs, distortion)

    def compute_pts_diff(self, rvecs, tvecs, distortion):
        pts_diff = torch.zeros(2, self.num_images * self.num_pts)
        for i in range(self.num_images):
            pts_2d_cart, _ = \
                cv.projectPoints(
                    self.model_chessboard.cb_pts_3D_cart.numpy(),
                    rvecs[i], tvecs[i], self.K.intrinsic_matrix.numpy(),
                    distortion
                    )
            pts_2d_cart = torch.tensor(pts_2d_cart).squeeze(1).T
            # print("pts_2d_cart: ", pts_2d_cart)
            image_tensor = self.calibration_values.img_pts_tensor[i].squeeze(1).T
            # print("image_tensor.shape: ", image_tensor.shape)
            # print("image_tensor: ", image_tensor)
            pts_diff[:, i * self.num_pts:(i + 1) * self.num_pts] = \
                pts_2d_cart - image_tensor
        pts_diff = pts_diff.flatten()
        return pts_diff.numpy()

def load_calibration_pickle():
    config_dir = Path(__file__).resolve().parent.parent / 'config'
    pickle_path = config_dir / 'camera_calibration.pkl'
    with open(pickle_path, 'rb') as file:
        loaded_instance = pickle.load(file)
    return loaded_instance


if __name__ == "__main__":
    image_path_pattern='./homography/calibration_images/*.jpg'
    calibration = CVBasedCalibrator(image_path_pattern=image_path_pattern)
    # calibration_values = calibration.calibrate_camera(image_path_pattern)

    print("Camera matrix:\n", calibration.mtx)
    #print("Distortion coefficients:\n", results["distortion_coefficients"])
    #print("Rotation vectors:\n", results["rotation_vectors"])
    #print("Translation vectors:\n", results["translation_vectors"])
    print("Mean error:\n", calibration.mean_error)

    zhang_nonlinear = ZhangNonlinear(calibration)

    config_dir = Path(__file__).resolve().parent.parent / 'config'
    pickle_path = config_dir / 'camera_calibration.pkl'
    with open(pickle_path, 'wb') as file:
        pickle.dump(zhang_nonlinear, file)

    images = glob.glob(image_path_pattern)
    img = cv.imread(images[0])
    out_img = zhang_nonlinear.K.undistort(img)

    cv.imshow('Original', img)
    cv.imshow('Undistorted', out_img)
    cv.waitKey(0)
