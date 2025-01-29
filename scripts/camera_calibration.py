import cv2 as cv
import glob
import numpy as np
from scipy.optimize import least_squares
# from scipy.spatial.transform import Rotation as R
import sys
import torch

from chess_board_renderer import *
from homography_utils import *
from model_chessboard import *
from utils import *

script_dir = os.path.dirname(os.path.abspath(__file__))
pyceres_location = script_dir + "../../sterling_env/lib/python3.10/site-packages/PyCeres/"
sys.path.insert(0, pyceres_location)

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

    for fname in images:
        img = cv.imread(fname)
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

    return imgpoints

            # Draw and display corners

    #cv.destroyAllWindows()

class CameraCalibration:
    def __init__(self, image_path_pattern, rows=6, cols=8, square_size=100):
        self.rows = rows
        self.cols = cols
        self.square_size = square_size
        grid_size=(self.cols, self.rows)
        self.imgpoints = find_image_points(grid_size,image_path_pattern)
        
        """
        Perform camera calibration using detected points.
        """
        if not self.imgpoints:
            raise ValueError("Object points or image points are empty. Run find_image_points() first.")

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

# class RigidTransformation:
#     def __init__(self, RT):
#           self.rt_mat = RT
#           self.rot_mat = R.from_matrix(RT[:3, :3])
#           self.rot_rod = R.from_matrix(self.rot_mat).as_rotvec()

class CameraIntrinsicMatrix:
    def __init__(self, alpha, beta, u0, v0):
        self.alpha = alpha
        self.beta = beta
        self.u0 = u0
        self.v0 = v0

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

def projection(k_mat_torch, r,t):
    r, _ = cv.Rodrigues(r)
    r_torch = torch.from_numpy(r).float()
    t_torch = torch.from_numpy(t).float().view(3, 1)
    RT = torch.cat([r_torch, t_torch], dim=1)
    # RT = torch.cat([RT, torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)], dim=0)
    # print("r_torch:   ", r_torch)
    # print("t_torch:   ", t_torch)
    # print("RT:   ", RT)
    return k_mat_torch @ RT
        
class ZhangNonlinear:
    def __init__(self, calibration_values):
        super().__init__()
        # , cb_rows, cb_cols, image_path_pattern
        # grid_size = (cb_rows, cb_cols)
        print("ZhangNonlinear()")
        # print(" Loading Images: ", image_path_pattern)
        # images = glob.glob(image_path_pattern)
        # criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        # model_chessboard = compute_model_chessboard_2d(cb_rows, cb_cols, 1, center_at_zero=True)

        self.calibration_values = calibration_values
        self.model_chessboard = ModelChessboard(
            calibration_values.rows, calibration_values.cols, calibration_values.square_size)
        self.num_images = len(calibration_values.imgpoints)
        
        self.K = CameraIntrinsicMatrix(
            calibration_values.mtx[0,0],
            calibration_values.mtx[1,1],
            calibration_values.mtx[0,2],
            calibration_values.mtx[1,2])
        
        print("num_images: ", len(self.num_images))

        # print("calibration.rvecs[1]:    ", calibration.rvecs[1])
        # print("calibration.tvecs[1]:    ", calibration.tvecs[1])

        rvecs = [calibration.rvecs[i].flatten() for i in range(len(calibration.rvecs))]
        tvecs = [calibration.tvecs[i].flatten() for i in range(len(calibration.tvecs))]
        params_initial = np.hstack([
            calibration_values.mtx[0,0],
            calibration_values.mtx[1,1],
            calibration_values.mtx[0,2],
            calibration_values.mtx[1,2],
            *rvecs,
            *tvecs])
        
        # print("params_initial:  ", params_initial)
        self.residuals(params_initial)


    def residuals(self, params):
        alpha, beta, u0, v0 = params[:4]
        rvecs = params[4:4 + 3*self.num_images].reshape(self.num_images, 3)
        tvecs = params[4 + 3*self.num_images:].reshape(self.num_images, 3)
        # print("rvecs[1]:    ", rvecs[1])
        # print("tvecs[1]:    ", tvecs[1])

        # for i in range(0, len(calibration_values.rvecs)):
        #     # print("rvec:    ", self.calibration_values.rvecs[i])
        #     # print("tvec:    ", self.calibration_values.tvecs[i])
        #     p = projection(self.K.intrinsic_matrix,
        #         self.calibration_values.rvecs[i],
        #         self.calibration_values.tvecs[i])
        #     projected_hom = p @ self.model_chessboard.cb_pts_3D
        #     projected_cartesian = projected_hom[:-1, :] / projected_hom[-1, :]
        #     img_cartesian = torch.Tensor(self.calibration_values.imgpoints[i]).squeeze(1).T
        #     # print("p:  ", p)
        #     # print("projected_hom:  ", projected_hom)
        #     print("projected_cartesian:  ", projected_cartesian)
        #     print("img_cartesian:  ", img_cartesian)
        #     print("minus:  ", projected_cartesian - img_cartesian)
        
        # self.set_num_residuals(calibration_values.rows * calibration_values.cols * )
        
        # print("K.intrinsic_matrix:  ", K.intrinsic_matrix)
        
        # K_inv = np.linalg.inv(K)

        # corner_list = []
        # h_list = []
        # rt_list = []
        # for fname in images:
        #     img = cv.imread(fname)
        #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #     ret, corners = cv.findChessboardCorners(gray, grid_size, None)

        #     if ret:
        #         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #         H, mask = cv2.findHomography(model_chessboard.cb_pts_2D_cart.T.numpy(), corners2, cv2.RANSAC)
        #         rt, _, _ = decompose_homography(H, K)
        #         corner_list.append(corners2)
        #         h_list.append(H)
        #         rt_list.append(rt)
        #         print("GOOD")
        #     else:
        #         print("BAD")

        

        # cbr = ChessboardRenderer()
        # while cbr.running:
        #     cbr.display_iteration(rt_list)
        # pygame.quit()        


# Example usage

if __name__ == "__main__":
    image_path_pattern='./homography/calibration_images/*.jpg'
    calibration = CameraCalibration(image_path_pattern=image_path_pattern)
    # calibration_values = calibration.calibrate_camera(image_path_pattern)

    print("Camera matrix:\n", calibration.mtx)
    #print("Distortion coefficients:\n", results["distortion_coefficients"])
    #print("Rotation vectors:\n", results["rotation_vectors"])
    #print("Translation vectors:\n", results["translation_vectors"])
    print("Mean error:\n", calibration.mean_error)

    zhang_nonlinear = ZhangNonlinear(calibration)
