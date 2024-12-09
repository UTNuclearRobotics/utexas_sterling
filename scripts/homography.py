'''
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


class IMURTsByTimestamp

class ImagesByTimestamp

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

'''

import cv2
import numpy as np
import os
import torch

class Homography:
    def __init__(self, homography_tensor):
        self.homography_tensor = homography_tensor

class HomographyFromChessboardImage(Homography):
    def __init__(self, image, cb_rows, cb_cols):
        super().__init__(torch.eye(3))
        model_chessboard = compute_model_chessboard(cb_rows, cb_cols)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        chessboard_size = (cb_rows, cb_cols)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # self.draw_corner_image(image, ret, chessboard_size, corners)

        H, mask = cv2.findHomography(corners, model_chessboard, cv2.RANSAC)

        points_out = np.linalg.inv(H) @ cart_to_hom(model_chessboard.T)

        cart_pts_out = hom_to_cart(points_out)
        wonky_pts_out = cart_pts_out.T.reshape(-1, 1, 2)
        wonky_pts_out = np.array(wonky_pts_out, dtype=np.float32)
        print("wonky_pts_out: ", wonky_pts_out)

        #self.draw_corner_image(image, ret, chessboard_size, wonky_pts_out)
        self.draw_corner_image(image, ret, chessboard_size, corners)


    def draw_corner_image(self, image, ret, chessboard_size, corners):
        if ret:
            print("Chessboard corners found!")
            # Draw the corners on the image
            # print(corners)
            cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
            cv2.imshow('Chessboard Corners', image)
        else:
            cv2.imshow('Loaded Image', image)
            print("Chessboard corners not found.")

        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

def compute_model_chessboard(rows, cols):
    model_chessboard = np.zeros((rows * cols, 2), dtype=np.float32)
    midpoint_row = rows / 2
    midpoint_col = cols / 2
    for row in range(0,rows):
        for col in range(0,cols):
            model_chessboard[row * cols + col, 0] = (col + 0.5) - midpoint_col
            model_chessboard[row * cols + col, 1] = (row + 0.5) - midpoint_row
    return model_chessboard

def cart_to_hom(input_pts):
    row_of_ones = np.ones((1, input_pts.shape[1]))
    return np.vstack((input_pts, row_of_ones))
    
def hom_to_cart(input_pts):
    w = input_pts[-1, :]
    cart_pts = input_pts / w
    return cart_pts[:-1, :]

if __name__ == "__main__":
    model_chessboard = compute_model_chessboard(6, 8)
    print(model_chessboard)
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    image_dir = script_dir + "/homography/"
    path_to_image = image_dir + "raw_image.jpg"

    image = cv2.imread(path_to_image)
    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)

    # if ret:
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    #     corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
    #         criteria)

