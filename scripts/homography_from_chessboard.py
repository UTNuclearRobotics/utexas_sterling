import cv2
import numpy as np

from camera_intrinsics import CameraIntrinsics
from homography_utils import *
from utils import *
from scipy.optimize import minimize


class HomographyFromChessboardImage:
    def __init__(self, image, cb_rows, cb_cols):
        # super().__init__(torch.eye(3))
        self.image = image
        self.cb_rows = cb_rows
        self.cb_cols = cb_cols
        self.chessboard_size = (cb_rows, cb_cols)

        # Get image chessboard corners, cartesian NX2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
        self.corners = corners.reshape(-1, 2)
        self.cb_tile_width = int(self.chessboard_tile_width())

        # Get model chessboard corners, cartesian NX2
        model_chessboard_2d = compute_model_chessboard_2d(cb_rows, cb_cols, self.cb_tile_width, center_at_zero=True)

        self.H, mask = cv2.findHomography(model_chessboard_2d, self.corners, cv2.RANSAC)
        self.K, K_inv = CameraIntrinsics().get_camera_calibration_matrix()
        self.RT = decompose_homography(self.H, self.K)

        self.validate_chessboard_2d(model_chessboard_2d)

        # Transform model chessboard 3D points to image points
        model_chessboard_3d = compute_model_chessboard_3d(cb_rows, cb_cols, self.cb_tile_width, center_at_zero=True)
        self.validate_chessboard_3d(model_chessboard_3d)

    def validate_chessboard_2d(self, model_chessboard_2d):
        # Transform model chessboard 2D points to image points
        self.transformed_model_chessboard_2d = self.transform_points(model_chessboard_2d.T, self.H)
        self.transformed_model_chessboard_2d = self.transformed_model_chessboard_2d.T.reshape(-1, 2).astype(np.float32)

    def validate_chessboard_3d(self, model_chessboard_3d):
        RT = self.get_rigid_transform()
        K = self.get_camera_intrinsics()
        self.IEK = K @ RT[:3] @ model_chessboard_3d.T
        self.model_cb_3d_to_2d = hom_to_cart(self.IEK)
        return self.model_cb_3d_to_2d

    def get_rigid_transform(self):
        return self.RT

    def get_camera_intrinsics(self):
        return self.K

    def chessboard_tile_width(self):
        """Calculate the maximum distance between two consecutive corners in each row of the chessboard."""
        # Sort corners by y value to group them by rows
        sorted_corners = sorted(self.corners, key=lambda x: x[1])

        # Split sorted_corners into rows
        interval = self.cb_cols
        rows = [sorted_corners[i * interval : (i + 1) * interval] for i in range(len(sorted_corners) // interval)]

        # Calculate distances between consecutive points in each row
        cb_tile_width = 0
        for row in rows:
            row.sort(key=lambda x: x[0])
            for i in range(len(row) - 1):
                distance = np.linalg.norm(np.array(row[i]) - np.array(row[i + 1]))
                cb_tile_width = max(cb_tile_width, distance)

        return cb_tile_width

    def validate_homography(self):
        keepRunning = True
        counter = 0
        cv2.namedWindow("Chessboard")
        while keepRunning:
            match counter % 3:
                case 0:
                    rend_image = draw_points(self.image, self.corners, color=(255, 0, 0))
                    cv2.setWindowTitle("Chessboard", "Original corners")
                case 1:
                    rend_image = draw_points(self.image, self.transformed_model_chessboard_2d, color=(0, 255, 0))
                    cv2.setWindowTitle("Chessboard", "Transformed 2D model chessboard corners")
                case 2:
                    rend_image = draw_points(self.image, self.model_cb_3d_to_2d.T, color=(0, 0, 255))
                    cv2.setWindowTitle("Chessboard", "Transformed 3D model chessboard corners")
            counter += 1
            cv2.imshow("Chessboard", rend_image)
            key = cv2.waitKey(0)
            if key == 113:  # Hitting 'q' quits the program
                keepRunning = False
        exit(0)

    def transform_points(self, points, H):
        """Transform points using the homography matrix."""
        hom_points = cart_to_hom(points)
        transformed_points = H @ hom_points
        return hom_to_cart(transformed_points)

    def plot_BEV_chessboard(self):
        image = self.image.copy()
        model_chessboard_2d_centered = compute_model_chessboard_2d(
            self.cb_rows, self.cb_cols, self.cb_tile_width, center_at_zero=False
        )
        H, mask = cv2.findHomography(model_chessboard_2d_centered, self.corners, cv2.RANSAC)
        dimensions = (int(self.cb_tile_width * (self.cb_cols - 1)), int(self.cb_tile_width * (self.cb_rows - 1)))
        warped_image = cv2.warpPerspective(image, np.linalg.inv(H), dsize=dimensions)

        cv2.imshow("BEV", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)

    def plot_BEV_full(self):
        """
        Notes:
            Optimize a new 3d rectangle (4 corners) to fit full image, maximum possible seen
            Non-linear optimizer in Python
            Functions to pass:
            - generate a rectangle in 3D, centered at 0, rotated a little
            - x1, y1, and theta to take up as much of the image as possible
        """

        def objective(params, RT, K, image):
            """
            params: [theta, scalar]
            Returns the absolute difference between the rectangle's projected width
            and image_width.
            """
            theta, scalar = params
            image_height, image_width = image.shape[:2]

            # 1) Build your 3D rectangle (homogeneous). You may need to pass theta in:
            model_rect_3d_hom = compute_model_rectangle_3d_hom(theta, scalar)

            # 2) Apply RT & K
            model_rect_3d_applied_RT = K @ RT[:3] @ model_rect_3d_hom.T

            # 3) Convert to 2D
            model_rect_3d_to_2d = hom_to_cart(model_rect_3d_applied_RT)

            # Check bounds
            # xs, ys = model_rect_3d_to_2d[:, 0], model_rect_3d_to_2d[:, 1]
            # if np.any(xs < 0) or np.any(xs > image_width) or np.any(ys < 0) or np.any(ys > image_height):
            # # Return large penalty if out of frame
            #     return 1e6

            # 4) Measure width of bottom edge
            bottom_points = model_rect_3d_to_2d.T[-2:]  # last two corners
            width = np.linalg.norm(bottom_points[0] - bottom_points[1])

            # Return difference from desired image width
            return abs(width - image_width)

        RT = self.get_rigid_transform()
        K = self.get_camera_intrinsics()

        result = minimize(
            objective,
            x0=(0.0, 50.0),  # [theta, scalar]
            args=(RT, K, self.image),
            method="Nelder-Mead",
        )
        theta, scalar = result.x
        print(f"Theta: {theta}, Scalar: {scalar}")

        # Show optimized rectangle corners on image
        model_rect_3d_hom = compute_model_rectangle_3d_hom(theta, scalar)
        model_rect_3d_applied_RT = K @ RT[:3] @ model_rect_3d_hom.T
        model_rect_3d_to_2d = hom_to_cart(model_rect_3d_applied_RT)

        # Make top left or model rectangle (0,0) so you can view it in warp perspective
        model_rect_2d = model_rect_3d_hom[:, :2]
        model_rect_2d -= model_rect_2d.min(axis=0)

        # Warp the image
        H, mask = cv2.findHomography(model_rect_2d, model_rect_3d_to_2d.T, cv2.RANSAC)
        image_height, image_width = self.image.shape[:2]
        warped_image = cv2.warpPerspective(self.image, np.linalg.inv(H), dsize=(image_width, image_height))

        keepRunning = True
        counter = 0
        cv2.namedWindow("Full BEV")
        while keepRunning:
            match counter % 2:
                case 0:
                    rend_image = draw_points(self.image, model_rect_3d_to_2d.T, color=(255, 255, 0))
                    cv2.setWindowTitle("Full BEV", "Rectangle corners")
                case 1:
                    rend_image = warped_image
                    cv2.setWindowTitle("Full BEV", "Warped perspective")
            counter += 1
            cv2.imshow("Full BEV", rend_image)
            key = cv2.waitKey(0)
            if key == 113:  # Hitting 'q' quits the program
                keepRunning = False
        exit(0)
