import cv2
import numpy as np

from camera_intrinsics import CameraIntrinsics
from homography_utils import *
from utils import *


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
        # H = self.get_homography_model_to_image()
        # Transform model chessboard 2D points to image points
        self.transformed_model_chessboard_2d = self.transform_points(model_chessboard_2d.T, self.H)
        self.transformed_model_chessboard_2d = self.transformed_model_chessboard_2d.T.reshape(-1, 2).astype(np.float32)
        # print("transformed_model_chessboard_2d:   ", transformed_model_chessboard_2d)
        # print("corners:   ", corners)
        # print("Diff:    ", transformed_model_chessboard_2d - corners)

    def validate_chessboard_3d(self, model_chessboard_3d):
        RT = self.get_rigid_transform()
        K = self.get_camera_intrinsics()
        print("RT:   ", RT)
        print("K:   ", K)
        print("model_chessboard_3d:   ", model_chessboard_3d)

        # Apply the rigid transformation
        self.IEK = (K @ RT[:3] @ model_chessboard_3d.T)
        #self.transformed_model_chessboard_3d = (RT @ model_chessboard_3d.T)
        #print("applied rigid transform:   ", self.transformed_model_chessboard_3d)
        ideal_camera = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        #P = K @ ideal_camera
        self.model_cb_3d_to_2d = hom_to_cart(self.IEK)
        print("self.model_cb_3d_to_2d:  ", self.model_cb_3d_to_2d)

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

    def plot_BEV_full(self, image, K, H):
        return
        # TODO: Get BEV of entire image
        image = image.copy()
        height, width = image.shape[:2]

        # Define the corners of the original image
        img_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        # print("Image corners:   ", img_corners)

        # Compute the transformed corners
        # K_inv * RRT
        inv_H = np.linalg.inv(H)
        transformation_matrix = K @ inv_H
        transformed_corners = cv2.perspectiveTransform(np.array([img_corners]), transformation_matrix)[0]
        # print("Transformed image corners:   ", transformed_corners)

        # Calculate the bounding box of the transformed corners
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        # print("Translated image corners:   ", translated_corners)

        # Compute new dimensions
        new_width = int(np.ceil(max_x - min_x))
        new_height = int(np.ceil(max_y - min_y))

        # Adjust the transformation matrix to account for translation
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

        # Scale the translated corners to the desired width
        translated_corners = cv2.perspectiveTransform(np.array([transformed_corners]), translation_matrix)[0]
        scale_factor = width / new_width
        scaled_corners = translated_corners * scale_factor

        # Warp the image to get BEV
        combined_matrix = translation_matrix @ transformation_matrix
        warped_image = cv2.warpPerspective(image, transformation_matrix, dsize=(new_width, new_height))
        # Print the warped image dimensions
        warped_image = cv2.resize(warped_image, dsize=(width, int(new_height * (width / new_width))))
        warped_image = cv2.resize(warped_image, (width, height))
        cv2.polylines(warped_image, [scaled_corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the result
        cv2.imshow("Translated Corners", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)
