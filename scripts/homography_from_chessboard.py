# from homography import *
from cam_calibration import CameraIntrinsics
from utils import *

class HomographyFromChessboardImage:
# class HomographyFromChessboardImage(Homography):
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
        self.pixel_width = self.corner_pixel_width()

        # Get model chessboard corners, cartesian NX2
        self.model_chessboard = compute_model_chessboard(cb_rows, cb_cols, self.pixel_width, center_at_zero=False)

        self.H, mask = cv2.findHomography(self.corners, self.model_chessboard, cv2.RANSAC)
        self.K, K_inv = CameraIntrinsics().get_camera_calibration_matrix()
        self.RT = self.decompose_homography(self.H, self.K)

        # Transform model chessboard points to image points
        self.transformed_model_corners = self.transform_points(
            self.model_chessboard.T, self.get_homography_model_to_image()
        )
        self.transformed_model_corners = self.transformed_model_corners.T.reshape(-1, 2).astype(np.float32)
        # print("transformed_model_corners:   ", transformed_model_corners)
        # print("corners:   ", corners)
        # print("Diff:    ", transformed_model_corners - corners)

    def get_homography_image_to_model(self):
        return self.H

    def get_homography_model_to_image(self):
        return np.linalg.inv(self.H)

    def get_rotation_translation_matrix(self):
        return self.RT

    def corner_pixel_width(self):
        """Calculate the maximum distance between two consecutive corners in each row of the chessboard."""
        # Sort corners by y value to group them by rows
        sorted_corners = sorted(self.corners, key=lambda x: x[1])

        # Split sorted_corners into rows
        interval = self.cb_cols
        rows = [sorted_corners[i * interval : (i + 1) * interval] for i in range(len(sorted_corners) // interval)]

        # Calculate distances between consecutive points in each row
        max_distance = 0
        for row in rows:
            row = sorted(row, key=lambda x: x[0])
            for i in range(len(row) - 1):
                distance = np.linalg.norm(np.array(row[i]) - np.array(row[i + 1]))
                max_distance = max(max_distance, distance)

        return max_distance

    def validate_homography(self):
        keepRunning = True
        renderTransformed = False
        while keepRunning:
            if renderTransformed:
                rend_image = draw_points(self.image, self.transformed_model_corners, color=(0, 255, 255))
            else:
                rend_image = draw_points(self.image, self.corners, color=(255, 0, 0))
            cv2.imshow("Chessboard", rend_image)
            key = cv2.waitKey(0)
            renderTransformed = not renderTransformed
            if key == 113:  # Hitting 'q' quits the program
                keepRunning = False
        exit(0)

    def transform_points(self, points, H):
        """Transform points using the homography matrix."""
        hom_points = cart_to_hom(points)
        transformed_points = H @ hom_points
        return hom_to_cart(transformed_points)

    def decompose_homography(self, H, K):
        """
        Decomposes a homography matrix H into a 4x4 transformation matrix RT
        using OpenCV's decomposeHomographyMat and selecting the valid decomposition.

        Args:
            H (np.ndarray): 3x3 homography matrix.
            K (np.ndarray): 3x3 intrinsic camera matrix.

        Returns:
            np.ndarray: 4x4 transformation matrix RT combining rotation and translation.
        """
        # Normalize the homography using the intrinsic matrix
        K_inv = np.linalg.inv(K)
        normalized_H = K_inv @ H

        # Decompose the homography matrix
        num_decompositions, rotations, translations, normals = cv2.decomposeHomographyMat(normalized_H, K)

        # Logic to select the correct decomposition
        best_index = -1
        max_z_translation = -np.inf  # Example criterion: largest positive translation in Z-axis
        for i in range(num_decompositions):
            # Ensure the plane normal points towards the camera (positive Z-axis)
            normal_z = normals[i][2]
            translation_z = translations[i][2]

            if normal_z > 0 and translation_z > max_z_translation:
                max_z_translation = translation_z
                best_index = i

        if best_index == -1:
            raise ValueError("No valid decomposition found.")

        # Use the selected decomposition
        R = rotations[best_index]
        t = translations[best_index].flatten()

        # Create the 4x4 transformation matrix
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = R
        RT[:3, 3] = t

        return RT

    def get_BEV_chessboard_region(self):
        image = self.image.copy()
        H = self.get_homography_image_to_model()
        dimensions = (int(self.pixel_width * (self.cb_cols - 1)), int(self.pixel_width * (self.cb_rows - 1)))
        warped_image = cv2.warpPerspective(image, H, dsize=dimensions)

        cv2.imshow("BEV", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)

    def plot_BEV_entire_image(self, image, K, H):
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