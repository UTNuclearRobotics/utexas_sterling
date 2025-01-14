import cv2
import numpy as np
from robot_data_at_timestep import RobotDataAtTimestep
import os
from camera_intrinsics import CameraIntrinsics
from homography_from_chessboard import HomographyFromChessboardImage
from homography_utils import *
from robot_data_at_timestep import RobotDataAtTimestep
from tqdm import tqdm
from utils import *
from math import atan2, degrees


class BirdseyeCanvas:
    def __init__(self, homography_matrix, canvas_size=(2000, 2000), scale_factor=10, dsize=(100, 100)):
        """
        Initialize the BirdseyeCanvas object.

        :param homography_matrix: 3x3 matrix for bird's-eye-view transformation.
        :param canvas_size: Initial size of the canvas (width, height) in pixels.
        :param scale_factor: Number of pixels per real-world unit (e.g., pixels/meter).
        :param dsize: Size of the transformed bird's-eye image (width, height) in pixels.
        """
        self.homography_matrix = homography_matrix
        self.scale_factor = scale_factor
        self.dsize = dsize
        self.canvas = None  # Will be initialized with the first image
        self.coverage_mask = None
        self.canvas_center = None
        self.canvas_size = None

    def _extract_odometry(self, transformation_matrix):
        """
        Extract (x, y, theta) from the 4x4 transformation matrix.

        :param transformation_matrix: 4x4 matrix representing the robot's position and orientation.
        :return: (x, y, theta) in real-world units and radians.
        """
        x = transformation_matrix[0, 3]
        y = transformation_matrix[1, 3]
        theta = atan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
        return x, y, theta

    def _transform_position(self, x, y):
        """
        Convert real-world (x, y) coordinates to canvas coordinates.

        :param x: X coordinate in real-world units.
        :param y: Y coordinate in real-world units.
        :return: (canvas_x, canvas_y) in pixel coordinates.
        """
        canvas_x = int(self.canvas_center[0] + x * self.scale_factor)
        canvas_y = int(self.canvas_center[1] - y * self.scale_factor)
        return canvas_x, canvas_y

    def _expand_canvas(self, width_margin, height_margin):
        """
        Expand the canvas dynamically to fit new data.
        """
        new_width = self.canvas_size[0] + 2 * width_margin
        new_height = self.canvas_size[1] + 2 * height_margin
        new_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Copy old canvas into the new one
        new_canvas[height_margin:height_margin + self.canvas_size[1],
                   width_margin:width_margin + self.canvas_size[0]] = self.canvas

        self.canvas = new_canvas
        self.canvas_size = (new_width, new_height)
        self.canvas_center = (self.canvas_center[0] + width_margin, self.canvas_center[1] + height_margin)

    def check_and_expand_canvas(self, canvas_x, canvas_y, margin=100):
        """
        Check if the canvas needs expansion to accommodate the new image.
        """
        if (canvas_x < margin or canvas_y < margin or
                canvas_x > self.canvas_size[0] - margin or
                canvas_y > self.canvas_size[1] - margin):
            self._expand_canvas(margin, margin)

    def blend_images(self, canvas, image, top_left_x, top_left_y, alpha=0.99):
        """
        Blend the given image with the canvas at the specified position.
        """
        h, w = image.shape[:2]
        y1, y2 = max(0, top_left_y), min(top_left_y + h, canvas.shape[0])
        x1, x2 = max(0, top_left_x), min(top_left_x + w, canvas.shape[1])

        # Define regions of interest (ROI) on the image
        img_y1 = max(0, -top_left_y)
        img_y2 = img_y1 + (y2 - y1)
        img_x1 = max(0, -top_left_x)
        img_x2 = img_x1 + (x2 - x1)

        # Validate ROI dimensions
        if y1 >= y2 or x1 >= x2 or img_y1 >= img_y2 or img_x1 >= img_x2:
            # Skip blending if ROI is invalid (e.g., no overlap)
            return

        # Extract the valid regions
        roi_canvas = canvas[y1:y2, x1:x2]
        roi_image = image[img_y1:img_y2, img_x1:img_x2]

        # Create a mask for blending
        mask = (roi_image > 0).astype(np.float32)

        # Blend using vectorized operations
        roi_canvas[:] = roi_canvas * (1 - alpha * mask) + roi_image * (alpha * mask)


    def add_image(self, image, transformation_matrix):
        """
        Add a bird's-eye view of an image to the canvas.

        :param image: Input image.
        :param transformation_matrix: Transformation matrix for robot position and orientation.
        """
        x, y, theta = self._extract_odometry(transformation_matrix)

        # Warp perspective and crop bottom quarter
        birdseye_image = cv2.warpPerspective(image, self.homography_matrix, self.dsize)
        start_row = int(3 * birdseye_image.shape[0] / 4)
        birdseye_image = birdseye_image[start_row:, :]

        # Rotate the image
        rotation_matrix = cv2.getRotationMatrix2D(
            (birdseye_image.shape[1] // 2, birdseye_image.shape[0] // 2),
            theta * (180 / np.pi), 1.0
        )
        rotated_image = cv2.warpAffine(
            birdseye_image, rotation_matrix,
            (birdseye_image.shape[1], birdseye_image.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        # Initialize the canvas on the first image
        if self.canvas is None:
            h, w = rotated_image.shape[:2]
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.coverage_mask = np.zeros((h, w), dtype=np.uint8)
            self.canvas_size = (w, h)
            self.canvas_center = (w // 2, h // 2)

        canvas_x, canvas_y = self._transform_position(x, y)
        self.check_and_expand_canvas(canvas_x, canvas_y)

        # Place the image on the canvas
        top_left_x = canvas_x - rotated_image.shape[1] // 2
        top_left_y = canvas_y - rotated_image.shape[0] // 2
        self.blend_images(self.canvas, rotated_image, top_left_x, top_left_y)


    def get_canvas(self):
        """
        Get the current canvas with all placed images.

        :return: The canvas as a numpy array.
        """
        # Generate a coverage mask (non-zero pixels)
        coverage_mask = (self.canvas > 0).any(axis=-1).astype(np.uint8)

        # Find the bounding box of the area of interest
        y_coords, x_coords = np.where(coverage_mask)
        if y_coords.size == 0 or x_coords.size == 0:
            # No area of interest, return an empty canvas
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # Determine the bounding box
        top, bottom = y_coords.min(), y_coords.max()
        left, right = x_coords.min(), x_coords.max()

        # Crop the canvas to the bounding box
        cropped_canvas = self.canvas[top:bottom+1, left:right+1]

        # Optionally resize the cropped canvas
        resized_canvas = cv2.resize(cropped_canvas, (1920, 1080), interpolation=cv2.INTER_AREA)

        return resized_canvas


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)
    #H = np.linalg.inv(chessboard_homography.H)  # get_homography_image_to_model()
    H, dsize = chessboard_homography.plot_BEV_full(plot_BEV_full=False)
    RT = chessboard_homography.get_rigid_transform()
    plane_normal = chessboard_homography.get_plane_norm()
    plane_distance = chessboard_homography.get_plane_dist()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    robot_data = RobotDataAtTimestep(
        os.path.join(script_dir, "../bags/panther_ahg_courtyard/panther_ahg_courtyard.pkl")
    )

    # Initialize the BirdseyeCanvas object
    canvas = BirdseyeCanvas(H, canvas_size=None, scale_factor=700, dsize=dsize)

    for timestep in tqdm(range(0,1200), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)

        canvas.add_image(cur_image, cur_rt)

    final_canvas = canvas.get_canvas()

    # Display the canvas using OpenCV
    cv2.imshow("Birdseye Canvas", final_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
