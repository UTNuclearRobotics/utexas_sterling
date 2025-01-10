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
        self.canvas_size = canvas_size
        self.scale_factor = scale_factor
        self.dsize = dsize

        # Initialize the canvas and coverage mask
        self.canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        self.coverage_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
        self.canvas_center = (canvas_size[0] // 2, canvas_size[1] // 2)

    def _extract_odometry(self, transformation_matrix):
        """
        Extract (x, y, theta) from the 4x4 transformation matrix.

        :param transformation_matrix: 4x4 matrix representing the robot's position and orientation.
        :return: (x, y, theta) in real-world units and radians.
        """
        x = transformation_matrix[0, 3]
        y = transformation_matrix[1, 3]
        # Extract yaw (2D rotation)
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
        # Negative for image coordinates
        canvas_y = int(self.canvas_center[1] - y * self.scale_factor)
        return canvas_x, canvas_y
    
    def blend_images(self, canvas, image, top_left_x, top_left_y, alpha=0.99):
        """
        Blend the given image with the canvas at the specified position.
        
        :param canvas: The canvas to blend the image onto.
        :param image: The image to blend.
        :param top_left_x: The x-coordinate of the top-left corner where the image should be placed.
        :param top_left_y: The y-coordinate of the top-left corner where the image should be placed.
        :param alpha: The blending factor (0.0 = canvas only, 1.0 = image only).
        """
        h, w = image.shape[:2]
        for y in range(h):
            for x in range(w):
                if (
                    0 <= top_left_y + y < canvas.shape[0] and
                    0 <= top_left_x + x < canvas.shape[1]
                ):
                    canvas_pixel = canvas[top_left_y + y, top_left_x + x]
                    image_pixel = image[y, x]
                    # Perform blending only for non-black pixels of the image
                    if np.any(image_pixel > 0):
                        blended_pixel = (1 - alpha) * canvas_pixel + alpha * image_pixel
                        canvas[top_left_y + y, top_left_x + x] = blended_pixel

    def _expand_canvas(self, width_margin, height_margin):
        """
        Expand the canvas dynamically to fit new data.

        :param width_margin: Additional width required.
        :param height_margin: Additional height required.
        """
        new_width = self.canvas_size[0] + 2 * width_margin
        new_height = self.canvas_size[1] + 2 * height_margin
        new_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Copy old canvas to the new center
        new_canvas[height_margin:height_margin + self.canvas_size[1],
                    width_margin:width_margin + self.canvas_size[0]] = self.canvas

        # Update canvas attributes
        self.canvas = new_canvas
        self.canvas_size = (new_width, new_height)
        self.canvas_center = (self.canvas_center[0] + width_margin, self.canvas_center[1] + height_margin)
    
    def check_and_expand_canvas(self, canvas_x, canvas_y, margin=100):
        if (canvas_x < margin or canvas_y < margin or
            canvas_x > self.canvas_size[0] - margin or
            canvas_y > self.canvas_size[1] - margin):
            self._expand_canvas(margin, margin)

    def update_coverage_mask(self, top_left_x, top_left_y, image):
        h, w = image.shape[:2]
        self.coverage_mask[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = 1

    def add_image(self, image, transformation_matrix):
        x, y, theta = self._extract_odometry(transformation_matrix)

        birdseye_image = cv2.warpPerspective(image, self.homography_matrix, self.dsize)
        #birdseye_image = cv2.resize(birdseye_image, (256, 256))

        # Rotate and pad
        pad_size = max(birdseye_image.shape[0], birdseye_image.shape[1])
        padded_image = cv2.copyMakeBorder(
            birdseye_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        rotation_matrix = cv2.getRotationMatrix2D(
            (padded_image.shape[1] // 2, padded_image.shape[0] // 2), theta * (180 / np.pi), 1.0
        )
        rotated_image = cv2.warpAffine(
            padded_image, rotation_matrix, (padded_image.shape[1], padded_image.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        h, w = birdseye_image.shape[:2]
        crop_x_start = (rotated_image.shape[1] - w) // 2
        crop_y_start = (rotated_image.shape[0] - h) // 2
        rotated_image = rotated_image[crop_y_start:crop_y_start + h, crop_x_start:crop_x_start + w]

        canvas_x, canvas_y = self._transform_position(x, y)

        self.check_and_expand_canvas(canvas_x, canvas_y)

        top_left_x = canvas_x - w // 2
        top_left_y = canvas_y - h // 2

        self.blend_images(self.canvas, rotated_image, top_left_x, top_left_y)

        self.update_coverage_mask(top_left_x, top_left_y, rotated_image)

    def get_canvas(self):
        """
        Get the current canvas with all placed images.

        :return: The canvas as a numpy array.
        """
        return self.canvas


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)
    H = np.linalg.inv(chessboard_homography.H)  # get_homography_image_to_model()
    #H, dsize = chessboard_homography.plot_BEV_full(plot_BEV_full=False)
    RT = chessboard_homography.get_rigid_transform()
    plane_normal = chessboard_homography.get_plane_norm()
    plane_distance = chessboard_homography.get_plane_dist()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    robot_data = RobotDataAtTimestep(
        os.path.join(script_dir, "../bags/panther_ahg_courtyard_0/panther_ahg_courtyard_0.pkl")
    )

    # Initialize the BirdseyeCanvas object
    canvas = BirdseyeCanvas(H, canvas_size=(2000, 1000), scale_factor=100, dsize=(128,128))

    for timestep in tqdm(range(1, 300), desc="Processing patches at timesteps"):
        cur_image = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)

        canvas.add_image(cur_image, cur_rt)

    final_canvas = canvas.get_canvas()

    # Display the canvas using OpenCV
    cv2.imshow("Birdseye Canvas", final_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
