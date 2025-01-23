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
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from numba import njit


class GlobalMap:
    def __init__(self, first_bev_image, output_height_times=3, output_width_times=3.0, detector_type="sift", visualize=True):
        """
        Initialize the GlobalMap object with the first BEV image.
        Args:
            first_bev_image (np.ndarray): The first bird's eye view image.
            output_height_times (float): Scale factor for output height relative to input image.
            output_width_times (float): Scale factor for output width relative to input image.
            detector_type (str): Feature detector type ('sift' or 'orb').
            visualize (bool): Whether to visualize matches and intermediate outputs.
        """
        self.detector_type = detector_type
        self.visualize = visualize

        # Initialize detector and matcher
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(500)
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(500)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Initialize global map canvas
        self.output_img = np.zeros(
            (int(output_height_times * first_bev_image.shape[0]),
             int(output_width_times * first_bev_image.shape[1]),
             first_bev_image.shape[2]),
            dtype=np.uint8
        )

        # Center the first BEV image on the canvas
        self.w_offset = int(self.output_img.shape[0] / 2 - first_bev_image.shape[0] / 2)
        self.h_offset = int(self.output_img.shape[1] / 2 - first_bev_image.shape[1] / 2)
        self.output_img[self.w_offset:self.w_offset + first_bev_image.shape[0],
                        self.h_offset:self.h_offset + first_bev_image.shape[1], :] = first_bev_image

        # Initialize transformation matrix
        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

        # Process the first frame
        self.process_first_frame(first_bev_image)

    def process_first_frame(self, first_bev_image):
        """
        Process the first BEV image to initialize keypoints and descriptors.
        Args:
            first_bev_image (np.ndarray): The first BEV image.
        """
        self.frame_prev = first_bev_image
        frame_gray_prev = cv2.cvtColor(first_bev_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def match(self, des_cur, des_prev):
        """
        Match descriptors between the current and previous frames.
        Args:
            des_cur (np.ndarray): Descriptors from the current frame.
            des_prev (np.ndarray): Descriptors from the previous frame.

        Returns:
            list: List of good matches between descriptors.
        """
        if self.detector_type == "sift":
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            matches = [m for m, n in pair_matches if m.distance < 0.7 * n.distance]
        elif self.detector_type == "orb":
            matches = self.bf.match(des_cur, des_prev)

        # Sort and limit matches
        matches = sorted(matches, key=lambda x: x.distance)[:20]

        # Visualize matches if enabled
        if self.visualize:
            match_img = cv2.drawMatches(self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
            cv2.imshow('matches', match_img)
            cv2.waitKey(1)

        return matches
    
    def expand_canvas(self, x_min, y_min, x_max, y_max):
        x_min = int(np.floor(x_min))
        y_min = int(np.floor(y_min))
        x_max = int(np.ceil(x_max))
        y_max = int(np.ceil(y_max))

        # Determine new canvas dimensions
        new_width = max(self.output_img.shape[1], x_max + abs(x_min))
        new_height = max(self.output_img.shape[0], y_max + abs(y_min))

        # Create a new larger canvas
        new_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Calculate offsets for copying the old canvas
        x_offset = abs(x_min) if x_min < 0 else 0
        y_offset = abs(y_min) if y_min < 0 else 0

        # Copy the existing map into the new canvas
        new_canvas[y_offset:y_offset + self.output_img.shape[0],
                x_offset:x_offset + self.output_img.shape[1]] = self.output_img
        self.output_img = new_canvas

        # Update offsets in the homography matrix
        self.H_old[0, 2] += x_offset
        self.H_old[1, 2] += y_offset
        self.H_old /= self.H_old[2, 2]  # Normalize


    def process_frame(self, frame_cur, timestep, reset_interval=50, odom_matrix=None):
        """
        Process a new BEV frame, compute its keypoints and descriptors, and update the global map.
        Args:
            frame_cur (np.ndarray): The current BEV frame.
            timestep (int): Current timestep in the process.
            reset_interval (int): Number of timesteps after which the alignment is reset.
            odom_matrix (np.ndarray): 4x4 odometry matrix for initial homography estimation.
        """
        # Compute keypoints and descriptors for the current frame
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        # Initialize the homography with odometry if provided
        if odom_matrix is not None:
            H_odom = self.odom_to_homography(odom_matrix)
            H_initial = np.matmul(self.H_old, H_odom)  # Use odometry to estimate H_initial
        else:
            H_initial = self.H_old  # Default fallback to the cumulative homography

        # Periodically reset the homography alignment
        if timestep % reset_interval == 0:
            print(f"Resetting homography alignment at timestep {timestep}")
            kp_map, des_map = self.detector.detectAndCompute(
                cv2.cvtColor(self.output_img, cv2.COLOR_BGR2GRAY), None
            )
            matches = self.match(self.des_cur, des_map)

            if len(matches) >= 4:
                H_reset = self.findHomography(self.kp_cur, kp_map, matches)
                if H_reset is not None:
                    # Align reset homography with the global map
                    H_reset_aligned = np.matmul(self.H_old, H_reset)
                    H_reset_aligned[0, 2] -= self.h_offset
                    H_reset_aligned[1, 2] -= self.w_offset
                    H_reset_aligned /= H_reset_aligned[2, 2]  # Normalize
                    self.H_old = H_reset_aligned
                else:
                    print(f"Failed to reset alignment at timestep {timestep}. Falling back to H_initial.")
                    self.H_old = H_initial
            else:
                print(f"Insufficient matches for alignment reset at timestep {timestep}. Falling back to H_initial.")
                self.H_old = H_initial
        else:
            # Compute homography relative to the previous frame
            matches = self.match(self.des_cur, self.des_prev)
            if len(matches) >= 4:
                H = self.findHomography(self.kp_cur, self.kp_prev, matches)
                if H is not None:
                    H = np.matmul(self.H_old, H)  # Update with previous cumulative transformation
                    H /= H[2, 2]  # Normalize
                    self.H_old = H
                else:
                    print(f"Failed to compute homography at timestep {timestep}. Falling back to H_initial.")
                    self.H_old = H_initial
            else:
                print(f"Skipping frame at timestep {timestep}: Not enough matches. Falling back to H_initial.")
                self.H_old = H_initial

        # Determine bounds for canvas expansion
        transformed_corners = self.get_transformed_corners(frame_cur, self.H_old)
        x_min, y_min = transformed_corners.min(axis=0).squeeze()
        x_max, y_max = transformed_corners.max(axis=0).squeeze()

        # Expand canvas if necessary
        if x_min < 0 or y_min < 0 or x_max > self.output_img.shape[1] or y_max > self.output_img.shape[0]:
            self.expand_canvas(x_min, y_min, x_max, y_max)

        # Warp the current frame and update the global map
        self.warp(frame_cur, self.H_old)

        # Save keypoints/descriptors for the next frame
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur


    def warp(self, frame_cur, H):
        """
        Warp the current BEV frame and merge it into the global map using feather blending.
        Args:
            frame_cur (np.ndarray): The current BEV frame.
            H (np.ndarray): The homography matrix.

        Returns:
            np.ndarray: The updated global map.
        """
        warped_img = cv2.warpPerspective(frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]),
                                        flags=cv2.INTER_LINEAR)

        # Create a binary mask for the warped image
        warped_mask = (warped_img > 0).any(axis=2).astype(np.uint8) * 255

        # Perform feather blending
        self.output_img = self.feather_blend(warped_img, self.output_img, warped_mask)

        # Optional visualization
        if self.visualize:
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', self.output_img)
            cv2.waitKey(1)

        return self.output_img

    def feather_blend(self, warped_img, global_map, mask):
        """
        Feather blending for smoothing transitions between overlapping images.
        Args:
            warped_img (np.ndarray): The warped image to blend.
            global_map (np.ndarray): The existing global map.
            mask (np.ndarray): Binary mask of the warped image.

        Returns:
            np.ndarray: The blended image.
        """
        # Distance transform to calculate weights
        distance_map = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        distance_map = distance_map / (distance_map.max() + 1e-6)  # Normalize to 0-1

        # Blend images based on distance weights
        blended_img = global_map * (1 - distance_map[:, :, None]) + warped_img * distance_map[:, :, None]
        return blended_img.astype(np.uint8)

    @staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        """
        Compute the homography matrix between two sets of keypoints using matches.
        Args:
            image_1_kp (list): Keypoints from the current frame.
            image_2_kp (list): Keypoints from the previous frame.
            matches (list): Matches between the two sets of keypoints.

        Returns:
            np.ndarray: Homography matrix.
        """
        src_pts = np.float32([image_1_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([image_2_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    @staticmethod
    def get_transformed_corners(frame_cur, H):
        """
        Finds the corners of the current frame after applying the homography.
        Args:
            frame_cur (np.ndarray): Current BEV frame.
            H (np.ndarray): Homography matrix.

        Returns:
            np.ndarray: Transformed corner points.
        """
        # Define the four corners of the input frame
        h, w = frame_cur.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)

        # Transform the corners using the homography
        transformed_corners = cv2.perspectiveTransform(corners, H)

        return transformed_corners

    @staticmethod
    def odom_to_homography(odom_matrix):
        """
        Convert a 4x4 odometry matrix to a 3x3 homography for 2D alignment.
        Args:
            odom_matrix (np.ndarray): The 4x4 transformation matrix from odometry.

        Returns:
            np.ndarray: A 3x3 homography matrix.
        """
        tx = odom_matrix[0, 3]
        ty = odom_matrix[1, 3]
        cos_theta = odom_matrix[0, 0]
        sin_theta = odom_matrix[1, 0]

        homography = np.array([
            [cos_theta, -sin_theta, tx],
            [sin_theta, cos_theta, ty],
            [0, 0, 1]
        ])
        return homography

    def get_map(self):
        """
        Retrieve the current global map.
        Returns:
            np.ndarray: The current global map.
        """
        return self.output_img


class MapViewer:
    def __init__(self, global_map):
        self.global_map = global_map
        self.view_window = self.get_non_black_bounding_box()  # Initial view set to non-black region
        self.dragging = False
        self.start_x = 0
        self.start_y = 0

    def get_non_black_bounding_box(self):
        """
        Calculate the bounding box of the non-black area in the global map.

        Returns:
            tuple: (x, y, width, height) of the bounding box for the non-black region.
        """
        # Find non-black pixels
        gray_map = cv2.cvtColor(self.global_map, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        non_black_pixels = np.argwhere(gray_map > 0)  # Identify non-black pixels

        if non_black_pixels.size == 0:
            # If no non-black pixels exist, show the top-left corner by default
            return (0, 0, 1280, 720)

        # Calculate bounding box
        y_min, x_min = non_black_pixels.min(axis=0)
        y_max, x_max = non_black_pixels.max(axis=0)

        # Expand the bounding box slightly for better visibility
        margin = 50
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(self.global_map.shape[1], x_max + margin)
        y_max = min(self.global_map.shape[0], y_max + margin)

        # Width and height of the view
        width = min(x_max - x_min, 1280)
        height = min(y_max - y_min, 720)

        return (x_min, y_min, width, height)

    def show_map(self):
        def mouse_callback(event, x, y, flags, param):
            nonlocal self

            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.start_x, self.start_y = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging:
                    dx = self.start_x - x
                    dy = self.start_y - y
                    self.start_x, self.start_y = x, y
                    self.update_view(dx, dy)

            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False

        cv2.namedWindow("Scrollable Map")
        cv2.setMouseCallback("Scrollable Map", mouse_callback)

        while True:
            # Extract the current view from the global map
            x, y, w, h = self.view_window
            view = self.global_map[y:y + h, x:x + w]

            # Resize for consistent viewing if the view is smaller
            display = cv2.resize(view, (w, h), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Scrollable Map", display)

            # Exit on 'q' key
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def update_view(self, dx, dy):
        """Update the view window based on the drag motion."""
        x, y, w, h = self.view_window

        # Update x and y positions
        new_x = max(0, min(x + dx, self.global_map.shape[1] - w))
        new_y = max(0, min(y + dy, self.global_map.shape[0] - h))

        self.view_window = (new_x, new_y, w, h)


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)
    #H = np.linalg.inv(chessboard_homography.H)  # get_homography_image_to_model()
    H, dsize,_ = chessboard_homography.plot_BEV_full(image)
    RT = chessboard_homography.get_rigid_transform()
    plane_normal = chessboard_homography.get_plane_norm()
    plane_distance = chessboard_homography.get_plane_dist()
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    robot_data = RobotDataAtTimestep(
        os.path.join(script_dir, "../bags/ahg_courtyard_1/ahg_courtyard_1_synced.pkl")
    )

    # Initialize the global map with the first BEV image
    first_bev_image = cv2.warpPerspective(robot_data.getImageAtTimestep(0), H, dsize)
    global_map = GlobalMap(first_bev_image, detector_type="sift", visualize=True)

    # Process subsequent BEV images
    for timestep in tqdm(range(1, 700), desc="Processing patches at timesteps"):
        try:
            cur_img = robot_data.getImageAtTimestep(timestep)
            cur_rt = robot_data.getOdomAtTimestep(timestep)
            if cur_img is None:
                print(f"Missing image data at timestep {timestep}")
                continue

            bev_img = cv2.warpPerspective(cur_img, H, dsize)  # Create BEV image
            global_map.process_frame(bev_img, timestep=timestep, reset_interval = 50)

        except Exception as e:
            print(f"Error at timestep {timestep}: {e}")
            continue

    # Retrieve and save the final stitched map
    final_map = global_map.get_map()
    # Ensure the map is properly scaled for viewing
    viewer = MapViewer(final_map)
    viewer.show_map()
# Add Z buffer, so that pixels off in the distance are replaced with the most recent. 
# We can get pixel and depth of the pixel using homography, once we reach that pixel, replace that portion of the image with most current