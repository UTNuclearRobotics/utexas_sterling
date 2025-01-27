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
from scipy.optimize import minimize


class GlobalMap:
    def __init__(self, output_height_times=3, output_width_times=3.0, visualize=True):
        """
        Initialize the GlobalMap object without requiring an explicit first BEV image.
        Args:
            output_height_times (float): Scale factor for output height relative to input image.
            output_width_times (float): Scale factor for output width relative to input image.
            visualize (bool): Whether to visualize matches and intermediate outputs.
        """
        self.visualize = visualize
        self.output_img = None  # Global map canvas initialized dynamically
        self.w_offset = None  # Horizontal offset, initialized dynamically
        self.h_offset = None  # Vertical offset, initialized dynamically
        self.H_old = None  # Homography matrix, initialized dynamically
        self.output_height_times = output_height_times
        self.output_width_times = output_width_times
        self.frame_previous = None
        self.odom_previous = None

    def initialize_canvas(self, first_bev_image):
        """
        Dynamically initialize the global map canvas using the first BEV image.
        Args:
            first_bev_image (np.ndarray): The first bird's-eye view image.
        """
        height, width, channels = first_bev_image.shape
        canvas_height = int(self.output_height_times * height)
        canvas_width = int(self.output_width_times * width)

        # Initialize the global map canvas and offsets
        self.output_img = np.zeros((canvas_height, canvas_width, channels), dtype=np.uint8)

        # Calculate offsets to center the first frame on the canvas
        self.w_offset = (self.output_img.shape[0] - first_bev_image.shape[0]) // 2
        self.h_offset = (self.output_img.shape[1] - first_bev_image.shape[1]) // 2

        # Place the first frame at the center
        self.output_img[self.w_offset:self.w_offset + first_bev_image.shape[0],
        self.h_offset:self.h_offset + first_bev_image.shape[1]] = first_bev_image

        # Initialize the transformation matrix
        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

    def expand_canvas(self, x_min, y_min, x_max, y_max):
        x_min = int(np.floor(x_min))
        y_min = int(np.floor(y_min))
        x_max = int(np.ceil(x_max))
        y_max = int(np.ceil(y_max))

        # Determine the required offsets for expansion
        x_offset = max(0, -x_min)  # Only expand if x_min < 0
        y_offset = max(0, -y_min)  # Only expand if y_min < 0

        # Determine new canvas dimensions
        new_width = max(self.output_img.shape[1] + x_offset, x_max)
        new_height = max(self.output_img.shape[0] + y_offset, y_max)

        # Create a new larger canvas
        new_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Copy the existing map into the new canvas
        new_canvas[y_offset:y_offset + self.output_img.shape[0],
                x_offset:x_offset + self.output_img.shape[1]] = self.output_img

        # Update the global map and offsets
        self.output_img = new_canvas
        self.h_offset += x_offset  # Update horizontal offset
        self.w_offset += y_offset  # Update vertical offset

        # Update the homography matrix offsets
        self.H_old[0, 2] += x_offset
        self.H_old[1, 2] += y_offset
        self.H_old /= self.H_old[2, 2]  # Normalize


    def process_frame(self, frame_cur, timestep, odom_data=None, scale=100):
        """
        Process a new BEV frame using odometry data and SSD minimization to update the global map.
        Args:
            frame_cur (np.ndarray): The current BEV frame.
            timestep (int): Current timestep in the process.
            odom_data (np.ndarray): Current odometry data as a 4x4 transformation matrix.
            scale (float): Conversion factor from meters to pixels.
        """
        if self.output_img is None:
            # Initialize the global map using the first frame
            self.initialize_canvas(frame_cur)
            self.frame_previous = frame_cur
            self.odom_previous = odom_data
            print(f"Initialized global map at timestep {timestep}")
            return

        if odom_data is not None and self.odom_previous is not None:
            # Compute relative odometry transformation
            H_relative = self.compute_relative_odometry(
                self.odom_previous, odom_data, frame_cur.shape[1], frame_cur.shape[0], scale
            )
        else:
            print(f"Missing odometry data at timestep {timestep}")
            return

        # Refine the initial homography using SSD minimization
        H_refined = self.refine_homography_with_ssd(frame_cur, self.output_img, H_relative)

        # Update the homography matrix
        self.H_old = np.matmul(self.H_old, H_refined)
        self.H_old /= self.H_old[2, 2]
        

        # Determine transformed corners of the new frame in the global map
        transformed_corners = self.get_transformed_corners(frame_cur, self.H_old)
        x_min, y_min = transformed_corners.min(axis=0).squeeze()
        x_max, y_max = transformed_corners.max(axis=0).squeeze()

        # Expand the canvas if necessary
        if x_min < 0 or y_min < 0 or x_max > self.output_img.shape[1] or y_max > self.output_img.shape[0]:
            self.expand_canvas(x_min, y_min, x_max, y_max)

        # Warp the current frame into the global map
        self.warp(frame_cur, self.H_old)

        # Update the previous frame and odometry
        self.odom_previous = odom_data
        self.frame_previous = frame_cur

        # Optional: Visualize the updated global map
        if self.visualize:
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', self.output_img)
            cv2.waitKey(1)

    def warp(self, frame_cur, H):
        """
        Warp the current BEV frame into the global map and blend overlapping regions,
        ensuring no black borders contribute to the final output.
        Args:
            frame_cur (np.ndarray): The current BEV frame.
            H (np.ndarray): The homography matrix.
        """
        # Warp the current frame into the global map space
        warped_img = cv2.warpPerspective(
            frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Generate a binary mask for the valid pixels in the warped image
        mask = (frame_cur > 0).any(axis=2).astype(np.uint8)  # Valid region of the input image
        warped_mask = cv2.warpPerspective(
            mask, H, (self.output_img.shape[1], self.output_img.shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Identify overlapping and non-overlapping regions
        global_mask = (self.output_img > 0).any(axis=2).astype(np.uint8)
        overlap_mask = (warped_mask & global_mask).astype(np.uint8)
        non_overlap_mask = (warped_mask & (~global_mask)).astype(np.uint8)

        # Blend overlapping regions
        if np.any(overlap_mask):
            self.output_img = self.blend_with_validity(warped_img, self.output_img, overlap_mask, warped_mask)

        # Copy non-overlapping regions
        self.output_img[non_overlap_mask == 1] = warped_img[non_overlap_mask == 1]

        # Optional visualization
        if self.visualize:
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', self.output_img)
            cv2.waitKey(1)

        return self.output_img

    def blend_with_validity(self, warped_img, global_map, overlap_mask, warped_mask):
        """
        Blend the warped image with the global map using a validity mask to exclude black borders.
        Args:
            warped_img (np.ndarray): The warped image.
            global_map (np.ndarray): The global map.
            overlap_mask (np.ndarray): Binary mask for overlapping regions.
            warped_mask (np.ndarray): Binary mask for valid pixels in the warped image.
        Returns:
            np.ndarray: Updated global map with blended regions.
        """
        # Normalize the overlap mask to compute alpha blending
        distance_to_edge = cv2.distanceTransform(overlap_mask, distanceType=cv2.DIST_L2, maskSize=3)
        max_distance = np.max(distance_to_edge) if np.max(distance_to_edge) > 0 else 1
        alpha = (distance_to_edge / max_distance)[..., None]  # Add a channel dimension

        # Ensure black pixels are excluded from blending
        warped_img[warped_mask == 0] = 0  # Mask out invalid pixels in the warped image

        # Blend the images
        blended_map = global_map.copy()
        blended_map[overlap_mask == 1] = (
            alpha[overlap_mask == 1] * warped_img[overlap_mask == 1] +
            (1 - alpha[overlap_mask == 1]) * global_map[overlap_mask == 1]
        ).astype(np.uint8)

        return blended_map

    def feather_blend(self, warped_img, global_map, overlap_mask):
        """
        Blend the warped image with the global map using distance-based alpha blending,
        prioritizing the latest image in overlapping regions.
        Args:
            warped_img (np.ndarray): The warped image to be blended.
            global_map (np.ndarray): The current global map.
            overlap_mask (np.ndarray): Binary mask for overlapping regions.
        Returns:
            np.ndarray: Updated global map with blended regions.
        """
        # Compute a distance transform for the overlap mask
        distance_to_edge = cv2.distanceTransform(overlap_mask, distanceType=cv2.DIST_L2, maskSize=3)
        max_distance = np.max(distance_to_edge)  # Normalize distances
        distance_to_edge = distance_to_edge / max_distance if max_distance > 0 else distance_to_edge

        # Adjust alpha weights to prioritize the latest image
        beta = 0.7  # Higher beta prioritizes the latest image
        alpha = (1 - beta) + beta * (distance_to_edge[..., None])  # Bias toward `warped_img`

        # Ensure alpha is clipped between 0 and 1
        alpha = np.clip(alpha, 0, 1)

        # Blend the images using the biased alpha weights
        blended_map = global_map.copy()
        blended_map[overlap_mask == 1] = (
            alpha[overlap_mask == 1] * warped_img[overlap_mask == 1] +
            (1 - alpha[overlap_mask == 1]) * global_map[overlap_mask == 1]
        ).astype(np.uint8)

        return blended_map

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


    def compute_relative_odometry(self, odom_matrix_prev, odom_matrix_cur, image_width, image_height, scale):
        """
        Compute the relative transformation between two odometry matrices, considering
        the center of the bottom edge of the BEV image as the reference point.
        
        Args:
            odom_matrix_prev (np.ndarray): The 4x4 odometry matrix of the previous timestep.
            odom_matrix_cur (np.ndarray): The 4x4 odometry matrix of the current timestep.
            meters_to_pixels (float): Conversion factor from meters to pixels.
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.
        
        Returns:
            np.ndarray: A 3x3 homography matrix representing the relative transformation.
        """
        # Extract relative motion in meters
        relative_transform = np.linalg.inv(odom_matrix_prev) @ odom_matrix_cur
        tx = relative_transform[1, 3] * scale
        ty = relative_transform[0, 3] * scale
        cos_theta = relative_transform[0, 0]
        sin_theta = relative_transform[1, 0]

        # Translation to shift the origin to the bottom-center of the image
        to_bottom_center = np.array([
            [1, 0, -image_width / 2],
            [0, 1, -image_height],
            [0, 0, 1]
        ])

        # Relative transformation matrix in the new coordinate system
        relative_homography = np.array([
            [cos_theta, sin_theta, -tx],
            [-sin_theta, cos_theta, -ty],
            [0, 0, 1]
        ])

        # Translate back to the original coordinate system
        to_original = np.array([
            [1, 0, image_width / 2],
            [0, 1, image_height],
            [0, 0, 1]
        ])

        # Combined transformation
        final_homography = to_original @ relative_homography @ to_bottom_center
        return final_homography
    
    def refine_homography_with_ssd(self, frame_cur, global_map, H_initial):
        """
        Refine the homography matrix using SSD minimization in overlapping regions.
        Args:
            frame_cur (np.ndarray): Current BEV frame.
            global_map (np.ndarray): Current global map.
            H_initial (np.ndarray): Initial homography matrix.
        Returns:
            np.ndarray: Refined homography matrix.
        """
        # Define the loss function for optimization
        def ssd_loss(params):
            # Construct the homography matrix from parameters
            H = np.array([
                [params[0], params[1], params[2]],
                [params[3], params[4], params[5]],
                [0,         0,         1]
            ])
            
            # Warp the current frame using the candidate homography
            warped_img = cv2.warpPerspective(frame_cur, H, (global_map.shape[1], global_map.shape[0]))
            
            # Create masks for overlapping regions
            warped_mask = (warped_img > 0).any(axis=2).astype(np.uint8)
            global_mask = (global_map > 0).any(axis=2).astype(np.uint8)
            overlap_mask = (warped_mask & global_mask).astype(np.uint8)
            
            if np.any(overlap_mask):  # Only compute SSD in overlapping regions
                global_overlap = global_map[overlap_mask == 1]
                warped_overlap = warped_img[overlap_mask == 1]
                ssd = np.sum((global_overlap - warped_overlap) ** 2)
                return ssd
            else:
                return 1e6  # Penalize cases with no overlap

        # Flatten the initial homography matrix to use as optimization parameters
        initial_params = H_initial.flatten()[:6]

        # Minimize SSD using L-BFGS-B optimization
        result = minimize(ssd_loss, initial_params, method='L-BFGS-B')

        # Reconstruct the refined homography matrix
        refined_H = np.array([
            [result.x[0], result.x[1], result.x[2]],
            [result.x[3], result.x[4], result.x[5]],
            [0,           0,           1]
        ])
        
        return refined_H

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

def calculate_meters_to_pixels(image_prev, image_cur, odom_matrix_prev, odom_matrix_cur):
    """
    Calculate the meters-to-pixels scaling factor using relative odometry and image displacement.

    Args:
        image_prev (np.ndarray): The previous bird's-eye view (BEV) image.
        image_cur (np.ndarray): The current bird's-eye view (BEV) image.
        odom_matrix_prev (np.ndarray): The 4x4 odometry matrix of the previous timestep.
        odom_matrix_cur (np.ndarray): The 4x4 odometry matrix of the current timestep.

    Returns:
        float: The calculated meters-to-pixels scaling factor.
    """
    # Compute relative odometry transformation (translation in meters)
    relative_transform = np.linalg.inv(odom_matrix_prev) @ odom_matrix_cur
    tx_meters = relative_transform[1, 3]
    ty_meters = relative_transform[0, 3]

    # Convert images to grayscale
    gray_prev = cv2.cvtColor(image_prev, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(image_cur, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors using ORB
    orb = cv2.ORB_create()
    keypoints_prev, descriptors_prev = orb.detectAndCompute(gray_prev, None)
    keypoints_cur, descriptors_cur = orb.detectAndCompute(gray_cur, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_prev, descriptors_cur)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points_prev = np.float32([keypoints_prev[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points_cur = np.float32([keypoints_cur[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Calculate average pixel displacement between matched points
    displacements = np.linalg.norm(points_cur - points_prev, axis=1)
    avg_pixel_displacement = np.mean(displacements)

    # Calculate relative translation in meters
    translation_meters = np.sqrt(tx_meters**2 + ty_meters**2)

    # Compute meters-to-pixels scaling factor
    meters_to_pixels = avg_pixel_displacement / translation_meters

    print(f"Translation in meters: {translation_meters}")
    print(f"Average pixel displacement: {avg_pixel_displacement}")
    print(f"Calculated meters-to-pixels scaling factor: {meters_to_pixels}")

    return meters_to_pixels


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

    start = 850
    end = 1500

    # Initialize the global map
    global_map = GlobalMap(visualize=True)
    bev_image_prev = cv2.warpPerspective(robot_data.getImageAtTimestep(849), H, dsize)
    bev_image_cur = cv2.warpPerspective(robot_data.getImageAtTimestep(850), H, dsize)

    meters_to_pixels = calculate_meters_to_pixels(bev_image_prev, bev_image_cur,
                                                  robot_data.getOdomAtTimestep(849),
                                                  robot_data.getOdomAtTimestep(850))

    # Process subsequent BEV images
    for timestep in tqdm(range(start+1, end), desc="Processing patches at timesteps"):
        try:
            cur_img = robot_data.getImageAtTimestep(timestep)
            cur_rt = robot_data.getOdomAtTimestep(timestep)
            if cur_img is None:
                print(f"Missing image data at timestep {timestep}")
                continue

            bev_img = cv2.warpPerspective(cur_img, H, dsize)  # Create BEV image
            global_map.process_frame(bev_img, timestep, odom_data=cur_rt, scale=252)

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

# Interpolate between IMU homography
#SUm of squared differences of the pixel intensites between images, use minimization function to minimize the difference in pixel intensities
# This would prevent overlap between sequential images
