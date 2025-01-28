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
import torch
import torch.nn.functional as F


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
        self.z_buffer = np.full((canvas_height, canvas_width), -np.inf, dtype=np.float32)

        # Initialize the global map canvas and offsets
        self.output_img = np.zeros((canvas_height, canvas_width, channels), dtype=np.uint8)

        # Calculate offsets to center the first frame on the canvas
        self.w_offset = (self.output_img.shape[0] - first_bev_image.shape[0]) // 2
        self.h_offset = (self.output_img.shape[1] - first_bev_image.shape[1]) // 2

        # Place the first frame at the center
        self.output_img[self.w_offset:self.w_offset + first_bev_image.shape[0],
        self.h_offset:self.h_offset + first_bev_image.shape[1]] = first_bev_image

        self.edge_mask = np.zeros((first_bev_image.shape[:2]), dtype=np.uint8)
        height, width = first_bev_image.shape[:2]
        cv2.rectangle(self.edge_mask, (0, 0), (width, 10), 1, -1)  # Top edge
        cv2.rectangle(self.edge_mask, (0, height - 10), (width, height), 1, -1)  # Bottom edge
        cv2.rectangle(self.edge_mask, (0, 0), (10, height), 1, -1)  # Left edge
        cv2.rectangle(self.edge_mask, (width - 10, 0), (width, height), 1, -1)  # Right edge

        # Initialize the transformation matrix
        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

    def expand_canvas(self, x_min, y_min, x_max, y_max, buffer=720):
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        x_offset = max(0, -x_min) + buffer
        y_offset = max(0, -y_min) + buffer
        new_width = max(self.output_img.shape[1] + x_offset, x_max + buffer)
        new_height = max(self.output_img.shape[0] + y_offset, y_max + buffer)

        if new_width > self.output_img.shape[1] or new_height > self.output_img.shape[0]:
            expanded_canvas = np.zeros((new_height, new_width, self.output_img.shape[2]), dtype=self.output_img.dtype)
            expanded_canvas[y_offset:y_offset + self.output_img.shape[0],
                            x_offset:x_offset + self.output_img.shape[1]] = self.output_img
            self.output_img = expanded_canvas

            # Expand the z-buffer: Resize and initialize new areas with -inf (no depth)
            expanded_z_buffer = np.full((new_height, new_width), -np.inf, dtype=np.float32)
            expanded_z_buffer[y_offset:y_offset + self.z_buffer.shape[0],
                            x_offset:x_offset + self.z_buffer.shape[1]] = self.z_buffer
            self.z_buffer = expanded_z_buffer

            self.h_offset += x_offset
            self.w_offset += y_offset
            self.H_old[0, 2] += x_offset
            self.H_old[1, 2] += y_offset


    def process_frame(self, frame_cur, timestep, odom_data=None, scale=100):
        """
        Process a new BEV frame using odometry data and SSD minimization to update the global map.
        """
        translation_threshold=0.01
        rotation_threshold=0.01
        if self.output_img is None:
            self.initialize_canvas(frame_cur)
            self.frame_previous = frame_cur
            self.odom_previous = odom_data
            print(f"Initialized global map at timestep {timestep}")
            return

        if odom_data is not None and self.odom_previous is not None:
            # Compute relative odometry transformation
            relative_transform = np.linalg.inv(self.odom_previous) @ odom_data

            # Compute translation and rotation differences
            tx, ty = relative_transform[0, 3], relative_transform[1, 3]
            translation_distance = np.sqrt(tx**2 + ty**2)  # Translation in meters
            rotation_angle = np.arctan2(relative_transform[1, 0], relative_transform[0, 0])  # Rotation in radians

            # Skip the frame if movement is below the threshold
            if translation_distance < translation_threshold and abs(rotation_angle) < rotation_threshold:
                print(f"Skipping timestep {timestep}: No significant movement detected (translation={translation_distance:.3f}, rotation={rotation_angle:.3f})")
                return

            # Compute relative homography based on odometry
            H_relative = self.compute_relative_odometry(
                self.odom_previous, odom_data, frame_cur.shape[1], frame_cur.shape[0], scale
            )
        else:
            print(f"Missing odometry data at timestep {timestep}")
            return


        # **Improvement Highlight**: Avoid re-warping large regions unnecessarily
        H_refined = self.refine_homography_with_ssd(frame_cur, self.frame_previous, H_relative)

        self.H_old = np.matmul(self.H_old, H_refined)
        self.H_old /= self.H_old[2, 2]

        transformed_corners = self.get_transformed_corners(frame_cur, self.H_old)
        x_min, y_min = transformed_corners.min(axis=0).squeeze()
        x_max, y_max = transformed_corners.max(axis=0).squeeze()

        if x_min < 0 or y_min < 0 or x_max > self.output_img.shape[1] or y_max > self.output_img.shape[0]:
            self.expand_canvas(x_min, y_min, x_max, y_max, 720)

        self.warp(frame_cur, self.H_old, timestep)
        self.odom_previous = odom_data
        self.frame_previous = frame_cur

        # Optional: Visualize the updated global map
        if self.visualize & timestep % 100 ==0:
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', self.output_img)
            cv2.waitKey(1)

    def warp(self, frame_cur, H, depth):
        """
        Warp the current BEV frame into the global map using a z-buffer to handle overlaps.
        Args:
            frame_cur (np.ndarray): The current BEV frame.
            H (np.ndarray): The homography matrix.
            depth (float): The depth value for the current frame (e.g., timestamp or priority).
        """
        height, width = self.output_img.shape[:2]

        # Warp the current frame into the global map space
        warped_img = cv2.warpPerspective(
            frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )

        # Generate a binary mask for valid pixels in the warped image
        mask = (frame_cur > 0).any(axis=2).astype(np.uint8)
        warped_mask = cv2.warpPerspective(
            mask, H, (self.output_img.shape[1], self.output_img.shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Replace edges of the warped image with the corresponding regions from the previous frame
        if self.frame_previous is not None:
            warped_edge_mask = cv2.warpPerspective(
                self.edge_mask, H, (self.output_img.shape[1], self.output_img.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        # Get only non-zero pixel coordinates from the warped mask
        valid_pixels = cv2.findNonZero(warped_mask)

        if valid_pixels is not None:  # Check if there are any valid pixels
            for px in valid_pixels:
                x, y = px[0]  # Extract pixel coordinates

                # Ensure indices are within bounds
                if not (0 <= x < width and 0 <= y < height):
                    continue

                # Handle edges from the previous frame
                if self.frame_previous is not None and warped_edge_mask is not None and warped_edge_mask[y, x] == 1:
                    if 0 <= y < self.frame_previous.shape[0] and 0 <= x < self.frame_previous.shape[1]:
                        self.output_img[y, x] = self.frame_previous[y, x]
                    continue  # Skip further processing for edge pixels
                
                # Depth-based update (Z-buffer)
                if depth > self.z_buffer[y, x]:  
                    self.output_img[y, x] = warped_img[y, x]
                    self.z_buffer[y, x] = depth

        return self.output_img

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
        relative_transform = np.linalg.inv(odom_matrix_prev) @ odom_matrix_cur
        tx = relative_transform[1, 3] * scale
        ty = relative_transform[0, 3] * scale
        cos_theta = relative_transform[0, 0]
        sin_theta = relative_transform[1, 0]

        to_bottom_center = np.array([
            [1, 0, -image_width / 2],
            [0, 1, -image_height],
            [0, 0, 1]])
        relative_homography = np.array([
            [cos_theta, sin_theta, -tx],
            [-sin_theta, cos_theta, -ty],
            [0, 0, 1]])
        to_original = np.array([
            [1, 0, image_width / 2],
            [0, 1, image_height],
            [0, 0, 1]])
        return to_original @ relative_homography @ to_bottom_center
    
    def refine_homography_with_ssd(self, frame_cur, prev_frame, H_initial, patch_size=32):
        """
        Refine the homography matrix using SSD loss in a parallelized manner with PyTorch.
        
        Args:
            frame_cur (np.ndarray): Current BEV frame (H, W, C).
            global_map (np.ndarray): Global map (H, W, C).
            H_initial (np.ndarray): Initial homography matrix.
            patch_size (int): Size of patches for SSD computation.
        
        Returns:
            np.ndarray: Refined homography matrix.
        """
        # Convert inputs to PyTorch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame_cur_tensor = torch.tensor(frame_cur.transpose(2, 0, 1), dtype=torch.float32, device=device) / 255.0  # (C, H, W)
        prev_frame_tensor = torch.tensor(prev_frame.transpose(2, 0, 1), dtype=torch.float32, device=device) / 255.0  # (C, H, W)
        H_initial_tensor = torch.tensor(H_initial, dtype=torch.float32, device=device)

        def ssd_loss(params):
            # Reconstruct the homography matrix from params
            H = torch.tensor([
                [params[0], params[1], params[2]],
                [params[3], params[4], params[5]],
                [0,         0,         1]
            ], dtype=torch.float32, device=device)

            # Warp the current frame using the homography matrix
            warped_frame = self.warp_with_homography(frame_cur_tensor, H, prev_frame_tensor.shape[1:])

            # Compute the SSD loss over patches
            ssd_loss = self.compute_patchwise_ssd(prev_frame_tensor, warped_frame, patch_size)
            return ssd_loss.item()  # Convert to scalar for optimization

        # Optimize using scipy's minimize
        initial_params = H_initial_tensor[:2, :].flatten().cpu().numpy()  # Flatten first 6 elements
        result = minimize(ssd_loss, initial_params, method='L-BFGS-B', options={"maxiter": 100})
        refined_params = result.x

        # Reconstruct refined homography matrix
        refined_H = torch.tensor([
            [refined_params[0], refined_params[1], refined_params[2]],
            [refined_params[3], refined_params[4], refined_params[5]],
            [0,                 0,                 1]
        ], dtype=torch.float32, device=device)

        return refined_H.cpu().numpy()

    def warp_with_homography(self, image, homography, output_shape):
        """
        Warp an image using a given homography matrix with PyTorch.

        Args:
            image (torch.Tensor): Input image (C, H, W).
            homography (torch.Tensor): Homography matrix (3x3).
            output_shape (tuple): Shape of the output image (H, W).

        Returns:
            torch.Tensor: Warped image (C, H, W).
        """
        H, W = output_shape
        device = image.device

        # Create a grid of normalized coordinates (-1 to 1)
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device)
        )
        coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1)  # (H, W, 3)
        coords = coords.view(-1, 3).T  # (3, H*W)

        # Transform coordinates using the homography matrix
        warped_coords = homography @ coords  # (3, H*W)
        warped_coords = warped_coords / warped_coords[2, :]  # Normalize by z-coordinate
        warped_coords = warped_coords[:2, :].T  # (H*W, 2)

        # Convert normalized coordinates back to image space
        warped_coords = warped_coords.view(H, W, 2).unsqueeze(0)  # (1, H, W, 2)

        # Sample the image using the warped coordinates
        warped_image = F.grid_sample(
            image.unsqueeze(0),  # Add batch dimension
            warped_coords,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros'
        )
        return warped_image.squeeze(0)  # Remove batch dimension

    def compute_patchwise_ssd(self, prev_frame, warped_frame, patch_size):
        """
        Compute the SSD loss between the global map and warped frame in patches.

        Args:
            global_map (torch.Tensor): Global map (C, H, W).
            warped_frame (torch.Tensor): Warped frame (C, H, W).
            patch_size (int): Size of patches.

        Returns:
            torch.Tensor: Total SSD loss (scalar).
        """
        C, H, W = prev_frame.shape
        prev_patches = prev_frame.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)  # (C, N, M, patch_size, patch_size)
        warped_patches = prev_frame.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)  # (C, N, M, patch_size, patch_size)

        # Compute SSD loss for each patch
        ssd = torch.sum((prev_patches - warped_patches) ** 2, dim=(-1, -2, -3))  # (N, M)

        # Sum up all patch losses
        return torch.sum(ssd)

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

    scale_start = 490

    # Initialize the global map
    global_map = GlobalMap(visualize=True)
    bev_image_prev = cv2.warpPerspective(robot_data.getImageAtTimestep(scale_start), H, dsize)
    bev_image_cur = cv2.warpPerspective(robot_data.getImageAtTimestep(scale_start+1), H, dsize)

    meters_to_pixels = calculate_meters_to_pixels(bev_image_prev, bev_image_cur,
                                                  robot_data.getOdomAtTimestep(scale_start),
                                                  robot_data.getOdomAtTimestep(scale_start+1))

    start = 850
    end = 1500
    
    # Process subsequent BEV images
    for timestep in tqdm(range(start+1, end), desc="Processing patches at timesteps"):
        try:
            cur_img = robot_data.getImageAtTimestep(timestep)
            cur_rt = robot_data.getOdomAtTimestep(timestep)
            if cur_img is None:
                print(f"Missing image data at timestep {timestep}")
                continue

            bev_img = cv2.warpPerspective(cur_img, H, dsize)  # Create BEV image
            global_map.process_frame(bev_img, timestep, odom_data=cur_rt, scale=meters_to_pixels)

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