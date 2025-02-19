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
from collections import deque
import math
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy.linalg import solve
from collections import Counter
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

class GlobalMap:
    def __init__(self, tile_size=1024, channels=3, visualize=False):
        """
        A global mapping system using a tiled structure for efficient large-scale stitching.

        Args:
            tile_size (int): Size of each tile in pixels.
            channels (int): Number of color channels (e.g., 3 for RGB).
            visualize (bool): Whether to display the stitched map periodically.
        """
        self.tile_size = tile_size
        self.channels = channels
        self.visualize = visualize
        self.tiles = {}  # Dictionary: (tile_x, tile_y) -> {'image', 'z_buffer'}
        self.lock = threading.Lock()  # Ensures thread safety

        # Odometry & frame history
        self.H_old = None
        self.frame_previous = None
        self.odom_previous = None
        self.frame_history = deque(maxlen=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_or_create_tile(self, tile_x, tile_y):
        """Thread-safe function to retrieve or initialize a tile."""
        if (tile_x, tile_y) not in self.tiles:
            with self.lock:  # Only lock when modifying shared state
                if (tile_x, tile_y) not in self.tiles:  # Double-check inside lock
                    self.tiles[(tile_x, tile_y)] = {  # Ensure it's a dictionary
                        'image': np.zeros((self.tile_size, self.tile_size, self.channels), dtype=np.uint8),
                        'z_buffer': np.full((self.tile_size, self.tile_size), -np.inf, dtype=np.float32),
                        'frame_history': deque(maxlen=3)
                    }
        return self.tiles[(tile_x, tile_y)]

    def _global_to_tile_coords(self, x, y):
        """Converts global pixel coordinates to tile indices."""
        return int(x // self.tile_size), int(y // self.tile_size)

    def get_transformed_corners(self, frame_cur, H):
        """Transforms image corners using a homography matrix."""
        h, w = frame_cur.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        return cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
    
    def compute_depth_map(self, H, image):
        """Computes a depth map using the homography matrix."""
        h, w, _ = image.shape
        coords = np.indices((h, w), dtype=np.float32).reshape(2, -1)  # Faster than meshgrid
        depth_values = H[2, 0] * coords[1] + H[2, 1] * coords[0] + H[2, 2]
        depth_values = ((depth_values - depth_values.min()) * (255.0 / (depth_values.max() + 1e-5))).astype(np.uint8)
        return depth_values.reshape(h, w)

    def compute_sharpness(self, image):
        """Computes image sharpness using Sobel filters."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sharpness = cv2.magnitude(sobel_x, sobel_y)  # Faster than hypot()
        sharpness = ((sharpness / (sharpness.max() + 1e-5)) * 255).astype(np.uint8)
        return sharpness

    def process_tile(self, tx, ty, frame_cur, depth_map, sharpness_map, H, tile_size):
        """Processes a single tile - warping, depth check, and blending."""
        tile_data = self._get_or_create_tile(tx, ty)
        tile_img = tile_data['image']
        tile_zb = tile_data['z_buffer']
        
        tile_origin_x, tile_origin_y = tx * tile_size, ty * tile_size
        T_offset = np.array([[1, 0, -tile_origin_x], [0, 1, -tile_origin_y], [0, 0, 1]], dtype=np.float32)
        H_tile = T_offset @ H
        H_tile /= H_tile[2, 2]

        # Initialize a rolling history of past 5 frames per tile
        if 'frame_history' not in tile_data:
            tile_data['frame_history'] = deque(maxlen=3)  # Store last 5 frames

        # Warp images to the tile
        warped_img = cv2.warpPerspective(
            frame_cur, H_tile, (tile_size, tile_size),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        warped_depth = cv2.warpPerspective(
            depth_map, H_tile, (tile_size, tile_size),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
        )
        warped_sharpness = cv2.warpPerspective(
            sharpness_map, H_tile, (tile_size, tile_size),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Mask Alignment
        if warped_depth.shape[:2] != warped_img.shape[:2]:
            warped_depth = cv2.resize(warped_depth, (warped_img.shape[1], warped_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if warped_sharpness.shape[:2] != warped_img.shape[:2]:
            warped_sharpness = cv2.resize(warped_sharpness, (warped_img.shape[1], warped_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = (warped_img > 0).any(axis=2)
        update_mask = mask & (warped_depth >= tile_zb) & (warped_sharpness > 0)

        # 🔹 **Detect the boundary of update_mask using edge detection**
        boundary_mask = cv2.Canny(update_mask.astype(np.uint8) * 255, 50, 150)
        
        # 🔹 **Adaptive Dilation: Use different kernel sizes based on edge density**
        close_edges = cv2.dilate(boundary_mask, np.ones((1,1), np.uint8), iterations=1).astype(bool)
        far_edges = cv2.dilate(boundary_mask, np.ones((3,3), np.uint8), iterations=1).astype(bool)
        
        adaptive_mask = np.where(close_edges, close_edges, far_edges)

        # 🔹 **Fix: Ensure `adaptive_mask` is the same size as `warped_img`**
        if adaptive_mask.shape[:2] != warped_img.shape[:2]:
            adaptive_mask = cv2.resize(adaptive_mask.astype(np.uint8), (warped_img.shape[1], warped_img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        # 🔹 **Expand only boundary pixels from warped_img without blurring**
        expanded_warped = warped_img.copy()
        expanded_warped[adaptive_mask] = cv2.dilate(warped_img, np.ones((3,3), np.uint8), iterations=1)[adaptive_mask]

        if tile_img.ndim == 3:
            update_mask_3d = np.repeat(update_mask[:, :, np.newaxis], 3, axis=2)

            # **Use Past Frames for Blending**
            frame_history = tile_data['frame_history']
            frame_history.append(warped_img.copy())  # Store the current frame

            # Compute weighted average of past frames
            weight_sum = sum((i + 1) for i in range(len(frame_history)))
            blended_img = frame_history[-1].copy()
            for i in range(len(frame_history) - 1):
                alpha = (i + 1) / weight_sum
                cv2.addWeighted(frame_history[i], alpha, blended_img, 1 - alpha, 0, blended_img)

            # **Apply expanded_warped only in update_mask regions**
            tile_img[update_mask_3d] = expanded_warped[update_mask_3d]
        else:
            tile_img[update_mask] = expanded_warped[update_mask]

        tile_zb[update_mask] = warped_depth[update_mask]  # Update z-buffer

    def warp_frame_into_tiles(self, frame_cur, H):
        depth_map, sharpness_map = self.compute_depth_map(H, frame_cur), self.compute_sharpness(frame_cur)

        # Compute once, reuse results
        corners = self.get_transformed_corners(frame_cur, H)
        x_min, x_max = np.floor(corners[:, 0, 0].min()), np.ceil(corners[:, 0, 0].max())
        y_min, y_max = np.floor(corners[:, 0, 1].min()), np.ceil(corners[:, 0, 1].max())

        tx_min, ty_min = self._global_to_tile_coords(x_min, y_min)
        tx_max, ty_max = self._global_to_tile_coords(x_max - 1, y_max - 1)

        tile_tasks = [(tx, ty, frame_cur, depth_map, sharpness_map, H, self.tile_size)
                    for ty in range(ty_min, ty_max + 1)
                    for tx in range(tx_min, tx_max + 1)]

        if not tile_tasks:
            print("Warning: No tiles were scheduled for processing!")

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_tile, *args) for args in tile_tasks]

            # Ensure all tasks finish before continuing
            for future in futures:
                future.result()

    def get_full_map(self):
        """Reconstructs the full stitched map from all tiles."""
        with self.lock:
            if not self.tiles:
                print("No tiles found! Returning empty map.")
                return np.zeros((self.tile_size, self.tile_size, self.channels), dtype=np.uint8)

            all_coords = list(self.tiles.keys())

        tx_min, tx_max = min(c[0] for c in all_coords), max(c[0] for c in all_coords)
        ty_min, ty_max = min(c[1] for c in all_coords), max(c[1] for c in all_coords)

        big_width, big_height = (tx_max - tx_min + 1) * self.tile_size, (ty_max - ty_min + 1) * self.tile_size
        big_map = np.zeros((big_height, big_width, self.channels), dtype=np.uint8)

        with self.lock:
            for (tx, ty), tile_data in self.tiles.items():
                offset_x, offset_y = (tx - tx_min) * self.tile_size, (ty - ty_min) * self.tile_size
                big_map[offset_y:offset_y + self.tile_size, offset_x:offset_x + self.tile_size] = tile_data['image']

        return big_map

    def compute_relative_odometry(self, odom_matrix_prev, odom_matrix_cur, image_width, image_height, scale):
        # Compute relative transform
        relative_transform = solve(odom_matrix_prev, odom_matrix_cur)

        # Extract translation and rotation
        tx = relative_transform[1, 3] * scale  # X-direction (forward/backward)
        ty = relative_transform[0, 3] * scale  # Y-direction (left/right)
        cos_theta = relative_transform[0, 0]
        sin_theta = relative_transform[1, 0]

        # Camera offset (only X matters, forward direction)
        camera_offset_x = -0.2286  # Camera is in front (-X direction)

        # Adjust translation based on camera offset
        tx += -sin_theta * camera_offset_x * scale  # Corrected sign
        ty += sin_theta * camera_offset_x * scale  # Corrected sign

        # Transformations to bottom center
        to_bottom_center = np.array([
            [1, 0, -image_width / 2],
            [0, 1, -image_height],
            [0, 0, 1]
        ])

        # Updated relative homography including camera offset
        relative_homography = np.array([
            [cos_theta, sin_theta, -tx],
            [-sin_theta, cos_theta, -ty],
            [0, 0, 1]
        ])

        # Transform back to the original image coordinates
        to_original = np.array([
            [1, 0, image_width / 2],
            [0, 1, image_height],
            [0, 0, 1]
        ])

        return to_original @ relative_homography @ to_bottom_center

    def warp_image(self, frame, H):
        """Warp an image using a homography matrix in PyTorch."""
        b, c, h, w = frame.shape  # Ensure shape is [batch, channels, height, width]

        # Create normalized grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                            torch.linspace(-1, 1, w, device=self.device),
                            indexing="ij")

        ones = torch.ones_like(x)
        grid = torch.stack([x, y, ones], dim=-1).reshape(-1, 3, 1)  # [h*w, 3, 1]

        # Apply homography
        H = H.to(self.device)
        warped_coords = (H @ grid).squeeze(-1)  # [h*w, 3]
        warped_coords = warped_coords[:, :2] / warped_coords[:, 2:3]  # Normalize

        # Reshape to [h, w, 2] and flip for (x, y)
        warped_coords = warped_coords.view(h, w, 2).flip(-1)

        # Perform sampling
        warped_image = F.grid_sample(frame, warped_coords.unsqueeze(0), align_corners=True, mode="bilinear")
        
        return warped_image

    def ssd_loss(self, H, frame_cur):
        """Computes SSD loss using multiple past frames."""
        if isinstance(frame_cur, np.ndarray):
            frame_cur = torch.tensor(frame_cur, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)

        H = H.reshape(3, 3)
        warped = self.warp_image(frame_cur, H)  # [1, C, H, W]

        total_loss = 0
        for frame_prev in self.frame_history:
            frame_prev_tensor = torch.tensor(frame_prev, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
            total_loss += torch.sum((warped - frame_prev_tensor) ** 2)

        return total_loss / len(self.frame_history)  # Normalize loss

    def refine_homography(self, H_init, frame_cur):
        """Optimizes the homography matrix using past frames."""
        H_opt = torch.tensor(H_init, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([H_opt], lr=0.1)

        def closure():
            optimizer.zero_grad()
            loss = self.ssd_loss(H_opt, frame_cur)
            loss.backward()
            return loss

        optimizer.step(closure)
        return H_opt.detach().cpu().numpy()


    def process_frame(self, frame_cur, timestep, odom_data=None, scale=100):
        """
        1. Compute relative odometry -> get H_relative
        2. Refine homography
        3. Compose with self.H_old
        4. Warp into tile map
        5. Optional: visualize
        """
        translation_threshold = 0.006
        rotation_threshold = 0.006

        if self.H_old is None:
            # First frame: set identity transform, etc.
            self.H_old = np.eye(3, dtype=np.float32)
            self.frame_previous = frame_cur
            self.odom_previous = odom_data
            self.frame_history.append(frame_cur)
            print(f"Initialized tile-based global map at timestep {timestep}")
            return

        if odom_data is not None and self.odom_previous is not None:
            # Solve for the relative transformation instead of inverting the matrix
            relative_transform = solve(self.odom_previous, odom_data)  # Faster than np.linalg.inv() @ odom_data

            # Compute translation and rotation differences
            tx, ty = relative_transform[0, 3], relative_transform[1, 3]
            translation_distance = np.hypot(tx, ty)  # Faster Euclidean distance
            rotation_angle = np.arctan2(relative_transform[1, 0], relative_transform[0, 0])  # Faster rotation calc
            
            # Skip frame if below movement threshold
            if translation_distance < translation_threshold and abs(rotation_angle) < rotation_threshold:
                print(f"Skipping timestep {timestep}: No significant movement (translation={translation_distance:.3f}, rotation={rotation_angle:.3f})")
                return
            H_relative = self.compute_relative_odometry(
                self.odom_previous, odom_data, frame_cur.shape[1], frame_cur.shape[0], scale
            )
        else:
            print(f"Missing odometry data at timestep {timestep}")
            return
        
        H_refined = self.refine_homography(H_relative, frame_cur)
        self.H_old = self.H_old @ H_refined
        self.H_old /= self.H_old[2,2]

        # Warp into tile map
        self.warp_frame_into_tiles(frame_cur, self.H_old)

        # Update references
        self.odom_previous = odom_data
        self.frame_previous = frame_cur
        self.frame_history.append(frame_cur)

        # Display full map (Only every 100 frames)
        if self.visualize:
            big_map = self.get_full_map()
            if big_map is not None and big_map.size > 0:
                cv2.namedWindow("Stitched Map", cv2.WINDOW_NORMAL)
                cv2.imshow("Stitched Map", big_map)
                cv2.waitKey(10)

class MapViewer:
    def __init__(self, global_map):
        self.global_map = global_map
        self.view_window = self.get_non_background_bounding_box()
        self.dragging = False
        self.start_x = 0
        self.start_y = 0
        self.zoom_factor = 1.2
        self.scale = 1.0  # Start with 100% scale

    def get_non_background_bounding_box(self):
        """
        Calculate the bounding box of the non-background area in the global map.
        It dynamically detects the most common color (background) and removes it.
        """
        if len(self.global_map.shape) == 2:  # Already grayscale
            gray_map = self.global_map
        else:
            gray_map = cv2.cvtColor(self.global_map, cv2.COLOR_BGR2GRAY)


        # Detect most frequent color (background color)
        pixel_counts = Counter(gray_map.flatten())
        background_color = max(pixel_counts, key=pixel_counts.get)

        # Find pixels that are NOT the background color
        non_background_pixels = np.argwhere(gray_map != background_color)

        if non_background_pixels.size == 0:
            return (0, 0, min(1920, self.global_map.shape[1]), min(1280, self.global_map.shape[0]))

        # Bounding box calculation
        y_min, x_min = non_background_pixels.min(axis=0)
        y_max, x_max = non_background_pixels.max(axis=0)

        # Add some margin for better visibility
        margin = 50
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(self.global_map.shape[1], x_max + margin)
        y_max = min(self.global_map.shape[0], y_max + margin)

        width = min(x_max - x_min, 1920)
        height = min(y_max - y_min, 1280)

        return (x_min, y_min, width, height)

    def update_view(self, dx, dy):
        """ Moves the view based on dragging. """
        x, y, w, h = self.view_window
        x = np.clip(x - dx, 0, self.global_map.shape[1] - w)
        y = np.clip(y - dy, 0, self.global_map.shape[0] - h)
        self.view_window = (x, y, w, h)

    def resize_view(self, dx, dy):
        """ Resizes the viewport while keeping it inside the image bounds. """
        x, y, w, h = self.view_window
        img_h, img_w, _ = self.global_map.shape

        if self.resize_mode == "right":
            w = np.clip(w + dx, 100, img_w - x)  # Minimum width = 100
        elif self.resize_mode == "bottom":
            h = np.clip(h + dy, 100, img_h - y)  # Minimum height = 100
        elif self.resize_mode == "corner":
            w = np.clip(w + dx, 100, img_w - x)
            h = np.clip(h + dy, 100, img_h - y)

        self.view_window = (x, y, w, h)

    def zoom(self, zoom_in, cursor_x, cursor_y):
        """
        Zooms in or out while keeping the viewport size fixed but adjusting the scale.
        """
        prev_scale = self.scale

        if zoom_in:
            self.scale *= self.zoom_factor
        else:
            self.scale /= self.zoom_factor

        self.scale = max(0.1, min(self.scale, 5.0))  # Prevent extreme zooming

    def save_current_view(self, filename="current_view.png"):
        """ Saves the currently visible section of the map. """
        x, y, w, h = self.view_window
        view = self.global_map[y:y + h, x:x + w]
        cv2.imwrite(filename, view)
        print(f"Saved current view as {filename}")

    def save_full_map(self, filename="full_map.png"):
        """ Saves the full map image. """
        cv2.imwrite(filename, self.global_map)
        print(f"Saved full map as {filename}")

    def show_map(self):
        """Displays the interactive map."""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.start_x, self.start_y = x, y
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                dx = x - self.start_x
                dy = y - self.start_y
                self.start_x, self.start_y = x, y
                self.update_view(dx, dy)
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False
            elif event == cv2.EVENT_MOUSEWHEEL:
                self.zoom(flags > 0, x, y)

        cv2.namedWindow("Scrollable Map")
        cv2.setMouseCallback("Scrollable Map", mouse_callback)

        while True:
            x, y, w, h = self.view_window

            # Resize the global map to simulate zooming out while keeping the viewport the same
            scaled_map = cv2.resize(self.global_map, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            # Extract the viewport from the resized image
            if len(scaled_map.shape) == 2:  # Grayscale image (no channel dimension)
                scaled_h, scaled_w = scaled_map.shape
            else:
                scaled_h, scaled_w, _ = scaled_map.shape
            x = np.clip(int(x * self.scale), 0, max(0, scaled_w - w))
            y = np.clip(int(y * self.scale), 0, max(0, scaled_h - h))

            view = scaled_map[y:y + h, x:x + w]
            cv2.imshow("Scrollable Map", view)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.zoom(True, w // 2, h // 2)  # Zoom in at center
            elif key == ord('-'):
                self.zoom(False, w // 2, h // 2)  # Zoom out at center

        cv2.destroyAllWindows()

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

    robot_data = RobotDataAtTimestep(os.path.join(script_dir, "../bags/panther_recording_20250218_175547/panther_recording_20250218_175547_synced.pkl"))
    #robot_data = RobotDataAtTimestep(os.path.join(script_dir, "../bags/panther_recording_sim_loop_2/panther_recording_sim_loop_2_synced.pkl"))

    scale_start = 490
    bev_image_prev = cv2.warpPerspective(robot_data.getImageAtTimestep(scale_start), H, dsize)
    bev_image_cur = cv2.warpPerspective(robot_data.getImageAtTimestep(scale_start+1), H, dsize)
    meters_to_pixels = calculate_meters_to_pixels(bev_image_prev, bev_image_cur,
                                                  robot_data.getOdomAtTimestep(scale_start),
                                                  robot_data.getOdomAtTimestep(scale_start+1))
    #522 for sim
    #261 for actual
    # Initialize the global map
    global_map = GlobalMap(visualize=True)
    start, end = 0, 4000

    # Check if an image path is provided
    image_path = None#"full_map.png"  # Change this to None if no image path is given

    if image_path:  # Show image if path exists
        clean_map = cv2.imread(image_path)
        if clean_map is not None:
            viewer = MapViewer(clean_map)
            viewer.show_map()
    else:
        # Process subsequent BEV images
        for timestep in tqdm(range(start+1, end), desc="Processing patches at timesteps"):
            try:
                cur_img = robot_data.getImageAtTimestep(timestep)
                cur_rt = robot_data.getOdomAtTimestep(timestep)
                if cur_img is None:
                    print(f"Missing image data at timestep {timestep}")
                    continue

                bev_img = cv2.warpPerspective(cur_img, H, dsize)  # Create BEV image
                #global_map.process_frame(bev_img, timestep, odom_data=cur_rt, scale=meters_to_pixels)
                global_map.process_frame(bev_img, timestep, odom_data=cur_rt, scale=522)

            except Exception as e:
                print(f"Error at timestep {timestep}: {e}")
                continue

        # Retrieve and save the final stitched map
        final_map = global_map.get_full_map()

        if final_map is None or final_map.size == 0:
            print("Error: Final map is empty!")
        else:
            print("Final map successfully retrieved.")

        # Pass `final_map` to `MapViewer`, NOT `clean_map`
        viewer = MapViewer(final_map)
        viewer.show_map()
        viewer.save_full_map()
# Add Z buffer, so that pixels off in the distance are replaced with the most recent. 
# We can get pixel and depth of the pixel using homography, once we reach that pixel, replace that portion of the image with most current

# Interpolate between IMU homography
#SUm of squared differences of the pixel intensites between images, use minimization function to minimize the difference in pixel intensities
# This would prevent overlap between sequential images