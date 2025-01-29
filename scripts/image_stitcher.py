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
from scipy.spatial.transform import Rotation as R
from numba import njit
from scipy.optimize import minimize
import torch
import torch.nn.functional as F
import math

import math
import numpy as np
import cv2

class TiledMapper:
    def __init__(self, tile_size=1024, channels=3):
        """
        tile_size : Size in pixels for each tile (square).
        channels  : Number of channels (e.g., 3 for RGB).
        """
        self.tile_size = tile_size
        self.channels = channels
        # Dictionary: (tile_x, tile_y) -> {'image': np.ndarray, 'z_buffer': np.ndarray}
        self.tiles = {}  
    
    def _get_or_create_tile(self, tile_x, tile_y):
        """Return tile at (tile_x, tile_y), create if it doesn't exist."""
        if (tile_x, tile_y) not in self.tiles:
            tile_img = np.zeros((self.tile_size, self.tile_size, self.channels), dtype=np.uint8)
            tile_zb = np.full((self.tile_size, self.tile_size), -np.inf, dtype=np.float32)
            self.tiles[(tile_x, tile_y)] = {'image': tile_img, 'z_buffer': tile_zb}
        return self.tiles[(tile_x, tile_y)]

    def _global_to_tile_coords(self, x, y):
        """
        Convert a global pixel coordinate (x,y) into
        integer tile indices (tile_x, tile_y) and local offsets within that tile.
        """
        tile_x = x // self.tile_size
        tile_y = y // self.tile_size
        return tile_x, tile_y

    def get_transformed_corners(self, frame_cur, H):
        """Same as your existing function, or see below usage."""
        h, w = frame_cur.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(corners, H)

    def warp_frame_into_tiles(self, frame_cur, H, depth):
        """
        Warp the current frame into all overlapping tiles. 
        Only updates tiles that the warped frame touches.
        """
        # 1. Compute corners in global space
        corners = self.get_transformed_corners(frame_cur, H)
        xs = corners[:, 0, 0]
        ys = corners[:, 0, 1]
        
        x_min, x_max = math.floor(xs.min()), math.ceil(xs.max())
        y_min, y_max = math.floor(ys.min()), math.ceil(ys.max())
        
        if x_min >= x_max or y_min >= y_max:
            return  # No valid region

        # 2. Figure out which tiles we need to update
        tx_min, ty_min = self._global_to_tile_coords(x_min, y_min)
        tx_max, ty_max = self._global_to_tile_coords(x_max - 1, y_max - 1)
        
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                tile_data = self._get_or_create_tile(tx, ty)
                tile_img = tile_data['image']
                tile_zb = tile_data['z_buffer']
                
                # 3. Compute offset homography to warp into this tile's local coordinate system
                tile_origin_x = tx * self.tile_size
                tile_origin_y = ty * self.tile_size
                T_offset = np.array([
                    [1, 0, -tile_origin_x],
                    [0, 1, -tile_origin_y],
                    [0, 0, 1]
                ], dtype=np.float32)
                H_tile = T_offset @ H

                # 4. Warp the frame into the size of the tile
                warped_img = cv2.warpPerspective(
                    frame_cur,
                    H_tile,
                    (self.tile_size, self.tile_size),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                
                # 5. Mask + z-buffer checks
                mask = (frame_cur > 0).any(axis=2).astype(np.uint8)
                warped_mask = cv2.warpPerspective(
                    mask,
                    H_tile,
                    (self.tile_size, self.tile_size),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                
                valid_indices = (warped_mask == 1)
                depth_mask = (depth > tile_zb)
                update_mask = valid_indices & depth_mask
                
                # For color images, broadcast update_mask to shape (H,W,3)
                if tile_img.ndim == 3:
                    update_mask_3d = np.repeat(update_mask[:, :, np.newaxis], 3, axis=2)
                    tile_img[update_mask_3d] = warped_img[update_mask_3d]
                else:
                    tile_img[update_mask] = warped_img[update_mask]
                
                tile_zb[update_mask] = depth  # update depth

    def get_full_map(self):
        """
        Reconstruct a large mosaic from all existing tiles.
        Useful for occasional visualization or saving to disk.
        """
        if not self.tiles:
            return None

        all_coords = list(self.tiles.keys())
        txs = [c[0] for c in all_coords]
        tys = [c[1] for c in all_coords]
        tx_min, tx_max = min(txs), max(txs)
        ty_min, ty_max = min(tys), max(tys)
        
        # Overall width/height in tiles
        tile_range_x = tx_max - tx_min + 1
        tile_range_y = ty_max - ty_min + 1
        
        big_width = tile_range_x * self.tile_size
        big_height = tile_range_y * self.tile_size
        
        big_map = np.zeros((big_height, big_width, self.channels), dtype=np.uint8)
        
        for (tx, ty), tile_data in self.tiles.items():
            tile_img = tile_data['image']
            offset_x = (tx - tx_min) * self.tile_size
            offset_y = (ty - ty_min) * self.tile_size
            big_map[offset_y:offset_y+self.tile_size, offset_x:offset_x+self.tile_size] = tile_img
        
        return big_map


class GlobalMap:
    def __init__(self, tile_size=1024, visualize=True):
        self.visualize = visualize

        # Instead of one monolithic output_img:
        self.tiled_mapper = TiledMapper(tile_size=tile_size, channels=3)

        self.H_old = None
        self.frame_previous = None
        self.odom_previous = None

        # If you still want frame history for homography refinement:
        self.frame_history = deque(maxlen=3)

    def get_map(self):
        """
        Retrieve the current global map.
        Returns:
            np.ndarray: The current global map.
        """
        return self.tiled_mapper.get_full_map()

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

    def process_frame(self, frame_cur, timestep, odom_data=None, scale=100):
        """
        1. Compute relative odometry -> get H_relative
        2. Refine homography
        3. Compose with self.H_old
        4. Warp into tile map
        5. Optional: visualize
        """
        translation_threshold = 0.01
        rotation_threshold = 0.01

        if self.H_old is None:
            # First frame: set identity transform, etc.
            self.H_old = np.eye(3, dtype=np.float32)
            self.frame_previous = frame_cur
            self.odom_previous = odom_data
            self.frame_history.append(frame_cur)
            print(f"Initialized tile-based global map at timestep {timestep}")
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

        # Optionally refine H_relative using your SSD approach with self.frame_history
        # Let's call that H_refined:
        H_refined = self.refine_homography_with_ssd(frame_cur, list(self.frame_history), H_relative)

        # Update global homography
        self.H_old = self.H_old @ H_refined
        self.H_old /= self.H_old[2,2]

        # Warp into tile map
        self.warp(frame_cur, self.H_old, depth=timestep)

        # Update references
        self.odom_previous = odom_data
        self.frame_previous = frame_cur
        self.frame_history.append(frame_cur)

        # Visualize only every 100 steps
        if self.visualize:
            # 1. Reconstruct a single large map from all tiles
            big_map = self.tiled_mapper.get_full_map()
            if big_map is not None:
                cv2.namedWindow('output', cv2.WINDOW_NORMAL)
                cv2.imshow('output', big_map)
                cv2.waitKey(1)

    def warp(self, frame_cur, H, depth):
        """
        Warp the current frame into the tile map using the TiledMapper.
        """
        self.tiled_mapper.warp_frame_into_tiles(frame_cur, H, depth)

    def refine_homography_with_ssd(self, frame_cur, prev_frames, H_initial, patch_size=32):
        """
        Refine the homography matrix using SSD loss across the last 3 frames.
        
        Args:
            frame_cur (np.ndarray): Current BEV frame (H, W, C).
            prev_frames (list of np.ndarray): List of last 3 previous frames (each H, W, C).
            H_initial (np.ndarray): Initial homography matrix.
            patch_size (int): Size of patches for SSD computation.
        
        Returns:
            np.ndarray: Refined homography matrix.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert current frame and all previous frame to PyTorch tensor
        frame_cur_tensor = torch.tensor(frame_cur.transpose(2, 0, 1), dtype=torch.float32, device=device) / 255.0  # (C, H, W)
        prev_frames_tensor = [
            torch.tensor(f.transpose(2, 0, 1), dtype=torch.float32, device=device) / 255.0 for f in prev_frames
        ]  # List of (C, H, W) tensors

        # Retrieve height and width
        _, H, W = frame_cur_tensor.shape

        # Cache a coordinate grid once (normalized [-1,1] range)
        #    shape of coords => (3, H*W)
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'  # For PyTorch 1.10+; omit if on older PyTorch
        )
        coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1)
        coords = coords.view(-1, 3).T  # (3, H*W)
        
        # A local function to warp with a cached grid
        def warp_with_homography_cached(image, H_mat):
            """
            Warp 'image' using homography 'H_mat', reusing the cached 'coords'.
            image shape: (C, H, W)
            H_mat shape: (3, 3)
            returns: warped image (C, H, W)
            """
            # Transform the base coords => (3, H*W)
            warped_coords = H_mat @ coords
            # Normalize by z
            warped_coords = warped_coords / warped_coords[2, :]
            # Drop z => (2, H*W)
            warped_coords = warped_coords[:2, :].T.view(H, W, 2)  # (H, W, 2)
            
            # grid_sample expects shape (N, H, W, 2)
            warped_coords = warped_coords.unsqueeze(0)  # (1, H, W, 2)

            # Add batch dimension to image => (1, C, H, W)
            image_for_sampling = image.unsqueeze(0)

            # Sample
            warped = F.grid_sample(
                image_for_sampling,
                warped_coords,
                mode='bilinear',
                align_corners=True,
                padding_mode='zeros'
            )
            return warped.squeeze(0)  # => (C, H, W)

        # 4. Efficient patchwise SSD using pooling
        def compute_patchwise_ssd_pool(a, b, patch):
            """
            Compute patchwise sum of squared differences between 'a' and 'b'.
            a, b: (C, H, W)
            """
            diff_sq = (a - b)**2
            # Use average pooling over each patch, then multiply back patch_size**2 to get the sum.
            patch_sum = F.avg_pool2d(diff_sq, kernel_size=patch, stride=patch) * (patch**2)
            return patch_sum.sum()  # scalar

        # Convert initial homography to torch (we only optimize 6 parameters => top 2 rows)
        H_initial_tensor = torch.tensor(H_initial, dtype=torch.float32, device=device)
        initial_params = H_initial_tensor[:2, :].reshape(-1).cpu().numpy()  # shape (6,)

        # 5. Define the loss function for SciPy
        def ssd_loss(params):
            """
            Given the 6 homography parameters (top two rows),
            compute average patchwise SSD over 'prev_frames_tensor'.
            """
            # Rebuild a 3x3 homography
            H_mat = torch.tensor([
                [params[0], params[1], params[2]],
                [params[3], params[4], params[5]],
                [0.0,       0.0,       1.0]
            ], device=device, dtype=torch.float32)

            total_loss = 0.0
            for prev_frame_tensor in prev_frames_tensor:
                # Warp current frame to the "prev_frame" coordinates
                warped_cur = warp_with_homography_cached(frame_cur_tensor, H_mat)
                # Compute patchwise SSD
                total_loss += compute_patchwise_ssd_pool(prev_frame_tensor, warped_cur, patch_size)

            # Return the average as a Python float
            avg_loss = total_loss / len(prev_frames_tensor)
            return avg_loss.item()

        # 6. Use scipy to minimize
        result = minimize(ssd_loss, initial_params, method='L-BFGS-B', options={"maxiter": 100})
        refined_params = result.x

        # 7. Rebuild the full 3x3 homography from the refined params
        refined_H_tensor = torch.tensor([
            [refined_params[0], refined_params[1], refined_params[2]],
            [refined_params[3], refined_params[4], refined_params[5]],
            [0.0,               0.0,               1.0]
        ], device=device, dtype=torch.float32)

        # Return as numpy on CPU
        return refined_H_tensor.cpu().numpy()

class MapViewer:
    def __init__(self, global_map):
        self.global_map = global_map
        self.view_window = self.get_non_black_bounding_box()
        self.dragging = False
        self.start_x = 0
        self.start_y = 0
        self.zoom_factor = 1.2
        self.scale = 1.0  # Start with 100% scale

    def get_non_black_bounding_box(self):
        """
        Calculate the bounding box of the non-black area in the global map.
        """
        gray_map = cv2.cvtColor(self.global_map, cv2.COLOR_BGR2GRAY)
        non_black_pixels = np.argwhere(gray_map > 0)

        if non_black_pixels.size == 0:
            return (0, 0, min(1920, self.global_map.shape[1]), min(1280, self.global_map.shape[0]))

        y_min, x_min = non_black_pixels.min(axis=0)
        y_max, x_max = non_black_pixels.max(axis=0)

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


def remove_lines(image, 
                canny_threshold1=25, 
                canny_threshold2=125,
                hough_threshold=40,
                min_line_length=75,
                max_line_gap=5,
                line_thickness=2,
                inpaint_radius=1):
    """
    Detects lines of any orientation using Canny + Hough transform, 
    then inpaints them out of the original image.

    Args:
        image (np.ndarray): BGR or grayscale input image.
        canny_threshold1 (int): Lower threshold for Canny edge detection.
        canny_threshold2 (int): Upper threshold for Canny edge detection.
        hough_threshold (int): Minimum number of intersecting 'votes' 
                               to detect a line in Hough space.
        min_line_length (int): Minimum length of a line to accept (in pixels).
        max_line_gap (int): Max allowed gap between segments to link them (HoughLinesP).
        line_thickness (int): Thickness (in pixels) used to draw lines on the mask.
        inpaint_radius (int): Radius for OpenCV inpainting.

    Returns:
        np.ndarray: The input image with detected lines removed via inpainting.
    """
    
    # 1. Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 2. Detect edges
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # 3. Run Hough line transform (probabilistic version HoughLinesP)
    lines = cv2.HoughLinesP(edges, 
                            1, 
                            np.pi / 180, 
                            threshold=hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    if lines is None:
        # No lines found => return original
        return image

    # 4. Create a mask of detected lines
    line_mask = np.zeros_like(gray)  # single channel, same size

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw white lines on black mask
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=line_thickness)

    # 5. Inpaint lines in the original color image
    #    If the image is grayscale, inpainting will also be grayscale
    if len(image.shape) == 2:  
        # grayscale inpaint
        inpainted = cv2.inpaint(image, line_mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        # color BGR inpaint
        inpainted = cv2.inpaint(image, line_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return inpainted


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
    bev_image_prev = cv2.warpPerspective(robot_data.getImageAtTimestep(scale_start), H, dsize)
    bev_image_cur = cv2.warpPerspective(robot_data.getImageAtTimestep(scale_start+1), H, dsize)

    meters_to_pixels = calculate_meters_to_pixels(bev_image_prev, bev_image_cur,
                                                  robot_data.getOdomAtTimestep(scale_start),
                                                  robot_data.getOdomAtTimestep(scale_start+1))
    # Initialize the global map
    global_map = GlobalMap(tile_size = 1024, visualize=False)

    start = 0
    end = 4000
    #robot_data.getNTimesteps()

    # Check if an image path is provided
    image_path = "full_map.png"  # Change this to None if no image path is given

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
                    global_map.process_frame(bev_img, timestep, odom_data=cur_rt, scale=meters_to_pixels)

                except Exception as e:
                    print(f"Error at timestep {timestep}: {e}")
                    continue

            # Retrieve and save the final stitched map
            final_map = global_map.get_map()
            clean_map = remove_lines(final_map)
            # Ensure the map is properly scaled for viewing
            viewer = MapViewer(clean_map)
            viewer.show_map()
            viewer.save_full_map()
# Add Z buffer, so that pixels off in the distance are replaced with the most recent. 
# We can get pixel and depth of the pixel using homography, once we reach that pixel, replace that portion of the image with most current

# Interpolate between IMU homography
#SUm of squared differences of the pixel intensites between images, use minimization function to minimize the difference in pixel intensities
# This would prevent overlap between sequential images